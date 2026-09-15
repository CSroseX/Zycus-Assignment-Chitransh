# Zycus Bookable Payable Pipeline

## How to Run

### Full folder run (the submission path)

```
python main.py documents/
```

This reads every PDF in `documents/`, processes each one, and writes one JSON file per PDF into `output/`. When it finishes, you will see a summary line with counts of payables extracted and segments declined. The `output/` folder will contain files like `INV-01.json`, `DU-02.json`, etc., one per input PDF.

### Single file run (for testing)

```
python run_single.py INV-01
```

This takes a document stem (no `.pdf` extension needed) and runs it through classification, LLM extraction, grounding, and ERP booking check with verbose output. Useful for debugging one file at a time. It reads from `parsed_files/` (the cached OCR text), not from `documents/` directly.

You can also limit `main.py` to specific files:

```
python main.py documents/ --only INV-01,INV-02
```

### Setup

Python 3.10 or later. Install dependencies:

```
pip install -r requirements.txt
```

The system needs at least one LLM API key in a `.env` file at the project root. It tries providers in this order: OpenRouter, Groq, Cloudflare Workers AI, Gemini. If one fails or hits a rate limit, it falls through to the next. The `.env` file should have one or more of:

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
OPEN_ROUTER_API=...
CLOUDFLARE_WORKERS_AI=...
```

The `master_data/` folder (suppliers, chart of books, tax master, payment terms, PO master) must be present. It ships with the repo.

### What "done" looks like

The `output/` folder contains one JSON file per input PDF. Each file has this shape:

```json
{
  "file": "INV-01.pdf",
  "payables": [ ... ],
  "declined": [ ... ]
}
```

`payables` contains one entry per bookable payable found in that PDF. `declined` contains one entry per non-payable segment (reminders, estimates, delivery notes, etc.) with a reason string.


## Decision Criteria

### How it decides whether something is a payable

The classifier runs before any LLM call. It scores the document text using a weighted point system. Positive points come from: having an invoice or credit note title in the first 1200 characters (+40), having table-like row structures with decimal amounts (+25), having tax identifiers like VAT or MwSt (+15), having buyer/seller party labels (+10), having total/subtotal keywords (+10). Negative points come from: having a non-payable title like "estimate," "mahnung," "delivery note," "quotation," or "donations and charitable contributions" (-100).

If the score is 40 or above, the segment is classified as payable and sent to the LLM for extraction. If below 40, it goes into `declined[]` with its document type (REMINDER_STATEMENT, ESTIMATE_QUOTE, DELIVERY_NOTE, etc.) and the scoring breakdown as the reason.

Credit memos are a special case. They are payable, not declined. The classifier checks for credit note keywords (credit note, gutschrift, kreeditarve, nota de credito, avoir) and if found alongside a passing score, sets the type to CREDIT_MEMO. The schema is identical to an invoice. All amounts are extracted as positive magnitudes.

### How it decides one payable versus several

Before classification, the segmenter splits the OCR text on page breaks. It looks at the first few lines of each page for signals of a new sub-document starting: a "Page 1 of N" restart, or a distinct sub-document title (consolidated invoice, delivery note, etc.). It also extracts invoice/PO/SO numbers from page headers and checks whether consecutive pages share the same document identifier. If they do, they are continuation pages of the same document. If a page shows "Page 2 of 3" or similar, it is treated as a continuation, not a new start.

Each resulting segment is classified and extracted independently. A single PDF with two invoices stapled together produces two entries in `payables[]`.

### How it places taxes at line level versus header level

The LLM prompt instructs the model to look at where the tax appears on the document. If tax is printed per line item (each line showing its own tax rate or amount), those go into `line_items[].taxes[]`. If tax is printed as a summary section at the bottom of the document (one block listing rates and totals), those go into the header `taxes[]` array.

After extraction, a deduplication pass checks for conflicts. If the same tax information appears in both places (because the LLM duplicated it), the system resolves the conflict. If line items have multiple different tax rates, it keeps them at the line level and clears the header taxes. If the sum of line totals already equals gross_total (meaning line totals are tax-inclusive), it removes the header taxes to prevent double-counting. This distinction matters because the ERP applies header taxes against the net base of all lines combined, while line taxes apply against each individual line's base. Placing a tax in the wrong location changes the calculated gross even if the rate and amount are identical.

### How it decides whether to fill in a master-data code

Each master-data field is resolved against its reference file with specific matching logic.

**Supplier ID**: First tries an exact match on VAT ID (cleaned and uppercased). If that fails, tries an exact match on supplier name (lowercased, whitespace-normalized). If that also fails, uses SequenceMatcher similarity against all known supplier names. The threshold is 85%. Below that, it returns blank. There is no "close enough" zone. It either clears 85% or the field stays empty.

**Buyer codes** (company_code, business_unit_code, location_code): The system tokenizes the document text and the master data's business unit names, location names, and addresses. It picks the record with the highest token overlap. If no tokens overlap at all, it falls back to the first entry in the chart of books.

**PO ID**: Exact match of the cleaned PO number against `po_master.json`. No fuzzy matching.

**Payment term ID**: Tries alias matching first (checking if known payment term text aliases appear in the document text). If that fails, computes the day difference between invoice_date and due_date and looks for a matching term by days. If neither works, blank.

**Tax type code**: Matches by country code (derived from the supplier's VAT ID prefix) and tax rate against `tax_master.json`. If no country-specific match, falls back to matching rate alone across all countries.

In all cases, the raw printed value stays in its own field (supplier name, PO number, tax rate). The resolved code goes in the `_id` or `_code` field. If there is no match, the code field is an empty string. It never guesses.

### What it does not do

The system never derives a missing value by working backward from a total. If a line item has a total printed but no unit price, the unit price field stays blank. It does not divide total by quantity to fill it in. If a quantity is missing but a total and price are present, the quantity stays blank. This is not a style preference. The grounding rule says every emitted value must appear on the document. A number computed just to balance the arithmetic did not appear on the document. Emitting it would mean the payable contains an invented figure, which disqualifies it under the grading rules even if the ERP total happens to come out right.

The grounding verifier enforces this after extraction. It checks every numeric field (invoice number, gross total, subtotal, quantities, unit prices, tax amounts) against the raw OCR text. If a value cannot be found in the text in any reasonable format (dot-decimal, comma-decimal, integer part), it is blanked out and a warning is logged.

### Where it falls short

Tax placement is still the weakest area. On documents where the same tax information is presented both per-line and in a summary block, the deduplication heuristics sometimes make the wrong call about which location to keep, which causes the ERP recompute to diverge from the stated gross. Multi-page invoices with complex table continuations and mixed languages occasionally lose line items at page boundaries.
