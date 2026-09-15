# DESIGN.md

## 1. Design Questions

### What did I understand on day three that I did not understand on day one

At the start, the task looked like extraction. Read the fields, fill the record, done. That approach books maybe a third of documents and then stops working, and the failures do not look related to each other.

The real problem is different. A record can copy the document faithfully, line by line, and still fail to book, because the ERP checks structure, not just the total. Two records can reach the same gross total through different paths, and only one of them is correct. A tax stated on a line has to stay on the line. A tax stated once at the header has to stay at the header. Collapsing three different line rates into one blended header rate produces the same total but is still wrong.

The harder lesson came from watching my own process fail in a specific, repeatable way. More than once, a fix got proposed by trying a value, checking if erp_book() matched, adjusting, and checking again. That shape is disqualifying regardless of how reasonable the final number looks, because the value was chosen to satisfy the oracle, not because the document stated it. This happened with a proposed unit_price backward-derived from a total, with a fabricated quantity on a document where quantity was genuinely unprinted, and with a test script that injected discount and freight values specifically to make totals match. Each time, the fix looked plausible until traced back to an actual quoted line in the document, and the trace failed.

By day three the operating rule was that every value in the output has to be traceable to an exact quoted fragment of the document. If a number only exists because it makes the total balance, it does not belong in the output, no matter how close it gets.

### What the system does when it meets a document unlike anything it has seen, and why that generalizes instead of guessing

The system applies one consistent rule across every document. Extract only what is printed, resolve master data codes only where there is a genuine match, and leave a field blank when the document does not state it. A blank field is treated as a correct, honest answer. A guessed field is treated as a wrong answer even if it happens to make the total balance.

This is why the system generalizes. It is not trying to recognize a document type and apply a matching template. It is applying the same grounding check to every document, seen or unseen. Does this value appear on the page, in this exact form, or does it not. That check does not depend on having seen the document's shape before.

### A document I concluded could not be solved the way the others were

I want to be honest here rather than overstate this with a specific example I have not fully verified against the source page. Across the assignment, I ran into a meaningful number of documents that resisted the same treatment as the rest, for a mix of reasons. Some had values that were genuinely not printed anywhere on the page, forcing a choice between leaving a field blank and derivi ng a number just to make the total balance. Some had structural ambiguity, like tax or charges that could plausibly sit at either the line level or the header level, or duplicated figures across multiple pages of the same document. Some had raw OCR misreads on key digits that were hard to distinguish from a genuinely different printed value without very close inspection. And in at least one case, arithmetic alone did not tell me whether the document simply had a discrepancy or whether my extraction had missed something, since erp.py only reports the final gross and nothing about where or why it diverges.

The consistent choice across all of these was the same. Where a value was not clearly grounded in something printed on the document, I left it blank rather than deriving it to satisfy the oracle, even when a derived value would have made the record book correctly. I would rather report an honest gap than submit a number I cannot point to on the page.


## 2. Architecture Design

Given the problem statement, the most efficient architecture for this complex and varied docs is this -> Parse -> feed to AI -> auto_draft -> erp, from an umbrella POV. But i'm going to specify in slightly more detailed below,

The system runs as a pipeline with the following phases.

1. OCR Engine. Extracts text from each PDF page, using vector text extraction first and falling back to OCR where the page is scanned or image based.

2. Spatial Layout Builder. Groups extracted text by position into readable tables, so that columns and rows can be identified independent of font size or page scale.

3. Multi-Document Pre-Segmenter. Splits a single PDF into separate sub-documents where the PDF contains more than one distinct payable, using page restart markers and reference number boundaries to decide where one document ends and another begins.

4. Payable Classifier. Evaluates each segmented section and decides whether it is a bookable payable or should be set aside in declined, based on structural signals rather than a fixed keyword list.

5. Structured Extractor. Extracts line items, taxes, discounts, and charges into the schema fields, keeping components decomposed rather than pre-summed, and leaving a field blank whenever the document does not state it.

6. Master Data Resolver. Matches extracted suppliers, tax codes, buyer codes, payment terms, and PO references against the provided master data, and returns a match only where one genuinely exists, leaving the field blank otherwise.

7. Pipeline Orchestrator. Runs all phases over the full documents folder in one command and writes one output JSON per input PDF to the output folder.