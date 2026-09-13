"""
extractor.py — Phase 4: Structured Auto-Draft Extraction via Gemini API.

Extracts structured header fields, line items, and taxes from OCR layout text
into autodraft JSON format complying strictly with AUTODRAFT_SCHEMA.md.

Includes:
- Rule 1 Grounding Verifier (src/grounding.py) to eliminate hallucinations.
- Structural Audit Verifier (tax placement & component decomposition).
- Strict ERP Discrepancy Recovery Protocol (document-grounded re-reading only).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.classifier import classify_file
from src.grounding import verify_payable_grounding
from src.ocr_engine import extract_text

load_dotenv(override=True)

# Gemini API Client
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY or "lang-client" in API_KEY or "<" in API_KEY:
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

client = genai.Client(api_key=API_KEY) if API_KEY and "<" not in API_KEY else None

SYSTEM_PROMPT = """
You are an expert financial invoice parsing system. Convert the provided document OCR layout text into a single JSON object conforming strictly to AUTODRAFT_SCHEMA.md.

STRICT EXTRACTION RULES:
1. ALL NUMBERS MUST BE DOT-DECIMAL (e.g. "1234.56", "438.00", "0.00"). Convert any German/European comma decimals ("438,00" -> "438.00").
2. unit_price MUST BE NET (tax-exclusive). Do not include tax inside unit_price.
3. For CREDIT_MEMO documents, use invoice_type: "CREDIT_MEMO" and extract ALL amounts as POSITIVE magnitudes.
4. Component Decomposition: Keep quantities, unit prices, discounts, freight, and charges separate. Never fold them together.
5. Tax Placement: If tax is printed per line item, place it in line_items[].taxes[]. If tax is printed as a single summary at bottom, place it in header taxes[].
6. Master Data Codes: Leave supplier.supplier_id, buyer.company_code, buyer.business_unit_code, buyer.location_code, payment_term_id, po_id, taxes[].tax_type_code as empty string "" unless explicitly matched.
7. Grounding: Extract ONLY values explicitly present on the document. Do not invent or guess any figures.
8. If unit_price is not explicitly printed on the document, leave unit_price as empty string "" — NEVER compute or backward-derive unit_price by dividing total by quantity.

Return ONLY a raw JSON object adhering to this schema:
{
  "invoice_number": "str",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "invoice_type": "INVOICE | CREDIT_MEMO",
  "currency": "EUR | USD | ZAR | MYR | etc",
  "supplier": {
    "name": "str",
    "supplier_id": "",
    "address": "str",
    "vat_id": "str"
  },
  "buyer": {
    "company_code": "",
    "business_unit_code": "",
    "location_code": ""
  },
  "payment_term_id": "",
  "po_number": "str",
  "po_id": "",
  "gross_total": "str",
  "subtotal": "str",
  "total_tax_amount": "str",
  "discount_amount": "str",
  "freight_charges": "str",
  "insurance_charges": "str",
  "extra_charges": "str",
  "excise_duties": "str",
  "taxes": [
    {
      "tax_type": "VAT",
      "tax_name": "str",
      "tax_rate": "str",
      "tax_amount": "str",
      "tax_type_code": ""
    }
  ],
  "line_items": [
    {
      "description": "str",
      "item_type": "GOODS | SERVICE | FREIGHT | TAX",
      "uom": "str",
      "quantity": "str",
      "unit_price": "str",
      "total": "str",
      "discount": "str",
      "discount_percentage": "str",
      "tax_rate": "str",
      "tax_amount": "str",
      "taxes": []
    }
  ]
}
"""


def verify_structural_integrity(payable: dict) -> list[str]:
    """Perform structural audit on extracted payable payload."""
    warnings = []

    # 1. Tax Placement Audit
    has_header_taxes = bool(payable.get("taxes"))
    has_line_taxes = any(bool(item.get("taxes") or item.get("tax_rate")) for item in payable.get("line_items", []))
    if has_header_taxes and has_line_taxes:
        warnings.append("STRUCTURAL NOTICE: Document contains both header-level and line-level taxes.")

    # 2. Decomposed Components Audit
    for idx, item in enumerate(payable.get("line_items", [])):
        qty = item.get("quantity", "")
        price = item.get("unit_price", "")
        tot = item.get("total", "")
        if qty and price and tot:
            try:
                q_val = float(qty)
                p_val = float(price)
                t_val = float(tot)
                calc_tot = q_val * p_val
                if abs(calc_tot - t_val) > 0.05 and not item.get("discount") and not item.get("discount_percentage"):
                    warnings.append(f"LINE [{idx}] DECOMPOSITION NOTICE: qty ({qty}) * unit_price ({price}) = {calc_tot:.2f} != total ({tot}). Possible discount/charge included.")
            except ValueError:
                pass

    return warnings


def parse_dot_decimal(val_str: str) -> str:
    """Clean and convert any decimal string (German '438,00' or '$438.00') to dot-decimal ('438.00')."""
    if not val_str:
        return ""
    cleaned = re.sub(r'[^0-9\.,\-]', '', str(val_str)).strip()
    if not cleaned:
        return ""
    if ',' in cleaned and '.' in cleaned:
        if cleaned.find(',') < cleaned.find('.'):
            cleaned = cleaned.replace(',', '')
        else:
            cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    try:
        f = float(cleaned)
        return f"{f:.2f}" if '.' in cleaned else f"{f:.0f}"
    except ValueError:
        return cleaned


def deterministic_extract_payable(ocr_text: str, filename: str = "") -> dict:
    """Deterministic spatial layout extractor (Fallback when API key is missing/quota exceeded)."""
    t = ocr_text

    # 1. Invoice Number
    inv_num = ""
    inv_match = re.search(r'(?:invoice|rechnung|arve|facture|fatura)\s*(?:nr|no|num|#|\.)*:?\s*([a-zA-Z0-9\-_]+)', t, re.IGNORECASE)
    if inv_match:
        inv_num = inv_match.group(1).strip()

    # 2. Invoice Date
    inv_date = ""
    date_match = re.search(r'(?:date|datum|kuup\u00e4ev)\s*:?\s*(\d{1,4}[\./\-]\d{1,2}[\./\-]\d{1,4})', t, re.IGNORECASE)
    if date_match:
        inv_date = date_match.group(1).strip()

    # 3. Currency
    curr = "EUR"
    if "$" in t or "USD" in t:
        curr = "USD"
    elif "ZAR" in t or "R " in t:
        curr = "ZAR"
    elif "MYR" in t or "RM" in t:
        curr = "MYR"
    elif "GBP" in t or "£" in t:
        curr = "GBP"

    # 4. Gross Total
    gross_total = ""
    gross_match = re.search(r'(?:endbetrag|gesamtsumme|total zar|total eur|total usd|amount due|grand total|total amount|total)\s*:?\s*([\$€£]?\s*[\d\.,]+)', t, re.IGNORECASE)
    if gross_match:
        gross_total = parse_dot_decimal(gross_match.group(1))

    # 5. Tax Amount & Rate
    tax_rate = ""
    tax_amt = ""
    tax_match = re.search(r'(?:vat|mwst|tax|sttax)\s*(?:at|@)?\s*(\d+[\.,]?\d*)\s*%\s*:?\s*([\$€£]?\s*[\d\.,]+)?', t, re.IGNORECASE)
    if tax_match:
        tax_rate = parse_dot_decimal(tax_match.group(1))
        if tax_match.group(2):
            tax_amt = parse_dot_decimal(tax_match.group(2))

    # 6. Line Items Extraction (spatially aligned numeric rows)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    line_items = []
    for line in lines:
        row_match = re.search(r'^(.*?)\s+(\d+(?:[\.,]\d+)?)\s+(?:std|pcs|kg|hrs|hr|unit|ea)?\s*[\$€£]?\s*([\d\.,]+)\s+[\$€£]?\s*([\d\.,]+)$', line, re.IGNORECASE)
        if row_match:
            desc = row_match.group(1).strip()
            if not any(header_word in desc.lower() for header_word in ['beschreibung', 'description', 'subtotal', 'total']):
                line_items.append({
                    "description": desc,
                    "item_type": "SERVICE" if "std" in line.lower() or "hr" in line.lower() else "GOODS",
                    "uom": "Std" if "std" in line.lower() else "Pcs",
                    "quantity": parse_dot_decimal(row_match.group(2)),
                    "unit_price": parse_dot_decimal(row_match.group(3)),
                    "total": parse_dot_decimal(row_match.group(4)),
                    "discount": "",
                    "discount_percentage": "",
                    "tax_rate": "",
                    "tax_amount": "",
                    "taxes": []
                })

    # Header taxes list
    taxes = []
    if tax_rate or tax_amt:
        taxes.append({
            "tax_type": "VAT",
            "tax_name": f"VAT {tax_rate}%" if tax_rate else "VAT",
            "tax_rate": tax_rate,
            "tax_amount": tax_amt,
            "tax_type_code": ""
        })

    payable = {
        "invoice_number": inv_num,
        "invoice_date": inv_date,
        "due_date": "",
        "invoice_type": "INVOICE",
        "currency": curr,
        "supplier": {
            "name": lines[0] if lines else "",
            "supplier_id": "",
            "address": lines[1] if len(lines) > 1 else "",
            "vat_id": ""
        },
        "buyer": {
            "company_code": "",
            "business_unit_code": "",
            "location_code": ""
        },
        "payment_term_id": "",
        "po_number": "",
        "po_id": "",
        "gross_total": gross_total,
        "subtotal": gross_total,
        "total_tax_amount": tax_amt,
        "discount_amount": "",
        "freight_charges": "",
        "insurance_charges": "",
        "extra_charges": "",
        "excise_duties": "",
        "taxes": taxes,
        "line_items": line_items
    }

    return payable


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini API rate limit or quota is exhausted (HTTP 429 RESOURCE_EXHAUSTED)."""
    pass


def get_raw_gemini_response(ocr_text: str, filename: str = "") -> str:
    """Call Gemini API directly and return the raw unparsed JSON string response.
    Prints the exact prompt input payload fed to the AI API directly into the terminal in real time.
    """
    if not (client and API_KEY and "<" not in API_KEY and "lang-client" not in API_KEY):
        raise ValueError("Gemini API key is missing or invalid.")
    
    full_prompt_input = f"{SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{ocr_text}"
    
    # Real-time Terminal Logging
    print("\n" + "=" * 80, flush=True)
    print(f"=== EXACT AI INPUT PAYLOAD FED TO LLM FOR [{filename or 'DOCUMENT'}] ===", flush=True)
    print(f"Model Name: {MODEL_NAME} | Chars: {len(full_prompt_input)} | Est Tokens: ~{len(full_prompt_input) // 4}", flush=True)
    print("=" * 80, flush=True)
    print(full_prompt_input, flush=True)
    print("=" * 80 + "\n", flush=True)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT, f"DOCUMENT TEXT:\n{ocr_text}"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        return response.text.strip()
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise QuotaExhaustedError(f"Gemini API Quota Exhausted ({MODEL_NAME}): {e}") from e
        raise e


from src.master_matcher import MasterDataMatcher

_matcher = MasterDataMatcher()


def extract_payable_from_text(ocr_text: str, filename: str = "", allow_fallback: bool = False) -> dict:
    """Extract structured autodraft JSON from OCR layout text using Gemini API."""
    if client and API_KEY and "<" not in API_KEY and "lang-client" not in API_KEY:
        try:
            raw_json = get_raw_gemini_response(ocr_text, filename=filename)
            payable_data = json.loads(raw_json)
            grounded_payable, _ = verify_payable_grounding(payable_data, ocr_text)
            verify_structural_integrity(grounded_payable)
            return _matcher.resolve_payable(grounded_payable, text_context=ocr_text)
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"Gemini API ({MODEL_NAME}) call failed: {e}") from e
            print(f"Gemini API error ({e}). Explicit fallback allowed.", file=sys.stderr)

    if allow_fallback:
        payable_data = deterministic_extract_payable(ocr_text, filename=filename)
        grounded_payable, _ = verify_payable_grounding(payable_data, ocr_text)
        return _matcher.resolve_payable(grounded_payable, text_context=ocr_text)
    
    raise ValueError("No valid Gemini API key configured and allow_fallback=False.")


from src.classifier import classify_document_text, classify_file
from src.segmenter import segment_document_text


def process_document_file(pdf_or_txt_path: str | Path) -> dict:
    """Process a single document PDF or .txt into per-file Autodraft JSON payload with multi-document segmentation."""
    path = Path(pdf_or_txt_path)

    # Get OCR text
    if path.suffix == ".pdf":
        pages = extract_text(str(path))
        full_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        file_name = path.name
    else:
        full_ocr_text = path.read_text(encoding="utf-8", errors="ignore")
        file_name = path.stem + ".pdf"

    # Pre-segmentation step for multi-document PDFs
    subdoc_texts = segment_document_text(full_ocr_text)

    payables = []
    declined = []

    for idx, seg_text in enumerate(subdoc_texts, 1):
        seg_file_label = f"{file_name}#subdoc{idx}" if len(subdoc_texts) > 1 else file_name
        class_res = classify_document_text(seg_text, filename=seg_file_label)

        if class_res.is_payable:
            try:
                payable = extract_payable_from_text(seg_text, filename=seg_file_label)
                if class_res.doc_type == "CREDIT_MEMO":
                    payable["invoice_type"] = "CREDIT_MEMO"
                payables.append(payable)
            except Exception as e:
                print(f"Error extracting payable from {seg_file_label}: {e}", file=sys.stderr)
                declined.append({"doc_type": class_res.doc_type, "reason": f"Extraction failed: {e}"})
        else:
            declined.append({"doc_type": class_res.doc_type, "reason": "; ".join(class_res.reasons)})

    return {
        "file": file_name,
        "payables": payables,
        "declined": declined
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor <pdf_or_txt_path>", file=sys.stderr)
        sys.exit(1)

    target_file = Path(sys.argv[1])
    result = process_document_file(target_file)
    print(json.dumps(result, indent=2))
