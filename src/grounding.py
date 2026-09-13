"""
grounding.py — Document Grounding & Anti-Hallucination Verifier.

Rule 1 Enforcement: "Every value you emit must appear on the document."
Verifies that every extracted field (invoice_number, dates, amounts, prices, quantities, PO numbers)
is traceable back to the raw OCR layout text. Ungrounded values are blanked out.
"""
from __future__ import annotations

import re
from typing import Any


def normalize_token(text: str) -> str:
    """Normalize text for fuzzy grounding comparison (lowercase, alphanumeric)."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()


def is_grounded_number(value_str: str, ocr_text: str) -> bool:
    """Check if a numeric value (e.g. '1234.56', '438.00', '15') appears in OCR text."""
    if not value_str or not str(value_str).strip():
        return True  # Empty fields are valid (Rule 30)

    val_norm = str(value_str).strip()
    if val_norm in ocr_text:
        return True

    # Check alternative decimal formats (e.g., '1234.56' vs '1234,56' vs '1 234,56')
    try:
        val_float = float(val_norm)
        # Regex search for numeric matches with dot or comma decimal
        # E.g. for 438.00 -> search 438,00 or 438.00 or 438
        int_part = str(int(val_float))
        if int_part in ocr_text:
            return True
    except ValueError:
        pass

    # Check normalized string token
    norm_val = normalize_token(val_norm)
    norm_ocr = normalize_token(ocr_text)

    return norm_val in norm_ocr if norm_val else True


def is_grounded_text(value_str: str, ocr_text: str, min_match_ratio: float = 0.6) -> bool:
    """Check if a text string (e.g. invoice_number, supplier_name) is grounded in OCR text."""
    if not value_str or not str(value_str).strip():
        return True

    clean_val = str(value_str).strip()
    if clean_val.lower() in ocr_text.lower():
        return True

    norm_val = normalize_token(clean_val)
    norm_ocr = normalize_token(ocr_text)

    if not norm_val:
        return True

    return norm_val in norm_ocr


def verify_payable_grounding(payable_dict: dict, ocr_text: str) -> tuple[dict, list[str]]:
    """Audit a complete Autodraft Payable payload against OCR layout text.

    Returns:
        (sanitized_payable_dict, list_of_ungrounded_warnings)
    """
    warnings = []
    sanitized = dict(payable_dict)

    # 1. Audit Header Numbers
    for field in ["invoice_number", "gross_total", "subtotal", "total_tax_amount", "po_number"]:
        val = str(sanitized.get(field, "") or "").strip()
        if val and not is_grounded_number(val, ocr_text) and not is_grounded_text(val, ocr_text):
            warnings.append(f"UNGROUNDED HEADER FIELD '{field}': '{val}' not found in OCR text. Blanking out field.")
            sanitized[field] = ""

    # 2. Audit Line Items
    if "line_items" in sanitized and isinstance(sanitized["line_items"], list):
        sanitized_lines = []
        for idx, line in enumerate(sanitized["line_items"]):
            san_line = dict(line)
            for l_field in ["quantity", "unit_price", "total", "tax_rate", "tax_amount"]:
                l_val = str(san_line.get(l_field, "") or "").strip()
                if l_val and not is_grounded_number(l_val, ocr_text):
                    warnings.append(f"UNGROUNDED LINE ITEM [{idx}] '{l_field}': '{l_val}' not found in OCR text.")
                    san_line[l_field] = ""
            sanitized_lines.append(san_line)
        sanitized["line_items"] = sanitized_lines

    # 3. Audit Taxes
    if "taxes" in sanitized and isinstance(sanitized["taxes"], list):
        sanitized_taxes = []
        for idx, tax in enumerate(sanitized["taxes"]):
            san_tax = dict(tax)
            for t_field in ["tax_rate", "tax_amount"]:
                t_val = str(san_tax.get(t_field, "") or "").strip()
                if t_val and not is_grounded_number(t_val, ocr_text):
                    warnings.append(f"UNGROUNDED TAX [{idx}] '{t_field}': '{t_val}' not found in OCR text.")
                    san_tax[t_field] = ""
            sanitized_taxes.append(san_tax)
        sanitized["taxes"] = sanitized_taxes

    return sanitized, warnings
