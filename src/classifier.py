"""
classifier.py — Phase 3: Document Classification Engine (Payable vs Non-Payable).

Classifies documents into PAYABLE (Invoices) vs NON-PAYABLE (Credit Notes,
Reminders, Estimates, Purchase Orders, Delivery Notes, Donation Forms) using
multi-lingual pattern analysis and structural heuristics.

Usage:
    python -m src.classifier test_output/
    python -m src.classifier test_output/INV-01.txt
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ClassificationResult:
    filename: str
    is_payable: bool
    doc_type: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


# Keywords mapped to True Non-Payable Document Types (to be placed in declined[])
NON_PAYABLE_PATTERNS = {
    "REMINDER_STATEMENT": [
        "mahnung", "payment reminder", "statement of account",
        "account statement", "dunning letter", "reminder notice"
    ],
    "ESTIMATE_QUOTE": [
        "estimate", "price quote", "quotation", "angebot", "devis",
        "proforma invoice", "proforma"
    ],
    "INTERNAL_FORM": [
        "donations and charitable contributions", "charitable contribution",
        "donation request", "internal request form"
    ],
    "DELIVERY_NOTE": [
        "delivery note", "lieferschein", "guia de remessa", "packing slip",
        "packing list"
    ],
    "PURCHASE_ORDER": [
        "purchase order", "bestellung", "bon de commande"
    ],
    "REMITTANCE_ADVICE": [
        "remittance advice", "payment advice"
    ]
}

# Credit Note patterns (Bookable Payables with invoice_type: "CREDIT_MEMO")
CREDIT_MEMO_PATTERNS = [
    "credit note", "credit memo", "kreedit", "kreeditarve",
    "gutschrift", "nota de crédito", "nota de credito", "avoir",
    "stornorechnung", "credit_note"
]

# Keywords indicating Payable Tax Invoices
PAYABLE_PATTERNS = [
    "tax invoice", "invoice", "rechnung", "arve", "facture",
    "faktura", "rechnung nr", "arve nr", "tax_invoice"
]


import re

def compute_payable_score(text: str) -> tuple[int, list[str]]:
    """Compute structural & intent Payable Score for a document text."""
    t = text.lower()
    header = t[:1200]

    score = 0
    breakdown = []

    non_payable_titles = [
        'estimate', 'quotation', 'price quote', 'mahnung', 'payment reminder',
        'statement of account', 'delivery note', 'lieferschein', 'donations and charitable'
    ]
    payable_titles = [
        'tax invoice', 'invoice', 'rechnung', 'arve', 'facture', 'faktura',
        'fatura', 'fac ', 'credit note', 'kreedit', 'kreeditarve', 'gutschrift',
        'nota de crédito', 'avoir'
    ]

    has_non_payable_title = any(kw in header for kw in non_payable_titles)
    has_payable_title = any(kw in header for kw in payable_titles)

    if has_non_payable_title:
        score -= 100
        breakdown.append('NonPayableTitle(-100)')
    elif has_payable_title:
        score += 40
        breakdown.append('PayableTitle(+40)')

    # Table structure check
    table_rows = re.findall(r'(\d+[\.,]\d{2}|\d+\s+Std|\d+\s+Pcs|\d+\s+kg|\d+\s+unit)', t)
    if len(table_rows) >= 2:
        score += 25
        breakdown.append('TableStructure(+25)')
    elif len(table_rows) == 1:
        score += 10
        breakdown.append('TableStructure(+10)')

    # Tax ID check
    tax_ids = ['vat', 'mwst', 'nif', 'tin', 'kmkr', 'reg. no', 'company reg', 'eori', 'gst', 'sst', 'contribuinte', 'pt']
    has_tax_id = any(kw in t for kw in tax_ids)
    if has_tax_id:
        score += 15
        breakdown.append('TaxID(+15)')

    # Entity Parties check
    parties = ['bill to', 'billed to', 'ship to', 'buyer', 'kunden-nr', 'kunde', 'destinatar', 'direcao fiscal', 'prepared for', 'vendor details', 'customer', 'cliente', 'exmo']
    has_party = any(kw in t for kw in parties)
    if has_party:
        score += 10
        breakdown.append('EntityParties(+10)')

    # Financial Total check
    totals = ['total', 'subtotal', 'endbetrag', 'gesamtsumme', 'amount due', 'sub-total', 'summe', 'endsumme', 'valor', 'quantia']
    has_total = any(kw in t for kw in totals)
    if has_total:
        score += 10
        breakdown.append('FinancialTotal(+10)')

    return score, breakdown


def classify_document_text(text: str, filename: str = "") -> ClassificationResult:
    """Classify a document's extracted text into PAYABLE (INVOICE / CREDIT_MEMO) or NON_PAYABLE (declined[])."""
    if not text or not text.strip():
        return ClassificationResult(
            filename=filename,
            is_payable=False,
            doc_type="EMPTY_DOCUMENT",
            confidence=1.0,
            reasons=["Document contains no text"]
        )

    text_lower = text.lower()
    header_text = text_lower[:1200]

    # Check for Credit Memo explicitly first
    is_credit_memo = any(kw in header_text or kw in text_lower for kw in CREDIT_MEMO_PATTERNS)

    score, breakdown = compute_payable_score(text)

    if score >= 40:
        doc_type = "CREDIT_MEMO" if is_credit_memo else "INVOICE"
        confidence = 0.95 if score >= 80 else 0.85
        return ClassificationResult(
            filename=filename,
            is_payable=True,
            doc_type=doc_type,
            confidence=confidence,
            reasons=breakdown
        )
    else:
        # Determine non-payable category
        doc_type = "UNKNOWN_DECLINED"
        for cat, keywords in NON_PAYABLE_PATTERNS.items():
            if any(kw in header_text or kw in text_lower for kw in keywords):
                doc_type = cat
                break
        return ClassificationResult(
            filename=filename,
            is_payable=False,
            doc_type=doc_type,
            confidence=0.95 if "NonPayableTitle(-100)" in breakdown else 0.70,
            reasons=breakdown
        )


def classify_file(file_path: str | Path) -> ClassificationResult:
    """Classify a single extracted .txt file or PDF."""
    path = Path(file_path)
    if path.suffix == ".pdf":
        from src.ocr_engine import extract_text
        text = "\n".join(extract_text(str(path)))
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return classify_document_text(text, filename=path.name)


def main():
    if len(sys.argv) < 2:
        target = Path("test_output")
    else:
        target = Path(sys.argv[1])

    if not target.exists():
        print(f"Error: Path '{target}' does not exist.", file=sys.stderr)
        sys.exit(1)

    files = [target] if target.is_file() else sorted(list(target.glob("*.txt")))

    print(f"{'FILENAME':<15} {'PAYABLE?':<10} {'DOC_TYPE':<20} {'CONF':<6} {'REASONS'}")
    print("=" * 80)

    payable_count = 0
    non_payable_count = 0

    for f in files:
        res = classify_file(f)
        if res.is_payable:
            payable_count += 1
            status_str = "YES"
        else:
            non_payable_count += 1
            status_str = "NO"

        reason_summary = "; ".join(res.reasons[:2])
        print(f"{res.filename:<15} {status_str:<10} {res.doc_type:<20} {res.confidence:<6.2f} {reason_summary}")

    print("=" * 80)
    print(f"TOTAL: {len(files)} | PAYABLE: {payable_count} | NON-PAYABLE: {non_payable_count}")


if __name__ == "__main__":
    main()
