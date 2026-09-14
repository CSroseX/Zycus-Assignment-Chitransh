"""
segmenter.py — Phase 1/2: Multi-Document PDF Pre-Segmentation Engine.

Splits multi-page OCR document text into distinct sub-document segments
wherever a new sub-document header block or page-sequence restart occurs
(e.g., "Page 1 of N", "Page 1/1", "Consolidated Invoice", "Detailed Invoice").
"""
from __future__ import annotations
import re
from typing import Sequence

PAGE_START_PATTERNS = [
    r'\bpage\s*1\s*(?:of|0f|/|out of)\s*\d+\b',  # Page 1 of N, Page 10f 2, Page 1/3
    r'\bpage\s*1\s*of\s*1\b',
]

SUBDOC_TITLE_PATTERNS = [
    r'^\s*(?:consolidated invoice|detailed invoice|lieferschein|delivery note|packing slip|packing list|mahnung|payment reminder|quotation|price quote|estimate|purchase order)\b'
]


def extract_doc_identifiers(page_text: str) -> set[str]:
    """Extract invoice/PO/SO numbers from page header for sub-doc continuity."""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()][:8]
    text_hdr = " ".join(lines)
    ids = set()
    matches = re.findall(r'(?:po|so|order|invoice|rechnung|arve|facture|fatura|#)\s*(?:nr|no|num|#|\.)*:?\s*([a-zA-Z0-9\-_]{4,})', text_hdr, re.IGNORECASE)
    for m in matches:
        if not any(w in m.lower() for w in ['number', 'date', 'total', 'page', 'amount']):
            ids.add(m.lower())
    return ids


def is_new_subdocument_start(page_text: str, prev_page_text: str = "", page_num: int = 0) -> bool:
    """Determine if a page text signals the start of a new sub-document."""
    if page_num == 0:
        return True

    header_lines = [l.strip() for l in page_text.splitlines() if l.strip()][:5]
    header_str = " ".join(header_lines).lower()

    # 0. Check if page explicitly states it is a continuation page (e.g. Page 2 of 2, Page 3 of 5)
    if re.search(r'\bpage\s*[2-9]\d*\s*(?:of|0f|/|out of)\s*\d+\b', header_str, re.IGNORECASE):
        return False

    # Check if page shares the exact same PO/SO number as previous page
    if prev_page_text:
        prev_ids = extract_doc_identifiers(prev_page_text)
        curr_ids = extract_doc_identifiers(page_text)
        if prev_ids and curr_ids and prev_ids.intersection(curr_ids):
            # Same PO/SO number -> continuation page of the same document!
            return False

    # 1. Check for explicit Page 1 restart (e.g. "Page 1 of 2", "Page 1 of 1")
    for pat in PAGE_START_PATTERNS:
        if re.search(pat, header_str, re.IGNORECASE):
            return True

    # 2. Check for distinct sub-document titles at line start
    for line in header_lines:
        for pat in SUBDOC_TITLE_PATTERNS:
            if re.search(pat, line.lower()):
                return True

    return False


def segment_document_text(ocr_text: str) -> list[str]:
    """Split multi-page document OCR text into individual sub-document text segments."""
    pages = ocr_text.split("--- PAGE BREAK ---")
    if len(pages) <= 1:
        return [ocr_text.strip()]

    segments: list[list[str]] = []
    current_segment: list[str] = []

    for i, page in enumerate(pages):
        page_clean = page.strip()
        if not page_clean:
            continue

        prev_page = pages[i-1].strip() if i > 0 else ""
        if current_segment and is_new_subdocument_start(page_clean, prev_page_text=prev_page, page_num=i):
            segments.append(current_segment)
            current_segment = [page_clean]
        else:
            current_segment.append(page_clean)

    if current_segment:
        segments.append(current_segment)

    return ["\n\n--- PAGE BREAK ---\n\n".join(seg) for seg in segments]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("parsed_files/DU-02.txt")
    text = target.read_text(encoding="utf-8", errors="ignore")
    subdocs = segment_document_text(text)
    print(f"Segmented '{target.name}' into {len(subdocs)} sub-document(s).")
    for idx, doc in enumerate(subdocs, 1):
        lines = [l.strip() for l in doc.splitlines() if l.strip()]
        print(f"  Sub-Doc [{idx}]: {len(doc.split('--- PAGE BREAK ---'))} page(s), {len(doc)} chars | First line: '{lines[0] if lines else ''}'")
