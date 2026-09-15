"""
main.py — Single Command Orchestrator for Zycus Bookable Payable Pipeline.

Executes end-to-end processing across documents/ folder:
1. OCR Text & Spatial Layout Extraction (src/ocr_engine.py)
2. Multi-Document Pre-Segmentation (src/segmenter.py)
3. Payable vs Non-Payable Classification (src/classifier.py)
4. Structured Autodraft Extraction & Rule 1 Grounding (src/extractor.py)
5. Master Data Matching & Resolution (src/master_matcher.py)
6. Output Generation: Writes AUTODRAFT JSON payload per PDF into output/<pdf_stem>.json

Usage:
    python main.py documents/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.classifier import classify_document_text
from src.extractor import extract_payable_from_text, QuotaExhaustedError
from src.ocr_engine import extract_text
from src.segmenter import segment_document_text


def process_all_documents(input_dir: str | Path = "documents", output_dir: str | Path = "output", only_files: list[str] | None = None) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        pdf_files = [in_path]
    else:
        pdf_files = sorted(list(in_path.glob("*.pdf")))

    if only_files:
        targets = {f.replace(".pdf", "").strip() for f in only_files}
        pdf_files = [p for p in pdf_files if p.stem in targets or p.name in targets]

    print("=" * 80, flush=True)
    print(f"=== ZYCUS BOOKABLE PAYABLE PIPELINE: PROCESSING {len(pdf_files)} PDF(S) ===", flush=True)
    print(f"Input Directory:  {in_path.resolve()}", flush=True)
    print(f"Output Directory: {out_path.resolve()}", flush=True)
    print("=" * 80 + "\n", flush=True)

    total_payables_count = 0
    total_declined_count = 0

    for idx, pdf in enumerate(pdf_files, 1):
        print("-" * 80, flush=True)
        print(f"DOCUMENT [{idx:02d}/{len(pdf_files)}]: {pdf.name}", flush=True)
        print("-" * 80, flush=True)

        txt_cache = Path("parsed_files") / f"{pdf.stem}.txt"
        
        # STEP 1: OCR Extraction / Cache Load
        if txt_cache.exists():
            full_ocr_text = txt_cache.read_text(encoding="utf-8", errors="ignore")
            print(f"[STEP 1: OCR & LAYOUT EXTRACTION] -> Loaded cached text ({len(full_ocr_text)} chars)", flush=True)
        else:
            pages = extract_text(str(pdf))
            full_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
            print(f"[STEP 1: OCR & LAYOUT EXTRACTION] -> Extracted {len(pages)} page(s) via PyMuPDF ({len(full_ocr_text)} chars)", flush=True)

        # STEP 2: Multi-Document Pre-Segmentation
        subdoc_texts = segment_document_text(full_ocr_text)
        print(f"[STEP 2: PRE-SEGMENTATION]       -> Segmented into {len(subdoc_texts)} sub-document(s)", flush=True)

        payables = []
        declined = []

        for s_idx, seg_text in enumerate(subdoc_texts, 1):
            sub_label = f"{pdf.name}#subdoc{s_idx}" if len(subdoc_texts) > 1 else pdf.name
            
            # STEP 3: Classification
            class_res = classify_document_text(seg_text, filename=sub_label)
            conf_val = getattr(class_res, 'confidence', getattr(class_res, 'score', 1.0))
            print(f"\n  >>> Sub-Doc {s_idx}/{len(subdoc_texts)} [{sub_label}]", flush=True)
            print(f"  [STEP 3: CLASSIFICATION]         -> Payable: {class_res.is_payable} | Type: {class_res.doc_type} (Confidence: {conf_val:.2f})", flush=True)

            if class_res.is_payable:
                try:
                    # STEP 4 & STEP 5: Extraction, Grounding & Master Data Matching
                    print(f"  [STEP 4: AI EXTRACTION & GROUNDING] -> Sending OCR text to LLM API...", flush=True)
                    payable = extract_payable_from_text(seg_text, filename=sub_label, allow_fallback=True)
                    if class_res.doc_type == "CREDIT_MEMO":
                        payable["invoice_type"] = "CREDIT_MEMO"
                    payables.append(payable)
                    total_payables_count += 1

                    supp_id = payable.get("supplier", {}).get("supplier_id", "")
                    comp_code = payable.get("buyer", {}).get("company_code", "")
                    print(f"  [STEP 5: MASTER DATA MATCHING]   -> Supplier ID: '{supp_id}' | Company Code: '{comp_code}'", flush=True)
                    print(f"  [STATUS]: BOOKABLE PAYABLE      -> Gross Total: {payable.get('gross_total')} {payable.get('currency')} | Line Items: {len(payable.get('line_items', []))}", flush=True)

                except QuotaExhaustedError as qe:
                    print(f"\n[QUOTA EXHAUSTED STOPPER]: {qe}", flush=True)
                    print("API Quota limit reached. Stopping batch execution immediately.", flush=True)
                    sys.exit(1)
                except Exception as e:
                    print(f"  [STEP 4: EXTRACTION FAILED]     -> Error: {e}", flush=True)
                    declined.append({"doc_type": class_res.doc_type, "reason": f"Extraction exception: {e}"})
                    total_declined_count += 1
            else:
                declined.append({"doc_type": class_res.doc_type, "reason": "; ".join(class_res.reasons)})
                total_declined_count += 1
                print(f"  [STATUS]: DECLINED              -> Reasons: {'; '.join(class_res.reasons)}", flush=True)

        # STEP 6: Save Final AUTODRAFT JSON Payload
        file_payload = {
            "file": pdf.name,
            "payables": payables,
            "declined": declined
        }

        out_json_path = out_path / f"{pdf.stem}.json"
        out_json_path.write_text(json.dumps(file_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n LAST [STEP 6: JSON SAVED]               -> Saved JSON to '{out_json_path}'\n", flush=True)

    print("=" * 80, flush=True)
    print("=== PIPELINE EXECUTION SUMMARY ===", flush=True)
    print(f"Total Documents Processed: {len(pdf_files)}", flush=True)
    print(f"Total Payables Extracted:  {total_payables_count}", flush=True)
    print(f"Total Declined Segments:   {total_declined_count}", flush=True)
    print(f"Output Directory:          {out_path.resolve()}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    target_dir = "documents"
    only_list = None
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--only" and idx + 1 < len(args):
            only_list = [f.strip() for f in args[idx+1].split(",") if f.strip()]
            idx += 2
        elif not arg.startswith("--"):
            target_dir = arg
            idx += 1
        else:
            idx += 1
    process_all_documents(target_dir, only_files=only_list)

