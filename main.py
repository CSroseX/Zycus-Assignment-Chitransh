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


def process_all_documents(input_dir: str | Path = "documents", output_dir: str | Path = "output") -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        pdf_files = [in_path]
    else:
        pdf_files = sorted(list(in_path.glob("*.pdf")))

    print(f"=== ZYCUS BOOKABLE PAYABLE PIPELINE: PROCESSING {len(pdf_files)} PDF(S) ===")
    print(f"Input Directory:  {in_path.resolve()}")
    print(f"Output Directory: {out_path.resolve()}\n")

    for idx, pdf in enumerate(pdf_files, 1):
        txt_cache = Path("test_output") / f"{pdf.stem}.txt"
        
        # 1. OCR Extraction / Cache Load
        if txt_cache.exists():
            full_ocr_text = txt_cache.read_text(encoding="utf-8", errors="ignore")
        else:
            pages = extract_text(str(pdf))
            full_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)

        # 2. Multi-Document Pre-Segmentation
        subdoc_texts = segment_document_text(full_ocr_text)

        payables = []
        declined = []

        print(f"[{idx:02d}/{len(pdf_files)}] Processing {pdf.name} ({len(subdoc_texts)} sub-document segment(s))...", flush=True)

        for s_idx, seg_text in enumerate(subdoc_texts, 1):
            sub_label = f"{pdf.name}#subdoc{s_idx}" if len(subdoc_texts) > 1 else pdf.name
            class_res = classify_document_text(seg_text, filename=sub_label)

            if class_res.is_payable:
                try:
                    payable = extract_payable_from_text(seg_text, filename=sub_label, allow_fallback=False)
                    if class_res.doc_type == "CREDIT_MEMO":
                        payable["invoice_type"] = "CREDIT_MEMO"
                    payables.append(payable)
                    print(f"   -> Sub-Doc {s_idx}: BOOKABLE PAYABLE ({payable.get('gross_total')} {payable.get('currency')})", flush=True)
                except QuotaExhaustedError as qe:
                    print(f"\n[QUOTA EXHAUSTED STOPPER]: {qe}", flush=True)
                    print("API Quota limit reached (HTTP 429 RESOURCE_EXHAUSTED). Stopping batch execution immediately.", flush=True)
                    sys.exit(1)
                except Exception as e:
                    print(f"   -> Sub-Doc {s_idx}: Extraction Failed ({e})", flush=True)
                    declined.append({"doc_type": class_res.doc_type, "reason": f"Extraction exception: {e}"})
            else:
                declined.append({"doc_type": class_res.doc_type, "reason": "; ".join(class_res.reasons)})
                print(f"   -> Sub-Doc {s_idx}: DECLINED ({class_res.doc_type})", flush=True)

        # Build final AUTODRAFT JSON payload for this PDF file
        file_payload = {
            "file": pdf.name,
            "payables": payables,
            "declined": declined
        }

        # Write output JSON file
        out_json_path = out_path / f"{pdf.stem}.json"
        out_json_path.write_text(json.dumps(file_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nPipeline processing complete. Generated {len(pdf_files)} JSON file(s) in '{out_path}/'.")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "documents"
    process_all_documents(target_dir)
