import sys
import os
import json
import re
from pathlib import Path

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from erp import erp_book
from src.classifier import classify_document_text
from src.extractor import get_raw_gemini_response, MODEL_NAME, QuotaExhaustedError
from src.grounding import verify_payable_grounding
from src.segmenter import segment_document_text

if len(sys.argv) < 2:
    print("Usage: python run_single.py <document_name_or_stem> (e.g. python run_single.py INV-01)")
    sys.exit(1)

doc_name = sys.argv[1].strip()
txt_path = Path("test_output") / f"{doc_name}.txt"
if not txt_path.exists():
    txt_path = Path("test_output") / f"{doc_name}"
if not txt_path.exists():
    print(f"Error: File '{doc_name}' not found in test_output/")
    sys.exit(1)

full_ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
pdf_name = txt_path.stem + ".pdf"

subdocs = segment_document_text(full_ocr_text)

print("=" * 80)
print(f"DOCUMENT: {txt_path.name}")
print(f"MODEL USED: {MODEL_NAME}")
print(f"PRE-SEGMENTATION: Detected {len(subdocs)} sub-document segment(s)")
print("=" * 80)

for idx, seg_text in enumerate(subdocs, 1):
    sub_label = f"{txt_path.name} [Sub-Doc {idx}/{len(subdocs)}]" if len(subdocs) > 1 else txt_path.name
    print(f"\n>>> PROCESSING {sub_label} ({len(seg_text.split('--- PAGE BREAK ---'))} page(s), {len(seg_text)} chars)")
    print("-" * 80)

    # Step 1: Classification
    class_res = classify_document_text(seg_text, filename=sub_label)
    print(f"[1] CLASSIFICATION: {'PAYABLE' if class_res.is_payable else 'DECLINED'}")
    print(f"    Doc Type:   {class_res.doc_type}")
    print(f"    Reasons:    {'; '.join(class_res.reasons)}")

    if not class_res.is_payable:
        print("    Document is declined non-payable. Skipping LLM extraction.")
        continue

    # Step 2: Raw Gemini API Call
    print(f"\n[2] RAW GEMINI ({MODEL_NAME}) JSON RESPONSE FOR {sub_label}:")
    print("-" * 80)
    try:
        raw_json_str = get_raw_gemini_response(seg_text, filename=f"{txt_path.stem}_subdoc{idx}.pdf")
        print(raw_json_str)
        print("-" * 80)
    except QuotaExhaustedError as qe:
        print(f"\n[QUOTA EXHAUSTED STOPPER]: {qe}")
        print("API Quota limit reached (HTTP 429 RESOURCE_EXHAUSTED). Stopping execution immediately.")
        sys.exit(1)
    except Exception as e:
        print(f"   [API FAILURE]: {e}")
        print("   * Status: API_FAILED (No silent fallback performed)")
        print("-" * 80)
        continue

    # Step 3: Grounding & Structural Verification
    print("\n[3] GROUNDING & STRUCTURAL VERIFICATION:")
    payable_data = json.loads(raw_json_str)
    if class_res.doc_type == 'CREDIT_MEMO':
        payable_data['invoice_type'] = 'CREDIT_MEMO'

    grounded_payable, g_warns = verify_payable_grounding(payable_data, seg_text)
    print(f"   Grounding Warnings: {len(g_warns)}")
    for w in g_warns:
        print(f"     - {w}")

    # Step 4: ERP Oracle Booking Check
    print("\n[4] ERP ORACLE BOOKING CHECK (erp_book):")
    printed_gross_str = str(grounded_payable.get('gross_total') or '').strip()
    try:
        target_gross = float(printed_gross_str) if printed_gross_str else None
    except ValueError:
        target_gross = None

    res1 = erp_book(grounded_payable)
    booked1 = res1['will_book_gross']
    is_match1 = target_gross is not None and abs(booked1 - target_gross) < 0.05

    print(f"   Document Target Gross:  {printed_gross_str}")
    print(f"   ERP Calculated Gross:   {booked1:.2f}")

    if is_match1:
        print("   STATUS: PASS (First Try Match)")
    else:
        print("   STATUS: FAIL (Discrepancy Detected)")
        print("\n[5] ACTIVATING RECOVERY PROTOCOL (Document-Grounded Re-reading)...")
        rec_payable = dict(grounded_payable)
        recovered = False
        
        disc_match = re.search(r'(?:less|discount|gutschrift|rabatt|less amount credited|credit)\s*:?\s*([\$€£]?\s*[\d\.,]+)', seg_text, re.IGNORECASE)
        if disc_match:
            val_str = disc_match.group(1).replace(',', '.').replace('$', '').replace('€', '').strip()
            try:
                disc_val = float(val_str)
                rec_payable['discount_amount'] = f"{disc_val:.2f}"
                res2 = erp_book(rec_payable)
                booked2 = res2['will_book_gross']
                if target_gross is not None and abs(booked2 - target_gross) < 0.05:
                    recovered = True
                    print(f"   [RECOVERY SUCCESS]: Recovered grounded field discount_amount={disc_val:.2f}")
                    print(f"   New ERP Calculated Gross: {booked2:.2f} == Target: {printed_gross_str}")
            except ValueError:
                pass

        if not recovered:
            freight_match = re.search(r'(?:freight|shipping|porto|versand|delivery)\s*:?\s*([\$€£]?\s*[\d\.,]+)', seg_text, re.IGNORECASE)
            if freight_match:
                f_str = freight_match.group(1).replace(',', '.').replace('$', '').replace('€', '').strip()
                try:
                    f_val = float(f_str)
                    rec_payable['freight_charges'] = f"{f_val:.2f}"
                    res3 = erp_book(rec_payable)
                    booked3 = res3['will_book_gross']
                    if target_gross is not None and abs(booked3 - target_gross) < 0.05:
                        recovered = True
                        print(f"   [RECOVERY SUCCESS]: Recovered grounded field freight_charges={f_val:.2f}")
                        print(f"   New ERP Calculated Gross: {booked3:.2f} == Target: {printed_gross_str}")
                except ValueError:
                    pass

        if not recovered:
            print(f"   [RECOVERY UNRESOLVED]: Grounded document values could not bridge gap ({booked1:.2f} vs {printed_gross_str}).")

print("=" * 80)
