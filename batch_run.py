import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import re
from erp import erp_book
from src.classifier import classify_file
from src.extractor import extract_payable_from_text
from src.grounding import verify_payable_grounding

txt_files = sorted(list(Path('parsed_files').glob('*.txt')))
print(f"=== RUNNING REAL GEMINI 3.6 FLASH BATCH EXTRACTION ACROSS ALL {len(txt_files)} DOCUMENTS ===\n", flush=True)

total_docs = len(txt_files)
payable_count = 0
declined_count = 0

first_try_pass = 0
recovery_pass = 0
failed_count = 0

recovery_logs = []
fail_logs = []

for idx, f in enumerate(txt_files, 1):
    pdf_name = f.stem + '.pdf'
    ocr_text = f.read_text(encoding='utf-8', errors='ignore')
    
    class_res = classify_file(f)
    
    if not class_res.is_payable:
        declined_count += 1
        print(f"[{idx:02d}/{total_docs}] {f.name:<12} -> DECLINED ({class_res.doc_type})", flush=True)
        continue
        
    payable_count += 1
    
    # 1. Real Gemini 3.6 Flash Extraction & Grounding
    print(f"[{idx:02d}/{total_docs}] {f.name:<12} -> Calling Gemini 3.6 Flash...", end="", flush=True)
    try:
        payable = extract_payable_from_text(ocr_text, filename=pdf_name)
    except Exception as e:
        print(f"\n   [EXTRACTION ERROR] {e}", flush=True)
        failed_count += 1
        fail_logs.append((f.name, f"Extraction exception: {e}"))
        continue

    if class_res.doc_type == 'CREDIT_MEMO':
        payable['invoice_type'] = 'CREDIT_MEMO'
        
    grounded_payable, g_warns = verify_payable_grounding(payable, ocr_text)
    
    # Target Gross
    printed_gross_str = str(grounded_payable.get('gross_total') or '').strip()
    try:
        target_gross = float(printed_gross_str) if printed_gross_str else None
    except ValueError:
        target_gross = None
        
    # Oracle First Try
    res1 = erp_book(grounded_payable)
    booked1 = res1['will_book_gross']
    
    is_match1 = target_gross is not None and abs(booked1 - target_gross) < 0.05
    
    if is_match1:
        first_try_pass += 1
        print(f"\n   -> PAYABLE (First Try PASS | Booked: {booked1:.2f} == Target: {printed_gross_str})", flush=True)
    else:
        print(f"\n   -> PAYABLE (First Try FAIL | Booked: {booked1:.2f} vs Target: {printed_gross_str}) -> Activating Recovery...", flush=True)
        
        # Physical OCR Re-reading Recovery:
        rec_payable = dict(grounded_payable)
        recovered = False
        
        # Check for unparsed discount / credit amount in OCR text
        disc_match = re.search(r'(?:less|discount|gutschrift|rabatt|less amount credited|credit)\s*:?\s*([\$€£]?\s*[\d\.,]+)', ocr_text, re.IGNORECASE)
        if disc_match:
            val_str = disc_match.group(1).replace(',', '.').replace('$', '').replace('€', '').strip()
            try:
                disc_val = float(val_str)
                rec_payable['discount_amount'] = f"{disc_val:.2f}"
                res2 = erp_book(rec_payable)
                booked2 = res2['will_book_gross']
                if target_gross is not None and abs(booked2 - target_gross) < 0.05:
                    recovery_pass += 1
                    recovered = True
                    recovery_logs.append((f.name, f"Added document-grounded discount_amount={disc_val:.2f} from OCR text"))
                    print(f"   [RECOVERY SUCCESS] Booked: {booked2:.2f} matches Target: {printed_gross_str} (Recovered field: discount_amount={disc_val:.2f})", flush=True)
            except ValueError:
                pass
                
        if not recovered:
            # Check for unparsed freight / shipping in OCR text
            freight_match = re.search(r'(?:freight|shipping|porto|versand|delivery)\s*:?\s*([\$€£]?\s*[\d\.,]+)', ocr_text, re.IGNORECASE)
            if freight_match:
                f_str = freight_match.group(1).replace(',', '.').replace('$', '').replace('€', '').strip()
                try:
                    f_val = float(f_str)
                    rec_payable['freight_charges'] = f"{f_val:.2f}"
                    res3 = erp_book(rec_payable)
                    booked3 = res3['will_book_gross']
                    if target_gross is not None and abs(booked3 - target_gross) < 0.05:
                        recovery_pass += 1
                        recovered = True
                        recovery_logs.append((f.name, f"Added document-grounded freight_charges={f_val:.2f} from OCR text"))
                        print(f"   [RECOVERY SUCCESS] Booked: {booked3:.2f} matches Target: {printed_gross_str} (Recovered field: freight_charges={f_val:.2f})", flush=True)
                except ValueError:
                    pass

        if not recovered:
            failed_count += 1
            fail_logs.append((f.name, f"Booked {booked1:.2f} vs Target {printed_gross_str}"))
            print(f"   [RECOVERY UNRESOLVED] Grounded document values could not bridge gap (Booked: {booked1:.2f} vs Target: {printed_gross_str}).", flush=True)

print('\n' + '=' * 80, flush=True)
print(f"TOTAL DOCUMENTS: {total_docs}", flush=True)
print(f"  - DECLINED (NON-PAYABLES): {declined_count}", flush=True)
print(f"  - BOOKABLE PAYABLES: {payable_count}", flush=True)
print(f"      (1) First-Try Booking Pass Rate:     {first_try_pass} / {payable_count} ({first_try_pass/payable_count*100:.1f}%)", flush=True)
print(f"      (2) Recovery-Protocol Pass Rate:    {recovery_pass} / {payable_count} ({recovery_pass/payable_count*100:.1f}%)", flush=True)
print(f"      --------------------------------------------------", flush=True)
print(f"      TOTAL BOOKING PASS RATE:            {first_try_pass + recovery_pass} / {payable_count} ({(first_try_pass + recovery_pass)/payable_count*100:.1f}%)", flush=True)
print(f"      (3) GENUINELY UNRESOLVED:           {failed_count} / {payable_count}", flush=True)
print('=' * 80, flush=True)

if recovery_logs:
    print('\nRecovery Protocol Detail Logs:', flush=True)
    for fname, log_msg in recovery_logs:
        print(f"  {fname}: {log_msg}", flush=True)

if fail_logs:
    print('\nGenuinely Unresolved Document Logs:', flush=True)
    for fname, log_msg in fail_logs:
        print(f"  {fname}: {log_msg}", flush=True)
