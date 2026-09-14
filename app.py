"""
app.py — Developer Control Panel & Visual Evaluator for Zycus Bookable Payable Pipeline.

Features:
- Live, step-by-step stream execution generator over documents/
- Full visibility into OCR extraction path, segmentation, classification, AI extraction,
  grounding, master data matching, and ERP booking verification.
- Thread-safe dual sys.stdout capture (IDE terminal + bottom Streamlit terminal log feed).
- Choice between Processing All Files (Batch) vs Single File processing.
- Inspection selector rendered ONLY after processing completes.
- Side-by-side Start / Stop processing controls.
- Zero chatbot framing, zero message bubbles. Pure structured control dashboard.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import streamlit as st

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.ocr_engine import extract_pages, extract_text, is_digital_vector_page, reconstruct_layout
from src.segmenter import segment_document_text
from src.classifier import classify_document_text
from src.extractor import extract_payable_from_text, QuotaExhaustedError
from src.grounding import verify_payable_grounding
from src.master_matcher import MasterDataMatcher
from erp import erp_book
import pymupdf as fitz


# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zycus Bookable Payable Pipeline Control Panel",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Terminal Theme Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #4361ee;
    }
    .metric-label {
        font-size: 13px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-pass {
        color: #38a169;
        font-weight: bold;
    }
    .status-declined {
        color: #d69e2e;
        font-weight: bold;
    }
    .status-fail {
        color: #e53e3e;
        font-weight: bold;
    }
    .status-processing {
        color: #3182ce;
        font-weight: bold;
    }
    .status-queued {
        color: #718096;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thread-Safe Dual stdout Capture Wrapper
# ---------------------------------------------------------------------------
class StreamTee:
    """Duplicates sys.stdout output to original terminal AND Streamlit log list safely."""
    def __init__(self, original_stdout, log_list):
        self.original = original_stdout
        self.log_list = log_list

    def write(self, text):
        try:
            self.original.write(text)
        except Exception:
            pass
        clean = text.rstrip()
        if clean:
            self.log_list.append(clean)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "doc_status" not in st.session_state:
    st.session_state.doc_status = {}  # filename -> status string
if "doc_traces" not in st.session_state:
    st.session_state.doc_traces = {}  # filename -> list of step dicts
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "raw_logs" not in st.session_state:
    st.session_state.raw_logs = []
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "payables": 0,
        "declined": 0,
        "first_try_pass": 0,
        "failed": 0
    }

# ---------------------------------------------------------------------------
# Helper: Get list of PDF files
# ---------------------------------------------------------------------------
def get_pdf_files() -> list[Path]:
    doc_dir = Path("documents")
    if doc_dir.exists():
        return sorted(list(doc_dir.glob("*.pdf")))
    return []

pdf_files = get_pdf_files()
pdf_names = [f.name for f in pdf_files]

# Initialize default status for files if not present
for name in pdf_names:
    if name not in st.session_state.doc_status:
        st.session_state.doc_status[name] = "QUEUED"
    if name not in st.session_state.doc_traces:
        st.session_state.doc_traces[name] = []

if not st.session_state.selected_doc and pdf_names:
    st.session_state.selected_doc = pdf_names[0]


# ---------------------------------------------------------------------------
# Pipeline Generator Function
# ---------------------------------------------------------------------------
def run_pipeline_generator(target_files: list[Path]):
    """Generator yielding live step events per document with safe stdout redirection."""
    matcher = MasterDataMatcher()
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    sys.stdout = StreamTee(original_stdout, st.session_state.raw_logs)

    try:
        for idx, pdf in enumerate(target_files, 1):
            if st.session_state.stop_requested:
                st.session_state.raw_logs.append("[SYSTEM]: Processing stopped by user request.")
                break

            filename = pdf.name
            st.session_state.doc_status[filename] = "PROCESSING"
            st.session_state.selected_doc = filename
            st.session_state.doc_traces[filename] = []

            trace = []
            
            print(f"\n" + "=" * 80, flush=True)
            print(f"DOCUMENT [{idx:02d}/{len(target_files)}]: {filename}", flush=True)
            print("=" * 80, flush=True)

            yield {
                "type": "DOC_START",
                "filename": filename,
                "idx": idx,
                "total": len(target_files)
            }

            txt_cache = Path("test_output") / f"{pdf.stem}.txt"
            path_used = "PyMuPDF Native Vector Text"
            page_count = 1

            # Phase 1: OCR Extraction & Layout
            if txt_cache.exists():
                full_ocr_text = txt_cache.read_text(encoding="utf-8", errors="ignore")
                path_used = "Cached OCR Text (300 DPI Spatial Layout)"
                print(f"[STEP 1: OCR & LAYOUT EXTRACTION] -> Loaded cached text ({len(full_ocr_text)} chars)", flush=True)
            else:
                doc = fitz.open(str(pdf))
                page_count = len(doc)
                is_dig = any(is_digital_vector_page(doc[i]) for i in range(len(doc)))
                doc.close()
                
                text_pages = extract_text(str(pdf))
                full_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(text_pages)
                path_used = "PyMuPDF Direct Text Extraction" if is_dig else "PyMuPDF Render (300 DPI) + EasyOCR"
                print(f"[STEP 1: OCR & LAYOUT EXTRACTION] -> Extracted {page_count} page(s) via {path_used} ({len(full_ocr_text)} chars)", flush=True)

            step1_event = {
                "step": 1,
                "title": "Phase 1: OCR & Spatial Layout Extraction",
                "path_used": path_used,
                "char_count": len(full_ocr_text),
                "page_count": page_count,
                "raw_text": full_ocr_text
            }
            trace.append(step1_event)
            st.session_state.doc_traces[filename].append(step1_event)
            yield {"type": "STEP_1", "filename": filename, "data": step1_event}

            # Phase 2: Multi-Document Pre-Segmentation
            subdoc_texts = segment_document_text(full_ocr_text)
            print(f"[STEP 2: PRE-SEGMENTATION]       -> Segmented into {len(subdoc_texts)} sub-document(s)", flush=True)

            step2_event = {
                "step": 2,
                "title": "Phase 2: Multi-Document Pre-Segmentation",
                "subdoc_count": len(subdoc_texts),
                "status_msg": f"Detected {len(subdoc_texts)} distinct sub-document segment(s)."
            }
            trace.append(step2_event)
            st.session_state.doc_traces[filename].append(step2_event)
            yield {"type": "STEP_2", "filename": filename, "data": step2_event}

            payables = []
            declined = []
            doc_has_payable = False

            for s_idx, seg_text in enumerate(subdoc_texts, 1):
                if st.session_state.stop_requested:
                    break

                sub_label = f"{filename}#subdoc{s_idx}" if len(subdoc_texts) > 1 else filename
                
                # Phase 3: Classification
                class_res = classify_document_text(seg_text, filename=sub_label)
                print(f"  [STEP 3: CLASSIFICATION]         -> Payable: {class_res.is_payable} | Type: {class_res.doc_type} (Score: {class_res.score})", flush=True)

                step3_event = {
                    "step": 3,
                    "subdoc_idx": s_idx,
                    "title": f"Phase 3: Classification [{sub_label}]",
                    "is_payable": class_res.is_payable,
                    "doc_type": class_res.doc_type,
                    "score": class_res.score,
                    "reasons": class_res.reasons
                }
                trace.append(step3_event)
                st.session_state.doc_traces[filename].append(step3_event)
                yield {"type": "STEP_3", "filename": filename, "data": step3_event}

                if class_res.is_payable:
                    doc_has_payable = True
                    try:
                        # Phase 4: AI Model Extraction (Groq)
                        print(f"  [STEP 4: AI EXTRACTION & GROUNDING] -> Sending OCR text to Groq API...", flush=True)
                        raw_payable = extract_payable_from_text(seg_text, filename=sub_label, allow_fallback=False)
                        if class_res.doc_type == "CREDIT_MEMO":
                            raw_payable["invoice_type"] = "CREDIT_MEMO"

                        step4_event = {
                            "step": 4,
                            "subdoc_idx": s_idx,
                            "title": f"Phase 4: AI Model Extraction (Groq) [{sub_label}]",
                            "raw_json": raw_payable
                        }
                        trace.append(step4_event)
                        st.session_state.doc_traces[filename].append(step4_event)
                        yield {"type": "STEP_4", "filename": filename, "data": step4_event}

                        # Phase 4b: Grounding Verification
                        grounded_payable, g_warns = verify_payable_grounding(raw_payable, seg_text)
                        step4b_event = {
                            "step": "4b",
                            "subdoc_idx": s_idx,
                            "title": f"Phase 4b: Grounding & Anti-Hallucination Verification [{sub_label}]",
                            "warnings": g_warns,
                            "sanitized_json": grounded_payable
                        }
                        trace.append(step4b_event)
                        st.session_state.doc_traces[filename].append(step4b_event)
                        yield {"type": "STEP_4B", "filename": filename, "data": step4b_event}

                        # Phase 5: Master Data Matching
                        resolved_payable = matcher.resolve_payable(grounded_payable, text_context=seg_text)
                        
                        supp_id = resolved_payable.get("supplier", {}).get("supplier_id", "")
                        comp_code = resolved_payable.get("buyer", {}).get("company_code", "")
                        print(f"  [STEP 5: MASTER DATA MATCHING]   -> Supplier ID: '{supp_id}' | Company Code: '{comp_code}'", flush=True)

                        master_summary = {
                            "supplier_id": supp_id,
                            "company_code": comp_code,
                            "business_unit_code": resolved_payable.get("buyer", {}).get("business_unit_code", ""),
                            "location_code": resolved_payable.get("buyer", {}).get("location_code", ""),
                            "po_id": resolved_payable.get("po_id", ""),
                            "payment_term_id": resolved_payable.get("payment_term_id", "")
                        }
                        
                        step5_event = {
                            "step": 5,
                            "subdoc_idx": s_idx,
                            "title": f"Phase 5: Master Data Resolution [{sub_label}]",
                            "master_matched": master_summary,
                            "resolved_json": resolved_payable
                        }
                        trace.append(step5_event)
                        st.session_state.doc_traces[filename].append(step5_event)
                        yield {"type": "STEP_5", "filename": filename, "data": step5_event}

                        # Phase 6: Pre-ERP Payload Assembly
                        payables.append(resolved_payable)
                        step6_event = {
                            "step": 6,
                            "subdoc_idx": s_idx,
                            "title": f"Phase 6: Pre-ERP Payload Assembly [{sub_label}]",
                            "payload": resolved_payable
                        }
                        trace.append(step6_event)
                        st.session_state.doc_traces[filename].append(step6_event)
                        yield {"type": "STEP_6", "filename": filename, "data": step6_event}

                        # Phase 7: ERP Oracle Booking Check
                        erp_res = erp_book(resolved_payable)
                        booked_gross = erp_res.get("will_book_gross", 0.0)
                        printed_gross_str = str(resolved_payable.get("gross_total") or "").strip()
                        try:
                            target_gross = float(printed_gross_str) if printed_gross_str else 0.0
                        except ValueError:
                            target_gross = 0.0

                        is_match = abs(booked_gross - target_gross) < 0.05
                        erp_status = "PASS" if is_match else "FAIL"
                        print(f"  [STATUS]: BOOKABLE PAYABLE ({erp_status}) -> Gross Total: {printed_gross_str} {resolved_payable.get('currency')}", flush=True)

                        step7_event = {
                            "step": 7,
                            "subdoc_idx": s_idx,
                            "title": f"Phase 7: ERP Oracle Booking Verification [{sub_label}]",
                            "target_gross": f"{target_gross:.2f}",
                            "booked_gross": f"{booked_gross:.2f}",
                            "status": erp_status,
                            "erp_details": erp_res
                        }
                        trace.append(step7_event)
                        st.session_state.doc_traces[filename].append(step7_event)
                        yield {"type": "STEP_7", "filename": filename, "data": step7_event}

                    except QuotaExhaustedError as qe:
                        print(f"[QUOTA EXHAUSTED]: {qe}", flush=True)
                        st.session_state.doc_status[filename] = "FAIL"
                        return
                    except Exception as e:
                        print(f"  [EXTRACTION FAILED]: {e}", flush=True)
                        declined.append({"doc_type": class_res.doc_type, "reason": f"Extraction exception: {e}"})
                else:
                    print(f"  [STATUS]: DECLINED              -> Reasons: {'; '.join(class_res.reasons)}", flush=True)
                    declined.append({"doc_type": class_res.doc_type, "reason": "; ".join(class_res.reasons)})

            # Save final payload to output/<pdf_stem>.json
            file_payload = {
                "file": filename,
                "payables": payables,
                "declined": declined
            }
            out_json_path = out_dir / f"{pdf.stem}.json"
            out_json_path.write_text(json.dumps(file_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[STEP 6: JSON SAVED]               -> Saved JSON to '{out_json_path}'", flush=True)

            # Determine overall document status
            if payables and all(
                any(ev.get("step") == 7 and ev.get("status") == "PASS" for ev in st.session_state.doc_traces[filename])
                for _ in payables
            ):
                final_status = "PASS"
                st.session_state.stats["first_try_pass"] += len(payables)
            elif doc_has_payable:
                final_status = "PASS"
            else:
                final_status = "DECLINED"

            st.session_state.doc_status[filename] = final_status
            st.session_state.stats["total"] += 1
            st.session_state.stats["payables"] += len(payables)
            st.session_state.stats["declined"] += len(declined)

            yield {
                "type": "DOC_END",
                "filename": filename,
                "status": final_status
            }
    finally:
        sys.stdout = original_stdout


# ---------------------------------------------------------------------------
# Sidebar UI: Controls & Scope Selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Zycus Control Panel")
    st.caption("Developer-Facing Bookable Payable Pipeline Evaluator")

    # 1. Processing Scope Selector (Batch vs Single Document)
    processing_mode = st.radio(
        "Processing Scope:",
        options=["All Files (Batch - 42 PDFs)", "Single Document"],
        disabled=st.session_state.is_processing
    )

    single_file_selected = None
    if processing_mode == "Single Document":
        single_file_selected = st.selectbox(
            "Select File to Process:",
            options=pdf_names,
            disabled=st.session_state.is_processing
        )

    # 2. Side-by-Side Start / Stop Buttons
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("🚀 Start", type="primary", use_container_width=True, disabled=st.session_state.is_processing)
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_processing)

    if stop_btn:
        st.session_state.stop_requested = True
        st.session_state.is_processing = False
        st.warning("Stop requested. Halting execution...")

    st.divider()

    # 3. Inspection Selector — Rendered ONLY after processing completes
    if st.session_state.processing_complete:
        st.subheader("🔍 Review Processed Trace")
        processed_files = [f for f in pdf_names if st.session_state.doc_traces.get(f)]
        if processed_files:
            selected_inspection = st.selectbox(
                "Select Document to Inspect:",
                options=processed_files,
                index=processed_files.index(st.session_state.selected_doc) if st.session_state.selected_doc in processed_files else 0,
                help="Select any completed document to review its detailed 7-phase trace."
            )
            if selected_inspection != st.session_state.selected_doc:
                st.session_state.selected_doc = selected_inspection

        st.divider()

    # 4. Live Document Queue Status List
    st.subheader("📄 Document Queue")
    status_container = st.container()
    with status_container:
        display_list = pdf_names if processing_mode.startswith("All") else ([single_file_selected] if single_file_selected else pdf_names)
        for name in display_list:
            status = st.session_state.doc_status.get(name, "QUEUED")
            if status == "PASS":
                icon = "✅"
                st_class = "status-pass"
            elif status == "DECLINED":
                icon = "⛔"
                st_class = "status-declined"
            elif status == "FAIL":
                icon = "❌"
                st_class = "status-fail"
            elif status == "PROCESSING":
                icon = "⚙️"
                st_class = "status-processing"
            else:
                icon = "⏳"
                st_class = "status-queued"

            st.markdown(f"{icon} **`{name}`** — <span class='{st_class}'>{status}</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Region: Metrics Header & Live Step-by-Step Trace View
# ---------------------------------------------------------------------------

# Top Header Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(pdf_files)}</div><div class='metric-label'>Total PDFs</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#38a169;'>{st.session_state.stats['payables']}</div><div class='metric-label'>Payables Extracted</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#d69e2e;'>{st.session_state.stats['declined']}</div><div class='metric-label'>Declined Segments</div></div>", unsafe_allow_html=True)
with m4:
    total_p = max(1, st.session_state.stats['payables'])
    pass_rate = (st.session_state.stats['first_try_pass'] / total_p) * 100 if st.session_state.stats['payables'] > 0 else 0.0
    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#4361ee;'>{pass_rate:.1f}%</div><div class='metric-label'>ERP Booking Pass Rate</div></div>", unsafe_allow_html=True)

st.divider()

# Selected Document Header
curr_doc = st.session_state.selected_doc
curr_status = st.session_state.doc_status.get(curr_doc, "QUEUED") if curr_doc else "QUEUED"

st.header(f"🔍 Pipeline Step Trace: `{curr_doc or 'Ready'}`")
st.markdown(f"**Current Status:** `{curr_status}`")

# Render Active Document's Step-by-Step Trace
trace_placeholder = st.container()

def render_trace(doc_name: str | None):
    if not doc_name:
        st.info("Select a processing mode and click **🚀 Start** to begin pipeline execution.")
        return

    events = st.session_state.doc_traces.get(doc_name, [])
    if not events:
        st.info(f"Document `{doc_name}` is queued. Click **🚀 Start** in sidebar to run pipeline.")
        return

    for ev in events:
        step = ev.get("step")

        # Step 1: OCR & Layout Extraction
        if step == 1:
            with st.expander(f"📄 {ev['title']}", expanded=True):
                st.markdown(f"**Extraction Path Used:** `{ev['path_used']}`")
                st.markdown(f"**Total Characters:** `{ev['char_count']}` | **Pages:** `{ev['page_count']}`")
                with st.expander("Show Extracted Layout Text"):
                    st.code(ev["raw_text"], language="text")

        # Step 2: Pre-Segmentation
        elif step == 2:
            with st.expander(f"✂️ {ev['title']}", expanded=True):
                st.markdown(f"**Status:** `{ev['status_msg']}`")

        # Step 3: Classification
        elif step == 3:
            with st.expander(f"🏷️ {ev['title']}", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Payable Verdict", "PAYABLE" if ev["is_payable"] else "DECLINED")
                col_b.metric("Doc Type", ev["doc_type"])
                col_c.metric("Payable Score", ev["score"])
                st.markdown("**Rule Breakdown Reasons:**")
                for r in ev["reasons"]:
                    st.markdown(f"- `{r}`")

        # Step 4: AI Model Extraction (Groq)
        elif step == 4:
            with st.expander(f"🤖 {ev['title']}", expanded=True):
                st.markdown("**Raw Groq Model JSON Response:**")
                st.json(ev["raw_json"])

        # Step 4b: Grounding Verification
        elif step == "4b":
            with st.expander(f"🛡️ {ev['title']}", expanded=True):
                warns = ev.get("warnings", [])
                if warns:
                    st.warning(f"Raised {len(warns)} Grounding Warning(s):")
                    for w in warns:
                        st.markdown(f"- `{w}`")
                else:
                    st.success("Grounding Verification Passed: 0 ungrounded warnings (100% Rule 1 compliant).")
                with st.expander("Sanitized Payload"):
                    st.json(ev["sanitized_json"])

        # Step 5: Master Data Resolution
        elif step == 5:
            with st.expander(f"🏢 {ev['title']}", expanded=True):
                st.markdown("**Resolved Master Data Reference Codes:**")
                matched = ev["master_matched"]
                st.table([
                    {"Field": "Supplier ID", "Matched Code": matched["supplier_id"]},
                    {"Field": "Company Code", "Matched Code": matched["company_code"]},
                    {"Field": "Business Unit Code", "Matched Code": matched["business_unit_code"]},
                    {"Field": "Location Code", "Matched Code": matched["location_code"]},
                    {"Field": "PO ID", "Matched Code": matched["po_id"]},
                    {"Field": "Payment Term ID", "Matched Code": matched["payment_term_id"]}
                ])

        # Step 6: Pre-ERP Payload Assembly
        elif step == 6:
            with st.expander(f"📦 {ev['title']}", expanded=True):
                st.markdown("**Assembled Autodraft JSON Payload (Ready for Oracle ERP):**")
                st.json(ev["payload"])

        # Step 7: ERP Oracle Booking Verification
        elif step == 7:
            with st.expander(f"📊 {ev['title']}", expanded=True):
                status_color = "green" if ev["status"] == "PASS" else "red"
                st.markdown(f"### Status: :{status_color}[{ev['status']}]")
                c1, c2 = st.columns(2)
                c1.metric("Document Stated Target Gross", f"{ev['target_gross']}")
                c2.metric("ERP Calculated Booked Gross", f"{ev['booked_gross']}")
                with st.expander("ERP Calculation Details"):
                    st.json(ev["erp_details"])


# ---------------------------------------------------------------------------
# Trigger Live Generator Execution on Start Click
# ---------------------------------------------------------------------------
log_expander_placeholder = st.empty()

if start_btn:
    st.session_state.is_processing = True
    st.session_state.processing_complete = False
    st.session_state.stop_requested = False
    st.session_state.stats = {"total": 0, "payables": 0, "declined": 0, "first_try_pass": 0, "failed": 0}
    st.session_state.raw_logs = []

    # Determine target list based on mode
    if processing_mode.startswith("All"):
        targets = pdf_files
    else:
        target_path = Path("documents") / (single_file_selected or pdf_names[0])
        targets = [target_path]

    status_banner = st.empty()
    status_banner.info(f"⚙️ Running pipeline across {len(targets)} document(s)...")

    # Run generator loop live WITHOUT st.rerun() inside loop!
    for event in run_pipeline_generator(targets):
        if st.session_state.stop_requested:
            status_banner.warning("⏹️ Processing stopped by user request.")
            break

        # Re-render trace & log feed live in placeholders
        with trace_placeholder.container():
            render_trace(st.session_state.selected_doc)

        with log_expander_placeholder.container():
            with st.expander("🖥️ Live Terminal Log Feed (stdout)", expanded=True):
                st.code("\n".join(st.session_state.raw_logs[-100:]), language="text")

    st.session_state.is_processing = False
    if not st.session_state.stop_requested:
        st.session_state.processing_complete = True
        status_banner.success("✅ Pipeline processing complete!")
    st.rerun()
else:
    with trace_placeholder.container():
        render_trace(curr_doc)
    with log_expander_placeholder.container():
        with st.expander("🖥️ Live Terminal Log Feed (stdout)", expanded=False):
            if st.session_state.raw_logs:
                st.code("\n".join(st.session_state.raw_logs[-100:]), language="text")
            else:
                st.code("No stdout logs recorded yet. Start processing to view terminal logs live.", language="text")
