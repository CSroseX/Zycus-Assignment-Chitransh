"""
ocr_engine.py — Phase 1: PDF → Raw OCR Text with bounding boxes.

Renders each PDF page to a 300 DPI image via PyMuPDF, then runs OCR (EasyOCR / PaddleOCR)
or native text extraction to obtain text fragments with spatial coordinates and confidence scores.

Usage:
    from src.ocr_engine import extract_pages, reconstruct_layout

    pages = extract_pages("documents/INV-01.pdf")
    for i, page in enumerate(pages):
        print(f"--- Page {i+1} ---")
        for frag in page:
            print(f"  [{frag['confidence']:.2f}] {frag['text']}")
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
import pymupdf as fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Lazy OCR Reader initialization
# Supports EasyOCR (primary for Py 3.14+) and PaddleOCR if available.
# ---------------------------------------------------------------------------
_ocr_instance = None
_ocr_type = None


def _get_ocr():
    """Lazy-init a single OCR reader instance (reused across calls)."""
    global _ocr_instance, _ocr_type
    if _ocr_instance is None:
        try:
            import easyocr
            _ocr_instance = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ocr_type = "easyocr"
        except ImportError:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            _ocr_type = "paddleocr"
    return _ocr_instance, _ocr_type


# ---------------------------------------------------------------------------
# Core: PDF → pages of OCR fragments
# ---------------------------------------------------------------------------

def render_page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """Render a single PyMuPDF page to a numpy RGB array at the given DPI."""
    zoom = dpi / 72.0  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return np.array(img)


def ocr_image(img_array: np.ndarray) -> list[dict]:
    """Run OCR on a numpy RGB image array.

    Returns a list of fragment dicts:
        {
            "text": str,
            "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 4 corners
            "confidence": float
        }
    """
    ocr, ocr_type = _get_ocr()
    fragments = []

    if ocr_type == "easyocr":
        # easyocr readtext returns list of (bbox, text, prob)
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        results = ocr.readtext(img_array)
        for bbox, text, prob in results:
            # Convert bbox coordinates (int/float) to standard float list
            clean_bbox = [[float(pt[0]), float(pt[1])] for pt in bbox]
            fragments.append({
                "text": text.strip(),
                "bbox": clean_bbox,
                "confidence": float(prob),
            })

    elif ocr_type == "paddleocr":
        raw_result = ocr.ocr(img_array, cls=True)
        if raw_result and raw_result[0] is not None:
            for line in raw_result[0]:
                bbox, (text, confidence) = line
                fragments.append({
                    "text": text.strip(),
                    "bbox": bbox,
                    "confidence": float(confidence),
                })

    return fragments


def extract_native_text(page: fitz.Page) -> list[dict]:
    """Extract native selectable text from a PyMuPDF page if available."""
    words = page.get_text("words")  # returns (x0, y0, x1, y1, word, block_no, line_no, word_no)
    fragments = []
    for w in words:
        x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
        fragments.append({
            "text": word,
            "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "confidence": 1.0,
        })
    return fragments


def is_digital_vector_page(page: fitz.Page) -> bool:
    """Check if a PyMuPDF page is a digital PDF page with native vector text."""
    blocks = page.get_text("blocks")
    if not blocks:
        return False

    images = page.get_images()
    page_area = abs(page.rect.width * page.rect.height)

    # Check if page has a single full-page background image covering >80% of canvas
    has_fullpage_background = False
    if page_area > 0 and images:
        for img in images:
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                for r in img_rects:
                    if (abs(r.width * r.height) / page_area) > 0.80:
                        has_fullpage_background = True
                        break
            except Exception:
                pass

    # If page has native text blocks and NO full-page raster image overlay, it is digital
    if len(blocks) >= 1 and not has_fullpage_background:
        return True

    # Fallback: if native text has 10+ words, treat as digital
    words = page.get_text("words")
    return len(words) >= 10


def extract_pages(pdf_path: str, dpi: int = 300) -> list[list[dict]]:
    """Extract text from every page of a PDF.

    Uses native vector text if present (digital PDF), otherwise runs OCR (scanned PDF).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        if is_digital_vector_page(page):
            native_frags = extract_native_text(page)
            all_pages.append(native_frags)
        else:
            img_array = render_page_to_image(page, dpi=dpi)
            fragments = ocr_image(img_array)
            all_pages.append(fragments)

    doc.close()
    return all_pages


# ---------------------------------------------------------------------------
# Phase 2: Scale-Invariant Spatial Layout Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_layout(fragments: list[dict], line_threshold: float | None = None) -> str:
    """Convert OCR/Text fragments into a spatially-aware layout string.

    Uses dynamic scale-invariant line heights and character width calculations
    derived from the page's bounding box statistics (DPI-independent).
    """
    if not fragments:
        return ""

    annotated = []
    heights = []
    char_widths = []

    for frag in fragments:
        x0 = frag["bbox"][0][0]
        y0 = frag["bbox"][0][1]
        x1 = frag["bbox"][1][0]
        y1 = frag["bbox"][2][1]

        h = abs(y1 - y0)
        w = abs(x1 - x0)
        text_len = max(1, len(frag["text"]))

        if h > 0:
            heights.append(h)
        if w > 0:
            char_widths.append(w / text_len)

        annotated.append({
            "text": frag["text"],
            "x": x0,
            "y": y0,
            "x_end": x1,
            "confidence": frag.get("confidence", 1.0),
        })

    # Dynamic line height & character width calculation
    median_h = float(np.median(heights)) if heights else 16.0
    median_char_w = float(np.median(char_widths)) if char_widths else 8.0

    if line_threshold is None:
        line_threshold = max(6.0, 0.5 * median_h)

    char_width_px = max(4.0, median_char_w)

    annotated.sort(key=lambda f: (f["y"], f["x"]))

    lines: list[list[dict]] = []
    current_line: list[dict] = [annotated[0]]

    for frag in annotated[1:]:
        if abs(frag["y"] - current_line[0]["y"]) <= line_threshold:
            current_line.append(frag)
        else:
            lines.append(current_line)
            current_line = [frag]
    lines.append(current_line)

    output_lines = []
    for line_frags in lines:
        line_frags.sort(key=lambda f: f["x"])
        parts = []
        for i, frag in enumerate(line_frags):
            if i > 0:
                gap_px = frag["x"] - line_frags[i - 1]["x_end"]
                num_spaces = max(1, int(gap_px / char_width_px))
                parts.append(" " * num_spaces)
            parts.append(frag["text"])
        output_lines.append("".join(parts))

    return "\n".join(output_lines)


def extract_text(pdf_path: str, dpi: int = 300) -> list[str]:
    """Extract text from a PDF with spatial layout preserved."""
    pages = extract_pages(pdf_path, dpi=dpi)
    return [reconstruct_layout(page) for page in pages]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr_engine <pdf_path_or_dir> [output_dir]", file=sys.stderr)
        sys.exit(1)

    target_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_path.is_dir():
        pdf_files = list(target_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs in {target_path}. Processing into '{out_dir}'...\n")
        for pdf in sorted(pdf_files):
            print(f"Processing {pdf.name}...")
            text_pages = extract_text(str(pdf))
            out_file = out_dir / f"{pdf.stem}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n\n--- PAGE BREAK ---\n\n".join(text_pages))
            print(f"  -> Saved to {out_file}")
        print(f"\nCompleted! All text files saved in '{out_dir}/'")

    else:
        print(f"Extracting text from: {target_path}\n")
        pages = extract_pages(str(target_path))
        full_text = []
        for i, page_fragments in enumerate(pages):
            print(f"{'='*60}")
            print(f"PAGE {i + 1}  ({len(page_fragments)} fragments)")
            print(f"{'='*60}")

            layout = reconstruct_layout(page_fragments)
            print(layout)
            print()
            full_text.append(layout)

        out_file = out_dir / f"{target_path.stem}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n\n--- PAGE BREAK ---\n\n".join(full_text))

        print(f"Saved extracted text to: {out_file}")
