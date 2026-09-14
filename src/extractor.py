"""
extractor.py — Phase 4: Structured Auto-Draft Extraction via Groq / Gemini API.

Extracts structured header fields, line items, and taxes from OCR layout text
into autodraft JSON format complying strictly with AUTODRAFT_SCHEMA.md.

Includes:
- Dual LLM Client Support: Groq API (GROQ_API_KEY) & Gemini API (GEMINI_API_KEY).
- Rule 1 Grounding Verifier (src/grounding.py) to eliminate hallucinations.
- Rule 8 Prompt Constraint (unprinted unit_price remains blank "").
- Structural Audit Verifier (tax placement & component decomposition).
- Strict ERP Discrepancy Recovery Protocol (document-grounded re-reading only).
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.classifier import classify_file
from src.grounding import verify_payable_grounding
from src.ocr_engine import extract_text

load_dotenv(override=True)

# 1. Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# 2. OpenRouter API Configuration
OPEN_ROUTER_API_KEY = (os.getenv("OPEN_ROUTER_API") or os.getenv("OPENROUTER_API_KEY") or "").strip()
OPEN_ROUTER_MODEL = (os.getenv("OPEN_ROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3.2-3b-instruct").strip()

# 3. Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
MODEL_NAME = GEMINI_MODEL

# 4. Cloudflare Workers AI Configuration
CLOUDFLARE_WORKERS_AI_KEY = (os.getenv("CLOUDFLARE_WORKERS_AI") or os.getenv("CLOUDFLARE_API_KEY") or "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct").strip()

if not GEMINI_API_KEY or "lang-client" in GEMINI_API_KEY or "<" in GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and "<" not in GEMINI_API_KEY else None

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

    inv_num = ""
    inv_match = re.search(r'(?:invoice|rechnung|arve|facture|fatura)\s*(?:nr|no|num|#|\.)*:?\s*([a-zA-Z0-9\-_]+)', t, re.IGNORECASE)
    if inv_match:
        inv_num = inv_match.group(1).strip()

    inv_date = ""
    date_match = re.search(r'(?:date|datum|kuup\u00e4ev)\s*:?\s*(\d{1,4}[\./\-]\d{1,2}[\./\-]\d{1,4})', t, re.IGNORECASE)
    if date_match:
        inv_date = date_match.group(1).strip()

    curr = "EUR"
    if "$" in t or "USD" in t:
        curr = "USD"
    elif "ZAR" in t or "R " in t:
        curr = "ZAR"
    elif "MYR" in t or "RM" in t:
        curr = "MYR"
    elif "GBP" in t or "£" in t:
        curr = "GBP"

    gross_total = ""
    gross_match = re.search(r'(?:endbetrag|gesamtsumme|total zar|total eur|total usd|amount due|grand total|total amount|total)\s*:?\s*([\$€£]?\s*[\d\.,]+)', t, re.IGNORECASE)
    if gross_match:
        gross_total = parse_dot_decimal(gross_match.group(1))

    tax_rate = ""
    tax_amt = ""
    tax_match = re.search(r'(?:vat|mwst|tax|sttax)\s*(?:at|@)?\s*(\d+[\.,]?\d*)\s*%\s*:?\s*([\$€£]?\s*[\d\.,]+)?', t, re.IGNORECASE)
    if tax_match:
        tax_rate = parse_dot_decimal(tax_match.group(1))
        if tax_match.group(2):
            tax_amt = parse_dot_decimal(tax_match.group(2))

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
    """Raised when LLM API rate limit or quota is exhausted (HTTP 429 RESOURCE_EXHAUSTED)."""
    pass


def call_groq_api(ocr_text: str, filename: str = "") -> str:
    """Call Groq API (OpenAI-compatible Chat Completions) via stdlib urllib.request."""
    if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing or invalid.")

    full_prompt_input = f"{SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{ocr_text}"

    print("\n" + "=" * 80, flush=True)
    print(f"=== EXACT AI INPUT PAYLOAD FED TO GROQ FOR [{filename or 'DOCUMENT'}] ===", flush=True)
    print(f"Model Name: {GROQ_MODEL} | Chars: {len(full_prompt_input)} | Est Tokens: ~{len(full_prompt_input) // 4}", flush=True)
    print("=" * 80, flush=True)
    print(full_prompt_input, flush=True)
    print("=" * 80 + "\n", flush=True)

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT TEXT:\n{ocr_text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            return content.strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if e.code == 429 or "rate_limit_exceeded" in err_body.lower() or "quota" in err_body.lower():
            raise QuotaExhaustedError(f"Groq API Quota Exhausted ({GROQ_MODEL}): HTTP {e.code} - {err_body}") from e
        if e.code == 400 and ("json_validate_failed" in err_body.lower() or "validate json" in err_body.lower()):
            # Retry without response_format constraint
            payload_retry = dict(payload)
            payload_retry.pop("response_format", None)
            req_retry = urllib.request.Request(
                url,
                data=json.dumps(payload_retry).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                },
                method="POST"
            )
            try:
                with urllib.request.urlopen(req_retry) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    content = resp_data["choices"][0]["message"]["content"].strip()
                    # Clean markdown code block or extract JSON substring
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    start_idx = content.find("{")
                    end_idx = content.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        content = content[start_idx:end_idx+1]
                    return content
            except Exception as retry_err:
                raise RuntimeError(f"Groq API Call Failed: HTTP 400 - {err_body}") from retry_err

        raise RuntimeError(f"Groq API Call Failed: HTTP {e.code} - {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"Groq API Error: {e}") from e


def call_openrouter_api(ocr_text: str, filename: str = "") -> str:
    """Call OpenRouter API (OpenAI-compatible Chat Completions) via stdlib urllib.request."""
    if not OPEN_ROUTER_API_KEY or "<" in OPEN_ROUTER_API_KEY:
        raise ValueError("OPEN_ROUTER_API key is missing or invalid.")

    full_prompt_input = f"{SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{ocr_text}"

    print("\n" + "=" * 80, flush=True)
    print(f"=== EXACT AI INPUT PAYLOAD FED TO OPENROUTER FOR [{filename or 'DOCUMENT'}] ===", flush=True)
    print(f"Model Name: {OPEN_ROUTER_MODEL} | Chars: {len(full_prompt_input)} | Est Tokens: ~{len(full_prompt_input) // 4}", flush=True)
    print("=" * 80, flush=True)
    print(full_prompt_input, flush=True)
    print("=" * 80 + "\n", flush=True)

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": OPEN_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT TEXT:\n{ocr_text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            return content.strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if e.code == 429 or "rate_limit_exceeded" in err_body.lower() or "quota" in err_body.lower():
            raise QuotaExhaustedError(f"OpenRouter API Quota Exhausted ({OPEN_ROUTER_MODEL}): HTTP {e.code} - {err_body}") from e
        if e.code == 400 and ("json_validate_failed" in err_body.lower() or "validate json" in err_body.lower()):
            payload_retry = dict(payload)
            payload_retry.pop("response_format", None)
            req_retry = urllib.request.Request(
                url,
                data=json.dumps(payload_retry).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                },
                method="POST"
            )
            try:
                with urllib.request.urlopen(req_retry) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    content = resp_data["choices"][0]["message"]["content"].strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    start_idx = content.find("{")
                    end_idx = content.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        content = content[start_idx:end_idx+1]
                    return content
            except Exception as retry_err:
                raise RuntimeError(f"OpenRouter API Call Failed: HTTP 400 - {err_body}") from retry_err

        raise RuntimeError(f"OpenRouter API Call Failed: HTTP {e.code} - {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"OpenRouter API Error: {e}") from e


def get_raw_gemini_response(ocr_text: str, filename: str = "") -> str:
    """Call Gemini API directly and return the raw unparsed JSON string response."""
    if not (client and GEMINI_API_KEY and "<" not in GEMINI_API_KEY and "lang-client" not in GEMINI_API_KEY):
        raise ValueError("Gemini API key is missing or invalid.")

    full_prompt_input = f"{SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{ocr_text}"

    print("\n" + "=" * 80, flush=True)
    print(f"=== EXACT AI INPUT PAYLOAD FED TO GEMINI FOR [{filename or 'DOCUMENT'}] ===", flush=True)
    print(f"Model Name: {GEMINI_MODEL} | Chars: {len(full_prompt_input)} | Est Tokens: ~{len(full_prompt_input) // 4}", flush=True)
    print("=" * 80, flush=True)
    print(full_prompt_input, flush=True)
    print("=" * 80 + "\n", flush=True)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
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
            raise QuotaExhaustedError(f"Gemini API Quota Exhausted ({GEMINI_MODEL}): {e}") from e
        raise e


def call_cloudflare_workers_ai_api(ocr_text: str, filename: str = "") -> str:
    """Call Cloudflare Workers AI API via stdlib urllib.request."""
    if not CLOUDFLARE_WORKERS_AI_KEY or "<" in CLOUDFLARE_WORKERS_AI_KEY:
        raise ValueError("CLOUDFLARE_WORKERS_AI key is missing or invalid.")

    full_prompt_input = f"{SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{ocr_text}"

    print("\n" + "=" * 80, flush=True)
    print(f"=== EXACT AI INPUT PAYLOAD FED TO CLOUDFLARE WORKERS AI FOR [{filename or 'DOCUMENT'}] ===", flush=True)
    print(f"Model Name: {CLOUDFLARE_MODEL} | Chars: {len(full_prompt_input)} | Est Tokens: ~{len(full_prompt_input) // 4}", flush=True)
    print("=" * 80, flush=True)
    print(full_prompt_input, flush=True)
    print("=" * 80 + "\n", flush=True)

    if CLOUDFLARE_ACCOUNT_ID:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
        payload = {
            "model": CLOUDFLARE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENT TEXT:\n{ocr_text}"}
            ],
            "temperature": 0.1
        }
    else:
        # Fallback to general AI gateway / direct worker format if account_id is not specified
        url = f"https://api.cloudflare.com/client/v4/ai/v1/chat/completions"
        payload = {
            "model": CLOUDFLARE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENT TEXT:\n{ocr_text}"}
            ],
            "temperature": 0.1
        }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_WORKERS_AI_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            if "choices" in resp_data and resp_data["choices"]:
                content = resp_data["choices"][0]["message"]["content"].strip()
            elif "result" in resp_data and isinstance(resp_data["result"], dict) and "response" in resp_data["result"]:
                content = resp_data["result"]["response"].strip()
            else:
                content = str(resp_data)

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]
            return content
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        if e.code == 429 or "rate_limit" in err_body.lower() or "quota" in err_body.lower():
            raise QuotaExhaustedError(f"Cloudflare Workers AI Quota Exhausted ({CLOUDFLARE_MODEL}): HTTP {e.code} - {err_body}") from e
        raise RuntimeError(f"Cloudflare Workers AI Call Failed: HTTP {e.code} - {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"Cloudflare Workers AI Error: {e}") from e


def get_raw_llm_response(ocr_text: str, filename: str = "") -> str:
    """Route LLM extraction call to available API provider (OpenRouter -> Groq -> Cloudflare -> Gemini)."""
    if OPEN_ROUTER_API_KEY and "<" not in OPEN_ROUTER_API_KEY:
        try:
            return call_openrouter_api(ocr_text, filename=filename)
        except Exception as e:
            print(f"OpenRouter API failed ({e}), trying fallback providers...", file=sys.stderr)
    if GROQ_API_KEY and "gsk_" in GROQ_API_KEY and "<" not in GROQ_API_KEY:
        try:
            return call_groq_api(ocr_text, filename=filename)
        except Exception as e:
            print(f"Groq API failed ({e}), trying fallback providers...", file=sys.stderr)
    if CLOUDFLARE_WORKERS_AI_KEY and "<" not in CLOUDFLARE_WORKERS_AI_KEY:
        try:
            return call_cloudflare_workers_ai_api(ocr_text, filename=filename)
        except Exception as e:
            print(f"Cloudflare Workers AI API failed ({e}), trying fallback providers...", file=sys.stderr)
    return get_raw_gemini_response(ocr_text, filename=filename)


from src.master_matcher import MasterDataMatcher

_matcher = MasterDataMatcher()


def extract_payable_from_text(ocr_text: str, filename: str = "", allow_fallback: bool = False) -> dict:
    """Extract structured autodraft JSON from OCR layout text using OpenRouter, Groq, Cloudflare, or Gemini API."""
    has_openrouter = bool(OPEN_ROUTER_API_KEY and "<" not in OPEN_ROUTER_API_KEY)
    has_groq = bool(GROQ_API_KEY and "gsk_" in GROQ_API_KEY and "<" not in GROQ_API_KEY)
    has_cloudflare = bool(CLOUDFLARE_WORKERS_AI_KEY and "<" not in CLOUDFLARE_WORKERS_AI_KEY)
    has_gemini = bool(client and GEMINI_API_KEY and "<" not in GEMINI_API_KEY and "lang-client" not in GEMINI_API_KEY)

    if has_openrouter or has_groq or has_cloudflare or has_gemini:
        try:
            raw_json = get_raw_llm_response(ocr_text, filename=filename)
            try:
                payable_data = json.loads(raw_json, strict=False)
            except Exception:
                # Fallback: sanitize unescaped newlines inside JSON strings
                clean_json = re.sub(r'(?<!\\)[\r\n]+', ' ', raw_json)
                payable_data = json.loads(clean_json, strict=False)
            grounded_payable, _ = verify_payable_grounding(payable_data, ocr_text)
            verify_structural_integrity(grounded_payable)
            return _matcher.resolve_payable(grounded_payable, text_context=ocr_text)
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"LLM API Call Failed: {e}") from e
            print(f"LLM API error ({e}). Explicit fallback allowed.", file=sys.stderr)

    if allow_fallback:
        payable_data = deterministic_extract_payable(ocr_text, filename=filename)
        grounded_payable, _ = verify_payable_grounding(payable_data, ocr_text)
        return _matcher.resolve_payable(grounded_payable, text_context=ocr_text)

    raise ValueError("No valid OPEN_ROUTER_API, GROQ_API_KEY, CLOUDFLARE_WORKERS_AI or GEMINI_API_KEY configured and allow_fallback=False.")


from src.classifier import classify_document_text, classify_file
from src.segmenter import segment_document_text


def process_document_file(pdf_or_txt_path: str | Path) -> dict:
    """Process a single document PDF or .txt into per-file Autodraft JSON payload with multi-document segmentation."""
    path = Path(pdf_or_txt_path)

    if path.suffix == ".pdf":
        pages = extract_text(str(path))
        full_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        file_name = path.name
    else:
        full_ocr_text = path.read_text(encoding="utf-8", errors="ignore")
        file_name = path.stem + ".pdf"

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
