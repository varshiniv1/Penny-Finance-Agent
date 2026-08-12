"""Parse PDF, scanned PDF, and image statements into raw row dicts.

Strategy:
  1. Text PDF   → pdfplumber (fast, exact)
  2. Image PDF  → pdf2image → pytesseract OCR per page
  3. Image file → pytesseract OCR directly
  4. CSV        → csv.DictReader

Detection: if pdfplumber extracts < 50 chars on a page, fall back to OCR.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date as _Date, datetime
from pathlib import Path
from typing import Any

_DATE_PATTERNS = [
    r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\w{3}\.?\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}-\w{3}-\d{2,4})\b",
    r"\b(\d{1,2}/\d{1,2})\b",  # bare MM/DD (e.g. Chase) — year resolved from statement period
]
_AMOUNT_RE = re.compile(r"(-?\$?[\d,]+\.\d{2})")
_ACCT_RE = re.compile(r"\b\d{4,}(\d{4})\b")
_PERIOD_RE = re.compile(
    r"([A-Za-z]+ \d{1,2},? \d{4})\s*through\s*([A-Za-z]+ \d{1,2},? \d{4})", re.IGNORECASE
)
_BARE_MD_RE = re.compile(r"^\d{1,2}/\d{1,2}$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_file(path: Path | str) -> list[dict[str, Any]]:
    """Dispatch to the right parser based on file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(p)
    elif ext == ".csv":
        return parse_csv(p)
    elif ext in _IMAGE_EXTS:
        return parse_image(p)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def parse_file_bytes(data: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse from in-memory bytes (for Streamlit uploads)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf_bytes(data, filename)
    elif ext == ".csv":
        return _parse_csv_bytes(data, filename)
    elif ext in _IMAGE_EXTS:
        return _parse_image_bytes(data, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return _parse_pdf_bytes(f.read(), str(path))


def _parse_pdf_bytes(data: bytes, source_file: str) -> list[dict[str, Any]]:
    import pdfplumber

    rows: list[dict] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        account_last4 = _extract_account_hint_from_text(first_page_text)
        period = _extract_statement_period(first_page_text)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < 50:
                # Scanned page — fall back to OCR
                text = _ocr_pdf_page_bytes(data, page_num - 1)

            rows.extend(_parse_text(text, page_num, account_last4, source_file, period))

            for table in page.extract_tables() or []:
                rows.extend(_parse_table(table, page_num, account_last4, source_file, period))

    return _dedup(rows)


def _ocr_pdf_page_bytes(pdf_bytes: bytes, page_index: int) -> str:
    """Convert a single PDF page to text via OCR."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        raise ImportError("Install pdf2image and pytesseract for scanned PDF support.")

    images = convert_from_bytes(pdf_bytes, first_page=page_index + 1, last_page=page_index + 1, dpi=200)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


# ── Image ─────────────────────────────────────────────────────────────────────

def parse_image(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return _parse_image_bytes(f.read(), str(path))


def _parse_image_bytes(data: bytes, source_file: str) -> list[dict[str, Any]]:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ImportError("Install Pillow and pytesseract for image support.")

    img = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(img)
    rows = _parse_text(text, page_num=1, account_last4="", source_file=source_file)
    return _dedup(rows)


# ── CSV ───────────────────────────────────────────────────────────────────────

def parse_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return _parse_csv_bytes(f.read().encode(), str(path))


def _parse_csv_bytes(data: bytes, source_file: str) -> list[dict[str, Any]]:
    rows = []
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader):
        mapped = _map_csv_row(row, source_file, i + 1)
        if mapped:
            rows.append(mapped)
    return rows


def _map_csv_row(row: dict, source_file: str, page_num: int) -> dict | None:
    date = _first(row, "Date", "Transaction Date", "Posted Date", "date", "Trans Date")
    desc = _first(row, "Description", "Merchant", "Name", "description", "Memo", "Payee")
    amount_str = _first(row, "Amount", "Debit", "amount", "Transaction Amount") or "0"
    credit_str = _first(row, "Credit", "credit") or ""

    if not date or not desc:
        return None
    try:
        amount = float(amount_str.replace(",", "").replace("$", "") or "0")
    except ValueError:
        return None
    if credit_str:
        try:
            amount = float(credit_str.replace(",", "").replace("$", ""))
        except ValueError:
            pass

    acct = _first(row, "Account", "account", "Account Number") or ""
    m = _ACCT_RE.search(acct)
    return {
        "date": date.strip(),
        "description": desc.strip(),
        "amount": amount,
        "account_last4": m.group(1) if m else "",
        "source_file": source_file,
        "page_num": page_num,
    }


# ── Text parsing helpers ──────────────────────────────────────────────────────

def _extract_account_hint_from_text(text: str) -> str:
    m = _ACCT_RE.search(text)
    return m.group(1) if m else ""


def _extract_statement_period(text: str) -> tuple[_Date, _Date] | None:
    """Find a 'Month DD, YYYY through Month DD, YYYY' statement period, if present."""
    m = _PERIOD_RE.search(text)
    if not m:
        return None
    try:
        start = datetime.strptime(m.group(1).replace(",", ""), "%B %d %Y").date()
        end = datetime.strptime(m.group(2).replace(",", ""), "%B %d %Y").date()
        return start, end
    except ValueError:
        return None


def _resolve_year(date_str: str, period: tuple[_Date, _Date] | None) -> str:
    """Fill in the year for a bare MM/DD date using the statement period, if known."""
    if not _BARE_MD_RE.match(date_str):
        return date_str
    month, day = (int(x) for x in date_str.split("/"))
    if period is None:
        year = _Date.today().year
    else:
        start, end = period
        year = end.year
        for candidate_year in {start.year, end.year}:
            try:
                candidate = _Date(candidate_year, month, day)
            except ValueError:
                continue
            if start <= candidate <= end:
                year = candidate_year
                break
    return f"{month}/{day}/{year}"


def _parse_text(
    text: str, page_num: int, account_last4: str, source_file: str,
    period: tuple[_Date, _Date] | None = None,
) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 8:
            continue
        raw_date = _extract_date(line)
        amount = _extract_amount(line)
        if raw_date and amount is not None:
            desc = _clean_description(line, raw_date, amount)
            if desc:
                rows.append({
                    "date": _resolve_year(raw_date, period),
                    "description": desc,
                    "amount": amount,
                    "account_last4": account_last4,
                    "source_file": source_file,
                    "page_num": page_num,
                })
    return rows


def _parse_table(
    table: list[list], page_num: int, account_last4: str, source_file: str,
    period: tuple[_Date, _Date] | None = None,
) -> list[dict]:
    if not table or len(table) < 2:
        return []
    rows = []
    for row in table[1:]:
        cells = [str(c or "").strip() for c in row]
        line = "  ".join(cells)
        raw_date = _extract_date(line)
        amount = _extract_amount(line)
        if raw_date and amount is not None:
            desc = _clean_description(line, raw_date, amount)
            rows.append({
                "date": _resolve_year(raw_date, period),
                "description": desc,
                "amount": amount,
                "account_last4": account_last4,
                "source_file": source_file,
                "page_num": page_num,
            })
    return rows


def _extract_date(text: str) -> str | None:
    for pat in _DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def _extract_amount(text: str) -> float | None:
    hits = _AMOUNT_RE.findall(text)
    if not hits:
        return None
    # Lines with a running-balance column look like "... AMOUNT BALANCE" — the
    # transaction amount is second-to-last, the balance is last. With only one
    # number on the line, there's no balance column and it must be the amount.
    raw = hits[-2] if len(hits) >= 2 else hits[-1]
    try:
        # Statements print debits as negative and credits unsigned (or vice
        # versa) — flip to this app's convention: positive = expense/debit,
        # negative = credit/refund (see Transaction schema in agent/prompts.py).
        val = -float(raw.replace(",", "").replace("$", ""))
        if re.search(r"\b(CR|credit|refund)\b", text, re.IGNORECASE):
            val = -abs(val)
        return val
    except ValueError:
        return None


def _clean_description(line: str, date: str, amount: float) -> str:
    s = line
    s = re.sub(re.escape(date), "", s)
    s = re.sub(r"-?\$?[\d,]+\.\d{2}", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -|")
    return s or line[:60]


def _dedup(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["date"], r["description"][:30], r["amount"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _first(d: dict, *keys: str) -> str | None:
    for k in keys:
        v = d.get(k, "")
        if v and str(v).strip():
            return str(v).strip()
    return None
