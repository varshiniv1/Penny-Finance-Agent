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
import hashlib
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
_CREDIT_RE = re.compile(r"\b(CR|credit|refund)\b", re.IGNORECASE)

# Credit-card statements print purchases as plain positive amounts and
# payments/credits as negative (or CR-marked) — the opposite convention from
# a bank/checking register, where withdrawals print negative and deposits
# print positive. Requiring 2+ distinct CC-specific markers on the first page
# keeps this from misfiring on an ordinary bank statement that just happens
# to mention "payment" once.
_CC_MARKERS_RE = re.compile(
    r"\b(minimum payment due|credit limit|new balance|previous balance|"
    r"payment due date|available credit)\b",
    re.IGNORECASE,
)


def _detect_statement_type(first_page_text: str) -> str:
    """Return 'credit_card' if the statement clearly reads as a card statement,
    else 'bank' (the default, existing sign convention)."""
    hits = set(m.lower() for m in _CC_MARKERS_RE.findall(first_page_text))
    return "credit_card" if len(hits) >= 2 else "bank"


# ── Public API ────────────────────────────────────────────────────────────────

def parse_file(path: Path | str) -> list[dict[str, Any]]:
    """Dispatch to the right parser based on file extension."""
    p = Path(path)
    return parse_file_bytes(p.read_bytes(), str(p))


def parse_file_bytes(data: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse from in-memory bytes (for Streamlit uploads)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        rows = _parse_pdf_bytes(data, filename)
    elif ext == ".csv":
        rows = _parse_csv_bytes(data, filename)
    elif ext in _IMAGE_EXTS:
        rows = _parse_image_bytes(data, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # A content hash (not the filename) is what makes the same statement
    # re-uploaded under a different name (browser auto-rename, re-export)
    # dedup correctly downstream instead of double-counting — see extractor.py.
    content_hash = hashlib.sha1(data).hexdigest()[:16]
    for r in rows:
        r["content_hash"] = content_hash
    return rows


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return _parse_pdf_bytes(f.read(), str(path))


def _parse_pdf_bytes(data: bytes, source_file: str) -> list[dict[str, Any]]:
    import pdfplumber

    rows: list[dict] = []
    ocr_images = None  # lazily rasterized once, only if a scanned page is hit
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        account_last4 = _extract_account_hint_from_text(first_page_text)
        period = _extract_statement_period(first_page_text)
        statement_type = _detect_statement_type(first_page_text)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < 50:
                # Scanned page — fall back to OCR. Rasterize the whole document
                # once on first use rather than per page (pdf2image re-parses
                # the full PDF bytes on every call otherwise).
                if ocr_images is None:
                    ocr_images = _rasterize_pdf(data)
                text = _ocr_page(ocr_images, page_num - 1)

            rows.extend(
                _parse_text(text, page_num, account_last4, source_file, period, statement_type)
            )

            for table in page.extract_tables() or []:
                rows.extend(
                    _parse_table(table, page_num, account_last4, source_file, period, statement_type)
                )

    return _dedup(rows)


def _rasterize_pdf(pdf_bytes: bytes) -> list:
    """Convert every page of the PDF to an image, once."""
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise ImportError("Install pdf2image and pytesseract for scanned PDF support.")
    return convert_from_bytes(pdf_bytes, dpi=200)


def _ocr_page(images: list, page_index: int) -> str:
    """OCR a single already-rasterized page."""
    try:
        import pytesseract
    except ImportError:
        raise ImportError("Install pdf2image and pytesseract for scanned PDF support.")
    if page_index >= len(images):
        return ""
    return pytesseract.image_to_string(images[page_index])


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
        # A value in a dedicated Credit column is unambiguous — money in —
        # regardless of how it's signed in the source file; this app's
        # convention is negative = credit/refund/income (see _extract_amount).
        try:
            amount = -abs(float(credit_str.replace(",", "").replace("$", "")))
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
    period: tuple[_Date, _Date] | None = None, statement_type: str = "bank",
) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 8:
            continue
        raw_date = _extract_date(line)
        amount = _extract_amount(line, statement_type)
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
    period: tuple[_Date, _Date] | None = None, statement_type: str = "bank",
) -> list[dict]:
    if not table or len(table) < 2:
        return []
    rows = []
    for row in table[1:]:
        cells = [str(c or "").strip() for c in row]
        line = "  ".join(cells)
        raw_date = _extract_date(line)
        amount = _extract_amount(line, statement_type)
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


def _extract_amount(text: str, statement_type: str = "bank") -> float | None:
    hits = _AMOUNT_RE.findall(text)
    if not hits:
        return None
    # Lines with a running-balance column look like "... AMOUNT BALANCE" — the
    # transaction amount is second-to-last, the balance is last. With only one
    # number on the line, there's no balance column and it must be the amount.
    raw = hits[-2] if len(hits) >= 2 else hits[-1]
    try:
        parsed = float(raw.replace(",", "").replace("$", ""))
        is_credit = bool(_CREDIT_RE.search(text))
        if statement_type == "credit_card":
            # Purchases print as plain positive (expense, stays positive);
            # payments/credits print negative or carry a credit marker.
            val = -abs(parsed) if is_credit else parsed
        else:
            # Bank/checking register: withdrawals print negative and deposits
            # print positive — flip to this app's convention: positive =
            # expense/debit, negative = credit/refund/income (see Transaction
            # schema in agent/prompts.py).
            val = -parsed
            if is_credit:
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
    """Collapse rows that text- and table-extraction both picked up from the
    same page. Keying on page_num (not just date/description/amount) means two
    genuinely distinct same-day, same-amount transactions on different pages
    both survive — though two such transactions on the very same page are
    still indistinguishable from a duplicate and will collapse to one."""
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["date"], r["description"][:30], r["amount"], r.get("page_num"))
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
