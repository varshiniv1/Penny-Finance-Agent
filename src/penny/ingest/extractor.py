"""Normalise raw parser rows into clean Transaction dicts ready for DB insert."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

_ACCT_NUM_RE = re.compile(r"\b\d{5,}\d{4}\b")

_DATE_FORMATS = [
    "%m/%d/%Y", "%m/%d/%y",
    "%Y-%m-%d",
    "%b %d, %Y", "%b %d %Y", "%b. %d, %Y",
    "%d-%b-%Y", "%d-%b-%y",
    "%B %d, %Y",
]


@dataclass
class Transaction:
    id: str
    date: str
    description: str
    amount: float
    account_last4: str
    source_file: str
    page_num: int
    merchant: str = ""
    category: str = ""
    account: str = ""
    is_transfer: bool = False
    is_refund: bool = False
    is_internal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract(raw: dict) -> Transaction | None:
    date_str = _parse_date(str(raw.get("date", "")))
    if date_str is None:
        return None

    description = _mask_account_numbers(str(raw.get("description", "")).strip())
    if not description:
        return None

    try:
        amount = float(raw.get("amount", 0))
    except (ValueError, TypeError):
        return None

    account_last4 = str(raw.get("account_last4", ""))[-4:]
    source_file = str(raw.get("source_file", ""))
    page_num = int(raw.get("page_num", 0))

    tx_id = _make_id(date_str, description, amount, source_file, page_num)

    is_refund = amount < 0 or bool(
        re.search(r"\b(refund|credit|return|reversal)\b", description, re.IGNORECASE)
    )

    return Transaction(
        id=tx_id,
        date=date_str,
        description=description,
        amount=amount,
        account_last4=account_last4,
        source_file=source_file,
        page_num=page_num,
        is_refund=is_refund,
    )


def _parse_date(raw: str) -> str | None:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _mask_account_numbers(text: str) -> str:
    return _ACCT_NUM_RE.sub(lambda m: "****" + m.group()[-4:], text)


def _make_id(date_str: str, description: str, amount: float, source_file: str, page_num: int) -> str:
    raw = f"{date_str}|{description}|{amount:.2f}|{source_file}|{page_num}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
