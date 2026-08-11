"""Detect and tag internal transfers, CC payments, and refunds across accounts."""
from __future__ import annotations

from datetime import date, timedelta


def reconcile(transactions: list[dict]) -> list[dict]:
    """Tag is_internal=True on matching cross-account pairs (CC payments, transfers)."""
    tagged = [dict(t) for t in transactions]

    # Index by (amount, date window) for fast matching
    by_amount: dict[float, list[int]] = {}
    for i, t in enumerate(tagged):
        key = round(abs(t["amount"]), 2)
        by_amount.setdefault(key, []).append(i)

    for i, tx in enumerate(tagged):
        if tx.get("is_internal"):
            continue
        candidates = by_amount.get(round(abs(tx["amount"]), 2), [])
        for j in candidates:
            if i == j:
                continue
            other = tagged[j]
            if other.get("is_internal"):
                continue
            # Different accounts, opposite-sign or same amount on nearby dates
            if _same_account(tx, other):
                continue
            if _dates_close(tx["date"], other["date"], days=3) and _is_mirror(tx, other):
                tagged[i]["is_internal"] = True
                tagged[j]["is_internal"] = True
                break

    return tagged


def _same_account(a: dict, b: dict) -> bool:
    la, lb = a.get("account_last4", ""), b.get("account_last4", "")
    return bool(la and lb and la == lb)


def _is_mirror(a: dict, b: dict) -> bool:
    """True if one transaction looks like a payment of the other."""
    # Opposite signs: e.g. -500 debit + 500 credit
    if a["amount"] * b["amount"] < 0:
        return True
    # Same-sign but description contains payment keywords
    payment_kw = r"payment|transfer|xfer|pymt|autopay"
    import re
    for t in (a, b):
        if re.search(payment_kw, t.get("description", ""), re.IGNORECASE):
            return True
    return False


def _dates_close(d1: str, d2: str, days: int = 3) -> bool:
    try:
        dt1 = date.fromisoformat(d1)
        dt2 = date.fromisoformat(d2)
        return abs((dt1 - dt2).days) <= days
    except ValueError:
        return False
