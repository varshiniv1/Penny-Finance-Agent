"""Detect and tag internal transfers, CC payments, and refunds across accounts."""
from __future__ import annotations

import re
from datetime import date

_PAYMENT_KW_RE = re.compile(r"payment|transfer|xfer|pymt|autopay", re.IGNORECASE)


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
        # With 3+ same-amount transactions nearby, matching the first
        # candidate found (list order = arbitrary parse order) can tag the
        # wrong pair. Score every valid candidate and take the closest match
        # by date instead.
        best_j, best_dist = None, None
        for j in candidates:
            if i == j:
                continue
            other = tagged[j]
            if other.get("is_internal") or _same_account(tx, other):
                continue
            dist = _date_distance(tx["date"], other["date"])
            if dist is None or dist > 3 or not _is_mirror(tx, other):
                continue
            if best_dist is None or dist < best_dist:
                best_j, best_dist = j, dist

        if best_j is not None:
            tagged[i]["is_internal"] = True
            tagged[best_j]["is_internal"] = True

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
    return bool(_PAYMENT_KW_RE.search(a.get("description", ""))) or bool(
        _PAYMENT_KW_RE.search(b.get("description", ""))
    )


def _date_distance(d1: str, d2: str) -> int | None:
    try:
        return abs((date.fromisoformat(d1) - date.fromisoformat(d2)).days)
    except ValueError:
        return None
