"""Merchant enrichment: regex cleanup + one-time batch web-search via Claude."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger

# ── Rule-based cleanup ────────────────────────────────────────────────────────

_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^AMZN\s+Mktp", re.I),          "Amazon",           "Shopping"),
    (re.compile(r"^Amazon\.com", re.I),            "Amazon",           "Shopping"),
    (re.compile(r"^SQ\s*\*",     re.I),            None,               "Dining"),   # merchant kept as-is after SQ *
    (re.compile(r"^TST\*",       re.I),            None,               "Dining"),
    (re.compile(r"^SP\s*\*",     re.I),            None,               "Shopping"),
    (re.compile(r"NETFLIX",      re.I),            "Netflix",          "Subscriptions"),
    (re.compile(r"SPOTIFY",      re.I),            "Spotify",          "Subscriptions"),
    (re.compile(r"HULU",         re.I),            "Hulu",             "Subscriptions"),
    (re.compile(r"APPLE\.COM",   re.I),            "Apple",            "Subscriptions"),
    (re.compile(r"GOOGLE\s+(PLAY|ONE|STORAGE)", re.I), "Google",       "Subscriptions"),
    (re.compile(r"UBER\s*EATS",  re.I),            "Uber Eats",        "Dining"),
    (re.compile(r"^UBER\b",      re.I),            "Uber",             "Transport"),
    (re.compile(r"LYFT",         re.I),            "Lyft",             "Transport"),
    (re.compile(r"DOORDASH",     re.I),            "DoorDash",         "Dining"),
    (re.compile(r"WHOLE\s*FOODS",re.I),            "Whole Foods",      "Groceries"),
    (re.compile(r"TRADER\s*JOE", re.I),            "Trader Joe's",     "Groceries"),
    (re.compile(r"COSTCO",       re.I),            "Costco",           "Groceries"),
    (re.compile(r"STARBUCKS",    re.I),            "Starbucks",        "Dining"),
    (re.compile(r"DUNKIN",       re.I),            "Dunkin'",          "Dining"),
    (re.compile(r"CVS\s*(PHARM)?",re.I),           "CVS",              "Health"),
    (re.compile(r"WALGREENS",    re.I),            "Walgreens",        "Health"),
    (re.compile(r"TARGET",       re.I),            "Target",           "Shopping"),
    (re.compile(r"WALMART",      re.I),            "Walmart",          "Shopping"),
    (re.compile(r"(ATM|CASH\s*WITHDRAWAL)", re.I),"ATM Withdrawal",   "Cash"),
    (re.compile(r"PAYROLL|DIRECT\s*DEP", re.I),   "Payroll",          "Income"),
    (re.compile(r"VENMO",        re.I),            "Venmo",            "Transfer"),
    (re.compile(r"ZELLE",        re.I),            "Zelle",            "Transfer"),
    (re.compile(r"ACH\s+(PAYMENT|TRANSFER)", re.I),"ACH Transfer",    "Transfer"),
]


def apply_rules(descriptor: str) -> tuple[str | None, str | None]:
    """Return (merchant, category) from rules, or (None, None) if no match."""
    for pattern, merchant, category in _RULES:
        if pattern.search(descriptor):
            if merchant is None:
                # Extract the bit after the prefix (e.g. "SQ *Blue Bottle" → "Blue Bottle")
                merchant = re.sub(r"^(SQ\s*\*|TST\*|SP\s*\*)", "", descriptor, flags=re.I).strip()
                merchant = merchant.split()[0].title() if merchant else descriptor
            return merchant, category
    return None, None


# ── Batch web-search enrichment ───────────────────────────────────────────────

def enrich_batch(ledger: "Ledger", api_key: str, max_batch: int = 50) -> dict:
    """
    For each uncached descriptor:
      1. Try rule-based first.
      2. For unknowns, ask Claude with web_search.
    Returns {"rules": n, "web": n, "ambiguous": n}.
    """
    import anthropic

    descriptors = ledger.uncached_descriptors()
    if not descriptors:
        return {"rules": 0, "web": 0, "ambiguous": 0}

    rule_entries, web_descriptors = [], []
    for d in descriptors:
        merchant, category = apply_rules(d)
        if merchant:
            rule_entries.append({"descriptor": d, "merchant": merchant, "category": category})
        else:
            web_descriptors.append(d)

    ledger.upsert_merchants(rule_entries)

    client = anthropic.Anthropic(api_key=api_key)
    web_entries, ambiguous = [], []

    for descriptor in web_descriptors[:max_batch]:
        result = _web_lookup(client, descriptor)
        if result:
            web_entries.append({"descriptor": descriptor, **result})
        else:
            ambiguous.append(descriptor)

    ledger.upsert_merchants(web_entries)
    ledger.apply_merchant_names()

    return {"rules": len(rule_entries), "web": len(web_entries), "ambiguous": len(ambiguous)}


def _web_lookup(client, descriptor: str) -> dict | None:
    """Ask Claude to identify the merchant via web search."""
    prompt = (
        f'What business or merchant corresponds to this bank statement descriptor: "{descriptor}"? '
        "Reply with exactly two lines:\nMerchant: <business name>\nCategory: <one of: Dining, Shopping, "
        "Groceries, Transport, Health, Subscriptions, Utilities, Entertainment, Travel, Income, Transfer, Cash, Other>"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp)
        return _parse_lookup_response(text)
    except Exception:
        return None


def _extract_text(response) -> str:
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _parse_lookup_response(text: str) -> dict | None:
    merchant, category = None, None
    for line in text.splitlines():
        if line.lower().startswith("merchant:"):
            merchant = line.split(":", 1)[1].strip()
        elif line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
    if merchant and category:
        return {"merchant": merchant, "category": category}
    return None
