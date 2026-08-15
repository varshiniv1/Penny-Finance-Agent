"""Merchant enrichment: regex cleanup + one-time batch web-search via Claude."""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from penny.config import DEFAULT_MODEL

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger

# ── Rule-based cleanup ────────────────────────────────────────────────────────

_RULES: list[tuple[re.Pattern, str, str]] = [
    # More specific Amazon rules first — order matters, first match wins.
    (re.compile(r"AMAZON\s*PRIME", re.I),          "Amazon Prime",     "Subscriptions"),
    (re.compile(r"AMAZON|AMZN",   re.I),           "Amazon",           "Shopping"),
    (re.compile(r"COMCAST|XFINITY", re.I),         "Comcast",          "Utilities"),
    (re.compile(r"DISCOVER\s*E-?Payment", re.I),   "Discover Card Payment", "Transfer"),
    (re.compile(r"AMERICAN\s*EXPRESS|AMERICANEXPRESS", re.I), "American Express", "Transfer"),
    (re.compile(r"MINT\s*MOBILE", re.I),           "Mint Mobile",      "Utilities"),
    (re.compile(r"PLAYSTATION",  re.I),            "PlayStation Network", "Entertainment"),
    (re.compile(r"GEICO",        re.I),            "Geico",            "Insurance"),
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

def enrich_batch(
    ledger: "Ledger", api_key: str, max_batch: int = 50, model: str = DEFAULT_MODEL
) -> dict:
    """
    For each uncached descriptor:
      1. Try rule-based first.
      2. For unknowns, ask Claude to classify all of them in one batched request
         (web_search stays available for anything it's unsure of, but as a tool
         call within that single request, not a separate round-trip per item).
    Returns {"rules": n, "web": n, "ambiguous": n, "usage": <Usage | None>}.
    """
    import anthropic

    descriptors = ledger.uncached_descriptors()
    if not descriptors:
        return {"rules": 0, "web": 0, "ambiguous": 0, "usage": None}

    rule_entries, remaining = [], []
    for d in descriptors:
        merchant, category = apply_rules(d)
        if merchant:
            rule_entries.append({"descriptor": d, "merchant": merchant, "category": category})
        else:
            remaining.append(d)

    ledger.upsert_merchants(rule_entries)

    web_entries, ambiguous = [], []
    total_usage = None
    if remaining:
        client = anthropic.Anthropic(api_key=api_key)
        for start in range(0, len(remaining), max_batch):
            batch = remaining[start : start + max_batch]
            results, usage = _batch_lookup(client, batch, model)
            if usage is not None:
                total_usage = _add_usage(total_usage, usage)
            for d in batch:
                result = results.get(d)
                if result:
                    web_entries.append({"descriptor": d, **result})
                else:
                    ambiguous.append(d)

    ledger.upsert_merchants(web_entries)
    ledger.apply_merchant_names()

    return {
        "rules": len(rule_entries), "web": len(web_entries), "ambiguous": len(ambiguous),
        "usage": total_usage,
    }


def _add_usage(total, usage):
    """Accumulate token usage across multiple chunked batch calls into one
    object Observability logging can read the same way as a single-call Usage."""
    from types import SimpleNamespace

    if total is None:
        total = SimpleNamespace(
            input_tokens=0, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", None) or 0
    total.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", None) or 0
    return total


_CATEGORIES = (
    "Dining, Shopping, Groceries, Transport, Health, Insurance, Subscriptions, "
    "Utilities, Entertainment, Travel, Income, Transfer, Cash, Taxes & Government, Other"
)
_BATCH_LINE_RE = re.compile(r"^\s*(\d+)\.\s*Merchant:\s*(.+?)\s*\|\s*Category:\s*(.+?)\s*$", re.IGNORECASE)
# The prompt asks for a plain "Merchant: X | Category: Y" line, but the model
# sometimes wraps a field in markdown emphasis anyway (caught a real
# "**Entertainment**" leaking into merchant_cache this way) — strip it rather
# than store it verbatim.
_MARKDOWN_EMPHASIS_RE = re.compile(r"^[*_`]+|[*_`]+$")


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_EMPHASIS_RE.sub("", text)


def _batch_lookup(
    client, descriptors: list[str], model: str = DEFAULT_MODEL
) -> tuple[dict[str, dict], Any]:
    """Ask Claude to identify merchant + category for all descriptors in one request.

    Returns (results_by_descriptor, usage) — usage is None if the call failed.
    """
    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(descriptors, start=1))
    prompt = (
        "Identify the merchant and spending category for each of these bank statement "
        "line descriptions. Use web search only for names you don't already recognize. "
        f"Reply with exactly one line per item, in order, formatted as:\n"
        f"<number>. Merchant: <business name> | Category: <one of: {_CATEGORIES}>\n\n"
        f"{numbered}"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=min(4096, 60 * len(descriptors) + 256),
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return {}, None

    text = _extract_text(resp)
    by_index: dict[int, dict] = {}
    for line in text.splitlines():
        m = _BATCH_LINE_RE.match(line)
        if m:
            idx, merchant, category = m.groups()
            by_index[int(idx)] = {
                "merchant": _strip_markdown(merchant), "category": _strip_markdown(category)
            }

    results = {
        descriptor: by_index[i]
        for i, descriptor in enumerate(descriptors, start=1)
        if i in by_index
    }
    return results, resp.usage


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if hasattr(block, "text"))
