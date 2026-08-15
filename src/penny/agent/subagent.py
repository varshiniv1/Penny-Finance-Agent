"""On-demand sub-agent: a focused, independent Claude call the main chat
agent can delegate to mid-conversation, separate from its own context.

Same "standalone call" shape as agent/insights.py and ingest/enrich.py's
_batch_lookup, but invoked live for a single transaction from inside a chat
turn (via the categorize_transaction tool) rather than as a batch job.
"""
from __future__ import annotations

import re
from typing import Any

from penny.config import DEFAULT_MODEL

_CATEGORIES = (
    "Dining, Shopping, Groceries, Transport, Health, Subscriptions, "
    "Utilities, Entertainment, Travel, Income, Transfer, Cash, Other"
)
_SUBAGENT_SYSTEM_PROMPT = (
    "You are a merchant-classification specialist sub-agent. You receive a single "
    "raw bank statement line and identify the real-world merchant and spending "
    "category behind it. Use web search only for names you don't already recognize. "
    "Reply with exactly two lines:\n"
    "Merchant: <business name>\n"
    f"Category: <one of: {_CATEGORIES}>"
)
_LINE_RE = re.compile(r"^(merchant|category):\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# The prompt asks for a plain "Merchant: X" / "Category: Y" reply, but the
# model sometimes wraps a field in markdown emphasis anyway (caught a real
# "**Entertainment**" leaking into merchant_cache this way) — strip it rather
# than store it verbatim.
_MARKDOWN_EMPHASIS_RE = re.compile(r"^[*_`]+|[*_`]+$")


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_EMPHASIS_RE.sub("", text)


def categorize_merchant(
    descriptor: str, api_key: str, model: str = DEFAULT_MODEL
) -> tuple[dict, Any] | None:
    """Delegate classification of one transaction descriptor to a sub-agent.

    Returns ({"merchant": ..., "category": ...}, usage), or None if the
    sub-agent's reply couldn't be parsed.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        # Room for a web_search tool round (query + results) before the final
        # two-line answer — 128 was tight enough to truncate before the
        # Merchant/Category lines ever got emitted.
        max_tokens=1024,
        system=_SUBAGENT_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": f'Statement line: "{descriptor}"'}],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))

    fields = {m.group(1).lower(): _strip_markdown(m.group(2)) for m in _LINE_RE.finditer(text)}
    merchant, category = fields.get("merchant"), fields.get("category")
    if not merchant or not category:
        return None
    return {"merchant": merchant, "category": category}, resp.usage
