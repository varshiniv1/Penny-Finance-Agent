"""Dashboard agentic insights: a short narrative summary over already-computed spend data."""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from penny.config import DEFAULT_MODEL

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger

# Same aggregation shape as the dashboard charts (charts/templates.py) — small,
# already-summarized rows, not raw transactions, so this stays a cheap call.
#
# is_internal alone under-excludes: it's only set when reconcile.py found a
# matching transaction on another account, which needs the user to have
# uploaded every account involved. A single-statement upload (the common
# case) has real transfers/card payments correctly categorized 'Transfer'
# but never matched, so is_internal stays false for them — excluding the
# category too catches those without requiring multi-account data.
_EXCLUDE_TRANSFERS = "is_internal = false AND category != 'Transfer'"
_CATEGORY_SQL = (
    "SELECT category, SUM(amount) AS total FROM transactions "
    f"WHERE {_EXCLUDE_TRANSFERS} AND amount > 0 AND category IS NOT NULL AND category != '' "
    "GROUP BY category ORDER BY total DESC"
)
_MONTHLY_SQL = (
    "SELECT strftime(date, '%Y-%m') AS month, SUM(amount) AS total "
    f"FROM transactions WHERE {_EXCLUDE_TRANSFERS} AND amount > 0 "
    "GROUP BY month ORDER BY month"
)
_TOP_MERCHANTS_SQL = (
    "SELECT COALESCE(NULLIF(merchant, ''), description) AS name, SUM(amount) AS total "
    f"FROM transactions WHERE {_EXCLUDE_TRANSFERS} AND amount > 0 "
    "GROUP BY name ORDER BY total DESC LIMIT 10"
)


def insight(ledger: "Ledger", api_key: str, model: str = DEFAULT_MODEL) -> tuple[str, Any] | None:
    """Return (insight_text, usage), or None if there's nothing to summarize."""
    import anthropic

    summary = {
        "spending_by_category": ledger.query(_CATEGORY_SQL),
        "monthly_trend": ledger.query(_MONTHLY_SQL),
        "top_merchants": ledger.query(_TOP_MERCHANTS_SQL),
    }
    if not any(summary.values()):
        return None

    prompt = (
        "Here is a summary of someone's spending, already aggregated by category, "
        "month, and merchant:\n\n"
        f"{json.dumps(summary, default=str)}\n\n"
        "In 2-4 sentences, call out the most notable patterns or anomalies (e.g. a "
        "category spike, an unusually large transaction, a trend change). Be specific "
        "and concrete — cite actual numbers. No generic financial advice."
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return text, resp.usage
