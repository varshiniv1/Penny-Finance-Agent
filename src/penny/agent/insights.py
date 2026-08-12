"""Dashboard agentic insights: a short narrative summary over already-computed spend data."""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger

# Same aggregation shape as the dashboard charts (charts/templates.py) — small,
# already-summarized rows, not raw transactions, so this stays a cheap call.
_CATEGORY_SQL = (
    "SELECT category, SUM(amount) AS total FROM transactions "
    "WHERE is_internal = false AND amount > 0 AND category IS NOT NULL AND category != '' "
    "GROUP BY category ORDER BY total DESC"
)
_MONTHLY_SQL = (
    "SELECT strftime(date, '%Y-%m') AS month, SUM(amount) AS total "
    "FROM transactions WHERE is_internal = false AND amount > 0 "
    "GROUP BY month ORDER BY month"
)
_TOP_MERCHANTS_SQL = (
    "SELECT COALESCE(NULLIF(merchant, ''), description) AS name, SUM(amount) AS total "
    "FROM transactions WHERE is_internal = false AND amount > 0 "
    "GROUP BY name ORDER BY total DESC LIMIT 10"
)


def insight(ledger: "Ledger", api_key: str) -> tuple[str, Any] | None:
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
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return text, resp.usage
