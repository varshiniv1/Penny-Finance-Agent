"""Streamlit page: token usage and estimated cost across every API call this session."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from penny.ui.session import get_usage_log

# (input $/MTok, output $/MTok) — rough estimates for display only, not billed truth.
_PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}
_DEFAULT_PRICING = (3.00, 15.00)


def _tokens_last_minute(log: list[dict]) -> tuple[int, int]:
    """(input_tokens, output_tokens) summed over entries from the last 60s."""
    now = datetime.now(timezone.utc)
    inp = out = 0
    for e in log:
        ts = datetime.fromisoformat(e["timestamp"])
        if (now - ts).total_seconds() <= 60:
            inp += e["input_tokens"]
            out += e["output_tokens"]
    return inp, out


def _estimate_cost(entry: dict) -> float:
    input_price, output_price = _PRICING.get(entry["model"], _DEFAULT_PRICING)
    return (
        entry["input_tokens"] * input_price
        + entry["output_tokens"] * output_price
        + entry["cache_creation_input_tokens"] * input_price * 1.25
        + entry["cache_read_input_tokens"] * input_price * 0.1
    ) / 1_000_000


def show() -> None:
    st.header("Observability")
    st.caption("Token usage and estimated cost for this browser session only — resets on refresh.")

    log = get_usage_log()
    if not log:
        st.info("No API calls made yet this session.")
        return

    total_input = sum(e["input_tokens"] for e in log)
    total_output = sum(e["output_tokens"] for e in log)
    total_cache_write = sum(e["cache_creation_input_tokens"] for e in log)
    total_cache_read = sum(e["cache_read_input_tokens"] for e in log)
    total_cost = sum(_estimate_cost(e) for e in log)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Input tokens", f"{total_input:,}")
    c2.metric("Output tokens", f"{total_output:,}")
    c3.metric("Cache read tokens", f"{total_cache_read:,}")
    c4.metric("Cache write tokens", f"{total_cache_write:,}")
    c5.metric("Est. cost", f"${total_cost:,.4f}")

    st.caption(
        "Estimated cost is a rough guide based on published per-model rates — it's not "
        "your actual bill. Cache reads are priced at ~0.1x, cache writes at ~1.25x."
    )

    rate_in, rate_out = _tokens_last_minute(log)
    r1, r2 = st.columns(2)
    r1.metric("Input tokens / min (rolling)", f"{rate_in:,}")
    r2.metric("Output tokens / min (rolling)", f"{rate_out:,}")
    st.caption(
        "Sum of the last 60 seconds of calls — for comparing against your org's "
        "tokens-per-minute rate limit (Console → Settings → Rate Limits). At a "
        "Build-tier org this is generally 5M input / 1M output TPM per model, so a "
        "single-user session like this one is very unlikely to get close."
    )

    st.divider()

    st.subheader("By operation")
    st.caption(
        "Each source is a distinct call site: chat_turn (the main agent loop), "
        "chat_subagent_categorize (the categorize_transaction delegated sub-agent "
        "call), enrichment (merchant categorization on upload), and "
        "dashboard_insight (the Dashboard's \"Generate Insights\" button)."
    )
    import pandas as pd

    df = pd.DataFrame(log)
    df["est_cost"] = [round(_estimate_cost(e), 5) for e in log]
    by_source = (
        df.groupby("source")
        .agg(
            calls=("input_tokens", "count"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            cache_read_tokens=("cache_read_input_tokens", "sum"),
            est_cost=("est_cost", "sum"),
        )
        .sort_values("est_cost", ascending=False)
    )
    st.dataframe(by_source, use_container_width=True)

    with st.expander("All calls (most recent first, each row = one API request)"):
        st.dataframe(df.iloc[::-1], use_container_width=True)
