"""Streamlit page: pre-built spending dashboards."""
from __future__ import annotations

import streamlit as st

from penny.charts.templates import (
    spending_by_category,
    monthly_trend,
    top_merchants,
    category_pie,
    recurring_charges,
)
from penny.ui.session import get_ledger, tx_count


def _render(compute, empty_message: str) -> None:
    """Run a chart-building callable, showing an info message instead of a raw
    error when the underlying query has no rows yet (e.g. nothing categorized)."""
    try:
        fig = compute()
    except Exception as e:
        st.error(f"Chart error: {e}")
        return
    if fig is None:
        st.info(empty_message)
    else:
        st.plotly_chart(fig, use_container_width=True)


def show() -> None:
    st.header("Dashboard")

    if tx_count() == 0:
        st.warning("No transactions loaded yet. Go to **Upload Statements** first.")
        return

    ledger = get_ledger()

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    rows = ledger.query(
        "SELECT "
        "  SUM(CASE WHEN amount > 0 AND is_internal = false THEN amount END) AS total_spend, "
        "  COUNT(CASE WHEN amount > 0 AND is_internal = false THEN 1 END) AS tx_count, "
        "  MIN(date) AS earliest, MAX(date) AS latest "
        "FROM transactions"
    )
    if rows:
        r = rows[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Spend", f"${r['total_spend'] or 0:,.2f}")
        c2.metric("Transactions", r["tx_count"] or 0)
        c3.metric("From", str(r["earliest"] or "—"))
        c4.metric("To", str(r["latest"] or "—"))

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    no_categories_msg = (
        "No categorized spending yet — add your Anthropic API key and "
        "reprocess statements to enable merchant categorization."
    )

    with col_left:
        _render(lambda: spending_by_category(ledger), no_categories_msg)

    with col_right:
        _render(lambda: category_pie(ledger), no_categories_msg)

    _render(lambda: monthly_trend(ledger), "No spending data yet.")

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        _render(lambda: top_merchants(ledger), "No spending data yet.")

    with col_right2:
        _render(lambda: recurring_charges(ledger), "No recurring charges detected yet.")

    # ── Raw transactions table ─────────────────────────────────────────────────
    with st.expander("Browse all transactions"):
        import pandas as pd
        rows = ledger.query(
            "SELECT date, merchant, description, category, amount, account_last4, source_file "
            "FROM transactions WHERE is_internal = false ORDER BY date DESC LIMIT 500"
        )
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
