"""Plotly chart templates for Penny dashboards."""
from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


_PALETTE = px.colors.qualitative.Set3

_LAYOUT = dict(
    paper_bgcolor="#1E2130",
    plot_bgcolor="#1E2130",
    font=dict(color="#FAFAFA", family="sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def render_chart(
    chart_type: str,
    rows: list[dict[str, Any]],
    title: str,
    x_col: str | None = None,
    y_col: str | None = None,
    label_col: str | None = None,
    value_col: str | None = None,
) -> go.Figure | None:
    if not rows:
        return None

    df = pd.DataFrame(rows)

    if chart_type == "bar":
        x = x_col or df.columns[0]
        y = y_col or df.columns[1]
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=_PALETTE)

    elif chart_type == "line":
        x = x_col or df.columns[0]
        y = y_col or df.columns[1]
        fig = px.line(df, x=x, y=y, title=title, markers=True, color_discrete_sequence=_PALETTE)

    elif chart_type == "pie":
        lbl = label_col or df.columns[0]
        val = value_col or df.columns[1]
        fig = px.pie(df, names=lbl, values=val, title=title, color_discrete_sequence=_PALETTE)

    elif chart_type == "table":
        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color="#6C63FF",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[df[c].tolist() for c in df.columns],
                    fill_color="#1E2130",
                    font=dict(color="#FAFAFA", size=11),
                    align="left",
                ),
            )],
            layout=go.Layout(title=title),
        )
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    fig.update_layout(**_LAYOUT)
    return fig


# ── Pre-built dashboard charts ────────────────────────────────────────────────

def spending_by_category(ledger) -> go.Figure | None:
    rows = ledger.query(
        "SELECT category, SUM(amount) AS total FROM transactions "
        "WHERE is_internal = false AND amount > 0 AND category IS NOT NULL AND category != '' "
        "GROUP BY category ORDER BY total DESC"
    )
    return render_chart("bar", rows, "Spending by Category", "category", "total")


def monthly_trend(ledger) -> go.Figure | None:
    rows = ledger.query(
        "SELECT strftime(date, '%Y-%m') AS month, SUM(amount) AS total "
        "FROM transactions WHERE is_internal = false AND amount > 0 "
        "GROUP BY month ORDER BY month"
    )
    return render_chart("line", rows, "Monthly Spending Trend", "month", "total")


def top_merchants(ledger, n: int = 10) -> go.Figure | None:
    rows = ledger.query(
        f"SELECT COALESCE(NULLIF(merchant, ''), description) AS name, SUM(amount) AS total "
        f"FROM transactions WHERE is_internal = false AND amount > 0 "
        f"GROUP BY name ORDER BY total DESC LIMIT {n}"
    )
    return render_chart("bar", rows, f"Top {n} Merchants", "name", "total")


def category_pie(ledger) -> go.Figure | None:
    rows = ledger.query(
        "SELECT category, SUM(amount) AS total FROM transactions "
        "WHERE is_internal = false AND amount > 0 AND category IS NOT NULL AND category != '' "
        "GROUP BY category ORDER BY total DESC"
    )
    return render_chart("pie", rows, "Spending Breakdown", "category", "total")


def recurring_charges(ledger) -> go.Figure | None:
    rows = ledger.query(
        """
        SELECT COALESCE(NULLIF(merchant, ''), description) AS name,
               COUNT(*) AS occurrences,
               AVG(amount) AS avg_amount,
               SUM(amount) AS total
        FROM transactions
        WHERE is_internal = false AND amount > 0
        GROUP BY name
        HAVING COUNT(*) >= 2 AND STDDEV(amount) < 1.0
        ORDER BY total DESC
        LIMIT 20
        """
    )
    return render_chart("table", rows, "Likely Recurring Charges")
