"""Tool definitions and implementations for the Penny agent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger
    from penny.storage.fts import FTSIndex

# ── Anthropic tool schemas ────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "query_sql",
        "description": (
            "Run a SQL SELECT query against the transactions DuckDB database. "
            "Use this for ANY question involving amounts, counts, totals, averages, or trends. "
            "Always prefer this over guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid DuckDB SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "search_text",
        "description": (
            "Full-text keyword search over transaction descriptions and merchant names. "
            "Use for fuzzy merchant lookup or finding transactions by keyword."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_chart",
        "description": (
            "Generate a chart from a SQL query result. "
            "Returns a base64-encoded PNG image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "table"],
                    "description": "Type of chart.",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL query whose results feed the chart.",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title.",
                },
                "x_col": {"type": "string", "description": "Column for x-axis (bar/line)."},
                "y_col": {"type": "string", "description": "Column for y-axis (bar/line)."},
                "label_col": {"type": "string", "description": "Column for pie labels."},
                "value_col": {"type": "string", "description": "Column for pie values."},
            },
            "required": ["chart_type", "sql", "title"],
        },
    },
    {
        "type": "web_search_20250305",
        "name": "web_search",
    },
    {
        "type": "code_execution_20260521",
        "name": "code_execution",
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, ledger: "Ledger", fts: "FTSIndex"):
        self.ledger = ledger
        self.fts = fts

    def run(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "query_sql":
            return self._query_sql(**tool_input)
        elif tool_name == "search_text":
            return self._search_text(**tool_input)
        elif tool_name == "generate_chart":
            return self._generate_chart(**tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _query_sql(self, sql: str) -> dict:
        try:
            rows = self.ledger.query(sql)
            q_hash = hashlib.sha1(sql.encode()).hexdigest()[:12]
            self.ledger.cache_query(q_hash, sql)
            return {"rows": rows, "count": len(rows)}
        except Exception as e:
            return {"error": str(e)}

    def _search_text(self, query: str, top_k: int = 20) -> dict:
        try:
            hits = self.fts.search(query, top_k=top_k)
            return {"hits": hits, "count": len(hits)}
        except Exception as e:
            return {"error": str(e)}

    def _generate_chart(
        self,
        chart_type: str,
        sql: str,
        title: str,
        x_col: str | None = None,
        y_col: str | None = None,
        label_col: str | None = None,
        value_col: str | None = None,
    ) -> dict:
        from penny.charts.templates import render_chart
        try:
            rows = self.ledger.query(sql)
            if not rows:
                return {"error": "Query returned no data"}
            fig = render_chart(chart_type, rows, title, x_col, y_col, label_col, value_col)
            return {"chart_json": fig.to_json(), "row_count": len(rows)}
        except Exception as e:
            return {"error": str(e)}
