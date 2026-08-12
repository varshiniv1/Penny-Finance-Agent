"""DuckDB transaction ledger.

Supports two modes:
  - file mode  : duckdb.connect("data/penny.duckdb")  — persistent local use
  - memory mode: duckdb.connect(":memory:")            — Streamlit Cloud sessions
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

# query() is exposed to the LLM (query_sql / generate_chart tools) and to any
# MCP client — it must never be able to read/write outside the transactions
# table it's handed. Block multi-statement injection and anything beyond a
# plain read: DDL/DML, DuckDB's file/network-reading table functions, and
# extension loading (which could otherwise smuggle in httpfs to exfiltrate
# data or read arbitrary local files via read_csv_auto/read_parquet/glob).
_SQL_DISALLOWED_RE = re.compile(
    r"\b(attach|detach|copy|pragma|install|load|call|export|import|create|drop|alter|"
    r"delete|update|insert|read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|"
    r"read_ndjson|glob|httpfs)\b",
    re.IGNORECASE,
)

_DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    id            VARCHAR PRIMARY KEY,
    date          DATE    NOT NULL,
    description   VARCHAR NOT NULL,
    merchant      VARCHAR,
    category      VARCHAR,
    amount        DECIMAL(10, 2) NOT NULL,
    account       VARCHAR,
    account_last4 VARCHAR,
    source_file   VARCHAR,
    page_num      INTEGER,
    is_transfer   BOOLEAN DEFAULT false,
    is_refund     BOOLEAN DEFAULT false,
    is_internal   BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS merchant_cache (
    descriptor  VARCHAR PRIMARY KEY,
    merchant    VARCHAR,
    category    VARCHAR
);

CREATE TABLE IF NOT EXISTS query_cache (
    question_hash VARCHAR PRIMARY KEY,
    sql           VARCHAR,
    last_used     TIMESTAMP DEFAULT current_timestamp
);
"""


def _escape_sql_literal(path: Path) -> str:
    """Escape a path for embedding in a single-quoted SQL string literal.
    These paths are always our own tempfiles, never user-controlled, but this
    keeps the query from breaking (or worse) if one ever contains a quote."""
    return str(path).replace("'", "''")


class Ledger:
    """Thin wrapper around a DuckDB connection with schema auto-init."""

    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._con.execute(_DDL)

    # ── transactions ─────────────────────────────────────────────────────────

    def upsert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        self._con.executemany(
            """
            INSERT OR REPLACE INTO transactions
              (id, date, description, merchant, category, amount, account,
               account_last4, source_file, page_num, is_transfer, is_refund, is_internal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["id"], r["date"], r["description"],
                    r.get("merchant"), r.get("category"),
                    r["amount"], r.get("account"), r.get("account_last4"),
                    r.get("source_file"), r.get("page_num"),
                    r.get("is_transfer", False), r.get("is_refund", False),
                    r.get("is_internal", False),
                )
                for r in rows
            ],
        )
        return len(rows)

    def query(self, sql: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Execute a read-only SELECT query, capped at `limit` rows.

        Raises ValueError if `sql` isn't a single plain SELECT/WITH statement,
        or references a disallowed keyword/function — see _SQL_DISALLOWED_RE.
        """
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise ValueError("Multiple statements are not allowed.")
        if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        if _SQL_DISALLOWED_RE.search(stripped):
            raise ValueError("Query contains a disallowed keyword or function.")

        wrapped = f"SELECT * FROM ({stripped}) __q LIMIT {limit}"
        rel = self._con.execute(wrapped)
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, row)) for row in rel.fetchall()]

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def apply_merchant_names(self) -> int:
        result = self._con.execute(
            """
            UPDATE transactions
            SET merchant = mc.merchant,
                category = mc.category
            FROM merchant_cache mc
            WHERE transactions.description = mc.descriptor
              AND (transactions.merchant IS NULL OR transactions.merchant = '')
            """
        )
        return result.rowcount or 0

    # ── merchant cache ────────────────────────────────────────────────────────

    def upsert_merchants(self, entries: list[dict]) -> None:
        if not entries:
            return
        self._con.executemany(
            "INSERT OR REPLACE INTO merchant_cache (descriptor, merchant, category) VALUES (?, ?, ?)",
            [(e["descriptor"], e["merchant"], e["category"]) for e in entries],
        )

    def uncached_descriptors(self) -> list[str]:
        rows = self._con.execute(
            """
            SELECT DISTINCT t.description
            FROM transactions t
            LEFT JOIN merchant_cache mc ON mc.descriptor = t.description
            WHERE mc.descriptor IS NULL
            """
        ).fetchall()
        return [r[0] for r in rows]

    # ── query cache ───────────────────────────────────────────────────────────

    def cache_query(self, question_hash: str, sql: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO query_cache (question_hash, sql, last_used) VALUES (?, ?, current_timestamp)",
            [question_hash, sql],
        )

    def get_cached_sql(self, question_hash: str) -> str | None:
        row = self._con.execute(
            "SELECT sql FROM query_cache WHERE question_hash = ?", [question_hash]
        ).fetchone()
        return row[0] if row else None

    # ── export / import for session persistence ───────────────────────────────

    def export_parquet(self, path: Path) -> None:
        """Dump transactions to Parquet so users can download and re-upload."""
        self._con.execute(f"COPY transactions TO '{_escape_sql_literal(path)}' (FORMAT PARQUET)")

    def import_parquet(self, path: Path) -> int:
        self._con.execute(
            f"INSERT OR REPLACE INTO transactions SELECT * FROM read_parquet('{_escape_sql_literal(path)}')"
        )
        return self.count()

    def close(self) -> None:
        self._con.close()
