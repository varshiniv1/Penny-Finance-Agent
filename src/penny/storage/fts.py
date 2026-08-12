"""SQLite FTS5 text index over transaction descriptions."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS tx_text USING fts5(
    tx_id,
    description,
    merchant,
    category
);
"""


def _to_fts_phrase(query: str) -> str:
    """Turn free-text user input into a safe FTS5 phrase query: each token
    double-quoted (FTS5's escape for a literal `"` is `""`), joined so it's
    an AND of phrases rather than raw MATCH syntax the caller doesn't control."""
    tokens = query.split()
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


class FTSIndex:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute(_DDL)
        self._con.commit()

    def index(self, rows: list[dict]) -> None:
        if not rows:
            return
        ids = [(r["id"],) for r in rows]
        self._con.executemany("DELETE FROM tx_text WHERE tx_id = ?", ids)
        self._con.executemany(
            "INSERT INTO tx_text (tx_id, description, merchant, category) VALUES (?, ?, ?, ?)",
            [
                (r["id"], r.get("description", ""), r.get("merchant", ""), r.get("category", ""))
                for r in rows
            ],
        )
        self._con.commit()

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        # Raw user input as FTS5 MATCH syntax raises OperationalError on
        # special characters ("-", '"', ":", etc). Quoting each token turns
        # the query into a plain phrase match instead of FTS5 query syntax.
        fts_query = _to_fts_phrase(query)
        if not fts_query:
            return []
        rows = self._con.execute(
            "SELECT tx_id, description, merchant, category, rank "
            "FROM tx_text WHERE tx_text MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._con.close()
