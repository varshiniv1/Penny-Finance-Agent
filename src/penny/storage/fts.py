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
    category,
    content=''
);
"""


class FTSIndex:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path))
        self._con.row_factory = sqlite3.Row
        self._con.execute(_DDL)
        self._con.commit()

    def index(self, rows: list[dict]) -> None:
        for r in rows:
            self._con.execute("DELETE FROM tx_text WHERE tx_id = ?", (r["id"],))
            self._con.execute(
                "INSERT INTO tx_text (tx_id, description, merchant, category) VALUES (?, ?, ?, ?)",
                (r["id"], r.get("description", ""), r.get("merchant", ""), r.get("category", "")),
            )
        self._con.commit()

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT tx_id, description, merchant, category, rank "
            "FROM tx_text WHERE tx_text MATCH ? ORDER BY rank LIMIT ?",
            (query, top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._con.close()
