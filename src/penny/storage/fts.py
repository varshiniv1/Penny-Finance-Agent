"""Full-text search over transaction descriptions, via DuckDB's own `fts` extension.

Runs against the same connection as the Ledger (MotherDuck, or :memory: with no
persistence configured) — search is just another DuckDB feature over the shared
`_transactions` table, not a second storage engine. Verified working against a live
MotherDuck connection (INSTALL fts / LOAD fts / PRAGMA create_fts_index / match_bm25
all succeed there).

The fts index is a batch/snapshot build over the *physical*, all-users table — DuckDB's
create_fts_index isn't incremental — so `index()` does a full rebuild rather than
adding just the new rows. At personal scale (hundreds to low thousands of rows across
all of Penny's users combined) that's fast enough to just do after every upload.
User-scoping happens the same way as everywhere else in this class's queries: filtered
by this instance's own `user_id`, never trusting the caller to add that filter.
"""
from __future__ import annotations

from typing import Any

import duckdb


class FTSIndex:
    def __init__(self, connection: duckdb.DuckDBPyConnection, user_id: str):
        self._con = connection
        self.user_id = user_id
        self._con.execute("INSTALL fts")
        self._con.execute("LOAD fts")
        self._indexed = False

    def index(self, rows: list[dict] | None = None) -> None:
        """Rebuild the full-text index over the whole _transactions table.

        `rows` is accepted (and ignored) only for call-site compatibility with
        the app's upload flow, which always calls this right after an upsert —
        DuckDB's fts extension rebuilds from the table itself, not a row list.
        """
        # stopwords='none': the default English stopword list (571 words,
        # including plain words like "old"/"new"/"about") is built for prose
        # search, not merchant/description lookup — dropping any of those
        # terms would make a transaction silently unsearchable by a word a
        # user would reasonably expect to find it with.
        self._con.execute(
            "PRAGMA create_fts_index("
            "'_transactions', 'id', 'description', 'merchant', 'category', "
            "stopwords='none', overwrite=1)"
        )
        self._indexed = True

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if not self._indexed:
            self.index()
        rows = self._con.execute(
            """
            SELECT t.id AS tx_id, t.description, t.merchant, t.category,
                   fts_main__transactions.match_bm25(t.id, ?) AS score
            FROM _transactions t
            WHERE t.user_id = ? AND score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query, self.user_id, top_k],
        ).fetchall()
        cols = ["tx_id", "description", "merchant", "category", "score"]
        return [dict(zip(cols, r)) for r in rows]

    def close(self) -> None:
        # Connection lifecycle is owned by whoever constructed it (session.py) —
        # Ledger and FTSIndex now share one connection per user, so this class
        # must never close it out from under the Ledger still using it.
        pass
