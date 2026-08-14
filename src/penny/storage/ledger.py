"""DuckDB transaction ledger.

Backed by MotherDuck (a hosted DuckDB database, shared across all of Penny's users)
so transaction data survives across browser sessions and Streamlit Cloud restarts,
instead of vanishing when the tab closes. Falls back to a local `:memory:` connection
when no MotherDuck token is configured, or no user identity exists yet (no API key
entered) — same ephemeral behavior the app has always had in that case.

Multi-tenancy without per-row SQL rewriting: the physical table (`_transactions`) holds
every user's rows together, but `Ledger.__init__` creates a connection-local TEMP VIEW
named `transactions` that's pre-filtered to just this connection's `user_id`. Every
query this class runs — including arbitrary SQL the LLM writes via the query_sql tool —
references `transactions` and is therefore automatically scoped, with no SQL parsing
or rewriting needed. Verified empirically: the view doesn't leak across connections,
and aggregates (SUM, COUNT, ...) over it are correctly pre-filtered.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
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
CREATE TABLE IF NOT EXISTS _transactions (
    id            VARCHAR NOT NULL,
    user_id       VARCHAR NOT NULL,
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
    is_internal   BOOLEAN DEFAULT false,
    content_hash  VARCHAR,
    PRIMARY KEY (id)
);

-- Shared/global across every user, deliberately not partitioned by user_id: a
-- descriptor like "STARBUCKS #1234" maps to the same merchant/category regardless
-- of whose statement it came from, so caching it globally avoids redundant Claude
-- calls across users with no personal-data exposure (only business names/categories
-- ever live here).
CREATE TABLE IF NOT EXISTS merchant_cache (
    descriptor  VARCHAR PRIMARY KEY,
    merchant    VARCHAR,
    category    VARCHAR
);

-- One row per accepted upload — powers duplicate-upload rejection (match on
-- content_hash) and the month/year coverage label shown after a batch upload.
CREATE TABLE IF NOT EXISTS uploaded_statements (
    user_id      VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    filename     VARCHAR,
    uploaded_at  TIMESTAMP DEFAULT current_timestamp,
    min_date     DATE,
    max_date     DATE,
    row_count    INTEGER,
    PRIMARY KEY (user_id, content_hash)
);

-- Every Claude API call this user's session has made, for the self-service
-- usage view (no admin gate — a user only ever reads their own rows here).
-- Durable and genuinely unrestricted in row count: unlike the old session-only
-- list this replaces, this survives a restart/redeploy, and there's no cap.
CREATE TABLE IF NOT EXISTS usage_log (
    user_id                     VARCHAR NOT NULL,
    -- Stored as the raw ISO-8601 string (always UTC, from
    -- datetime.now(timezone.utc).isoformat()) rather than a TIMESTAMP column —
    -- DuckDB's plain TIMESTAMP type drops timezone info on round-trip, which
    -- silently turned every read-back timestamp naive and broke comparisons
    -- against datetime.now(timezone.utc) elsewhere. A string sidesteps that
    -- entirely and needs no conversion on the way back out.
    timestamp                   VARCHAR NOT NULL,
    source                      VARCHAR,
    model                       VARCHAR,
    input_tokens                INTEGER,
    output_tokens               INTEGER,
    cache_creation_input_tokens INTEGER,
    cache_read_input_tokens     INTEGER
);

-- One row per saved conversation. `history` is the full Anthropic-format
-- message list (JSON) exactly as passed to client.beta.messages.create — the
-- same shape already kept in st.session_state["history"] during a live
-- session, just made durable — so reopening a conversation can genuinely
-- continue it, not just display a read-only transcript.
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR NOT NULL,
    user_id         VARCHAR NOT NULL,
    title           VARCHAR,
    history         VARCHAR,
    created_at      VARCHAR NOT NULL,
    updated_at      VARCHAR NOT NULL,
    PRIMARY KEY (conversation_id)
);

-- One row per human-readable chat turn (text/op/note/thinking only — never
-- chart/image/file bytes, which stay out of this table entirely). Exists
-- purely to power full-text search and the "recent conversations" list;
-- the actual resumable state lives in conversations.history above.
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              VARCHAR NOT NULL,
    conversation_id VARCHAR NOT NULL,
    user_id         VARCHAR NOT NULL,
    role            VARCHAR NOT NULL,
    type            VARCHAR NOT NULL,
    content         VARCHAR,
    created_at      VARCHAR NOT NULL,
    PRIMARY KEY (id)
);
"""


def _escape_sql_literal(value: str) -> str:
    """Escape a value for embedding in a single-quoted SQL string literal."""
    return str(value).replace("'", "''")


class Ledger:
    """Thin wrapper around a DuckDB connection, scoped to one user_id.

    `connection_string` is either a MotherDuck database (e.g. "md:penny") or
    ":memory:" for the no-persistence fallback. `user_id` is the SHA-256-derived
    key from session.get_user_key() — never the raw API key.
    """

    def __init__(self, connection: "str | duckdb.DuckDBPyConnection", user_id: str):
        self.user_id = user_id
        # Accepts either a connection string (":memory:" or "md:penny") to open
        # fresh, or an already-open connection to share — the latter is what
        # makes it possible to test cross-user isolation locally without a live
        # MotherDuck token: two Ledgers sharing one :memory: connection behave
        # exactly like two users sharing one MotherDuck database, since separate
        # `duckdb.connect(":memory:")` calls are otherwise fully independent,
        # unshared databases.
        self._con = duckdb.connect(connection) if isinstance(connection, str) else connection
        self._con.execute(_DDL)
        # CREATE TABLE IF NOT EXISTS above is a no-op against a table already
        # created by an earlier version of this schema (i.e. the live
        # MotherDuck database from before content_hash existed) — add it here
        # so both fresh and already-deployed databases end up with the column.
        try:
            self._con.execute("ALTER TABLE _transactions ADD COLUMN content_hash VARCHAR")
        except duckdb.Error:
            pass  # already has the column
        # Connection-local — never visible to any other connection, including
        # another browser tab/session for the same user (verified empirically).
        # DDL can't be parameterized in DuckDB ("Unexpected prepared parameter"),
        # so user_id is inlined here — safe because it's our own hex-digest
        # string (see session.get_user_key()), never attacker-controlled input.
        self._con.execute(
            "CREATE OR REPLACE TEMP VIEW transactions AS "
            f"SELECT * FROM _transactions WHERE user_id = '{_escape_sql_literal(user_id)}'"
        )

    # ── transactions ─────────────────────────────────────────────────────────

    def upsert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        self._con.executemany(
            """
            INSERT OR REPLACE INTO _transactions
              (id, user_id, date, description, merchant, category, amount, account,
               account_last4, source_file, page_num, is_transfer, is_refund, is_internal,
               content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["id"], self.user_id, r["date"], r["description"],
                    r.get("merchant"), r.get("category"),
                    r["amount"], r.get("account"), r.get("account_last4"),
                    r.get("source_file"), r.get("page_num"),
                    r.get("is_transfer", False), r.get("is_refund", False),
                    r.get("is_internal", False), r.get("content_hash"),
                )
                for r in rows
            ],
        )
        return len(rows)

    def query(self, sql: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Execute a read-only SELECT query, capped at `limit` rows.

        Raises ValueError if `sql` isn't a single plain SELECT/WITH statement,
        or references a disallowed keyword/function — see _SQL_DISALLOWED_RE.
        Runs against the connection's user-scoped `transactions` view, so a
        query never sees another user's rows regardless of what it selects.
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
        # Targets the physical table (not the `transactions` view) with an explicit
        # user_id filter — DuckDB's UPDATE...FROM support over a simple filtered
        # view isn't something to rely on, so this stays explicit rather than
        # implicit like the SELECT-side scoping elsewhere in this class.
        result = self._con.execute(
            """
            UPDATE _transactions
            SET merchant = mc.merchant,
                category = mc.category
            FROM merchant_cache mc
            WHERE _transactions.description = mc.descriptor
              AND _transactions.user_id = ?
              AND (_transactions.merchant IS NULL OR _transactions.merchant = '')
            """,
            [self.user_id],
        )
        return result.rowcount or 0

    # ── merchant cache (global, not user-scoped) ────────────────────────────────

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

    def source_file_summary(self, filename: str) -> dict:
        """Date range + row count for one just-uploaded file, scoped to this
        user. Uses a real parameter binding (not the LLM-facing query() path,
        which is deliberately restricted and not meant for app-internal
        lookups involving a user-controlled filename)."""
        row = self._con.execute(
            "SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS row_count "
            "FROM transactions WHERE source_file = ?",
            [filename],
        ).fetchone()
        return {"min_date": row[0], "max_date": row[1], "row_count": row[2]}

    # ── duplicate-upload prevention + coverage labeling ─────────────────────────

    def is_duplicate_upload(self, content_hash: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM uploaded_statements WHERE user_id = ? AND content_hash = ?",
            [self.user_id, content_hash],
        ).fetchone()
        return row is not None

    def mark_uploaded(
        self, content_hash: str, filename: str, min_date: str | None,
        max_date: str | None, row_count: int,
    ) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO uploaded_statements
              (user_id, content_hash, filename, min_date, max_date, row_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [self.user_id, content_hash, filename, min_date, max_date, row_count],
        )

    def rename_upload(self, content_hash: str, new_filename: str) -> int:
        """Rename one prior upload — updates both the uploaded_statements
        record (what the "Delete my data" list shows) and source_file on
        every transaction from that upload (what the agent cites back to the
        user, e.g. "from your Chase statement, page 3" — stale source_file
        after a rename would make that citation lie). Scoped by content_hash,
        not the old filename, for the same reason as delete_upload. Returns
        the number of transactions updated."""
        n = self._con.execute(
            "SELECT COUNT(*) FROM _transactions WHERE user_id = ? AND content_hash = ?",
            [self.user_id, content_hash],
        ).fetchone()[0]
        self._con.execute(
            "UPDATE _transactions SET source_file = ? WHERE user_id = ? AND content_hash = ?",
            [new_filename, self.user_id, content_hash],
        )
        self._con.execute(
            "UPDATE uploaded_statements SET filename = ? WHERE user_id = ? AND content_hash = ?",
            [new_filename, self.user_id, content_hash],
        )
        return n

    def list_uploads(self) -> list[dict]:
        """This user's accepted uploads, most recent first — powers the
        per-file delete UI."""
        rows = self._con.execute(
            """
            SELECT content_hash, filename, uploaded_at, min_date, max_date, row_count
            FROM uploaded_statements WHERE user_id = ? ORDER BY uploaded_at DESC
            """,
            [self.user_id],
        ).fetchall()
        cols = ["content_hash", "filename", "uploaded_at", "min_date", "max_date", "row_count"]
        return [dict(zip(cols, r)) for r in rows]

    def delete_upload(self, content_hash: str) -> int:
        """Remove exactly the transactions from one prior upload (and its
        uploaded_statements record, freeing that file to be re-uploaded
        later) — scoped by content_hash, not filename, since two different
        files can share a name. Returns the number of transactions removed.
        Caller is responsible for re-running fts.index() afterward.

        Counts explicitly rather than trusting DELETE's rowcount — verified
        against a live MotherDuck connection that it reports -1 there (a
        real behavioral difference from local DuckDB, not a hypothetical
        one), so rowcount can't be relied on for this."""
        n = self._con.execute(
            "SELECT COUNT(*) FROM _transactions WHERE user_id = ? AND content_hash = ?",
            [self.user_id, content_hash],
        ).fetchone()[0]
        self._con.execute(
            "DELETE FROM _transactions WHERE user_id = ? AND content_hash = ?",
            [self.user_id, content_hash],
        )
        self._con.execute(
            "DELETE FROM uploaded_statements WHERE user_id = ? AND content_hash = ?",
            [self.user_id, content_hash],
        )
        return n

    # ── export / import for manual backups ──────────────────────────────────────

    def export_parquet(self, path: Path) -> None:
        """Dump this user's transactions (only — via the scoped view) to Parquet."""
        self._con.execute(f"COPY transactions TO '{_escape_sql_literal(str(path))}' (FORMAT PARQUET)")

    def import_parquet(self, path: Path) -> int:
        # Explicit column list, forcing user_id to the CURRENT connection's identity
        # regardless of whatever user_id (if any) is stored in the file — restoring
        # a backup is "these are my transactions," not an arbitrary cross-user write.
        self._con.execute(
            f"""
            INSERT OR REPLACE INTO _transactions
              (id, user_id, date, description, merchant, category, amount, account,
               account_last4, source_file, page_num, is_transfer, is_refund, is_internal)
            SELECT id, ?, date, description, merchant, category, amount, account,
                   account_last4, source_file, page_num, is_transfer, is_refund, is_internal
            FROM read_parquet('{_escape_sql_literal(str(path))}')
            """,
            [self.user_id],
        )
        return self.count()

    # ── usage log (self-service, no admin gate) ─────────────────────────────────

    def log_usage_event(self, entry: dict) -> None:
        self._con.execute(
            """
            INSERT INTO usage_log
              (user_id, timestamp, source, model, input_tokens, output_tokens,
               cache_creation_input_tokens, cache_read_input_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                self.user_id, entry["timestamp"], entry["source"], entry["model"],
                entry["input_tokens"], entry["output_tokens"],
                entry["cache_creation_input_tokens"], entry["cache_read_input_tokens"],
            ],
        )

    def get_usage_log(self) -> list[dict]:
        """This user's own usage history only — scoped by user_id, same as
        every other read in this class."""
        rows = self._con.execute(
            """
            SELECT timestamp, source, model, input_tokens, output_tokens,
                   cache_creation_input_tokens, cache_read_input_tokens
            FROM usage_log WHERE user_id = ? ORDER BY timestamp
            """,
            [self.user_id],
        ).fetchall()
        cols = ["timestamp", "source", "model", "input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens"]
        return [dict(zip(cols, r)) for r in rows]

    # ── conversations (persisted chat history) ──────────────────────────────────

    def save_conversation(self, conversation_id: str, title: str, history: list[dict]) -> None:
        """Upsert this conversation's resumable state. `ON CONFLICT ... DO
        UPDATE` (rather than INSERT OR REPLACE) so created_at is set once on
        first save and never overwritten on later turns."""
        now = datetime.now(timezone.utc).isoformat()
        self._con.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, title, history, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (conversation_id) DO UPDATE SET
                title = excluded.title,
                history = excluded.history,
                updated_at = excluded.updated_at
            """,
            [conversation_id, self.user_id, title, json.dumps(history, default=str), now, now],
        )

    def add_conversation_messages(self, conversation_id: str, entries: list[dict]) -> None:
        """Append the display-message entries new since the last save — each
        with role/type/content already in the shape chat_page._render_message
        expects, so a reopened conversation renders identically to how it
        first appeared."""
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._con.executemany(
            """
            INSERT INTO conversation_messages (id, conversation_id, user_id, role, type, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (str(uuid.uuid4()), conversation_id, self.user_id, e["role"], e["type"], e["content"], now)
                for e in entries
            ],
        )

    def recent_conversations(self, limit: int = 5) -> list[dict]:
        """Most recently active conversations, most recent first — powers the
        sidebar's "Recent conversations" list."""
        rows = self._con.execute(
            """
            SELECT conversation_id, title, updated_at
            FROM conversations WHERE user_id = ?
            ORDER BY updated_at DESC LIMIT ?
            """,
            [self.user_id, limit],
        ).fetchall()
        cols = ["conversation_id", "title", "updated_at"]
        return [dict(zip(cols, r)) for r in rows]

    def load_conversation(self, conversation_id: str) -> dict | None:
        """The resumable API-format history for one conversation, or None if
        it doesn't exist (or belongs to a different user)."""
        row = self._con.execute(
            "SELECT title, history FROM conversations WHERE user_id = ? AND conversation_id = ?",
            [self.user_id, conversation_id],
        ).fetchone()
        if row is None:
            return None
        return {"title": row[0], "history": json.loads(row[1]) if row[1] else []}

    def load_conversation_messages(self, conversation_id: str) -> list[dict]:
        """The rendered transcript for one conversation, oldest first — what
        chat_page.show() replays into display_messages on reopen."""
        rows = self._con.execute(
            """
            SELECT role, type, content FROM conversation_messages
            WHERE user_id = ? AND conversation_id = ? ORDER BY created_at
            """,
            [self.user_id, conversation_id],
        ).fetchall()
        cols = ["role", "type", "content"]
        return [dict(zip(cols, r)) for r in rows]

    # ── data deletion ────────────────────────────────────────────────────────

    def delete_all_user_data(self) -> None:
        """Wipe every row belonging to this user_id — transactions, upload
        history, usage log, and saved conversations. Does not touch
        merchant_cache (global, not personal data) or any other user's rows."""
        self._con.execute("DELETE FROM _transactions WHERE user_id = ?", [self.user_id])
        self._con.execute("DELETE FROM uploaded_statements WHERE user_id = ?", [self.user_id])
        self._con.execute("DELETE FROM usage_log WHERE user_id = ?", [self.user_id])
        self._con.execute("DELETE FROM conversations WHERE user_id = ?", [self.user_id])
        self._con.execute("DELETE FROM conversation_messages WHERE user_id = ?", [self.user_id])

    def close(self) -> None:
        self._con.close()
