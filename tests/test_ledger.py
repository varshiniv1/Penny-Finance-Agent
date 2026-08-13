"""Tests for the DuckDB ledger wrapper: the query() SQL guard, and multi-tenant
isolation via the per-connection user-scoped `transactions` view."""
import duckdb
import pytest

from penny.storage.ledger import Ledger


@pytest.fixture
def ledger():
    l = Ledger(":memory:", "user_a")
    l.upsert([
        {"id": "a", "date": "2024-01-01", "description": "Coffee", "amount": 5.0},
        {"id": "b", "date": "2024-01-02", "description": "Groceries", "amount": 40.0},
    ])
    return l


def test_upsert_and_count(ledger):
    assert ledger.count() == 2


def test_query_select_works(ledger):
    rows = ledger.query("SELECT * FROM transactions ORDER BY date")
    assert len(rows) == 2
    assert rows[0]["description"] == "Coffee"


def test_query_respects_limit(ledger):
    rows = ledger.query("SELECT * FROM transactions", limit=1)
    assert len(rows) == 1


def test_query_with_cte_allowed(ledger):
    rows = ledger.query("WITH t AS (SELECT * FROM transactions) SELECT * FROM t")
    assert len(rows) == 2


@pytest.mark.parametrize("bad_sql", [
    "SELECT * FROM transactions; DROP TABLE transactions",
    "DELETE FROM transactions",
    "UPDATE transactions SET amount = 0",
    "INSERT INTO transactions VALUES ('x')",
    "DROP TABLE transactions",
    "ATTACH ':memory:' AS x",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_parquet('http://evil.example/x.parquet')",
    "PRAGMA database_list",
    "INSTALL httpfs",
])
def test_query_blocks_disallowed_sql(ledger, bad_sql):
    with pytest.raises(ValueError):
        ledger.query(bad_sql)


def test_export_import_roundtrip(ledger, tmp_path):
    path = tmp_path / "export.parquet"
    ledger.export_parquet(path)

    fresh = Ledger(":memory:", "user_a")
    n = fresh.import_parquet(path)
    assert n == 2
    assert fresh.count() == 2


def test_import_restamps_rows_under_importing_users_identity(ledger, tmp_path):
    """Restoring a backup is "these are my transactions" — the importing
    connection's user_id always wins, regardless of what's in the file."""
    path = tmp_path / "export.parquet"
    ledger.export_parquet(path)  # exported while scoped to "user_a"

    other = Ledger(":memory:", "user_b")
    other.import_parquet(path)
    assert other.count() == 2
    rows = other.query("SELECT id FROM transactions")
    assert {r["id"] for r in rows} == {"a", "b"}


def test_upsert_is_idempotent_on_same_id(ledger):
    ledger.upsert([{"id": "a", "date": "2024-01-01", "description": "Coffee", "amount": 5.0}])
    assert ledger.count() == 2


# ── Multi-tenant isolation ──────────────────────────────────────────────────
# Two Ledgers sharing one connection, exactly like two different users sharing
# one MotherDuck database in production (see Ledger.__init__'s docstring for
# why a shared connection, not two separate ":memory:" ones, is required for
# this to be a meaningful test).

@pytest.fixture
def shared_db():
    """A :memory: connection plus a factory for independent cursors onto it —
    cursors are DuckDB's equivalent of separate MotherDuck connections to the
    same database: independent TEMP view scoping, shared physical tables.
    Using the same connection object for two "users" instead (rather than two
    cursors) would make them silently overwrite each other's TEMP view."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()


def test_users_cannot_see_each_others_rows(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    a.upsert([{"id": "1", "date": "2024-01-01", "description": "A's coffee", "amount": 5.0}])
    b.upsert([{"id": "2", "date": "2024-01-01", "description": "B's rent", "amount": 1000.0}])

    assert a.count() == 1
    assert b.count() == 1
    assert a.query("SELECT id FROM transactions") == [{"id": "1"}]
    assert b.query("SELECT id FROM transactions") == [{"id": "2"}]


def test_aggregate_queries_are_also_scoped(shared_db):
    """A naive per-row filter could miss this — SUM/COUNT must be scoped too,
    not just raw SELECT * — this is exactly what the query_sql tool runs."""
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    a.upsert([{"id": "1", "date": "2024-01-01", "description": "x", "amount": 5.0}])
    b.upsert([{"id": "2", "date": "2024-01-01", "description": "y", "amount": 1000.0}])

    assert a.query("SELECT SUM(amount) AS total FROM transactions")[0]["total"] == 5.0
    assert b.query("SELECT SUM(amount) AS total FROM transactions")[0]["total"] == 1000.0


def test_delete_all_user_data_does_not_touch_other_users(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    a.upsert([{"id": "1", "date": "2024-01-01", "description": "x", "amount": 5.0}])
    b.upsert([{"id": "2", "date": "2024-01-01", "description": "y", "amount": 1000.0}])

    a.delete_all_user_data()
    assert a.count() == 0
    assert b.count() == 1


def test_merchant_cache_is_shared_across_users(shared_db):
    """Deliberately global, not user-scoped — see the _DDL comment for why."""
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    b.upsert([{"id": "1", "date": "2024-01-01", "description": "STARBUCKS #1", "amount": 5.0}])

    # Before any user caches this descriptor, it's uncached for b.
    assert "STARBUCKS #1" in b.uncached_descriptors()

    # user_a caches it (e.g. via merchant enrichment on their own upload)...
    a.upsert_merchants([{"descriptor": "STARBUCKS #1", "merchant": "Starbucks", "category": "Dining"}])

    # ...and it's now visible to user_b too, saving a redundant Claude call.
    assert "STARBUCKS #1" not in b.uncached_descriptors()


# ── Duplicate-upload prevention + coverage labeling ─────────────────────────

def test_duplicate_upload_detection(ledger):
    assert ledger.is_duplicate_upload("hash1") is False
    ledger.mark_uploaded("hash1", "statement.pdf", "2024-01-01", "2024-01-31", 10)
    assert ledger.is_duplicate_upload("hash1") is True


def test_duplicate_check_is_scoped_per_user(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    a.mark_uploaded("hash1", "statement.pdf", "2024-01-01", "2024-01-31", 10)
    assert a.is_duplicate_upload("hash1") is True
    assert b.is_duplicate_upload("hash1") is False


def test_delete_all_user_data_clears_upload_history(ledger):
    ledger.mark_uploaded("hash1", "statement.pdf", "2024-01-01", "2024-01-31", 10)
    ledger.delete_all_user_data()
    assert ledger.is_duplicate_upload("hash1") is False


def test_source_file_summary(ledger):
    ledger.upsert([
        {"id": "c", "date": "2024-02-15", "description": "Rent", "amount": 1000.0, "source_file": "feb.pdf"},
        {"id": "d", "date": "2024-02-20", "description": "Gym", "amount": 40.0, "source_file": "feb.pdf"},
    ])
    info = ledger.source_file_summary("feb.pdf")
    assert info["row_count"] == 2
    assert str(info["min_date"]) == "2024-02-15"
    assert str(info["max_date"]) == "2024-02-20"


# ── Usage log (self-service, no admin gate) ─────────────────────────────────

_USAGE_ENTRY = {
    "timestamp": "2024-01-01T00:00:00+00:00",
    "source": "chat_turn",
    "model": "claude-haiku-4-5-20251001",
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def test_log_and_read_usage_event(ledger):
    ledger.log_usage_event(_USAGE_ENTRY)
    log = ledger.get_usage_log()
    assert len(log) == 1
    assert log[0]["source"] == "chat_turn"
    assert log[0]["input_tokens"] == 100


def test_usage_log_is_scoped_per_user(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    a.log_usage_event(_USAGE_ENTRY)

    assert len(a.get_usage_log()) == 1
    assert b.get_usage_log() == []


def test_delete_all_user_data_clears_usage_log(ledger):
    ledger.log_usage_event(_USAGE_ENTRY)
    ledger.delete_all_user_data()
    assert ledger.get_usage_log() == []
