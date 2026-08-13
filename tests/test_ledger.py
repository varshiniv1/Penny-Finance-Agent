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


# ── Per-upload deletion ──────────────────────────────────────────────────────

def _seed_two_uploads(ledger):
    ledger.upsert([
        {"id": "fa-1", "date": "2024-01-05", "description": "Coffee A", "amount": 5.0,
         "source_file": "fileA.pdf", "content_hash": "hash_a"},
        {"id": "fa-2", "date": "2024-01-10", "description": "Groceries A", "amount": 40.0,
         "source_file": "fileA.pdf", "content_hash": "hash_a"},
    ])
    ledger.mark_uploaded("hash_a", "fileA.pdf", "2024-01-05", "2024-01-10", 2)
    ledger.upsert([
        {"id": "fb-1", "date": "2024-02-03", "description": "Gas B", "amount": 30.0,
         "source_file": "fileB.pdf", "content_hash": "hash_b"},
    ])
    ledger.mark_uploaded("hash_b", "fileB.pdf", "2024-02-03", "2024-02-03", 1)


def test_list_uploads_returns_this_users_uploads(ledger):
    _seed_two_uploads(ledger)
    uploads = ledger.list_uploads()
    assert {u["filename"] for u in uploads} == {"fileA.pdf", "fileB.pdf"}


def test_delete_upload_removes_only_that_files_transactions(ledger):
    _seed_two_uploads(ledger)
    removed = ledger.delete_upload("hash_a")
    assert removed == 2
    remaining_ids = {r["id"] for r in ledger.query("SELECT id FROM transactions")}
    # fileA's two rows are gone; fileB's row and the fixture's original two
    # rows (no content_hash, untouched by either upload) all survive.
    assert remaining_ids == {"a", "b", "fb-1"}


def test_delete_upload_removes_its_upload_history_entry(ledger):
    _seed_two_uploads(ledger)
    ledger.delete_upload("hash_a")
    uploads = ledger.list_uploads()
    assert {u["filename"] for u in uploads} == {"fileB.pdf"}
    assert ledger.is_duplicate_upload("hash_a") is False


def test_delete_upload_lets_the_same_file_be_reuploaded(ledger):
    _seed_two_uploads(ledger)
    ledger.delete_upload("hash_a")
    # Same content_hash as before deletion — re-upserting shouldn't collide
    # with anything still present (fileB's row uses a different id/hash).
    ledger.upsert([
        {"id": "fa-1-again", "date": "2024-03-01", "description": "Coffee A retry",
         "amount": 6.0, "source_file": "fileA.pdf", "content_hash": "hash_a"},
    ])
    assert ledger.is_duplicate_upload("hash_a") is False  # not re-marked until mark_uploaded() runs
    ledger.mark_uploaded("hash_a", "fileA.pdf", "2024-03-01", "2024-03-01", 1)
    assert ledger.is_duplicate_upload("hash_a") is True


def test_delete_upload_is_scoped_per_user(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    for l in (a, b):
        l.upsert([
            {"id": f"{l.user_id}-1", "date": "2024-01-05", "description": "Coffee",
             "amount": 5.0, "source_file": "statement.pdf", "content_hash": "same_hash"},
        ])
        l.mark_uploaded("same_hash", "statement.pdf", "2024-01-05", "2024-01-05", 1)

    removed = a.delete_upload("same_hash")
    assert removed == 1
    assert a.count() == 0
    assert b.count() == 1  # user_b's identical-hash upload is untouched


# ── Per-upload rename ────────────────────────────────────────────────────────

def test_rename_upload_updates_upload_history(ledger):
    _seed_two_uploads(ledger)
    n = ledger.rename_upload("hash_a", "January Chase Statement.pdf")
    assert n == 2
    uploads = {u["content_hash"]: u["filename"] for u in ledger.list_uploads()}
    assert uploads["hash_a"] == "January Chase Statement.pdf"
    assert uploads["hash_b"] == "fileB.pdf"  # untouched


def test_rename_upload_updates_transaction_source_file(ledger):
    _seed_two_uploads(ledger)
    ledger.rename_upload("hash_a", "January Chase Statement.pdf")
    rows = ledger.query("SELECT id, source_file FROM transactions ORDER BY id")
    by_id = {r["id"]: r["source_file"] for r in rows}
    assert by_id["fa-1"] == "January Chase Statement.pdf"
    assert by_id["fa-2"] == "January Chase Statement.pdf"
    assert by_id["fb-1"] == "fileB.pdf"  # different upload, untouched


def test_rename_upload_is_scoped_per_user(shared_db):
    a = Ledger(shared_db.cursor(), "user_a")
    b = Ledger(shared_db.cursor(), "user_b")
    for l in (a, b):
        l.upsert([
            {"id": f"{l.user_id}-1", "date": "2024-01-05", "description": "Coffee",
             "amount": 5.0, "source_file": "statement.pdf", "content_hash": "same_hash"},
        ])
        l.mark_uploaded("same_hash", "statement.pdf", "2024-01-05", "2024-01-05", 1)

    a.rename_upload("same_hash", "Renamed by A.pdf")
    a_name = next(u["filename"] for u in a.list_uploads() if u["content_hash"] == "same_hash")
    b_name = next(u["filename"] for u in b.list_uploads() if u["content_hash"] == "same_hash")
    assert a_name == "Renamed by A.pdf"
    assert b_name == "statement.pdf"  # user_b's record is untouched


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
