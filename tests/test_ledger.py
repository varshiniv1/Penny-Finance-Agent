"""Tests for the DuckDB ledger wrapper, especially the query() SQL guard."""
import pytest

from penny.storage.ledger import Ledger


@pytest.fixture
def ledger():
    l = Ledger(":memory:")
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

    fresh = Ledger(":memory:")
    n = fresh.import_parquet(path)
    assert n == 2
    assert fresh.count() == 2


def test_upsert_is_idempotent_on_same_id(ledger):
    ledger.upsert([{"id": "a", "date": "2024-01-01", "description": "Coffee", "amount": 5.0}])
    assert ledger.count() == 2
