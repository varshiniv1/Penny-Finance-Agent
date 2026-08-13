"""Tests for DuckDB-native full-text search (the fts extension), including
multi-tenant isolation between users sharing one physical table."""
import duckdb
import pytest

from penny.storage.fts import FTSIndex
from penny.storage.ledger import Ledger


@pytest.fixture
def ledger_and_fts():
    l = Ledger(":memory:", "user_a")
    l.upsert([
        {"id": "a", "date": "2024-01-01", "description": "STARBUCKS #1234 - downtown",
         "merchant": "Starbucks", "category": "Dining", "amount": 6.5},
        {"id": "b", "date": "2024-01-02", "description": "WHOLE FOODS MKT",
         "merchant": "Whole Foods", "category": "Groceries", "amount": 40.0},
    ])
    f = FTSIndex(l._con, "user_a")
    f.index()
    return l, f


def test_basic_search_finds_match(ledger_and_fts):
    _, fts = ledger_and_fts
    hits = fts.search("starbucks")
    assert len(hits) == 1
    assert hits[0]["tx_id"] == "a"


def test_search_no_match_returns_empty(ledger_and_fts):
    _, fts = ledger_and_fts
    assert fts.search("nonexistent merchant") == []


def test_search_empty_query_returns_empty(ledger_and_fts):
    _, fts = ledger_and_fts
    assert fts.search("") == []


@pytest.mark.parametrize("weird_query", [
    '"unterminated quote',
    "trailing-dash-",
    "colon:syntax",
    "AND OR NOT -weird*(",
    "()",
])
def test_search_special_characters_dont_raise(ledger_and_fts, weird_query):
    _, fts = ledger_and_fts
    fts.search(weird_query)  # should not raise


def test_reindex_picks_up_new_rows():
    l = Ledger(":memory:", "user_a")
    f = FTSIndex(l._con, "user_a")
    l.upsert([{"id": "a", "date": "2024-01-01", "description": "OLD DESC", "amount": 1.0}])
    f.index()
    assert f.search("old") != []

    l.upsert([{"id": "b", "date": "2024-01-02", "description": "NEW DESC", "amount": 1.0}])
    f.index()
    assert f.search("new") != []
    assert f.search("old") != []  # still there — rebuild covers the whole table


# ── Multi-tenant isolation ──────────────────────────────────────────────────

def test_search_is_scoped_per_user():
    # Independent cursors onto one :memory: database — DuckDB's equivalent of
    # two users' separate connections to the same shared MotherDuck database.
    con = duckdb.connect(":memory:")
    try:
        a = Ledger(con.cursor(), "user_a")
        b = Ledger(con.cursor(), "user_b")
        a.upsert([{"id": "1", "date": "2024-01-01", "description": "STARBUCKS", "amount": 5.0}])
        b.upsert([{"id": "2", "date": "2024-01-01", "description": "NETFLIX", "amount": 15.0}])

        fts_a = FTSIndex(a._con, "user_a")
        fts_a.index()
        fts_b = FTSIndex(b._con, "user_b")

        assert [h["tx_id"] for h in fts_a.search("starbucks")] == ["1"]
        # The physical fts index covers both users' rows, but the user_id
        # filter in the query must keep user_a from ever seeing user_b's hit.
        assert fts_a.search("netflix") == []
        assert [h["tx_id"] for h in fts_b.search("netflix")] == ["2"]
    finally:
        con.close()
