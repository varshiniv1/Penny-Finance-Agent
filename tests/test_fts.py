"""Tests for the FTS5 search index, especially special-character query safety."""
import pytest

from penny.storage.fts import FTSIndex


@pytest.fixture
def fts():
    f = FTSIndex(":memory:")
    f.index([
        {"id": "a", "description": "STARBUCKS #1234 - downtown", "merchant": "Starbucks", "category": "Dining"},
        {"id": "b", "description": "WHOLE FOODS MKT", "merchant": "Whole Foods", "category": "Groceries"},
    ])
    return f


def test_basic_search_finds_match(fts):
    hits = fts.search("starbucks")
    assert len(hits) == 1
    assert hits[0]["tx_id"] == "a"


def test_search_no_match_returns_empty(fts):
    assert fts.search("nonexistent merchant") == []


def test_search_empty_query_returns_empty(fts):
    assert fts.search("") == []


@pytest.mark.parametrize("weird_query", [
    '"unterminated quote',
    "trailing-dash-",
    "colon:syntax",
    "AND OR NOT -weird*(",
    "()",
])
def test_search_special_characters_dont_raise(fts, weird_query):
    # Previously raised sqlite3.OperationalError since raw input was passed
    # straight through as FTS5 MATCH syntax.
    fts.search(weird_query)  # should not raise


def test_reindex_replaces_old_entry():
    f = FTSIndex(":memory:")
    f.index([{"id": "a", "description": "OLD DESC", "merchant": "", "category": ""}])
    f.index([{"id": "a", "description": "NEW DESC", "merchant": "", "category": ""}])
    hits = f.search("new")
    assert len(hits) == 1
    assert f.search("old") == []
