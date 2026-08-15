"""Tests for merchant enrichment: rule-based cleanup, and the batch
web-search parsing path — specifically the markdown-stripping fix after a
real "**Entertainment**" (literal asterisks) value was found leaking into
merchant_cache from a model reply that wrapped a field in markdown emphasis
despite the prompt asking for a plain "Merchant: X | Category: Y" line.
"""
from __future__ import annotations

from types import SimpleNamespace

from penny.ingest.enrich import _batch_lookup, _strip_markdown, apply_rules


# ── _strip_markdown ──────────────────────────────────────────────────────────

def test_strip_markdown_removes_bold_asterisks():
    assert _strip_markdown("**Entertainment**") == "Entertainment"


def test_strip_markdown_removes_italic_underscores():
    assert _strip_markdown("_Dining_") == "Dining"


def test_strip_markdown_removes_backticks():
    assert _strip_markdown("`Shopping`") == "Shopping"


def test_strip_markdown_leaves_plain_text_unchanged():
    assert _strip_markdown("Groceries") == "Groceries"


def test_strip_markdown_leaves_internal_punctuation_alone():
    # Only strips leading/trailing markers — a merchant name with an
    # apostrophe or internal asterisk (unlikely, but not this function's
    # job) shouldn't be mangled.
    assert _strip_markdown("Trader Joe's") == "Trader Joe's"


# ── apply_rules (unaffected by the fix, sanity check) ───────────────────────

def test_apply_rules_matches_known_merchant():
    assert apply_rules("STARBUCKS #1234") == ("Starbucks", "Dining")


def test_apply_rules_no_match_returns_none_none():
    assert apply_rules("SOME UNKNOWN VENDOR XYZ") == (None, None)


# ── _batch_lookup: parsing + the markdown fix, end to end ──────────────────

class _FakeMessages:
    def __init__(self, text: str, usage=None):
        self._text = text
        self._usage = usage or SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text=self._text)], usage=self._usage
        )


class _FakeClient:
    def __init__(self, text: str):
        self.messages = _FakeMessages(text)


def test_batch_lookup_strips_markdown_from_merchant_and_category():
    client = _FakeClient("1. Merchant: **AMC Theatres** | Category: **Entertainment**")
    results, usage = _batch_lookup(client, ["AMC 1234"])
    assert results == {"AMC 1234": {"merchant": "AMC Theatres", "category": "Entertainment"}}
    assert usage is not None


def test_batch_lookup_plain_reply_unaffected():
    client = _FakeClient("1. Merchant: Chipotle | Category: Dining")
    results, _ = _batch_lookup(client, ["CHIPOTLE 987"])
    assert results == {"CHIPOTLE 987": {"merchant": "Chipotle", "category": "Dining"}}


def test_batch_lookup_maps_multiple_descriptors_by_index():
    client = _FakeClient(
        "1. Merchant: Starbucks | Category: Dining\n"
        "2. Merchant: Uber | Category: Transport"
    )
    results, _ = _batch_lookup(client, ["STARBUCKS #1", "UBER TRIP"])
    assert results["STARBUCKS #1"] == {"merchant": "Starbucks", "category": "Dining"}
    assert results["UBER TRIP"] == {"merchant": "Uber", "category": "Transport"}


def test_batch_lookup_unparseable_reply_returns_empty_results():
    client = _FakeClient("Sorry, I couldn't identify these.")
    results, usage = _batch_lookup(client, ["UNKNOWN VENDOR"])
    assert results == {}
    assert usage is not None  # the call still succeeded; just nothing matched


def test_batch_lookup_returns_none_usage_on_api_failure():
    class _RaisingMessages:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    client = SimpleNamespace(messages=_RaisingMessages())
    results, usage = _batch_lookup(client, ["ANYTHING"])
    assert results == {}
    assert usage is None
