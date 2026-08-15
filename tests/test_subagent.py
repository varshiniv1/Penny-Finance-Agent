"""Tests for the categorize_transaction sub-agent's reply parsing —
specifically the markdown-stripping fix (see tests/test_enrich.py's module
docstring for the real "**Entertainment**" bug this closes; both call sites
shared the same unstripped-regex-capture defect)."""
from __future__ import annotations

from types import SimpleNamespace

import anthropic

from penny.agent.subagent import _strip_markdown, categorize_merchant


def test_strip_markdown_removes_bold_asterisks():
    assert _strip_markdown("**Entertainment**") == "Entertainment"


def _fake_anthropic(monkeypatch, reply_text: str, usage=None):
    usage = usage or SimpleNamespace(input_tokens=10, output_tokens=5)

    class _FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=reply_text)], usage=usage)

    class _FakeClient:
        def __init__(self, api_key):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    return usage


def test_categorize_merchant_strips_markdown_from_reply(monkeypatch):
    usage = _fake_anthropic(monkeypatch, "Merchant: AMC Theatres\nCategory: **Entertainment**")
    result = categorize_merchant("AMC 1234", "fake-key")
    assert result == ({"merchant": "AMC Theatres", "category": "Entertainment"}, usage)


def test_categorize_merchant_plain_reply_unaffected(monkeypatch):
    usage = _fake_anthropic(monkeypatch, "Merchant: Chipotle\nCategory: Dining")
    result = categorize_merchant("CHIPOTLE 987", "fake-key")
    assert result == ({"merchant": "Chipotle", "category": "Dining"}, usage)


def test_categorize_merchant_returns_none_when_unparseable(monkeypatch):
    _fake_anthropic(monkeypatch, "I'm not sure what this merchant is.")
    assert categorize_merchant("???", "fake-key") is None
