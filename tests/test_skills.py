"""Tests for the xlsx Agent Skill's attachment logic (_request_extras /
_EXPORT_INTENT_RE in loop.py).

Per Anthropic's Skills-with-the-API guidance, a skill's description sits in
the model's context on every turn it's attached — so it should only be
attached when the turn actually looks like it needs it. This app decides
that with a single regex check against the user's own message; these tests
pin down exactly which messages do and don't trigger it, plus the shape of
the container/beta config sent to the API when it does.
"""
from __future__ import annotations

import pytest

from penny.agent.loop import _EXPORT_INTENT_RE, _request_extras


# ── _EXPORT_INTENT_RE: which messages count as "export intent" ─────────────

@pytest.mark.parametrize("message", [
    "export my transactions",
    "can you export this to a file",
    "I want to download my statement",
    "give me a spreadsheet of March spending",
    "can you make an excel file",
    "send this as xlsx",
    "EXPORT everything",  # case-insensitive
])
def test_matches_export_related_messages(message):
    assert _EXPORT_INTENT_RE.search(message)


@pytest.mark.parametrize("message", [
    "how much did I spend on coffee",
    "what's my biggest expense this month",
    "show me a chart of spending by category",
    "categorize this transaction",
    "",
])
def test_does_not_match_unrelated_messages(message):
    assert not _EXPORT_INTENT_RE.search(message)


def test_word_boundary_prevents_substring_false_positives():
    # "exporter"/"exports" contain "export" but aren't the standalone word —
    # \b(export|...)​\b is meant to catch deliberate requests, not any string
    # that happens to contain "export" as a substring.
    assert not _EXPORT_INTENT_RE.search("who is the biggest exporter of coffee")
    assert not _EXPORT_INTENT_RE.search("exports data")


# ── _request_extras: betas + container config ───────────────────────────────

def test_no_export_intent_leaves_container_unset():
    betas, extra = _request_extras("how much did I spend on coffee")
    assert "container" not in extra
    assert "skills-2025-10-02" not in betas


def test_export_intent_attaches_xlsx_skill_container():
    betas, extra = _request_extras("export my March statement to excel")
    assert extra["container"] == {"skills": [{"type": "anthropic", "skill_id": "xlsx"}]}


def test_export_intent_adds_the_skills_beta():
    betas, _ = _request_extras("download this as a spreadsheet")
    assert "skills-2025-10-02" in betas


def test_base_betas_always_present_regardless_of_export_intent():
    # code-execution/interleaved-thinking betas are unconditional — only the
    # skills beta is contingent on export intent.
    for message in ("export to excel", "how much did I spend"):
        betas, _ = _request_extras(message)
        assert "code-execution-2025-08-25" in betas
        assert "interleaved-thinking-2025-05-14" in betas


def test_request_extras_does_not_mutate_shared_base_betas_list():
    # Regression guard: _request_extras builds a fresh list each call (`list(_BASE_BETAS)`)
    # — if it appended to _BASE_BETAS directly instead, the skills beta from one
    # export-related turn would leak into every subsequent turn's request.
    _request_extras("export to excel")
    betas, _ = _request_extras("how much did I spend on coffee")
    assert "skills-2025-10-02" not in betas
