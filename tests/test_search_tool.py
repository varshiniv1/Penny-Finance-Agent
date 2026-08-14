"""Tests for the server-side search/execution tools: web_search and
code_execution — response parsing (loop.py) and how results get surfaced in
the UI (chat_page.py).

Result/error shapes here are pinned to what Anthropic's docs specify:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#errors
  - Common error codes across all server-side tools: unavailable,
    execution_time_exceeded, invalid_tool_input, too_many_requests.
  - Success content is a plain list of result blocks; error content is a
    single object with `.type == "..._tool_result_error"` and `.error_code`.

No MCP server exists in this codebase (grepped for `mcp_servers` — the actual
Anthropic SDK param name for wiring one up — with none found in loop.py's
request kwargs). test_no_mcp_server_is_configured below is a deliberate
tripwire: if an MCP server is added later, that assertion starts failing as a
reminder that this file's coverage needs to grow to match, rather than the
gap going unnoticed.
"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from penny.agent.loop import (
    _MAX_DISPLAYED_SEARCH_RESULTS,
    _summarize_code_execution_result,
    _summarize_web_search_result,
    _web_search_results,
)
from penny.ui.chat_page import _describe_tool_call, _render_search_results, _summarize_tool_result


def _search_result(title: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(title=title, url=url, type="web_search_result")


def _success_block(content) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _error_block(error_type: str, error_code: str) -> SimpleNamespace:
    return SimpleNamespace(content=SimpleNamespace(type=error_type, error_code=error_code))


# ── web_search: result parsing ──────────────────────────────────────────────

def test_web_search_results_extracts_title_and_url():
    block = _success_block([_search_result("Claude docs", "https://platform.claude.com/docs")])
    results = _web_search_results(block)
    assert results == [{"title": "Claude docs", "url": "https://platform.claude.com/docs"}]


def test_web_search_results_caps_at_display_limit():
    block = _success_block([_search_result(f"Result {i}", f"https://example{i}.com") for i in range(25)])
    results = _web_search_results(block)
    assert len(results) == _MAX_DISPLAYED_SEARCH_RESULTS == 10


def test_web_search_results_empty_on_error():
    block = _error_block("web_search_tool_result_error", "unavailable")
    assert _web_search_results(block) == []


def test_web_search_results_empty_when_no_hits():
    assert _web_search_results(_success_block([])) == []


def test_summarize_web_search_result_counts_hits():
    block = _success_block([_search_result("A", "https://a.com"), _search_result("B", "https://b.com")])
    assert _summarize_web_search_result(block) == "Found 2 results"


def test_summarize_web_search_result_singular_phrasing():
    block = _success_block([_search_result("A", "https://a.com")])
    assert _summarize_web_search_result(block) == "Found 1 result"


@pytest.mark.parametrize("error_code", ["unavailable", "execution_time_exceeded", "invalid_tool_input", "too_many_requests"])
def test_summarize_web_search_result_reports_documented_error_codes(error_code):
    block = _error_block("web_search_tool_result_error", error_code)
    summary = _summarize_web_search_result(block)
    assert "failed" in summary.lower()
    assert error_code in summary


# ── code_execution: result parsing ──────────────────────────────────────────

def test_summarize_code_execution_result_success():
    block = _success_block(SimpleNamespace(return_code=0))
    assert _summarize_code_execution_result(block) == "Code ran successfully"


def test_summarize_code_execution_result_nonzero_exit():
    block = _success_block(SimpleNamespace(return_code=1))
    assert "1" in _summarize_code_execution_result(block)


@pytest.mark.parametrize("error_code", ["unavailable", "execution_time_exceeded", "output_file_too_large"])
def test_summarize_code_execution_result_reports_documented_error_codes(error_code):
    block = _error_block("bash_code_execution_tool_result_error", error_code)
    summary = _summarize_code_execution_result(block)
    assert "failed" in summary.lower()
    assert error_code in summary


# ── UI: operation labels for the search/execution tools ────────────────────

def test_describe_tool_call_includes_web_search_query():
    label = _describe_tool_call("web_search", {"query": "anthropic certified associate"})
    assert "web_search" in label
    assert "anthropic certified associate" in label


def test_describe_tool_call_code_execution_has_no_raw_input_leaked():
    # Deliberately generic ("Ran code") — unlike query_sql/search_text, this
    # never echoes the tool_input back, since code_execution's input can be
    # an arbitrary shell command.
    label = _describe_tool_call("code_execution", {"command": "rm -rf /"})
    assert "Ran code" in label
    assert "rm -rf" not in label


def test_summarize_tool_result_passes_through_error():
    result = _summarize_tool_result("web_search", {"error": "boom"})
    assert result == "Failed: boom"


# ── UI: rich result card rendering ──────────────────────────────────────────

def test_render_search_results_does_not_raise(monkeypatch):
    # Streamlit calls no-op outside a real app run (verified: only logs a
    # "missing ScriptRunContext" warning, doesn't raise) — this just confirms
    # domain extraction/pluralization don't blow up on the shapes loop.py
    # actually produces.
    _render_search_results([
        {"title": "Claude docs", "url": "https://platform.claude.com/docs/en/foo"},
        {"title": "No path", "url": "https://example.com"},
    ])


def test_render_search_results_handles_empty_list():
    _render_search_results([])


# ── MCP: intentional absence, not an oversight ──────────────────────────────

def test_no_mcp_server_is_configured():
    """Tripwire, not a real feature test: Penny doesn't use MCP today (no
    `mcp_servers` in the request kwargs anywhere in loop.py). If that
    changes, this assertion starts failing — the reminder to add real MCP
    coverage here (server config validation, tool-result parsing for
    mcp_tool_use/mcp_tool_result blocks, connection-failure handling) instead
    of the gap going unnoticed.
    """
    loop_source = Path(__file__).parent.parent.joinpath("src", "penny", "agent", "loop.py").read_text()
    assert "mcp_servers" not in loop_source
