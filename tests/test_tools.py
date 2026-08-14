"""Tests for the custom tools exposed to the agent (TOOL_SCHEMAS + ToolExecutor).

Schema tests follow Anthropic's tool-definition contract: every client-side
tool needs name/description/input_schema with required fields actually
declared in properties, and server-side tools (web_search, code_execution)
take no input_schema at all — see
https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#tool-definition
("The code execution tool requires no additional parameters ... Both fields
are fixed: `type` selects the tool version, and `name` must be
`code_execution`.").

Executor tests focus on the one invariant the code itself calls out as
critical (see ToolExecutor.run's docstring): a tool_use block must always get
a matching tool_result, so malformed input must produce an {"error": ...}
dict, never an unhandled exception — an uncaught exception here would leave
`history` holding a tool_use with no result, which the API rejects on the
next turn.
"""
from __future__ import annotations

import pytest

from penny.agent.tools import TOOL_SCHEMAS, ToolExecutor
from penny.storage.fts import FTSIndex
from penny.storage.ledger import Ledger

_CLIENT_SIDE_TOOL_NAMES = {"query_sql", "search_text", "generate_chart", "categorize_transaction"}
_SERVER_SIDE_TOOL_NAMES = {"web_search", "code_execution"}


# ── TOOL_SCHEMAS: structural validation ─────────────────────────────────────

def test_no_duplicate_tool_names():
    names = [t["name"] for t in TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_every_tool_covered_by_executor_or_is_server_side():
    # Anything not handled server-side must be one ToolExecutor.run() branches
    # away from an "Unknown tool" error — this catches a tool added to
    # TOOL_SCHEMAS but never wired into the executor.
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == _CLIENT_SIDE_TOOL_NAMES | _SERVER_SIDE_TOOL_NAMES


@pytest.mark.parametrize("tool", [t for t in TOOL_SCHEMAS if t["name"] in _CLIENT_SIDE_TOOL_NAMES])
def test_client_side_tool_has_name_description_and_input_schema(tool):
    assert tool.get("name")
    assert tool.get("description"), f"{tool['name']} needs a description — it's the model's only guidance on when to call it"
    schema = tool.get("input_schema")
    assert schema is not None
    assert schema["type"] == "object"
    assert "properties" in schema


@pytest.mark.parametrize("tool", [t for t in TOOL_SCHEMAS if t["name"] in _CLIENT_SIDE_TOOL_NAMES])
def test_every_required_field_is_declared_in_properties(tool):
    schema = tool["input_schema"]
    for field in schema.get("required", []):
        assert field in schema["properties"], f"{tool['name']} requires '{field}' but never declares it"


@pytest.mark.parametrize("tool", [t for t in TOOL_SCHEMAS if t["name"] in _SERVER_SIDE_TOOL_NAMES])
def test_server_side_tools_have_no_input_schema(tool):
    # Per the docs: "Both fields are fixed: `type` selects the tool version,
    # and `name` must be `code_execution`." Same fixed-shape contract applies
    # to web_search. An input_schema here would be a sign someone tried to
    # configure it like a client-side tool, which the API doesn't support.
    assert "input_schema" not in tool
    assert "type" in tool
    assert tool["name"] in tool["type"] or tool["type"].startswith(tool["name"])


def test_code_execution_tool_version_is_dated():
    # type must be "code_execution_YYYYMMDD" — catches a typo'd or truncated
    # version string that would otherwise only surface as an opaque 400 from
    # the API at chat time.
    code_exec = next(t for t in TOOL_SCHEMAS if t["name"] == "code_execution")
    prefix, _, date_part = code_exec["type"].partition("code_execution_")
    assert prefix == ""
    assert date_part.isdigit() and len(date_part) == 8


def test_web_search_tool_version_is_dated():
    web_search = next(t for t in TOOL_SCHEMAS if t["name"] == "web_search")
    prefix, _, date_part = web_search["type"].partition("web_search_")
    assert prefix == ""
    assert date_part.isdigit() and len(date_part) == 8


# ── ToolExecutor: the tool_use -> tool_result invariant ─────────────────────

@pytest.fixture
def executor():
    ledger = Ledger(":memory:", "user_a")
    ledger.upsert([
        {"id": "a", "date": "2024-01-01", "description": "STARBUCKS #1", "merchant": "Starbucks",
         "category": "Dining", "amount": 5.0},
        {"id": "b", "date": "2024-01-02", "description": "WHOLE FOODS", "merchant": "Whole Foods",
         "category": "Groceries", "amount": 40.0},
    ])
    fts = FTSIndex(ledger._con, "user_a")
    fts.index()
    return ToolExecutor(ledger, fts)


def test_unknown_tool_returns_error_dict_not_exception(executor):
    result = executor.run("delete_everything", {})
    assert "error" in result


def test_missing_required_argument_returns_error_not_raises(executor):
    # query_sql requires "sql" — omitting it would raise TypeError from the
    # **tool_input unpacking if run() didn't guard against it.
    result = executor.run("query_sql", {})
    assert "error" in result


def test_unexpected_extra_argument_returns_error_not_raises(executor):
    result = executor.run("query_sql", {"sql": "SELECT 1", "made_up_arg": "x"})
    assert "error" in result


def test_wrong_argument_type_returns_error_not_raises(executor):
    # top_k should be an int; a model hallucinating a string shouldn't crash the turn.
    result = executor.run("search_text", {"query": "coffee", "top_k": "not-a-number"})
    assert "error" in result


def test_query_sql_happy_path(executor):
    result = executor.run("query_sql", {"sql": "SELECT * FROM transactions ORDER BY date"})
    assert result["count"] == 2
    assert result["truncated"] is False


def test_query_sql_surfaces_ledger_validation_as_error_dict(executor):
    # Ledger.query() raises ValueError for disallowed SQL (see test_ledger.py's
    # own injection-attempt coverage) — the executor's job is to catch that
    # and hand back a tool_result-shaped error, not let it propagate.
    result = executor.run("query_sql", {"sql": "DROP TABLE transactions"})
    assert "error" in result


def test_query_sql_row_cap_is_stricter_than_ledger_default(executor):
    # _SQL_ROW_CAP (100) is intentionally below Ledger.query's own default
    # limit (1000) and not exposed via the tool's input_schema, so the model
    # can't just ask for more rows than the chat surface is meant to show.
    assert ToolExecutor._SQL_ROW_CAP < 1000
    assert "limit" not in TOOL_SCHEMAS[0]["input_schema"]["properties"]


def test_search_text_happy_path(executor):
    # The fts index covers description/merchant/category text (see fts.py) —
    # "starbucks" is literally in both the description and merchant columns
    # of the fixture row, unlike a semantic association like "coffee" would be.
    result = executor.run("search_text", {"query": "starbucks"})
    assert result["count"] >= 1
    assert any("STARBUCKS" in h["description"] for h in result["hits"])


def test_search_text_default_top_k(executor):
    result = executor.run("search_text", {"query": "starbucks"})
    assert "hits" in result and "count" in result


def test_generate_chart_errors_on_empty_query_result(executor):
    result = executor.run(
        "generate_chart",
        {"chart_type": "bar", "sql": "SELECT * FROM transactions WHERE amount > 999999", "title": "Nothing"},
    )
    assert "error" in result


def test_generate_chart_happy_path(executor):
    result = executor.run(
        "generate_chart",
        {
            "chart_type": "bar",
            "sql": "SELECT category, SUM(amount) AS total FROM transactions GROUP BY category",
            "title": "Spending by category",
            "x_col": "category",
            "y_col": "total",
        },
    )
    assert "chart_json" in result
    assert result["row_count"] == 2


def test_categorize_transaction_requires_api_key(executor):
    # No api_key was passed to the executor fixture — this must fail fast
    # with a clear error instead of attempting a sub-agent API call with an
    # empty key.
    result = executor.run("categorize_transaction", {"descriptor": "UNKNOWN MERCHANT"})
    assert "error" in result
    assert "api key" in result["error"].lower()
