"""Agentic conversation loop: handles tool calls until stop_reason == end_turn."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Generator, TYPE_CHECKING

import anthropic

from penny.agent.prompts import SYSTEM_PROMPT
from penny.agent.tools import TOOL_SCHEMAS, ToolExecutor
from penny.config import DEFAULT_MODEL

if TYPE_CHECKING:
    from penny.storage.ledger import Ledger
    from penny.storage.fts import FTSIndex

# Static across every request — cache it. The breakpoint here also covers
# TOOL_SCHEMAS, since the API renders tools -> system -> messages and a
# breakpoint caches everything rendered before it.
_CACHED_SYSTEM = [
    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
]


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with a cache breakpoint on the last content block.

    Placed fresh on every call rather than persisted into `history`, so the
    marker always sits on the current last block instead of accumulating on
    old ones (max 4 breakpoints per request).
    """
    if not messages:
        return messages
    *rest, last = messages
    content = last["content"]
    content = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    if not content:
        return messages
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    return [*rest, {**last, "content": content}]


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# The xlsx skill's description sits in context on every turn it's attached,
# even when unused — cheap per-turn, but there's no reason to pay it on
# messages that plainly aren't about exporting. Checked once against the
# turn's initial user message (not re-checked per tool-calling round), so a
# multi-step "query then export" turn keeps the skill for its whole duration.
_EXPORT_INTENT_RE = re.compile(
    r"\b(export|download|spreadsheet|excel|xlsx)\b", re.IGNORECASE
)

# code-execution-2025-08-25/interleaved-thinking-2025-05-14: both current tool
# versions (code_execution_20260521 in tools.py) are GA and no longer require
# a beta header per Anthropic's docs — these are legacy opt-ins, kept as
# harmless no-ops rather than removed outright, since the docs state legacy
# code-execution beta headers "remain valid opt-ins."
_BASE_BETAS = ["code-execution-2025-08-25", "interleaved-thinking-2025-05-14"]
_XLSX_SKILL_BETA = "skills-2025-10-02"
_XLSX_SKILL_CONTAINER = {"skills": [{"type": "anthropic", "skill_id": "xlsx"}]}


def _request_extras(user_message: str) -> tuple[list[str], dict[str, Any]]:
    """Betas and extra request kwargs for one turn's API call. Split out from
    run_turn so the xlsx-skill attachment decision (see _EXPORT_INTENT_RE) is
    testable on its own, without mocking a whole streaming API call."""
    betas = list(_BASE_BETAS)
    extra: dict[str, Any] = {}
    if _EXPORT_INTENT_RE.search(user_message):
        betas.append(_XLSX_SKILL_BETA)
        extra["container"] = _XLSX_SKILL_CONTAINER
    return betas, extra

# Hard ceiling on tool-calling round trips within a single turn — a model
# stuck in a retry loop (or one that just won't stop calling tools) should
# fail closed instead of costing an unbounded number of API calls.
_MAX_TOOL_ROUNDS = 20

# Ceiling on extended-thinking tokens per API call — must stay below the
# 4096 max_tokens above (the API rejects otherwise), leaving plenty of room
# for the actual response afterward. A ceiling, not a fixed spend.
_THINKING_BUDGET_TOKENS = 1024

# Without this, `history` (and therefore the request body) grows for the
# whole browser session — every turn resends the entire conversation,
# including every past tool_result (a query_sql call alone can carry ~100
# rows of JSON). Prompt caching makes that cheaper per token, but the token
# *count* still climbs every turn and the cache itself expires after 5 min
# idle, so a long or resumed session still pays full price for all of it.
# Keeping only the most recent turns bounds both the cost and the context
# window regardless of session length.
#
# Trimmed in batches, not down to the target every turn: the cache matches
# the longest common *prefix* from the start of the message list, so if the
# front of `history` shifted on every turn, every request would miss the
# cache entirely and pay a fresh 1.25x write on top — worse than not
# trimming at all. Holding the window steady between trims (only cutting
# once _TRIM_SLACK extra turns have piled up, back down to
# _MAX_HISTORY_TURNS) lets caching keep working normally in between.
_MAX_HISTORY_TURNS = 20
_TRIM_SLACK = 10


def _trim_history(history: list[dict]) -> None:
    """Drop the oldest turns once more than _MAX_HISTORY_TURNS + _TRIM_SLACK
    have piled up, cutting back down to _MAX_HISTORY_TURNS — in place.

    Only cuts at a genuine user-turn boundary — a "user" message with plain
    string content — never at a tool_result continuation (also role="user"
    but with list content), so a tool_use is never left without its result.
    """
    boundaries = [
        i for i, m in enumerate(history) if m["role"] == "user" and isinstance(m["content"], str)
    ]
    if len(boundaries) <= _MAX_HISTORY_TURNS + _TRIM_SLACK:
        return
    cut = boundaries[-_MAX_HISTORY_TURNS]
    del history[:cut]


def _summarize_web_search_result(block) -> str:
    """One-line status for a completed web_search call — content is either a
    WebSearchToolResultError (has .type == "web_search_tool_result_error") or
    a plain list of result blocks (no .type attribute, hence the getattr
    default) on success."""
    content = block.content
    if getattr(content, "type", None) == "web_search_tool_result_error":
        return f"Web search failed ({content.error_code})"
    n = len(content) if isinstance(content, list) else 0
    return f"Found {n} result{'s' if n != 1 else ''}"


# UI-only cap on how many result cards render — the model itself still sees
# every result the API returned (this never touches what's sent back to it).
_MAX_DISPLAYED_SEARCH_RESULTS = 10


def _web_search_results(block) -> list[dict]:
    """title/url pairs for a completed web_search call, for the rich result
    card — [] on error or an empty result set. No extra API call: this is
    just reading data the response already carried, previously discarded
    after being reduced to a bare count."""
    content = block.content
    if not isinstance(content, list):
        return []
    return [{"title": r.title, "url": r.url} for r in content[:_MAX_DISPLAYED_SEARCH_RESULTS]]


def _summarize_code_execution_result(block) -> str:
    """One-line status for a completed code_execution call — content is
    either a BetaBashCodeExecutionToolResultError or a
    BetaBashCodeExecutionResultBlock (has .return_code) on success."""
    content = block.content
    if getattr(content, "type", None) == "bash_code_execution_tool_result_error":
        return f"Code execution failed ({content.error_code})"
    code = getattr(content, "return_code", None)
    return "Code ran successfully" if code == 0 else f"Code exited with status {code}"


def _emit_code_execution_files(client, block) -> Generator[dict, None, None]:
    """Download any files a code_execution call produced — a chart image, or (with
    the xlsx skill enabled) a generated spreadsheet.

    Best-effort: a download failure surfaces as a text note rather than
    breaking the turn, since the code execution itself already succeeded.
    """
    result = getattr(block, "content", None)
    file_refs = getattr(result, "content", None) or []
    for file_ref in file_refs:
        if getattr(file_ref, "type", None) != "bash_code_execution_output":
            continue
        try:
            metadata = client.beta.files.retrieve_metadata(file_ref.file_id)
            downloaded = client.beta.files.download(file_ref.file_id)
            suffix = Path(metadata.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            downloaded.write_to_file(tmp_path)
            file_bytes = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            if suffix.lower() in _IMAGE_EXTS:
                yield {"type": "image", "filename": metadata.filename, "image_bytes": file_bytes}
            else:
                yield {"type": "file", "filename": metadata.filename, "file_bytes": file_bytes}
        except Exception as e:
            yield {"type": "text", "text": f"\n_(couldn't retrieve generated file: {e})_\n"}


def run_turn(
    user_message: str,
    history: list[dict],
    ledger: "Ledger",
    fts: "FTSIndex",
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Generator[dict, None, None]:
    """
    Run one conversational turn, yielding events as they happen:
      {"type": "text",           "text": "..."}          (one per streamed chunk)
      {"type": "thinking_start"}
      {"type": "thinking_delta", "text": "..."}           (one per streamed chunk)
      {"type": "thinking_stop"}
      {"type": "tool_call",      "name": "...", "input": {...}}
      {"type": "tool_result",    "name": "...", "result": {...}}
      {"type": "op_result",      "name": "...", "text": "..."}
      {"type": "web_search_results", "results": [{"title": "...", "url": "..."}, ...]}
      {"type": "chart",          "chart_json": "..."}
      {"type": "sql",            "sql": "..."}
      {"type": "code_execution", "block": {...}}
      {"type": "image",          "filename": "...", "image_bytes": b"..."}
      {"type": "file",           "filename": "...", "file_bytes": b"..."}
      {"type": "usage",          "source": "...", "model": "...", "usage": ...}
      {"type": "done"}
    """
    client = anthropic.Anthropic(api_key=api_key)
    executor = ToolExecutor(ledger, fts, api_key)

    # Persist the user's message immediately, and keep `messages` (this turn's
    # working context) and `history` (the caller's persistent record) in sync
    # from here on — every assistant/tool turn below is appended to both.
    history.append({"role": "user", "content": user_message})
    _trim_history(history)
    messages = list(history)

    # interleaved-thinking lets a "thinking" block appear before *each* tool
    # call in a multi-round turn, not just once at the very start — verified
    # live against claude-haiku-4-5-20251001, which does support extended
    # thinking. budget_tokens is a ceiling, not something forced to be
    # spent — real usage tends to land well under it for a question this
    # size — but it does add real cost every round, billed as output
    # tokens, since this is a genuine tradeoff (transparency into the
    # model's reasoning vs. token cost), not a free win.
    betas, extra = _request_extras(user_message)

    for _round in range(_MAX_TOOL_ROUNDS):
        # .beta namespace: code_execution requires it; the xlsx skill (container.skills,
        # added above only when this turn looks export-related) needs the extra
        # skills-2025-10-02 beta too. Same params/behavior otherwise as
        # client.messages.create — a superset, not a different call shape.
        # Streamed (not .create()) so thinking and text render token-by-token
        # as the model produces them, the same live-typing feel as Claude
        # Code's own UI, instead of appearing all at once after the whole
        # round finishes. The SDK's helper stream fires both raw SSE events
        # and derived per-delta convenience events ("thinking"/"text") as it
        # accumulates them — only the derived ones are needed here, since
        # get_final_message() below hands back the exact same fully-assembled
        # Message the old .create() call returned, so everything after this
        # point (tool_use handling, history bookkeeping) is unchanged.
        with client.beta.messages.stream(
            model=model,
            max_tokens=4096,
            system=_CACHED_SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=_with_cache_breakpoint(messages),
            betas=betas,
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS},
            **extra,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "thinking":
                    yield {"type": "thinking_start"}
                elif event.type == "thinking":
                    yield {"type": "thinking_delta", "text": event.thinking}
                elif event.type == "text":
                    yield {"type": "text", "text": event.text}
            response = stream.get_final_message()
        yield {"type": "usage", "source": "chat_turn", "model": model, "usage": response.usage}

        # A single response can contain text AND multiple tool_use blocks —
        # collect them all before touching history, so the reconstructed
        # conversation matches what the model actually produced.
        assistant_content = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                # Already streamed live above via the "text" delta events —
                # only bookkeeping is left to do here.
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "thinking":
                # Deltas already streamed live above; this just tells the UI
                # the block is complete (e.g. so it can stop showing a live
                # cursor). model_dump() (not a hand-built dict) so the
                # signature field round-trips exactly as the API issued it —
                # required for a continued conversation to remain valid.
                yield {"type": "thinking_stop"}
                assistant_content.append(block.model_dump())

            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_use_blocks.append(block)

            else:
                # Server-side tools (web_search, code_execution, ...) are
                # already fully resolved by the API within this response —
                # no tool_result round-trip needed on our end. Preserve the
                # block as-is so it round-trips correctly if the conversation
                # continues, instead of silently dropping it from history.
                block_dict = block.model_dump()
                assistant_content.append(block_dict)
                if block.type == "server_tool_use":
                    # The invocation side (web_search, code_execution, ...) —
                    # surfaced the same lightweight way as a client-side tool_call,
                    # so it shows up as an operation indicator in the UI too.
                    yield {"type": "tool_call", "name": block.name, "input": block.input}
                if block.type == "bash_code_execution_tool_result":
                    yield {
                        "type": "op_result", "name": "code_execution",
                        "text": _summarize_code_execution_result(block),
                    }
                    yield {"type": "code_execution", "block": block_dict}
                    yield from _emit_code_execution_files(client, block)
                if block.type == "web_search_tool_result":
                    # Previously fell all the way through to the bare
                    # assistant_content.append() above with no UI event at
                    # all — a web_search call's invocation showed up (via the
                    # server_tool_use branch), but never its outcome.
                    yield {
                        "type": "op_result", "name": "web_search",
                        "text": _summarize_web_search_result(block),
                    }
                    results = _web_search_results(block)
                    if results:
                        yield {"type": "web_search_results", "results": results}

        history.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "assistant", "content": assistant_content})

        if tool_use_blocks:
            # All tool_result blocks for this turn's tool_use blocks must be
            # returned together in a single following user turn.
            tool_result_content = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_input = block.input

                # Emit SQL for transparency before running it
                if tool_name == "query_sql":
                    yield {"type": "sql", "sql": tool_input.get("sql", "")}

                yield {"type": "tool_call", "name": tool_name, "input": tool_input}

                result = executor.run(tool_name, tool_input)

                yield {"type": "tool_result", "name": tool_name, "result": result}

                # Emit chart JSON if generate_chart succeeded
                if tool_name == "generate_chart" and "chart_json" in result:
                    yield {"type": "chart", "chart_json": result["chart_json"]}

                # categorize_transaction makes its own independent API call (a
                # sub-agent) outside this loop's own request — surface its usage
                # the same way so Observability still sees it.
                if executor.last_subagent_usage is not None:
                    yield {
                        "type": "usage",
                        "source": "chat_subagent_categorize",
                        "model": DEFAULT_MODEL,
                        "usage": executor.last_subagent_usage,
                    }
                    executor.last_subagent_usage = None

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            history.append({"role": "user", "content": tool_result_content})
            messages.append({"role": "user", "content": tool_result_content})

        if response.stop_reason == "end_turn":
            yield {"type": "done"}
            break

        if response.stop_reason != "tool_use":
            yield {"type": "done"}
            break
    else:
        yield {
            "type": "text",
            "text": "\n\n_(Stopped after too many tool calls in one turn — try rephrasing "
            "or breaking this into a smaller question.)_",
        }
        yield {"type": "done"}
