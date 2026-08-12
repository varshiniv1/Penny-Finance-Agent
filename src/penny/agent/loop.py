"""Agentic conversation loop: handles tool calls until stop_reason == end_turn."""
from __future__ import annotations

import json
from typing import Any, Generator, TYPE_CHECKING

import anthropic

from penny.agent.prompts import SYSTEM_PROMPT
from penny.agent.tools import TOOL_SCHEMAS, ToolExecutor

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


def run_turn(
    user_message: str,
    history: list[dict],
    ledger: "Ledger",
    fts: "FTSIndex",
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> Generator[dict, None, None]:
    """
    Run one conversational turn, yielding events as they happen:
      {"type": "text",       "text": "..."}
      {"type": "tool_call",  "name": "...", "input": {...}}
      {"type": "tool_result","name": "...", "result": {...}}
      {"type": "chart",      "chart_json": "..."}
      {"type": "sql",        "sql": "..."}
      {"type": "done"}
    """
    client = anthropic.Anthropic(api_key=api_key)
    executor = ToolExecutor(ledger, fts)

    # Persist the user's message immediately, and keep `messages` (this turn's
    # working context) and `history` (the caller's persistent record) in sync
    # from here on — every assistant/tool turn below is appended to both.
    history.append({"role": "user", "content": user_message})
    messages = list(history)

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_CACHED_SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=_with_cache_breakpoint(messages),
        )

        # A single response can contain text AND multiple tool_use blocks —
        # collect them all before touching history, so the reconstructed
        # conversation matches what the model actually produced.
        assistant_content = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                yield {"type": "text", "text": block.text}
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_use_blocks.append(block)

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

                # web_search is handled by Anthropic natively; other tools run locally
                if tool_name == "web_search":
                    # The API handles this; we pass the tool_result block from the response
                    # This branch shouldn't be reached in normal flow
                    result = {"info": "Web search handled natively by the API"}
                else:
                    result = executor.run(tool_name, tool_input)

                yield {"type": "tool_result", "name": tool_name, "result": result}

                # Emit chart JSON if generate_chart succeeded
                if tool_name == "generate_chart" and "chart_json" in result:
                    yield {"type": "chart", "chart_json": result["chart_json"]}

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
