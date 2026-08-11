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

    messages = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Collect assistant content for history
        assistant_content = []

        for block in response.content:
            if block.type == "text":
                yield {"type": "text", "text": block.text}
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                # Emit SQL for transparency before running it
                if tool_name == "query_sql":
                    yield {"type": "sql", "sql": tool_input.get("sql", "")}

                yield {"type": "tool_call", "name": tool_name, "input": tool_input}
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": tool_name,
                    "input": tool_input,
                })

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

                messages = messages + [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str),
                            }
                        ],
                    },
                ]
                assistant_content = []  # reset for next loop

        if response.stop_reason == "end_turn":
            # Append the final assistant turn to history for the caller
            if assistant_content:
                history.append({"role": "assistant", "content": assistant_content})
            yield {"type": "done"}
            break

        if response.stop_reason != "tool_use":
            yield {"type": "done"}
            break
