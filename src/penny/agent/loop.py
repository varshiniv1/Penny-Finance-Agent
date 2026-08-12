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
    model: str = "claude-haiku-4-5-20251001",
) -> Generator[dict, None, None]:
    """
    Run one conversational turn, yielding events as they happen:
      {"type": "text",           "text": "..."}
      {"type": "tool_call",      "name": "...", "input": {...}}
      {"type": "tool_result",    "name": "...", "result": {...}}
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
    messages = list(history)

    betas = ["code-execution-2025-08-25"]
    extra: dict[str, Any] = {}
    if _EXPORT_INTENT_RE.search(user_message):
        betas.append("skills-2025-10-02")
        extra["container"] = {"skills": [{"type": "anthropic", "skill_id": "xlsx"}]}

    while True:
        # .beta namespace: code_execution requires it; the xlsx skill (container.skills,
        # added above only when this turn looks export-related) needs the extra
        # skills-2025-10-02 beta too. Same params/behavior otherwise as
        # client.messages.create — a superset, not a different call shape.
        response = client.beta.messages.create(
            model=model,
            max_tokens=4096,
            system=_CACHED_SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=_with_cache_breakpoint(messages),
            betas=betas,
            **extra,
        )
        yield {"type": "usage", "source": "chat_turn", "model": model, "usage": response.usage}

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

            else:
                # Server-side tools (web_search, code_execution, ...) are
                # already fully resolved by the API within this response —
                # no tool_result round-trip needed on our end. Preserve the
                # block as-is so it round-trips correctly if the conversation
                # continues, instead of silently dropping it from history.
                block_dict = block.model_dump()
                assistant_content.append(block_dict)
                if block.type == "bash_code_execution_tool_result":
                    yield {"type": "code_execution", "block": block_dict}
                    yield from _emit_code_execution_files(client, block)

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
                        "model": "claude-haiku-4-5-20251001",
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
