"""Streamlit page: a single chat surface — ask questions about statements uploaded
via the sidebar (see streamlit_app.py's "Upload statements" expander)."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
from urllib.parse import urlparse

import plotly.io as pio
import streamlit as st

from penny.agent.insights import insight
from penny.agent.loop import run_turn
from penny.config import DEFAULT_MODEL
from penny.ingest.enrich import enrich_batch
from penny.ingest.extractor import extract
from penny.ingest.parser import parse_file_bytes
from penny.ingest.reconcile import reconcile
from penny.ui.session import (
    friendly_api_error, get_conversation_id, get_ledger, get_fts, get_history, get_user_key,
    list_recent_conversations, load_conversation, load_conversation_messages,
    log_usage, persist_conversation, search_conversations as _search_conversations,
    tx_count,
)

# Overrides Streamlit's default chat avatars (a stylized red icon for the
# user, an orange robot for the assistant) with a neutral silhouette and the
# app's own money-bag emoji — ties the assistant's identity to "💰 Penny"
# rather than a generic bot icon. A plain person emoji (🧑) was tried first,
# but its gold/orange skin tone read as near-identical to 💰 at a glance —
# 👤 stays visually distinct from it while still reading as "person."
_USER_AVATAR = "👤"
_ASSISTANT_AVATAR = "💰"

_TITLE_MAX_LEN = 50
# Types that make it into the persisted transcript / search index — never
# chart/image/file, whose "content" is binary or a large JSON chart spec, not
# something worth storing per-turn or searching by text.
_PERSISTABLE_TYPES = ("text", "op", "note", "thinking", "search_results")

_FILE_TYPES = ["pdf", "csv", "jpg", "jpeg", "png", "tiff", "tif", "bmp"]
_MAX_FILE_MB = 20
# Parsing multiple files in one upload runs concurrently, not one at a time —
# real speedup for OCR specifically (pytesseract/pdf2image shell out to
# tesseract/poppler as subprocesses, releasing the GIL while they run), less
# so for plain text-layer PDFs (pdfplumber is pure Python, GIL-bound). Capped
# so a large batch doesn't spawn dozens of concurrent OCR subprocesses on
# Streamlit Cloud's modest CPU allocation.
_MAX_PARSE_WORKERS = 4
# Cap how many chat turns are kept rendered/in memory — display_messages can
# hold raw image/chart/file bytes per turn, which would otherwise grow
# unbounded over a long session.
_MAX_DISPLAY_MESSAGES = 200

_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _yearmonth(date_str: str) -> tuple[int, int]:
    y, m, _ = str(date_str).split("-")
    return int(y), int(m)


def _month_span(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    y, m = start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _format_ym(ym: tuple[int, int]) -> str:
    y, m = ym
    return f"{_MONTH_ABBR[m]} {y}"


def _coverage_label(per_file_ranges: list[tuple[str, str]]) -> str:
    """Given each accepted file's (min_date, max_date), return a label like
    "Jan 2026 – Apr 2026" if the combined months are one continuous span, or
    a plain list of the actual covered months if there's a real gap — never
    implying continuity that isn't there."""
    if not per_file_ranges:
        return ""
    covered: set[tuple[int, int]] = set()
    for min_d, max_d in per_file_ranges:
        if not min_d or not max_d:
            continue
        covered.update(_month_span(_yearmonth(min_d), _yearmonth(max_d)))
    if not covered:
        return ""
    months_sorted = sorted(covered)
    if _month_span(months_sorted[0], months_sorted[-1]) == months_sorted:
        if months_sorted[0] == months_sorted[-1]:
            return _format_ym(months_sorted[0])
        return f"{_format_ym(months_sorted[0])} – {_format_ym(months_sorted[-1])}"
    return ", ".join(_format_ym(ym) for ym in months_sorted)


_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.*)$", re.MULTILINE)


def _normalize_markdown(text: str) -> str:
    """Tame LLM-generated Markdown for a chat bubble, not a full document:

    - Downgrade ATX headings (#, ##, ...) to bold text. A heading renders as
      a full page-size title — wildly oversized inside a chat message — bold
      keeps it visually set apart as a section label without blowing up the
      font size.
    - Escape literal `$` so st.markdown doesn't treat a pair of them as
      inline LaTeX/KaTeX math (a reply mentioning two dollar amounts, e.g.
      "$11,411.67 ... $718,000", would otherwise have everything between
      them silently rendered as a math expression instead of plain text).

    Must run on the *whole* text, not per streamed chunk — a heading marker
    or `$` can land split across two chunks, and the heading pattern needs a
    full line to match reliably.
    """
    text = _HEADING_RE.sub(lambda m: f"**{m.group(1)}**", text)
    text = _DOLLAR_RE.sub(lambda m: "\\$", text)
    return text


def _describe_tool_call(name: str, tool_input: dict) -> str:
    """Human-readable label for a tool call — shown in light/muted text, like
    Claude Code's own operation indicators. Prefixed with the raw tool name
    (in backticks) so it's visible which underlying function actually ran —
    useful for anyone reading the trace to learn how the agent works, not
    just what it did. Deliberately never includes the raw SQL or full query
    text (chat stays free of implementation detail)."""
    if name == "query_sql":
        desc = "Queried your transactions"
    elif name == "search_text":
        desc = f'Searched for "{tool_input.get("query", "")}"'
    elif name == "generate_chart":
        desc = f'Built a {tool_input.get("chart_type", "")} chart'
    elif name == "categorize_transaction":
        descriptor = str(tool_input.get("descriptor", ""))[:40]
        desc = f'Looked up "{descriptor}"'
    elif name == "web_search":
        desc = f'Searched the web for "{tool_input.get("query", "")}"'
    elif name == "code_execution":
        desc = "Ran code"
    else:
        desc = "Ran"
    return f"`{name}` — {desc}"


def _summarize_tool_result(name: str, result: dict) -> str | None:
    """One-line outcome for a client-side tool call (query_sql, search_text,
    categorize_transaction), shown muted right after its invocation label —
    the "results" half of the trace, same spirit as op_result for
    web_search/code_execution. Returns None when nothing needs its own
    caption: generate_chart's success case renders the chart itself as the
    visible confirmation, so a redundant text line would just be noise."""
    if "error" in result:
        return f"Failed: {result['error']}"
    if name == "query_sql":
        n = result.get("count", 0)
        note = " (truncated)" if result.get("truncated") else ""
        return f"Found {n} row{'s' if n != 1 else ''}{note}"
    if name == "search_text":
        n = result.get("count", 0)
        return f"Found {n} match{'es' if n != 1 else ''}"
    if name == "categorize_transaction":
        merchant = result.get("merchant", "")
        category = result.get("category", "")
        return f"Categorized as {merchant} ({category})" if merchant else None
    return None


def _append(role: str, type_: str, **fields) -> None:
    # Defensive, not just show()'s job: process_uploads() is called from the
    # sidebar in streamlit_app.py, which renders before show() does — relying
    # on show() having already initialized this on some earlier run is fragile.
    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []
    msgs = st.session_state["display_messages"]
    msgs.append({"role": role, "type": type_, **fields})
    if len(msgs) > _MAX_DISPLAY_MESSAGES:
        del msgs[: len(msgs) - _MAX_DISPLAY_MESSAGES]


def _render_search_results(results: list[dict]) -> None:
    """A result card like the one on claude.ai: title + source domain per
    row, collapsed by default. Shared between the live-streaming render and
    replaying a saved conversation, so both look identical."""
    n = len(results)
    with st.expander(f"🌐 {n} result{'s' if n != 1 else ''}", expanded=False):
        for r in results:
            domain = urlparse(r["url"]).netloc
            st.markdown(f"[{r['title']}]({r['url']})")
            st.caption(domain)


def _render_message(msg: dict, key: str) -> None:
    if msg["type"] == "text":
        st.markdown(msg["content"])
    elif msg["type"] in ("op", "note"):
        st.caption(msg["content"])
    elif msg["type"] == "thinking":
        # Replayed from a saved conversation — the live view shows elapsed
        # thinking time in the title (see show()'s thinking_stop handling),
        # but that duration isn't persisted, so replay just shows a plain
        # label rather than a fabricated or missing number.
        with st.expander("🕐 Thinking", expanded=False):
            st.caption(msg["content"])
    elif msg["type"] == "search_results":
        _render_search_results(json.loads(msg["content"]))
    elif msg["type"] == "chart":
        st.plotly_chart(pio.from_json(msg["content"]))
    elif msg["type"] == "image":
        st.image(msg["content"])
    elif msg["type"] == "file":
        st.download_button(
            f"Download {msg['filename']}", data=msg["content"], file_name=msg["filename"], key=key
        )


def process_uploads(files: list, api_key: str) -> None:
    """Parse, reconcile and load attached statements, then post a summary as the
    assistant. Called from the sidebar's "Upload statements" expander in
    streamlit_app.py — uploading lives outside the chat input itself now."""
    user_key = get_user_key()
    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        try:
            ledger = get_ledger(user_key)
            fts = get_fts(user_key)
        except Exception as e:
            st.error(friendly_api_error(e))
            _append("assistant", "text", content=f"Couldn't open the transaction database: {e}")
            return

        with st.spinner(f"Reading {len(files)} file{'s' if len(files) != 1 else ''}…"):
            # Size/duplicate checks stay sequential — they need the ledger
            # connection, which isn't safe to share across threads. What's
            # left after this filter is genuinely independent, CPU-bound
            # parsing work (see _MAX_PARSE_WORKERS), safe to run concurrently
            # since it touches no Streamlit or database state.
            errors, duplicates, to_parse = [], [], []
            for f in files:
                if f.size > _MAX_FILE_MB * 1024 * 1024:
                    errors.append(f"{f.name}: file is over {_MAX_FILE_MB}MB, skipped")
                    continue
                data = f.read()
                content_hash = hashlib.sha1(data).hexdigest()[:16]
                if ledger.is_duplicate_upload(content_hash):
                    duplicates.append(f.name)
                    continue
                to_parse.append((f.name, data, content_hash))

            def _parse_one(name: str, data: bytes) -> list[dict]:
                raw_rows = parse_file_bytes(data, name)
                return [t.to_dict() for r in raw_rows if (t := extract(r)) is not None]

            all_transactions, accepted = [], []
            if to_parse:
                workers = min(_MAX_PARSE_WORKERS, len(to_parse))
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_parse_one, name, data) for name, data, _ in to_parse]
                    # Iterated in submission order (not as_completed) so the
                    # summary message lists files in the order they were
                    # uploaded, not whichever happened to finish parsing
                    # first — the parsing itself still runs concurrently in
                    # the background regardless of this iteration order.
                    for (name, _, content_hash), future in zip(to_parse, futures):
                        try:
                            all_transactions.extend(future.result())
                            accepted.append((name, content_hash))
                        except Exception as e:
                            errors.append(f"{name}: {e}")

            try:
                reconciled = reconcile(all_transactions)
                n_new = ledger.upsert(reconciled)
                fts.index(reconciled)
            except Exception as e:
                st.error(friendly_api_error(e))
                _append("assistant", "text", content=f"Couldn't save the parsed transactions: {e}")
                return

            # Record each accepted file so a future re-upload of the same bytes
            # is rejected as a duplicate, and build the month/year coverage label.
            ranges = []
            for filename, content_hash in accepted:
                info = ledger.source_file_summary(filename)
                ledger.mark_uploaded(
                    content_hash, filename, info["min_date"], info["max_date"], info["row_count"]
                )
                ranges.append((info["min_date"], info["max_date"]))
            coverage = _coverage_label(ranges)

            enrich_note = ""
            if api_key and reconciled:
                try:
                    stats = enrich_batch(ledger, api_key)
                    log_usage("enrichment", DEFAULT_MODEL, stats.get("usage"), user_key)
                    enrich_note = (
                        f" Categorized {stats['rules'] + stats['web']} merchants"
                        + (f", {stats['ambiguous']} left uncategorized." if stats["ambiguous"] else ".")
                    )
                except Exception as e:
                    enrich_note = f"\n\n_Merchant enrichment failed: {friendly_api_error(e)}_"

            insight_text = None
            if api_key and reconciled:
                try:
                    result = insight(ledger, api_key)
                    if result:
                        insight_text, usage = result
                        log_usage("upload_insight", DEFAULT_MODEL, usage, user_key)
                except Exception:
                    pass

        if accepted:
            names = ", ".join(name for name, _ in accepted)
            summary = f"Loaded **{n_new}** new transaction(s) from {names}.{enrich_note}"
            if coverage:
                summary += f"\n\nCoverage: **{coverage}**"
        else:
            summary = "No new transactions loaded."
        if duplicates:
            summary += "\n\n" + "\n".join(f"↩️ {name}: already uploaded — skipped" for name in duplicates)
        if errors:
            summary += "\n\n" + "\n".join(f"⚠️ Couldn't read {e}" for e in errors)
        if insight_text:
            summary += f"\n\n{insight_text}"
        elif not api_key:
            summary += "\n\nAdd your Anthropic API key in the sidebar to enable categorization and chat."

        summary = _normalize_markdown(summary)
        st.markdown(summary)
        _append("assistant", "text", content=summary)


def _conversation_title(display_messages: list[dict]) -> str:
    """First user message, truncated — same idea as how chat UIs (including
    Claude Code) label a saved session, so the recent-conversations list is
    scannable without opening each one."""
    for m in display_messages:
        if m["role"] == "user" and m["type"] == "text":
            text = m["content"].strip()
            return text[:_TITLE_MAX_LEN] + "…" if len(text) > _TITLE_MAX_LEN else text
    return "New conversation"


def recent_conversations() -> list[dict]:
    """Top 5 (by default) most recently active saved conversations for the
    sidebar's "Recent conversations" list."""
    return list_recent_conversations(get_user_key())


def search_conversations(query: str) -> list[dict]:
    """Conversations with a message matching `query`, best match first."""
    return _search_conversations(get_user_key(), query)


def open_conversation(conversation_id: str) -> None:
    """Load a saved conversation back into the active session — both the
    resumable API-format history (so chatting can continue exactly where it
    left off) and the rendered transcript (so it looks the same as it did
    originally). No-ops if the conversation doesn't exist or belongs to a
    different user."""
    user_key = get_user_key()
    saved = load_conversation(user_key, conversation_id)
    if saved is None:
        return
    display_msgs = load_conversation_messages(user_key, conversation_id)
    st.session_state["history"] = saved["history"]
    st.session_state["display_messages"] = display_msgs
    st.session_state["conversation_id"] = conversation_id
    st.session_state["_persisted_msg_count"] = len(display_msgs)


def list_uploads() -> list[dict]:
    """This user's uploaded statements, for the sidebar's per-file delete
    list — see streamlit_app.py's "Delete my data" expander."""
    return get_ledger(get_user_key()).list_uploads()


def delete_upload(content_hash: str) -> int:
    """Remove exactly one prior upload's transactions (not the whole
    account) and rebuild the search index to match. Returns the number of
    transactions removed."""
    user_key = get_user_key()
    n = get_ledger(user_key).delete_upload(content_hash)
    get_fts(user_key).index()
    return n


def rename_upload(content_hash: str, new_filename: str) -> int:
    """Rename one prior upload — no FTS reindex needed, since source_file
    isn't one of the indexed columns (description/merchant/category only).
    Returns the number of transactions updated."""
    return get_ledger(get_user_key()).rename_upload(content_hash, new_filename)


def show() -> None:
    user_key = get_user_key()

    st.header("💰 Penny")

    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []

    api_key = st.session_state.get("api_key", "")

    for i, msg in enumerate(st.session_state["display_messages"]):
        avatar = _USER_AVATAR if msg["role"] == "user" else _ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            _render_message(msg, key=f"history_dl_{i}")

    text = st.chat_input("Ask about your spending…")
    if not text:
        if not st.session_state["display_messages"] and tx_count(user_key) == 0:
            st.info(
                "👋 Attach a bank or credit card statement using **📎 Upload statements** "
                "in the sidebar to get started, then ask me anything about your spending."
            )
        return

    label = _normalize_markdown(text)
    _append("user", "text", content=label)
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(label)

    if not api_key:
        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            st.error("Add your Anthropic API key in the sidebar to chat.")
        return

    if tx_count(user_key) == 0:
        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            st.warning("Attach a statement first (📎 Upload statements in the sidebar) so I have something to answer questions about.")
        return

    history = get_history()
    ledger = get_ledger(user_key)
    fts = get_fts(user_key)

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        text_placeholder = st.empty()
        text_placeholder.caption("Thinking…")
        accumulated_text = ""
        thinking_placeholder = None
        accumulated_thinking = ""
        thinking_started_at = 0.0
        got_output = False
        thinking_cleared = False

        def _flush_text(final: bool = False) -> None:
            # Persist and lock in the current text segment into its existing
            # placeholder — does NOT create the next one. That has to happen
            # in the caller, and only *after* it renders its own content
            # (tool-call label / chart / image / file): st.empty() reserves
            # its DOM position the instant it's called, before it has
            # anything to show, so creating it here — before that content
            # renders — would reserve a slot ahead of it. The next segment
            # would then land in that earlier slot once filled, visually
            # jumping ahead of the tool call that actually preceded it.
            #
            # We can't know a segment is the *actual* final answer until the
            # turn ends with no further tool call after it — so every segment
            # streams muted (like the tool-call labels), and only the one
            # still open when "done" fires gets upgraded to full-weight text.
            # Everything earlier — the model's own "I'll check that..."
            # narration between tool calls — stays muted permanently, the
            # same visual language Claude Code uses for in-progress commentary
            # versus a finished deliverable.
            nonlocal accumulated_text
            if accumulated_text:
                normalized = _normalize_markdown(accumulated_text)
                if final:
                    text_placeholder.markdown(normalized)
                    _append("assistant", "text", content=normalized)
                else:
                    text_placeholder.caption(normalized)
                    _append("assistant", "note", content=normalized)
            else:
                text_placeholder.empty()
            accumulated_text = ""

        def _new_placeholder() -> None:
            nonlocal text_placeholder
            text_placeholder = st.empty()

        try:
            for event in run_turn(text, history, ledger, fts, api_key):
                if not thinking_cleared and event["type"] != "usage":
                    thinking_cleared = True
                    if event["type"] != "text":
                        text_placeholder.empty()

                if event["type"] == "usage":
                    log_usage(event["source"], event["model"], event["usage"], user_key)

                elif event["type"] == "text":
                    # Muted while streaming — we don't yet know if this segment
                    # is commentary or the final answer; _flush_text() decides
                    # that once the turn's outcome (more tool calls, or done)
                    # is known, and upgrades it then if so.
                    accumulated_text += event["text"]
                    text_placeholder.caption(_normalize_markdown(accumulated_text) + "▌")
                    got_output = True

                elif event["type"] == "thinking_start":
                    # Same live-typing treatment as "text" below, streamed
                    # token-by-token — the expander stays open (expanded=True)
                    # while reasoning is actively arriving, the same way
                    # Claude Code/claude.ai show a "Thought for Ns" block live
                    # rather than only after the fact. Duration is timed here
                    # (wall-clock, client-side) rather than asked of the API —
                    # no extra request, just local bookkeeping around events
                    # already being streamed.
                    _flush_text()
                    thinking_placeholder = st.empty()
                    accumulated_thinking = ""
                    thinking_started_at = time.monotonic()
                    with thinking_placeholder:
                        with st.expander("🕐 Thinking…", expanded=True):
                            st.caption("▌")
                    got_output = True

                elif event["type"] == "thinking_delta":
                    accumulated_thinking += event["text"]
                    with thinking_placeholder:
                        with st.expander("🕐 Thinking…", expanded=True):
                            st.caption(_normalize_markdown(accumulated_thinking) + "▌")
                    got_output = True

                elif event["type"] == "thinking_stop":
                    # Collapse now that reasoning is complete — available on
                    # demand without dominating the chat the way a
                    # multi-paragraph thinking trace would inline.
                    thinking_text = _normalize_markdown(accumulated_thinking)
                    elapsed = max(1, round(time.monotonic() - thinking_started_at))
                    with thinking_placeholder:
                        with st.expander(f"🕐 Thought for {elapsed}s", expanded=False):
                            st.caption(thinking_text)
                    _append("assistant", "thinking", content=thinking_text)
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "web_search_results":
                    # The API already returns full result data (title/url per
                    # hit) as part of the same web_search call — this renders
                    # more of what was already paid for and received, not an
                    # extra lookup.
                    _flush_text()
                    _render_search_results(event["results"])
                    _append("assistant", "search_results", content=json.dumps(event["results"]))
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "tool_call":
                    _flush_text()
                    label = _describe_tool_call(event["name"], event["input"])
                    st.caption(label)
                    _append("assistant", "op", content=label)
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "tool_result":
                    # The "results" half of the trace for client-side tools
                    # (query_sql, search_text, categorize_transaction) — the
                    # same idea as op_result below, just for the tools this
                    # app executes itself rather than Anthropic's server-side
                    # ones. generate_chart intentionally returns no caption
                    # here (see _summarize_tool_result) since the chart event
                    # right after this one already renders the confirmation.
                    summary = _summarize_tool_result(event["name"], event["result"])
                    if summary:
                        _flush_text()
                        label = f"`{event['name']}` — {summary}"
                        st.caption(label)
                        _append("assistant", "op", content=label)
                        _new_placeholder()
                        got_output = True

                elif event["type"] == "op_result":
                    # Completes a web_search/code_execution call announced by
                    # the "tool_call" branch above — e.g. "Found 5 results"
                    # or "Code ran successfully". Previously this outcome had
                    # no event at all for web_search, and no *handler* for
                    # code_execution even though loop.py already yielded one.
                    _flush_text()
                    label = f"`{event['name']}` — {event['text']}"
                    st.caption(label)
                    _append("assistant", "op", content=label)
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "chart":
                    _flush_text()
                    fig = pio.from_json(event["chart_json"])
                    st.plotly_chart(fig)
                    _append("assistant", "chart", content=event["chart_json"])
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "image":
                    _flush_text()
                    st.image(event["image_bytes"])
                    _append("assistant", "image", content=event["image_bytes"])
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "file":
                    _flush_text()
                    st.download_button(
                        f"Download {event['filename']}",
                        data=event["file_bytes"],
                        file_name=event["filename"],
                        key=f"live_dl_{len(st.session_state['display_messages'])}",
                    )
                    _append(
                        "assistant", "file", filename=event["filename"], content=event["file_bytes"]
                    )
                    _new_placeholder()
                    got_output = True

                elif event["type"] == "done":
                    _flush_text(final=True)
                    if not got_output:
                        st.warning("Penny didn't return a response — try rephrasing your question.")
        except Exception as e:
            _flush_text()
            st.error(friendly_api_error(e))

    # Keep history for next turn (already updated by run_turn)
    st.session_state["history"] = history

    # Persist this turn — the resumable history plus any display messages new
    # since the last save — so it survives a refresh and shows up in "Recent
    # conversations"/search. The user's own message guarantees at least one
    # new persistable entry, so this always has something to save here.
    display_msgs = st.session_state["display_messages"]
    already_persisted = st.session_state.get("_persisted_msg_count", 0)
    new_msgs = [m for m in display_msgs[already_persisted:] if m["type"] in _PERSISTABLE_TYPES]
    persist_conversation(
        user_key, get_conversation_id(), _conversation_title(display_msgs), history, new_msgs
    )
    st.session_state["_persisted_msg_count"] = len(display_msgs)
