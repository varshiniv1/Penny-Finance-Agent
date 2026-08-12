"""Streamlit page: a single chat surface — attach statements and ask questions in one thread."""
from __future__ import annotations

import re

import plotly.io as pio
import streamlit as st

from penny.agent.insights import insight
from penny.agent.loop import run_turn
from penny.ingest.enrich import enrich_batch
from penny.ingest.extractor import extract
from penny.ingest.parser import parse_file_bytes
from penny.ingest.reconcile import reconcile
from penny.ui.session import (
    friendly_api_error, get_ledger, get_fts, get_history, log_usage, reset_session, tx_count,
)

_FILE_TYPES = ["pdf", "csv", "jpg", "jpeg", "png", "tiff", "tif", "bmp"]
_MAX_FILE_MB = 20
# Cap how many chat turns are kept rendered/in memory — display_messages can
# hold raw image/chart/file bytes per turn, which would otherwise grow
# unbounded over a long session.
_MAX_DISPLAY_MESSAGES = 200


_DOLLAR_RE = re.compile(r"(?<!\\)\$")


def _escape_markdown_math(text: str) -> str:
    """st.markdown renders paired `$...$` as inline LaTeX/KaTeX math — a
    sentence with two dollar amounts like "$11,411.67 ... $718,000" gets torn
    apart and rendered as a math expression instead of plain text. Escaping
    literal `$` keeps dollar amounts in Claude's replies as plain text."""
    return _DOLLAR_RE.sub(lambda m: "\\$", text)


def _append(role: str, type_: str, **fields) -> None:
    msgs = st.session_state["display_messages"]
    msgs.append({"role": role, "type": type_, **fields})
    if len(msgs) > _MAX_DISPLAY_MESSAGES:
        del msgs[: len(msgs) - _MAX_DISPLAY_MESSAGES]


def _render_message(msg: dict, key: str) -> None:
    if msg["type"] == "text":
        st.markdown(msg["content"])
    elif msg["type"] == "sql":
        st.code(msg["content"], language="sql")
    elif msg["type"] == "chart":
        st.plotly_chart(pio.from_json(msg["content"]), use_container_width=True)
    elif msg["type"] == "image":
        st.image(msg["content"])
    elif msg["type"] == "file":
        st.download_button(
            f"Download {msg['filename']}", data=msg["content"], file_name=msg["filename"], key=key
        )


def _process_uploads(files: list, api_key: str) -> None:
    """Parse, reconcile and load attached statements, then post a summary as the assistant."""
    with st.chat_message("assistant"):
        try:
            ledger = get_ledger()
            fts = get_fts()
        except Exception as e:
            st.error(friendly_api_error(e))
            _append("assistant", "text", content=f"Couldn't open the transaction database: {e}")
            return

        with st.spinner(f"Reading {len(files)} file{'s' if len(files) != 1 else ''}…"):
            all_transactions, errors = [], []
            for f in files:
                if f.size > _MAX_FILE_MB * 1024 * 1024:
                    errors.append(f"{f.name}: file is over {_MAX_FILE_MB}MB, skipped")
                    continue
                try:
                    raw_rows = parse_file_bytes(f.read(), f.name)
                    transactions = [t for r in raw_rows if (t := extract(r)) is not None]
                    all_transactions.extend(t.to_dict() for t in transactions)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

            try:
                reconciled = reconcile(all_transactions)
                n_new = ledger.upsert(reconciled)
                fts.index(reconciled)
            except Exception as e:
                st.error(friendly_api_error(e))
                _append("assistant", "text", content=f"Couldn't save the parsed transactions: {e}")
                return

            enrich_note = ""
            if api_key and reconciled:
                try:
                    stats = enrich_batch(ledger, api_key)
                    log_usage("enrichment", "claude-haiku-4-5-20251001", stats.get("usage"))
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
                        log_usage("upload_insight", "claude-haiku-4-5-20251001", usage)
                except Exception:
                    pass

        names = ", ".join(f.name for f in files)
        summary = f"Loaded **{n_new}** new transaction(s) from {names}.{enrich_note}"
        if errors:
            summary += "\n\n" + "\n".join(f"⚠️ Couldn't read {e}" for e in errors)
        if insight_text:
            summary += f"\n\n{insight_text}"
        elif not api_key:
            summary += "\n\nAdd your Anthropic API key in the sidebar to enable categorization and chat."

        summary = _escape_markdown_math(summary)
        st.markdown(summary)
        _append("assistant", "text", content=summary)


def show() -> None:
    header_col, reset_col = st.columns([6, 1])
    header_col.header("💰 Penny")
    if st.session_state.get("display_messages") and reset_col.button("🗑️ New chat", help="Clear the conversation and loaded transactions"):
        reset_session()
        st.rerun()

    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []

    api_key = st.session_state.get("api_key", "")

    for i, msg in enumerate(st.session_state["display_messages"]):
        with st.chat_message(msg["role"]):
            _render_message(msg, key=f"history_dl_{i}")

    chat_value = st.chat_input(
        "Ask about your spending, or attach a statement…",
        accept_file="multiple",
        file_type=_FILE_TYPES,
    )
    if not chat_value:
        if not st.session_state["display_messages"] and tx_count() == 0:
            st.info(
                "👋 Attach a bank or credit card statement (PDF, CSV, or image) using the "
                "📎 icon below to get started, then ask me anything about your spending."
            )
        return

    text = chat_value.text
    files = chat_value.files

    if files or text:
        label = text
        if files:
            attachment_line = "📎 " + ", ".join(f.name for f in files)
            label = f"{attachment_line}\n\n{text}" if text else attachment_line
        label = _escape_markdown_math(label)
        _append("user", "text", content=label)
        with st.chat_message("user"):
            st.markdown(label)

    if files:
        _process_uploads(files, api_key)

    if not text:
        return

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Add your Anthropic API key in the sidebar to chat.")
        return

    if tx_count() == 0:
        with st.chat_message("assistant"):
            st.warning("Attach a statement first (📎 below) so I have something to answer questions about.")
        return

    history = get_history()
    ledger = get_ledger()
    fts = get_fts()

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        text_placeholder.markdown("_Thinking…_")
        accumulated_text = ""
        got_output = False
        thinking_cleared = False

        def _flush_text() -> None:
            # Persist and lock in the current text segment, then start a
            # fresh placeholder so the next segment renders below whatever
            # comes next (SQL/chart), instead of overwriting this one.
            nonlocal text_placeholder, accumulated_text
            if accumulated_text:
                text_placeholder.markdown(accumulated_text)
                _append("assistant", "text", content=accumulated_text)
            else:
                text_placeholder.empty()
            accumulated_text = ""
            text_placeholder = st.empty()

        try:
            for event in run_turn(text, history, ledger, fts, api_key):
                if not thinking_cleared and event["type"] != "usage":
                    thinking_cleared = True
                    if event["type"] != "text":
                        text_placeholder.empty()

                if event["type"] == "usage":
                    log_usage(event["source"], event["model"], event["usage"])

                elif event["type"] == "text":
                    accumulated_text += _escape_markdown_math(event["text"])
                    text_placeholder.markdown(accumulated_text + "▌")
                    got_output = True

                elif event["type"] == "sql":
                    _flush_text()
                    st.code(event["sql"], language="sql")
                    _append("assistant", "sql", content=event["sql"])
                    got_output = True

                elif event["type"] == "chart":
                    _flush_text()
                    fig = pio.from_json(event["chart_json"])
                    st.plotly_chart(fig, use_container_width=True)
                    _append("assistant", "chart", content=event["chart_json"])
                    got_output = True

                elif event["type"] == "image":
                    _flush_text()
                    st.image(event["image_bytes"])
                    _append("assistant", "image", content=event["image_bytes"])
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
                    got_output = True

                elif event["type"] == "done":
                    _flush_text()
                    if not got_output:
                        st.warning("Penny didn't return a response — try rephrasing your question.")
        except Exception as e:
            _flush_text()
            st.error(friendly_api_error(e))

    # Keep history for next turn (already updated by run_turn)
    st.session_state["history"] = history
