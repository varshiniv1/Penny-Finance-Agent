"""Streamlit page: chat with Penny agent."""
from __future__ import annotations

import json

import plotly.io as pio
import streamlit as st

from penny.agent.loop import run_turn
from penny.ui.session import friendly_api_error, get_ledger, get_fts, get_history, log_usage, tx_count


def show() -> None:
    st.header("Chat with Penny")

    if tx_count() == 0:
        st.warning("No transactions loaded yet. Go to **Upload Statements** first.")
        return

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar to use the chat.")
        return

    st.caption(f"{tx_count()} transactions in session.")

    # Render conversation history
    history = get_history()
    for i, msg in enumerate(st.session_state.get("display_messages", [])):
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "sql":
                st.code(msg["content"], language="sql")
            elif msg["type"] == "chart":
                fig = pio.from_json(msg["content"])
                st.plotly_chart(fig, use_container_width=True)
            elif msg["type"] == "image":
                st.image(msg["content"])
            elif msg["type"] == "file":
                st.download_button(
                    f"Download {msg['filename']}",
                    data=msg["content"],
                    file_name=msg["filename"],
                    key=f"history_dl_{i}",
                )

    # Input
    user_input = st.chat_input("Ask about your spending…")
    if not user_input:
        return

    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []

    st.session_state["display_messages"].append(
        {"role": "user", "type": "text", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    ledger = get_ledger()
    fts = get_fts()

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        accumulated_text = ""
        got_output = False

        def _flush_text() -> None:
            # Persist and lock in the current text segment, then start a
            # fresh placeholder so the next segment renders below whatever
            # comes next (SQL/chart), instead of overwriting this one.
            nonlocal text_placeholder, accumulated_text
            if accumulated_text:
                text_placeholder.markdown(accumulated_text)
                st.session_state["display_messages"].append(
                    {"role": "assistant", "type": "text", "content": accumulated_text}
                )
            accumulated_text = ""
            text_placeholder = st.empty()

        try:
            for event in run_turn(user_input, history, ledger, fts, api_key):
                if event["type"] == "usage":
                    log_usage(event["source"], event["model"], event["usage"])

                elif event["type"] == "text":
                    accumulated_text += event["text"]
                    text_placeholder.markdown(accumulated_text + "▌")
                    got_output = True

                elif event["type"] == "sql":
                    _flush_text()
                    st.code(event["sql"], language="sql")
                    st.session_state["display_messages"].append(
                        {"role": "assistant", "type": "sql", "content": event["sql"]}
                    )
                    got_output = True

                elif event["type"] == "chart":
                    _flush_text()
                    fig = pio.from_json(event["chart_json"])
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state["display_messages"].append(
                        {"role": "assistant", "type": "chart", "content": event["chart_json"]}
                    )
                    got_output = True

                elif event["type"] == "image":
                    _flush_text()
                    st.image(event["image_bytes"])
                    st.session_state["display_messages"].append(
                        {"role": "assistant", "type": "image", "content": event["image_bytes"]}
                    )
                    got_output = True

                elif event["type"] == "file":
                    _flush_text()
                    st.download_button(
                        f"Download {event['filename']}",
                        data=event["file_bytes"],
                        file_name=event["filename"],
                        key=f"live_dl_{len(st.session_state['display_messages'])}",
                    )
                    st.session_state["display_messages"].append(
                        {
                            "role": "assistant",
                            "type": "file",
                            "filename": event["filename"],
                            "content": event["file_bytes"],
                        }
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
