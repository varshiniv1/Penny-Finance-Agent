"""Penny — Personal Finance Agent.

Entry point for Streamlit Community Cloud.
Run locally:  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="Penny — Personal Finance Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Match Claude Code's compact chat text size (14px) rather than Streamlit's
# default — leaving color/theme entirely to .streamlit/config.toml.
#
# Inline code (the `tool_name` backticks in operation labels, e.g. "`query_sql`
# — Queried your transactions") also gets its own color here — Streamlit's
# default inline-code green clashes with the warm cream theme. A muted violet
# chip instead, tying it to the app's own primaryColor rather than an
# arbitrary syntax-highlighting color.
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        font-size: 0.875rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessageContent"] code {
        color: #5A50E0;
        background-color: rgba(108, 99, 255, 0.12);
        padding: 0.08em 0.4em;
        border-radius: 4px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from datetime import datetime

from penny.ui import chat_page, observability_page
from penny.ui.session import delete_all_user_data, get_display_label, get_user_key, reset_session

_FILE_TYPES = ["pdf", "csv", "jpg", "jpeg", "png", "tiff", "tif", "bmp"]


def _format_timestamp(iso_str: str) -> str:
    """updated_at (a UTC ISO-8601 string, see Ledger.save_conversation) as a
    short local-agnostic label for the recent-conversations list."""
    try:
        return datetime.fromisoformat(iso_str).strftime("%b %d, %I:%M %p")
    except (TypeError, ValueError):
        return ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Penny")
    st.caption("Your personal finance agent")

    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for the chat agent and merchant enrichment, and used to "
        "identify your saved data (a one-way hash of it, never the key itself).",
    )
    if api_key:
        st.session_state["api_key"] = api_key
    elif "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    user_key = get_user_key()
    display_label = get_display_label()
    if display_label:
        st.caption(f"Signed in as ••••{display_label}")

    st.divider()

    # Top-level, always visible — the single "start over" entry point (was
    # previously a small button buried in the main header, easy to miss).
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        reset_session(user_key)
        st.rerun()

    with st.expander("🕑 Recent conversations"):
        query = st.text_input(
            "Search your conversations",
            key="conv_search",
            label_visibility="collapsed",
            placeholder="🔍 Search your conversations…",
        )
        conversations = chat_page.search_conversations(query) if query else chat_page.recent_conversations()
        if not conversations:
            st.caption("No matches." if query else "No saved conversations yet.")
        for c in conversations:
            label = c["title"] or "Untitled"
            if st.button(label, key=f"open_conv_{c['conversation_id']}", use_container_width=True):
                chat_page.open_conversation(c["conversation_id"])
                st.rerun()
            st.caption(_format_timestamp(c["updated_at"]))

    with st.expander("📎 Upload statements"):
        st.caption("Supports PDF (text or scanned), CSV, and image statements.")
        uploads = st.file_uploader(
            "Add bank or credit card statements",
            type=_FILE_TYPES,
            accept_multiple_files=True,
            key="statement_uploader",
        )
        if uploads and st.button("Process statements", type="primary"):
            chat_page.process_uploads(uploads, api_key)
            st.rerun()

    with st.expander("📂 Manage uploads"):
        uploads = chat_page.list_uploads()
        if uploads:
            st.caption("Rename or remove a statement:")
            for u in uploads:
                date_range = (
                    f"{u['min_date']} – {u['max_date']}" if u["min_date"] else "no dated rows"
                )
                st.markdown(f"**{u['filename']}**  \n{u['row_count']} rows, {date_range}")
                name_col, rename_col, delete_col = st.columns([3, 1, 1])
                new_name = name_col.text_input(
                    "New name",
                    value=u["filename"],
                    key=f"rename_input_{u['content_hash']}",
                    label_visibility="collapsed",
                )
                if rename_col.button("Rename", key=f"rename_btn_{u['content_hash']}"):
                    cleaned = new_name.strip()
                    if not cleaned:
                        st.error("Name can't be empty.")
                    elif cleaned == u["filename"]:
                        st.info("That's already the name.")
                    else:
                        chat_page.rename_upload(u["content_hash"], cleaned)
                        st.success(f"Renamed to {cleaned}.")
                        st.rerun()
                if delete_col.button("Delete", key=f"del_upload_{u['content_hash']}"):
                    n = chat_page.delete_upload(u["content_hash"])
                    st.success(f"Removed {n} transaction(s) from {u['filename']}.")
                    st.rerun()
                st.divider()

        st.caption("Or permanently remove everything saved under this API key.")
        if st.button("Delete all my saved data"):
            delete_all_user_data(user_key)
            st.success("Deleted. Your saved transactions are gone.")
            st.rerun()

    with st.expander("📊 Your usage"):
        observability_page.show(user_key)

    st.divider()
    st.caption(
        "Raw statement files are discarded right after parsing — only the extracted "
        "transactions are saved, tied to a one-way hash of your API key so they're "
        "here again next time. Delete them any time above."
    )

# ── Main content ─────────────────────────────────────────────────────────────
chat_page.show()
