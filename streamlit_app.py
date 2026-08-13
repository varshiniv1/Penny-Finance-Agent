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

# Streamlit's default chat text is smaller than comfortable reading size — bump
# font-size/line-height only, leaving color/theme entirely to .streamlit/config.toml.
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from penny.ui import chat_page, observability_page
from penny.ui.session import (
    delete_all_user_data, get_display_label, get_ledger, get_user_key, tx_count,
)

_FILE_TYPES = ["pdf", "csv", "jpg", "jpeg", "png", "tiff", "tif", "bmp"]

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

    with st.expander("💾 Export data"):
        st.caption(
            "Your transactions are saved automatically and are here next time you "
            "sign in with the same API key — no need to restore a backup. Export a "
            "Parquet snapshot if you want an offline copy, or to query your data "
            "from an MCP client like Claude Desktop (see README)."
        )
        if tx_count(user_key) > 0:
            if st.button("Export transactions (Parquet)"):
                import tempfile, pathlib
                from datetime import datetime, timezone

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                try:
                    get_ledger(user_key).export_parquet(tmp_path)
                    # Timestamped so re-exporting later the same day (or a
                    # different session) never silently overwrites an earlier
                    # download sitting in the same Downloads folder.
                    export_name = (
                        "penny_session_"
                        + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
                        + ".parquet"
                    )
                    st.download_button(
                        f"Download {export_name}",
                        data=tmp_path.read_bytes(),
                        file_name=export_name,
                        mime="application/octet-stream",
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

        st.divider()
        st.caption("Permanently remove everything saved under this API key.")
        if st.button("🗑️ Delete all my saved data"):
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
