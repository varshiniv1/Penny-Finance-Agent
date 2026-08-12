"""Penny — Personal Finance Agent.

Entry point for Streamlit Community Cloud.
Run locally:  streamlit run streamlit_app.py
"""
import hmac
import os
import sys
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st


def _is_admin() -> bool:
    """Observability is owner-only: visible only with ?admin=<PENNY_ADMIN_TOKEN>
    matching a secret configured out-of-band (Streamlit secrets or an env var),
    never checked into the repo. No secret configured -> tab never appears."""
    try:
        secret = st.secrets.get("PENNY_ADMIN_TOKEN", "")
    except Exception:
        secret = ""
    secret = secret or os.environ.get("PENNY_ADMIN_TOKEN", "")
    if not secret:
        return False
    given = st.query_params.get("admin", "")
    return hmac.compare_digest(given, secret)

st.set_page_config(
    page_title="Penny — Personal Finance Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

from penny.ui import chat_page, observability_page
from penny.ui.session import get_fts, get_ledger, tx_count

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Penny")
    st.caption("Your personal finance agent")

    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for the chat agent and merchant enrichment. Never stored.",
    )
    if api_key:
        st.session_state["api_key"] = api_key
    elif "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    st.divider()

    with st.expander("💾 Session data"):
        st.caption("Your data lives only in this browser session. Export it to restore later.")
        if tx_count() > 0:
            if st.button("Export transactions (Parquet)"):
                import tempfile, pathlib
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = pathlib.Path(tmp.name)
                try:
                    get_ledger().export_parquet(tmp_path)
                    st.download_button(
                        "Download penny_data.parquet",
                        data=tmp_path.read_bytes(),
                        file_name="penny_data.parquet",
                        mime="application/octet-stream",
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
        restore = st.file_uploader("Restore from Parquet backup", type=["parquet"], key="restore")
        if restore and st.button("Restore"):
            import tempfile, pathlib
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp.write(restore.read())
                tmp_path = pathlib.Path(tmp.name)
            try:
                ledger = get_ledger()
                n = ledger.import_parquet(tmp_path)
                # import_parquet only touches the ledger — the FTS index needs
                # the same rows explicitly, or restored transactions become
                # invisible to search_text without any error to surface it.
                # limit set well above any realistic transaction count so a
                # large backup doesn't get silently truncated on reindex.
                rows = ledger.query(
                    "SELECT id, description, merchant, category FROM transactions",
                    limit=1_000_000,
                )
                get_fts().index(rows)
                st.success(f"Restored. Total transactions: {n}")
            finally:
                tmp_path.unlink(missing_ok=True)

    if _is_admin():
        st.divider()
        page = st.radio(
            "Navigate", ["Chat with Penny", "Observability"], label_visibility="collapsed"
        )
    else:
        page = "Chat with Penny"

    st.divider()
    st.caption(
        "All processing happens in your browser session. "
        "Raw PDFs never leave your device. "
        "Only query results are sent to the Claude API."
    )

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Observability":
    observability_page.show()
else:
    chat_page.show()
