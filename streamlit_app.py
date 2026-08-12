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

from penny.ui import upload_page, chat_page, dashboard_page, observability_page

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

    nav_options = ["Upload Statements", "Chat with Penny", "Dashboard"]
    if _is_admin():
        nav_options.append("Observability")

    page = st.radio(
        "Navigate",
        nav_options,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "All processing happens in your browser session. "
        "Raw PDFs never leave your device. "
        "Only query results are sent to the Claude API."
    )

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Upload Statements":
    upload_page.show()
elif page == "Chat with Penny":
    chat_page.show()
elif page == "Dashboard":
    dashboard_page.show()
elif page == "Observability":
    observability_page.show()
