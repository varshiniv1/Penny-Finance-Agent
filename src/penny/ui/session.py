"""Streamlit session-state helpers: initialise and access the in-memory DB."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from penny.storage.ledger import Ledger
from penny.storage.fts import FTSIndex


def get_ledger() -> Ledger:
    if "ledger" not in st.session_state:
        st.session_state["ledger"] = Ledger(":memory:")
    return st.session_state["ledger"]


def get_fts() -> FTSIndex:
    if "fts" not in st.session_state:
        st.session_state["fts"] = FTSIndex(":memory:")
    return st.session_state["fts"]


def get_history() -> list[dict]:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    return st.session_state["history"]


def tx_count() -> int:
    return get_ledger().count()


def log_usage(source: str, model: str, usage) -> None:
    """Record token usage from an Anthropic API response for the Observability tab."""
    if usage is None:
        return
    st.session_state.setdefault("usage_log", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    })


def get_usage_log() -> list[dict]:
    return st.session_state.get("usage_log", [])


def reset_session() -> None:
    """Clear the conversation, ledger, and search index — start over without
    a full browser refresh (which would also lose the API key entry)."""
    for key in ("ledger", "fts", "history", "display_messages"):
        st.session_state.pop(key, None)


def friendly_api_error(e: Exception) -> str:
    """Turn a raised Anthropic SDK exception into a message safe to show users."""
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return "Your Anthropic API key looks invalid — check it in the sidebar."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "Your Anthropic API key doesn't have permission for this request."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Anthropic API — wait a moment and try again."
    if isinstance(e, anthropic.BadRequestError):
        if "credit balance" in str(e).lower():
            return (
                "Your Anthropic account is out of credit balance. Add credits at "
                "console.anthropic.com/settings/billing, then try again."
            )
        return f"Anthropic API rejected the request: {e}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API — check your connection and try again."
    if isinstance(e, anthropic.APIStatusError):
        return f"Anthropic API error ({e.status_code}): {e.message}"
    return f"Unexpected error: {e}"
