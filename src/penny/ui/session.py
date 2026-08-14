"""Streamlit session-state helpers: user identity, and access to the per-user DB.

Identity without login/signup: a user is identified by a SHA-256 hash of their
Anthropic API key (never the raw key itself, and never just its last 4 characters —
see get_user_key()'s docstring for why). The hash is the real partition key used to
scope every query; the last 4 characters are shown in the UI purely as a
human-recognizable label ("Signed in as ••••ab12").

Storage falls back to :memory: (today's fully ephemeral behavior) whenever either no
MotherDuck token is configured, or no API key has been entered yet — so the app still
works, just without persistence, exactly as before this change.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import streamlit as st

from penny.identity import ensure_motherduck_token, hash_api_key
from penny.storage.fts import FTSIndex
from penny.storage.ledger import Ledger

_MOTHERDUCK_DB = "md:penny"


def _motherduck_token() -> str:
    try:
        token = st.secrets.get("MOTHERDUCK_TOKEN", "")
    except Exception:
        token = ""
    return token or os.environ.get("MOTHERDUCK_TOKEN", "") or os.environ.get("motherduck_token", "")


def _motherduck_available() -> bool:
    """True once a token is configured — wires it into the env var DuckDB's
    MotherDuck extension reads automatically."""
    token = _motherduck_token()
    if not token:
        return False
    ensure_motherduck_token(token)
    return True


def get_user_key() -> str:
    """The current user's real storage/partition key — see
    penny.identity.hash_api_key() for what this actually is and why.

    No API key entered yet: falls back to a random id stable for this browser
    session only (matches today's :memory:, non-persistent behavior).
    """
    api_key = st.session_state.get("api_key", "")
    if api_key:
        return hash_api_key(api_key)
    if "_ephemeral_user_key" not in st.session_state:
        st.session_state["_ephemeral_user_key"] = f"local-{uuid.uuid4().hex[:12]}"
    return st.session_state["_ephemeral_user_key"]


def get_display_label() -> str:
    api_key = st.session_state.get("api_key", "")
    return api_key[-4:] if api_key else ""


def get_ledger(user_key: str) -> Ledger:
    cache_key = f"ledger_{user_key}"
    if cache_key not in st.session_state:
        conn_str = _MOTHERDUCK_DB if _motherduck_available() and st.session_state.get("api_key") else ":memory:"
        st.session_state[cache_key] = Ledger(conn_str, user_key)
    return st.session_state[cache_key]


def get_fts(user_key: str) -> FTSIndex:
    cache_key = f"fts_{user_key}"
    if cache_key not in st.session_state:
        ledger = get_ledger(user_key)
        st.session_state[cache_key] = FTSIndex(ledger._con, user_key)
    return st.session_state[cache_key]


def get_history() -> list[dict]:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    return st.session_state["history"]


def get_conversation_id() -> str:
    """The active conversation's storage key — created lazily on the first
    turn that actually completes (see persist_conversation's call site in
    chat_page.show()), not on every page load."""
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = uuid.uuid4().hex
    return st.session_state["conversation_id"]


def persist_conversation(
    user_key: str, conversation_id: str, title: str, history: list[dict], new_messages: list[dict]
) -> None:
    """Save this turn's resumable state (history) and any display messages
    new since the last save, then refresh the search index so the turn is
    immediately findable."""
    ledger = get_ledger(user_key)
    ledger.save_conversation(conversation_id, title, history)
    ledger.add_conversation_messages(conversation_id, new_messages)
    get_fts(user_key).index_conversations()


def list_recent_conversations(user_key: str, limit: int = 5) -> list[dict]:
    return get_ledger(user_key).recent_conversations(limit)


def search_conversations(user_key: str, query: str) -> list[dict]:
    return get_fts(user_key).search_conversations(query)


def load_conversation(user_key: str, conversation_id: str) -> dict | None:
    return get_ledger(user_key).load_conversation(conversation_id)


def load_conversation_messages(user_key: str, conversation_id: str) -> list[dict]:
    return get_ledger(user_key).load_conversation_messages(conversation_id)


def tx_count(user_key: str) -> int:
    return get_ledger(user_key).count()


def log_usage(source: str, model: str, usage, user_key: str) -> None:
    """Record token usage from an Anthropic API response, persisted in this
    user's own usage_log table — durable across restarts (MotherDuck), or
    naturally session-scoped only when running on the :memory: fallback.
    No admin gate: a user only ever reads their own rows (see get_usage_log)."""
    if usage is None:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    get_ledger(user_key).log_usage_event(entry)


def get_usage_log(user_key: str) -> list[dict]:
    return get_ledger(user_key).get_usage_log()


def reset_session(user_key: str) -> None:
    """Clear the conversation, ledger, and search index for this user — start
    over without a full browser refresh (which would also lose the API key).
    Also drops conversation_id, so the next completed turn starts a new saved
    conversation rather than continuing to append to the old one."""
    for key in (
        f"ledger_{user_key}", f"fts_{user_key}", "history", "display_messages",
        "conversation_id", "_persisted_msg_count",
    ):
        st.session_state.pop(key, None)


def delete_all_user_data(user_key: str) -> None:
    """Wipe this user's persisted transactions/upload history, then drop the
    cached connections so the next access opens a fresh, empty one."""
    get_ledger(user_key).delete_all_user_data()
    for key in (f"ledger_{user_key}", f"fts_{user_key}"):
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
