"""Tests for session.py's identity helpers — get_user_key()'s hashing behavior
in particular, since it's the actual security boundary for per-user storage."""
import streamlit as st

from penny.ui.session import get_display_label, get_user_key


def _reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_same_api_key_produces_same_user_key():
    _reset_session_state()
    st.session_state["api_key"] = "sk-ant-abc123"
    first = get_user_key()
    second = get_user_key()
    assert first == second


def test_different_api_keys_produce_different_user_keys():
    _reset_session_state()
    st.session_state["api_key"] = "sk-ant-abc123"
    key_a = get_user_key()

    _reset_session_state()
    st.session_state["api_key"] = "sk-ant-xyz789"
    key_b = get_user_key()

    assert key_a != key_b


def test_user_key_is_not_the_raw_api_key_or_a_trivial_transform():
    _reset_session_state()
    st.session_state["api_key"] = "sk-ant-abc123"
    key = get_user_key()
    assert "sk-ant-abc123" not in key
    assert key != "sk-ant-abc123"[-4:]


def test_empty_api_key_falls_back_to_ephemeral_key():
    _reset_session_state()
    st.session_state["api_key"] = ""
    key = get_user_key()
    assert key  # non-empty
    # Stable within the same "session" (same session_state dict) even with no key.
    assert get_user_key() == key


def test_display_label_is_last_four_characters():
    _reset_session_state()
    st.session_state["api_key"] = "sk-ant-abc123"
    assert get_display_label() == "c123"


def test_display_label_empty_when_no_api_key():
    _reset_session_state()
    st.session_state["api_key"] = ""
    assert get_display_label() == ""
