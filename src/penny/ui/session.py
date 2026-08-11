"""Streamlit session-state helpers: initialise and access the in-memory DB."""
from __future__ import annotations

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
