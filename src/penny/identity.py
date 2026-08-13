"""Shared, Streamlit-independent identity helpers.

Used by both the web app (penny.ui.session) and the standalone MCP server
(mcp_server.py), so the hashing formula and MotherDuck token wiring can never
drift out of sync between the two entry points — they're the same account
system, just reached two different ways.
"""
from __future__ import annotations

import hashlib
import os


def hash_api_key(api_key: str) -> str:
    """The real storage/partition key: SHA-256 of the full API key, truncated
    to 16 hex chars.

    Not the raw last-4 characters — an Anthropic key's trailing 4 characters
    come from a small alphabet (~62 options/char, ~14.7M combinations), which
    collides at realistic user counts. For a finance app, a collision isn't
    cosmetic: it would silently merge two different people's transactions
    into one account. The hash is effectively collision-proof for this
    purpose; callers that want a human-recognizable label still use the raw
    last 4 characters for *display* only (see session.get_display_label()).
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def ensure_motherduck_token(token: str) -> None:
    """Set DuckDB's own `motherduck_token` env var (read automatically by the
    extension on connect) — idempotent, safe to call before every connection."""
    if token and os.environ.get("motherduck_token") != token:
        os.environ["motherduck_token"] = token
