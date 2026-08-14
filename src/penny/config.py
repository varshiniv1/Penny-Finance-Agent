"""App-wide configuration, loaded from environment variables — .env locally,
Streamlit Cloud secrets in production (secrets that need to stay secret, like
MOTHERDUCK_TOKEN and PENNY_ADMIN_TOKEN, still check st.secrets first in
session.py; PENNY_MODEL isn't sensitive, so a plain env var is enough and
keeps this module framework-independent).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# The model every Claude call in this app uses — main chat loop, sub-agent
# delegation, merchant enrichment, spending insights. Previously a literal
# string duplicated across six different files: changing models meant
# hunting down every occurrence instead of touching one place. Falls back to
# Haiku (the cheap, cost-conscious default) if PENNY_MODEL isn't set — set it
# in .env locally or Streamlit Cloud secrets/env vars in production to
# override, e.g. PENNY_MODEL=claude-sonnet-4-6.
DEFAULT_MODEL: str = os.environ.get("PENNY_MODEL", "claude-haiku-4-5-20251001")
