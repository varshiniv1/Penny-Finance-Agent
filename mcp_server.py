"""Penny as an MCP server: query and search your OWN live transaction data
from any MCP client (Claude Desktop, Claude Code, etc.), not just the
Streamlit app itself.

Connects directly to the same persistent MotherDuck database the web app
uses, scoped to your account the exact same way the web app scopes it —
by hashing your Anthropic API key (see penny/identity.py, shared by both
entry points so the two can never drift apart). No export step, no
staleness: whatever's in the Streamlit app right now is what this server
sees too, live, because it's the same underlying data.

Run standalone:
    python mcp_server.py <your-anthropic-api-key>
    # or: set PENNY_API_KEY and omit the argument

Also needs a MotherDuck token — the same one the web app uses — as either
MOTHERDUCK_TOKEN or motherduck_token in the environment.

Prefer the environment variable over the command-line argument where you
can: a key passed as a CLI arg can end up in shell history or a process
list, which the API key deserves better than.

See README.md for wiring this into claude_desktop_config.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.mcpserver import MCPServer

from penny.identity import ensure_motherduck_token, hash_api_key
from penny.storage.fts import FTSIndex
from penny.storage.ledger import Ledger

mcp = MCPServer(
    name="penny-finance",
    instructions=(
        "Query and search a personal finance transaction ledger — the same "
        "live data as the Penny web app, not a snapshot. Schema: "
        "transactions(id, date, description, merchant, category, amount, "
        "account, account_last4, source_file, page_num, is_transfer, "
        "is_refund, is_internal). amount sign convention: positive = "
        "expense/debit, negative = credit/refund/income. Exclude transfers "
        "from spending totals unless asked about them: add "
        "WHERE is_internal = false AND category != 'Transfer' — is_internal "
        "alone often isn't enough, since it's only set when a matching "
        "transaction was found on another uploaded account."
    ),
)

_ledger: Ledger | None = None
_fts: FTSIndex | None = None


def _connect(api_key: str) -> None:
    global _ledger, _fts
    token = os.environ.get("MOTHERDUCK_TOKEN") or os.environ.get("motherduck_token", "")
    if not token:
        print(
            "MOTHERDUCK_TOKEN (or motherduck_token) must be set in the environment — "
            "the same token the web app uses.",
            file=sys.stderr,
        )
        sys.exit(1)
    ensure_motherduck_token(token)

    user_id = hash_api_key(api_key)
    _ledger = Ledger("md:penny", user_id)
    _fts = FTSIndex(_ledger._con, user_id)


@mcp.tool(description="Run a read-only SQL SELECT query against the transactions table.")
def query_transactions(sql: str) -> list[dict[str, Any]]:
    """Run a DuckDB SELECT query against your live transaction data.

    Args:
        sql: A valid DuckDB SELECT statement over the `transactions` table.
    """
    return _ledger.query(sql)


@mcp.tool(description="Full-text keyword search over transaction descriptions and merchant names.")
def search_transactions(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """Search transactions by keyword (fuzzy merchant lookup, finding transactions by name).

    Args:
        query: Keywords to search for.
        top_k: Max results to return.
    """
    return _fts.search(query, top_k=top_k)


def main() -> None:
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PENNY_API_KEY")
    if not api_key:
        print("Usage: python mcp_server.py <your-anthropic-api-key>", file=sys.stderr)
        print("(or set the PENNY_API_KEY environment variable — preferred)", file=sys.stderr)
        sys.exit(1)
    _connect(api_key)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
