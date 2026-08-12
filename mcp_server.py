"""Penny as an MCP server: query and search your own exported transaction data
from any MCP client (Claude Desktop, Claude Code, etc.), not just the Streamlit app.

This is read-only over a snapshot you explicitly exported — nothing here talks
to a running Streamlit session, and no data is written back. Matches Penny's
existing "nothing persists automatically" privacy model: you choose to export,
you choose to point this at that file.

Run standalone:
    python mcp_server.py path/to/penny_data.parquet
    # or: set PENNY_DATA_PATH and omit the argument

Get a Parquet file via the "Export transactions (Parquet)" button in the
app's sidebar "Session data" section. See README.md for wiring this into
claude_desktop_config.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.mcpserver import MCPServer

from penny.storage.fts import FTSIndex
from penny.storage.ledger import Ledger

mcp = MCPServer(
    name="penny-finance",
    instructions=(
        "Query and search a personal finance transaction ledger exported from the "
        "Penny app. Schema: transactions(id, date, description, merchant, category, "
        "amount, account, account_last4, source_file, page_num, is_transfer, "
        "is_refund, is_internal). amount sign convention: positive = expense/debit, "
        "negative = credit/refund/income. Exclude internal transfers from spending "
        "totals unless asked about them: add WHERE is_internal = false."
    ),
)

_ledger = Ledger(":memory:")
_fts = FTSIndex(":memory:")


def _load(data_path: str) -> None:
    _ledger.import_parquet(Path(data_path))
    # limit set well above any realistic transaction count so a large
    # snapshot doesn't get silently truncated when building the search index.
    rows = _ledger.query(
        "SELECT id, description, merchant, category FROM transactions", limit=1_000_000
    )
    _fts.index(rows)


@mcp.tool(description="Run a read-only SQL SELECT query against the transactions table.")
def query_transactions(sql: str) -> list[dict[str, Any]]:
    """Run a DuckDB SELECT query against the loaded transaction snapshot.

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
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PENNY_DATA_PATH")
    if not data_path:
        print("Usage: python mcp_server.py <path-to-exported-parquet>", file=sys.stderr)
        print("(or set the PENNY_DATA_PATH environment variable)", file=sys.stderr)
        sys.exit(1)
    _load(data_path)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
