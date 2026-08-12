# Penny-Finance-Agent

A privacy-first personal finance agent: attach bank/credit-card statements
(PDF, scanned/OCR, CSV) directly in the chat box, then ask questions about
them in the same thread — or query the same data from your own MCP client.
Everything runs in-session — raw statements never leave your device.

## Local setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Paste your Anthropic API key into the sidebar each session — it's never
stored. No `.env` file is required for the Streamlit app itself; only the
standalone `mcp_server.py` reads an environment variable (`PENNY_DATA_PATH`,
see below). An optional `PENNY_ADMIN_TOKEN` (env var or Streamlit secret)
unlocks the hidden Observability page at `?admin=<token>`.

## Connect via MCP

Penny's transaction ledger is also available as an MCP server, so you can
query your own exported data from Claude Desktop, Claude Code, or any other
MCP client — not just Penny's own chat page.

1. **Export a snapshot.** In the app, attach and process your statements in
   chat, then open the sidebar's **💾 Session data** section and click
   **Export transactions (Parquet)**. Save the file somewhere on your
   machine, e.g. `~/penny_data.parquet`.
2. **Install the server's dependency** (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```
3. **Point a client at it.** For Claude Desktop, add this to
   `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "penny-finance": {
         "command": "python",
         "args": ["/absolute/path/to/mcp_server.py", "/absolute/path/to/penny_data.parquet"]
       }
     }
   }
   ```
   Or run it directly to sanity-check it starts:
   ```bash
   python mcp_server.py /absolute/path/to/penny_data.parquet
   ```

It exposes two read-only tools — `query_transactions` (SQL over the
snapshot) and `search_transactions` (full-text search) — the same
underlying `Ledger`/`FTSIndex` code the Streamlit app itself uses. It's
read-only over a file you explicitly exported: nothing here talks to a live
Streamlit session, and nothing gets written back.