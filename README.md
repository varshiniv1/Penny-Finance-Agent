# Penny-Finance-Agent

A personal finance agent: attach bank/credit-card statements (PDF,
scanned/OCR, CSV) from the sidebar, then ask questions about them in a chat
thread — or query the same data from your own MCP client. There's no
login/signup — you're identified by a one-way hash of the Anthropic API key
you paste in, so the same key brings your data back on your next visit.

Raw statement files are discarded immediately after parsing — only the
extracted transactions are kept. Those persist (on MotherDuck, a hosted
DuckDB database) so they're there next time you sign in with the same key,
until you delete them yourself from the sidebar's **💾 Export data**
section. Usage/cost data works the same way: a **📊 Your usage** section
shows only your own token usage — nobody else's, no admin access required.

## Local setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Paste your Anthropic API key into the sidebar each session — it's never
stored (only a SHA-256 hash of it is, as your account identifier). For
persistence to work locally, set `motherduck_token` in `.streamlit/secrets.toml`
or as an environment variable — get one free at motherduck.com (no credit
card required). Without it, the app still works, just without persistence
(the same in-memory-only behavior as before).

## Connect via MCP

Penny's transaction ledger is also available as an MCP server, so you can
query your own **live** data from Claude Desktop, Claude Code, or any other
MCP client — not just Penny's own chat page. It connects to the same
MotherDuck database the web app uses, scoped to your account the same way
(a hash of your API key) — no export step, and nothing goes stale: whatever
you see in the app is what the MCP client sees too.

1. **Install the server's dependency** (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```
2. **Point a client at it.** For Claude Desktop, add this to
   `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "penny-finance": {
         "command": "python",
         "args": ["/absolute/path/to/mcp_server.py"],
         "env": {
           "PENNY_API_KEY": "sk-ant-...",
           "MOTHERDUCK_TOKEN": "..."
         }
       }
     }
   }
   ```
   Or run it directly to sanity-check it starts:
   ```bash
   MOTHERDUCK_TOKEN=... PENNY_API_KEY=sk-ant-... python mcp_server.py
   ```

It exposes two read-only tools — `query_transactions` (SQL over your live
transactions) and `search_transactions` (full-text search) — the same
underlying `Ledger`/`FTSIndex` code the Streamlit app itself uses, and the
same identity-hashing logic (`penny/identity.py`), so it's always scoped to
your own data, never anyone else's.

If you'd rather keep an offline copy instead — for backup, or to hand data
to something that isn't MCP-aware — the sidebar's **💾 Export data** section
still produces a Parquet snapshot you can download.