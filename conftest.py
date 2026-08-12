"""Make src/ importable for pytest, same as streamlit_app.py / mcp_server.py do
at runtime — so `pytest` works without an editable install of the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
