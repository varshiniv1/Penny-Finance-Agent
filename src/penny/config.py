from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # src/penny -> Penny/

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
DB_PATH: Path = ROOT / os.environ.get("DB_PATH", "data/penny.duckdb")
FTS_PATH: Path = ROOT / os.environ.get("FTS_PATH", "data/fts.db")
STATEMENTS_DIR: Path = ROOT / os.environ.get("STATEMENTS_DIR", "data/statements")

CHART_DIR: Path = ROOT / "data" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
