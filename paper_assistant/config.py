import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENREVIEW_USERNAME = os.getenv("OPENREVIEW_USERNAME", "")
OPENREVIEW_PASSWORD = os.getenv("OPENREVIEW_PASSWORD", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
