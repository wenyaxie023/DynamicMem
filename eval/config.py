from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


# Data paths
DATA_DIR = ROOT / "eval" / "data"
HISTORY_DIR = ROOT / "eval" / "data" / "history"

BG_PATH = DATA_DIR / "dynamic_profiles_conflict_resolved.json"
QA_PATH = DATA_DIR / "QA.json"
GEN_OUTPUT_PATH = DATA_DIR / "QA.json"
EVAL_INPUT_PATH = DATA_DIR / "QA.json"
EVAL_OUTPUT_PATH = DATA_DIR / "eval_results.json"
EVAL_TABLE_PATH = DATA_DIR / "eval_table.csv"


LOG_DIR = ROOT / "eval" / "logs"


GEN_PROVIDER = "gemini"      # or "openai"
GEN_MODEL_NAME = "gemini-2.5-flash"   # or "gpt-5-mini"


