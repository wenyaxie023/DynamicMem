from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


# Data paths
DATA_DIR = ROOT / "eval" / "data"
HISTORY_DIR = ROOT / "eval" / "data" / "history"

BG_PATH = DATA_DIR / "schema.json"
QA_PATH = DATA_DIR / "QA.json"
GEN_OUTPUT_PATH = DATA_DIR / "QA_all_answered.json"
EVAL_INPUT_PATH = DATA_DIR / "rag_results.json"
EVAL_OUTPUT_PATH = DATA_DIR / "eval_results_rag.json"
EVAL_TABLE_PATH = DATA_DIR / "eval_table_rag.csv"
REAL_ATOMS_PATH = DATA_DIR / "atoms.json"


LOG_DIR = ROOT / "eval" / "logs"


GEN_PROVIDER = "gemini"      # or "openai"
GEN_MODEL_NAME = "gemini-3-flash-preview"   # or "gpt-5-mini"

# Concurrency
LLM_MAX_WORKERS = 5

EXPERIMENT_NAME = "MemBench_Evaluation_01"


