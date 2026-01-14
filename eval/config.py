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
REAL_ATOMS_PATH = DATA_DIR / "real_atoms.json"


LOG_DIR = ROOT / "eval" / "logs"


GEN_PROVIDER = "openai"      # or "openai"
GEN_MODEL_NAME = "gpt-4o"   # or "gpt-5-mini"

# Concurrency
LLM_MAX_WORKERS = 4

EXPERIMENT_NAME = "MemBench_Evaluation_01"


