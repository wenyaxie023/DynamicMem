from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


# Data paths
DATA_DIR = ROOT / "data"
HISTORY_DIR = ROOT / "data" / "history"

ATOM_LIST_PATH = DATA_DIR / "atom_list.json"
ATOM_BASE_SCHEMA_PATH = DATA_DIR / "atom_base_schema.json"
ORIGINAL_QA_OUTPUT_PATH = DATA_DIR / "QA.json"

BG_PATH = DATA_DIR / "schema.json"

GEN_INPUT_PATH = DATA_DIR / "QA.json"
GEN_OUTPUT_PATH = DATA_DIR / "QA.json"

EVAL_INPUT_PATH = DATA_DIR / "QA.json"
EVAL_OUTPUT_PATH = DATA_DIR / "eval_results.json"
EVAL_TABLE_PATH = DATA_DIR / "eval_table.csv"


LOG_DIR = ROOT / "logs"


GEN_PROVIDER = "gemini"      # or "openai"
GEN_MODEL_NAME = "gemini-3-flash-preview"   # or "gpt-5-mini"

# Concurrency
LLM_MAX_WORKERS = 5
LLM_JUDGE_PROVIDERS = ["gpt"]  # options: "gpt", "gemini"
LLM_JUDGE_GPT_PROVIDER = "openai"  # "openai" or "azure"
LLM_JUDGE_GPT_MODEL = "gpt-5-mini"
LLM_JUDGE_GEMINI_MODEL = "gemini-3-flash-preview"
LLM_JUDGE_MAX_RETRIES = 3

EXPERIMENT_NAME = "MemBench_Evaluation_01"
