"""Runtime-tunable config surface for vendored SimpleMem."""

from typing import Any, Dict


OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-5-mini"
LLM_MAX_WORKERS = 1
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0
LLM_TOP_K = None
ENABLE_THINKING = False
USE_STREAMING = False
USE_JSON_FORMAT = False

EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
RETRIEVER_BATCH_SIZE = 64

LANCEDB_PATH = ""
MEMORY_TABLE_NAME = "memory_entries"

WINDOW_SIZE = 5
OVERLAP_SIZE = 1
ENABLE_PARALLEL_PROCESSING = False
MAX_PARALLEL_WORKERS = 1

SEMANTIC_TOP_K = 10
KEYWORD_TOP_K = 10
STRUCTURED_TOP_K = 10
ENABLE_PLANNING = True
ENABLE_REFLECTION = True
MAX_REFLECTION_ROUNDS = 2
ENABLE_PARALLEL_RETRIEVAL = False
MAX_RETRIEVAL_WORKERS = 1


def apply_runtime_config(overrides: Dict[str, Any]) -> None:
    if not isinstance(overrides, dict):
        return
    globals_ns = globals()
    for key, value in overrides.items():
        if key in globals_ns and value is not None:
            globals_ns[key] = value
