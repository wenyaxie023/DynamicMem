from pathlib import Path
from typing import Any

from baseline_prediction.common.provider_config import load_repo_dotenv, setup_provider_env

try:
    from baseline_prediction.common.llm_client import LLMClient as _SharedLLMClient
    _LLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - allows tests to patch eval.client.LLMClient without provider deps
    _SharedLLMClient = object  # type: ignore[assignment]
    _LLM_IMPORT_ERROR = exc


class LLMClient(_SharedLLMClient):
    def __init__(self, provider: str = "openai", model_name: str = "gpt-5-mini", **kwargs: Any):
        if _LLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "eval.client.LLMClient is unavailable because generation.common.llm_client "
                f"failed to import: {_LLM_IMPORT_ERROR}"
            )
        repo_root = Path(__file__).resolve().parents[1]
        load_repo_dotenv(repo_root)
        if provider in {"openai", "azure"}:
            setup_provider_env(provider, repo_root=repo_root, require_api_key=True)
        super().__init__(provider=provider, model_name=model_name, **kwargs)


__all__ = ["LLMClient"]
