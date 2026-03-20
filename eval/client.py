from pathlib import Path
from typing import Any

from generation.common.llm_client import LLMClient as _SharedLLMClient
from generation.common.provider_config import load_repo_dotenv, setup_provider_env


class LLMClient(_SharedLLMClient):
    def __init__(self, provider: str = "openai", model_name: str = "gpt-5-mini", **kwargs: Any):
        repo_root = Path(__file__).resolve().parents[1]
        load_repo_dotenv(repo_root)
        if provider in {"openai", "azure"}:
            setup_provider_env(provider, repo_root=repo_root, require_api_key=True)
        super().__init__(provider=provider, model_name=model_name, **kwargs)


__all__ = ["LLMClient"]
