try:
    from .llm_client import LLMClient
except Exception:  # pragma: no cover - allows importing sibling modules in minimal test envs
    LLMClient = None  # type: ignore[assignment]

__all__ = ["LLMClient"]
