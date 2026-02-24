import os
from pathlib import Path
from typing import Optional, Tuple


def load_repo_dotenv(repo_root: Path) -> None:
    """Best-effort .env loader for repository-level configuration."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(repo_root / ".env", override=False)


def resolve_openai_compatible_credentials(
    provider: str,
    *,
    require_api_key: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve API key and base URL for OpenAI-compatible providers.

    For provider='azure':
      prefers AZURE_OPENAI_* then OPENAI_*
    For provider='openai' (or others):
      prefers OPENAI_* then AZURE_OPENAI_*
    """
    p = (provider or "").strip().lower()
    if p == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")

    if require_api_key and not api_key:
        raise RuntimeError(
            "Missing API key. Set AZURE_OPENAI_API_KEY for provider=azure, "
            "or OPENAI_API_KEY for provider=openai."
        )
    return api_key, base_url


def apply_openai_compat_env(api_key: Optional[str], base_url: Optional[str]) -> None:
    """
    Populate OPENAI_* env names for downstream libraries expecting OpenAI naming.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url


def setup_provider_env(
    provider: str,
    *,
    repo_root: Optional[Path] = None,
    require_api_key: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience one-shot setup for baseline scripts.

    - Optionally load repository .env
    - Resolve provider-specific OpenAI-compatible credentials
    - Populate OPENAI_* env names for downstream libraries
    """
    if repo_root is not None:
        load_repo_dotenv(repo_root)
    api_key, base_url = resolve_openai_compatible_credentials(
        provider,
        require_api_key=require_api_key,
    )
    apply_openai_compat_env(api_key, base_url)
    return api_key, base_url
