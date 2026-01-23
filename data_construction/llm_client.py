import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import requests
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
try:
    from google import genai as genai_client
    from google.genai import errors as genai_errors
except ImportError:
    genai_client = None
    genai_errors = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    GOOGLE = "google"
    OPENAI = "openai"
    AIMLAPI = "aimlapi"
    CUSTOM = "custom"


# Default base URLs for each provider
DEFAULT_BASE_URLS = {
    LLMProvider.GOOGLE: None,  # Google uses SDK, not REST
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.AIMLAPI: "https://api.aimlapi.com/v1",
}

# Environment variable names for each provider
ENV_VAR_NAMES = {
    LLMProvider.GOOGLE: {
        "api_key": "GOOGLE_API_KEY",
        "base_url": None,  # Google doesn't use base_url
    },
    LLMProvider.OPENAI: {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
    },
    LLMProvider.AIMLAPI: {
        "api_key": "AIMLAPI_API_KEY",
        "base_url": "AIMLAPI_BASE_URL",
    },
    LLMProvider.CUSTOM: {
        "api_key": "CUSTOM_API_KEY",
        "base_url": "CUSTOM_BASE_URL",
    },
}


class LLMGenerationError(RuntimeError):
    """Raised when the LLM response cannot be parsed as JSON."""


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` style fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence.
        lines = lines[1:]
        # Drop closing fence if present.
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON parsing that tolerates markdown fences."""
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _extract_usage_metadata(usage: Optional[Any]) -> Dict[str, int]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
        "completion_tokens": getattr(usage, "candidates_token_count", 0),
        "total_tokens": getattr(usage, "total_token_count", 0),
    }


def _extract_openai_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Extract usage from OpenAI-compatible API response."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


@dataclass
class LLMResult:
    data: Dict[str, Any]
    usage: Dict[str, int]
    prompt: str
    raw_text: str


# Model name mappings for AIMLAPI (requires provider prefix)
AIMLAPI_MODEL_MAPPING = {
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-2.5-flash-preview-05-20": "google/gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
}


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (OpenAI, AIMLAPI, etc.)."""

    def __init__(
        self,
        model_name: str,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name  # Keep original for output dirs, etc.
        self.provider = provider

        # Normalize model name for API requests (e.g., AIMLAPI needs google/ prefix)
        self._api_model_name = self._normalize_model_name(model_name, provider)

        # Get API key from param or env var
        env_vars = ENV_VAR_NAMES.get(provider, ENV_VAR_NAMES[LLMProvider.CUSTOM])
        self.api_key = api_key or os.getenv(env_vars["api_key"])
        if not self.api_key:
            raise ValueError(
                f"API key required. Provide via `api_key` param or {env_vars['api_key']} env var."
            )

        # Get base URL from param, env var, or default
        if base_url:
            self.base_url = base_url
        elif env_vars["base_url"]:
            self.base_url = os.getenv(env_vars["base_url"]) or DEFAULT_BASE_URLS.get(provider, "")
        else:
            self.base_url = DEFAULT_BASE_URLS.get(provider, "")

        if not self.base_url:
            raise ValueError(
                f"Base URL required. Provide via `base_url` param or {env_vars.get('base_url', 'env var')}."
            )

        # Ensure base_url doesn't end with /
        self.base_url = self.base_url.rstrip("/")

    def _normalize_model_name(self, model_name: str, provider: LLMProvider) -> str:
        """Normalize model name for the specific provider's API."""
        if provider == LLMProvider.AIMLAPI:
            # AIMLAPI requires provider prefix (e.g., google/gemini-3-flash-preview)
            return AIMLAPI_MODEL_MAPPING.get(model_name, model_name)
        return model_name

    def generate_json(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 5,
        retry_wait_seconds: int = 120,
    ) -> LLMResult:
        """Generate JSON response from OpenAI-compatible API.

        Args:
            prompt: The user prompt.
            response_schema: Optional JSON Schema to enforce structured output.
                When provided, uses OpenAI's Structured Outputs feature.
            system_prompt: Optional system prompt.
            max_retries: Number of retries on transient errors.
            retry_wait_seconds: Wait time between retries.
        """
        last_exception = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._api_model_name,
            "messages": messages,
        }

        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": response_schema,
                }
            }
        else:
            # Basic JSON mode (no schema enforcement)
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300,
                )

                if response.status_code == 429:
                    raise LLMGenerationError(f"Rate limited: {response.text}")
                elif response.status_code >= 500:
                    raise LLMGenerationError(f"Server error: {response.text}")
                elif response.status_code != 200:
                    raise LLMGenerationError(f"API error ({response.status_code}): {response.text}")

                result = response.json()
                raw_text = result["choices"][0]["message"]["content"]

                try:
                    data = _parse_json_from_text(raw_text)
                except json.JSONDecodeError as exc:
                    raise LLMGenerationError(
                        f"Failed to decode JSON from model response: {exc}\nRaw response:\n{raw_text}"
                    ) from exc

                usage = _extract_openai_usage(result.get("usage"))
                return LLMResult(data=data, usage=usage, prompt=prompt, raw_text=raw_text)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    print(f"[LLM] Connection error (attempt {attempt + 1}/{max_retries}): {exc}")
                    print(f"[LLM] Waiting {retry_wait_seconds} seconds before retry...")
                    time.sleep(retry_wait_seconds)
                else:
                    print(f"[LLM] Max retries ({max_retries}) exceeded.")
            except LLMGenerationError as exc:
                if "Rate limited" in str(exc) or "Server error" in str(exc):
                    last_exception = exc
                    if attempt < max_retries - 1:
                        print(f"[LLM] {exc} (attempt {attempt + 1}/{max_retries})")
                        print(f"[LLM] Waiting {retry_wait_seconds} seconds before retry...")
                        time.sleep(retry_wait_seconds)
                    else:
                        print(f"[LLM] Max retries ({max_retries}) exceeded.")
                else:
                    raise

        raise LLMGenerationError(
            f"Failed after {max_retries} retries: {last_exception}"
        ) from last_exception


class GeminiJSONClient:
    """Minimal wrapper around the Gemini SDK that always returns JSON payloads."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GeminiJSONClient requires an API key via `api_key` or GOOGLE_API_KEY env var."
            )
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
            or {
                "response_mime_type": "application/json",
            },
        )
        self._structured_client = (
            genai_client.Client(api_key=self.api_key) if genai_client else None
        )

    def generate_json(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        retry_wait_seconds: int = 120,
    ) -> LLMResult:
        last_exception = None

        for attempt in range(max_retries):
            try:
                if response_schema and self._structured_client is not None:
                    response = self._structured_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_json_schema": response_schema,
                        },
                    )
                else:
                    response = self.model.generate_content(prompt, generation_config={
                        "response_mime_type": "application/json",
                    })

                # Handle empty response (no valid Part returned)
                try:
                    raw_text = (response.text or "").strip()
                except ValueError as e:
                    # response.text raises ValueError when no valid Part is returned
                    raise google_exceptions.InternalServerError(
                        f"Empty response from model: {e}"
                    ) from e

                # Treat empty response as retryable error
                if not raw_text:
                    raise google_exceptions.InternalServerError(
                        "Empty response from model (no content returned)"
                    )

                try:
                    payload = _parse_json_from_text(raw_text)
                except json.JSONDecodeError as exc:
                    # If JSON parsing fails, it might be a transient issue - make it retryable
                    raise google_exceptions.InternalServerError(
                        f"Failed to decode JSON from model response: {exc}\nRaw response:\n{raw_text}"
                    ) from exc
                usage = _extract_usage_metadata(getattr(response, "usage_metadata", None))
                return LLMResult(data=payload, usage=usage, prompt=prompt, raw_text=raw_text)

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
                google_exceptions.DeadlineExceeded,
                google_exceptions.GoogleAPICallError,  # Base class for API errors
            ) as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    print(f"[LLM] Rate limit or server error (attempt {attempt + 1}/{max_retries}): {exc}")
                    print(f"[LLM] Waiting {retry_wait_seconds} seconds before retry...")
                    time.sleep(retry_wait_seconds)
                else:
                    print(f"[LLM] Max retries ({max_retries}) exceeded.")
            except Exception as exc:
                # Catch other exceptions that might be retryable (e.g., from google.genai SDK)
                exc_str = str(exc)
                is_retryable = any(keyword in exc_str for keyword in [
                    "503", "UNAVAILABLE", "overloaded", "429", "rate limit",
                    "500", "502", "504", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"
                ])
                if is_retryable:
                    last_exception = exc
                    if attempt < max_retries - 1:
                        print(f"[LLM] Retryable error (attempt {attempt + 1}/{max_retries}): {exc}")
                        print(f"[LLM] Waiting {retry_wait_seconds} seconds before retry...")
                        time.sleep(retry_wait_seconds)
                    else:
                        print(f"[LLM] Max retries ({max_retries}) exceeded.")
                else:
                    # Non-retryable error, raise immediately
                    raise

        raise LLMGenerationError(
            f"Failed after {max_retries} retries due to rate limiting or server errors: {last_exception}"
        ) from last_exception


def create_client(
    provider: str | LLMProvider,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> GeminiJSONClient | OpenAICompatibleClient:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: Provider name ("google", "openai", "aimlapi", "custom") or LLMProvider enum
        model_name: Model name/ID to use
        api_key: Optional API key (falls back to env var)
        base_url: Optional base URL (falls back to env var or default)
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        Configured LLM client instance

    Examples:
        # Google Gemini
        client = create_client("google", "gemini-3-flash-preview")

        # OpenAI
        client = create_client("openai", "gpt-4o")

        # AIMLAPI
        client = create_client("aimlapi", "gpt-4o")

        # Custom provider
        client = create_client("custom", "my-model", base_url="https://my-api.com/v1")
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    if provider == LLMProvider.GOOGLE:
        return GeminiJSONClient(
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )
    else:
        return OpenAICompatibleClient(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
