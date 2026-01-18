import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai
try:
    from google import genai as genai_client
except ImportError: 
    genai_client = None


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


@dataclass
class LLMResult:
    data: Dict[str, Any]
    usage: Dict[str, int]
    prompt: str
    raw_text: str


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
    ) -> LLMResult:
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
            response = self.model.generate_content(prompt)
        raw_text = (response.text or "").strip()
        try:
            payload = _parse_json_from_text(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(
                f"Failed to decode JSON from model response: {exc}\nRaw response:\n{raw_text}"
            ) from exc
        # import pdb; pdb.set_trace()
        usage = _extract_usage_metadata(getattr(response, "usage_metadata", None))
        return LLMResult(data=payload, usage=usage, prompt=prompt, raw_text=raw_text)
