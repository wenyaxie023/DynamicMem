"""LLM wrapper with upstream-compatible interface and TCE usage tracking."""

import json
import threading
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from generation.common.provider_config import resolve_openai_compatible_credentials

from .. import config

_AIMLAPI_MODEL_MAPPING = {
    "gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
}


def _normalize_aiml_model(model_name: str) -> str:
    return _AIMLAPI_MODEL_MAPPING.get(model_name, model_name)


def _usage_to_dict(raw_usage: Any) -> Dict[str, Any]:
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
    elif hasattr(raw_usage, "to_dict"):
        raw_usage = raw_usage.to_dict()
    return raw_usage if isinstance(raw_usage, dict) else {}


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        provider: Optional[str] = None,
        max_workers: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        del max_workers
        self.provider = str(provider or config.LLM_PROVIDER)
        self.model = str(model or config.LLM_MODEL)
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.top_p = config.LLM_TOP_P if top_p is None else top_p
        self.top_k = config.LLM_TOP_K if top_k is None else top_k
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

        if self.provider in {"openai", "azure"}:
            resolved_key, resolved_base_url = resolve_openai_compatible_credentials(self.provider)
            api_key = api_key or resolved_key
            self.base_url = self.base_url or resolved_base_url
        elif self.provider in {"aiml", "aimlapi"}:
            api_key = api_key or ""
            self.model = _normalize_aiml_model(self.model)
        elif self.provider == "vllm":
            api_key = api_key or "EMPTY"
            self.base_url = self.base_url or "http://localhost:8002/v1"
        else:
            raise ValueError(f"Unsupported SimpleMem LLM provider: {self.provider}")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            if self.provider == "azure":
                client_kwargs["default_headers"] = {"api-key": api_key}
        self.client = OpenAI(**client_kwargs)

    def _supports_sampling_controls(self) -> bool:
        provider = str(self.provider or "").strip().lower()
        model_name = str(self.model or "").strip().lower()
        if provider not in {"openai", "azure", "aiml", "aimlapi"}:
            return True
        return not model_name.startswith("gpt-5")

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        raw_usage = _usage_to_dict(getattr(response, "usage", None))
        input_details = _usage_to_dict(raw_usage.get("input_tokens_details"))
        output_details = _usage_to_dict(raw_usage.get("output_tokens_details"))
        prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or 0) or (prompt_tokens + completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": int(raw_usage.get("reasoning_tokens") or output_details.get("reasoning_tokens") or 0),
            "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or input_details.get("cached_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_write_tokens") or input_details.get("cache_write_tokens") or 0),
            "context_tokens": int(raw_usage.get("context_tokens") or 0),
            "total_tokens": total_tokens,
        }

    def _record_usage(self, *, phase: str, response: Any) -> None:
        usage = self._extract_usage(response)
        usage["phase"] = str(phase or "response")
        usage["provider"] = self.provider
        usage["model"] = self.model
        usage["timestamp_unix"] = time.time()
        with self._usage_lock:
            self._usage_records.append(usage)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        phase: str = "response",
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self._supports_sampling_controls():
            kwargs["temperature"] = float(temperature if temperature is not None else self.temperature or 0.0)
            if self.top_p is not None:
                kwargs["top_p"] = float(self.top_p)
        if response_format:
            kwargs["response_format"] = response_format
        if self.top_k is not None and self.provider == "vllm":
            kwargs.setdefault("extra_body", {})["top_k"] = int(self.top_k)

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_usage(phase=phase, response=response)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
                    time.sleep(wait_s)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {exc}")
        raise last_exception

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def usage_summary(self, phase: Optional[str] = None) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "request_count": 0,
            "turn_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": 0,
            "by_model": [],
        }
        with self._usage_lock:
            records = [dict(x) for x in self._usage_records]
        if phase is not None:
            records = [x for x in records if str(x.get("phase") or "") == str(phase)]
        summary["request_count"] = len(records)
        summary["turn_count"] = len(records)
        for record in records:
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
                "context_tokens",
                "total_tokens",
            ):
                summary[key] += int(record.get(key) or 0)
        if records:
            summary["by_model"] = [
                {
                    "request_kind": "response",
                    "model": self.model,
                    "request_count": len(records),
                    "prompt_tokens": summary["prompt_tokens"],
                    "completion_tokens": summary["completion_tokens"],
                    "total_tokens": summary["total_tokens"],
                }
            ]
        return summary

    def close(self) -> None:
        return None

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
