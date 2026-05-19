import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union
from .provider_config import resolve_openai_compatible_credentials

try:
    from google import genai
except ImportError:
    genai = None

from openai import OpenAI


_AIMLAPI_MODEL_MAPPING = {
    "gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-2.5-flash-preview-05-20": "google/gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
}


def _normalize_aiml_model(model_name: str) -> str:
    return _AIMLAPI_MODEL_MAPPING.get(model_name, model_name)


class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-5-mini",
        *,
        max_workers: int = 4,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        vllm_base_url: str = "http://localhost:8002/v1",
    ):
        self.provider = provider
        self.model_name = model_name
        self.vllm_base_url = vllm_base_url
        self.temperature = None if temperature is None else float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.top_k = None if top_k is None else int(top_k)
        if self.top_k is not None and self.top_k <= 0:
            self.top_k = None
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=max_workers)
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []

        if provider in {"openai", "azure"}:
            api_key, base_url = resolve_openai_compatible_credentials(provider)
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
                if os.getenv("AZURE_OPENAI_API_KEY") or "azure" in base_url:
                    client_kwargs["default_headers"] = {"api-key": api_key}
            self.client = OpenAI(**client_kwargs)
        elif provider in {"aiml", "aimlapi"}:
            api_key = os.getenv("AIMLAPI_API_KEY")
            if api_key and api_key.startswith("Bearer "):
                api_key = api_key[len("Bearer ") :]
            base_url = os.getenv("AIMLAPI_BASE_URL") or "https://api.aimlapi.com/v1"
            self.model_name = _normalize_aiml_model(self.model_name)
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "gemini":
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        elif provider == "vllm":
            self.client = OpenAI(base_url=self.vllm_base_url, api_key="EMPTY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _supports_sampling_controls(self) -> bool:
        model_name = str(self.model_name or "").strip().lower()
        provider = str(self.provider or "").strip().lower()
        if provider not in {"openai", "azure", "aiml", "aimlapi"}:
            return True
        if model_name.startswith("gpt-5"):
            return False
        return True

    def ask(self, prompt: str, *, response_type: str = "json") -> Union[Dict[str, Any], str]:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")
        future = self._executor.submit(self._ask_impl, prompt, response_type)
        return future.result()

    def ask_async(self, prompt: str, *, response_type: str = "json") -> Future:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")
        return self._executor.submit(self._ask_impl, prompt, response_type)

    def supports_structured_response(self) -> bool:
        return self.provider in {"openai", "azure"}

    def ask_structured(self, prompt: str, *, text_format: Type[Any]) -> Dict[str, Any]:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")
        future = self._executor.submit(self._ask_structured_impl, prompt, text_format)
        return future.result()

    def ask_structured_async(self, prompt: str, *, text_format: Type[Any]) -> Future:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")
        return self._executor.submit(self._ask_structured_impl, prompt, text_format)

    def ask_many_async(
        self,
        prompts: Iterable[str],
        *,
        response_type: str = "json",
    ) -> List[Future]:
        return [self.ask_async(prompt, response_type=response_type) for prompt in prompts]

    def ask_many(
        self,
        prompts: Iterable[str],
        *,
        response_type: str = "json",
    ) -> List[Union[Dict[str, Any], str, Exception]]:
        return self.collect(self.ask_many_async(prompts, response_type=response_type))

    def collect(self, futures: Iterable[Future]) -> List[Union[Dict[str, Any], str, Exception]]:
        results: List[Union[Dict[str, Any], str, Exception]] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
        return results

    def _with_retry(self, func: Callable, *args, **kwargs) -> Any:
        import time
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if "content_filter" in str(exc):
                    raise
                if attempt < max_attempts - 1:
                    print(f"[LLM] Attempt {attempt + 1} failed: {exc}. Retrying in 2s...")
                    time.sleep(2)
                    continue
                raise

    def _response_sampling_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if not self._supports_sampling_controls():
            return kwargs
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs

    def _call_with_typeerror_variants(self, func: Callable[..., Any], base_kwargs: Dict[str, Any]) -> Any:
        variants = [dict(base_kwargs)]
        if "truncation" in base_kwargs:
            no_trunc = dict(base_kwargs)
            no_trunc.pop("truncation", None)
            variants.append(no_trunc)
        if "temperature" in base_kwargs or "top_p" in base_kwargs:
            no_sampling = dict(base_kwargs)
            no_sampling.pop("temperature", None)
            no_sampling.pop("top_p", None)
            variants.append(no_sampling)
        if "truncation" in base_kwargs and ("temperature" in base_kwargs or "top_p" in base_kwargs):
            minimal = dict(base_kwargs)
            minimal.pop("truncation", None)
            minimal.pop("temperature", None)
            minimal.pop("top_p", None)
            variants.append(minimal)

        last_exc: Optional[TypeError] = None
        seen = set()
        for kwargs in variants:
            key = tuple(sorted(kwargs.keys()))
            if key in seen:
                continue
            seen.add(key)
            try:
                return func(**kwargs)
            except TypeError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return func(**base_kwargs)

    def _usage_to_dict(self, raw_usage: Any) -> Dict[str, Any]:
        if hasattr(raw_usage, "model_dump"):
            raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
        elif hasattr(raw_usage, "to_dict"):
            raw_usage = raw_usage.to_dict()
        if isinstance(raw_usage, dict):
            return raw_usage
        return {}

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        raw_usage = self._usage_to_dict(getattr(response, "usage", None))
        input_details = self._usage_to_dict(raw_usage.get("input_tokens_details"))
        output_details = self._usage_to_dict(raw_usage.get("output_tokens_details"))
        prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": int(raw_usage.get("reasoning_tokens") or output_details.get("reasoning_tokens") or 0),
            "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or input_details.get("cached_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_write_tokens") or input_details.get("cache_write_tokens") or 0),
            "context_tokens": int(raw_usage.get("context_tokens") or 0),
            "total_tokens": total_tokens,
        }

    def _record_usage(self, *, phase: str, response: Any) -> Dict[str, Any]:
        usage = self._extract_usage(response)
        usage["phase"] = str(phase or "")
        usage["provider"] = self.provider
        usage["model"] = self.model_name
        usage["timestamp_unix"] = time.time()
        with self._usage_lock:
            self._usage_records.append(dict(usage))
        return usage

    def usage_records(self) -> List[Dict[str, Any]]:
        with self._usage_lock:
            return [dict(x) for x in self._usage_records]

    def usage_summary(self) -> Dict[str, Any]:
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
                    "model": self.model_name,
                    "request_count": len(records),
                    "prompt_tokens": summary["prompt_tokens"],
                    "completion_tokens": summary["completion_tokens"],
                    "total_tokens": summary["total_tokens"],
                }
            ]
        return summary

    def _ask_structured_impl(self, prompt: str, text_format: Type[Any]) -> Dict[str, Any]:
        if not self.supports_structured_response():
            raise ValueError(
                f"Structured response is only supported for openai/azure provider, got {self.provider}"
            )

        def _parse_call():
            kwargs = {
                "model": self.model_name,
                "input": prompt,
                "text_format": text_format,
                "truncation": "auto",
                "timeout": 300,
            }
            kwargs.update(self._response_sampling_kwargs())
            return self._call_with_typeerror_variants(self.client.responses.parse, kwargs)

        response = self._with_retry(_parse_call)
        self._record_usage(phase="ask_structured", response=response)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("Structured response parsing returned no output_parsed payload.")
        if hasattr(parsed, "model_dump"):
            dumped = parsed.model_dump(mode="json", by_alias=True)
            if isinstance(dumped, dict):
                return dumped
            return {"parsed": dumped}
        if isinstance(parsed, dict):
            return parsed
        return {"parsed": str(parsed)}

    def _ask_impl(self, prompt: str, response_type: str) -> Union[Dict[str, Any], str]:
        if self.provider in {"openai", "azure", "aiml", "aimlapi"}:
            def _create_call():
                kwargs = {
                    "model": self.model_name,
                    "input": prompt,
                    "truncation": "auto",
                    "timeout": 300,
                }
                kwargs.update(self._response_sampling_kwargs())
                return self._call_with_typeerror_variants(self.client.responses.create, kwargs)

            response = self._with_retry(_create_call)
            self._record_usage(phase="ask", response=response)
            return self._parse_response(response.output_text, response_type)
        if self.provider == "gemini":
            config: Dict[str, Any] = {"response_mime_type": "application/json"}
            if self.temperature is not None:
                config["temperature"] = self.temperature
            if self.top_p is not None:
                config["top_p"] = self.top_p
            if self.top_k is not None:
                config["top_k"] = self.top_k
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return self._parse_response(response.text, response_type)
        if self.provider == "vllm":
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["extra_body"] = {"top_k": self.top_k}
            response = self.client.chat.completions.create(**kwargs)
            return self._parse_response(response.choices[0].message.content, response_type)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, text: str, response_type: str) -> Union[Dict[str, Any], str]:
        if response_type == "text":
            return text
        if response_type != "json":
            raise ValueError(f"Unknown response_type: {response_type}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            extracted = self._extract_json_block(text)
            if extracted is None:
                raise ValueError(f"Invalid JSON response:\n{text}")
            return json.loads(extracted)

    def _extract_json_block(self, text: str) -> Optional[str]:
        marker = "```"
        start = text.find(marker)
        if start == -1:
            return None
        end = text.find(marker, start + len(marker))
        if end == -1:
            return None
        block = text[start + len(marker) : end]
        if block.lstrip().lower().startswith("json"):
            block = block.lstrip()[4:]
        block = block.strip()
        return block if block else None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
