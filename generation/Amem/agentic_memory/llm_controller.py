from typing import Dict, Optional, Literal, Any, List, Tuple
import os
import json
import threading
import time
from abc import ABC, abstractmethod


_USAGE_LOCK = threading.Lock()
_USAGE_RECORDS: List[Dict[str, Any]] = []


def _usage_to_dict(raw_usage: Any) -> Dict[str, Any]:
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
    elif hasattr(raw_usage, "to_dict"):
        raw_usage = raw_usage.to_dict()
    if isinstance(raw_usage, dict):
        return raw_usage
    return {}


def _extract_usage(raw_usage: Any) -> Dict[str, Any]:
    payload = _usage_to_dict(raw_usage)
    input_details = _usage_to_dict(payload.get("input_tokens_details"))
    output_details = _usage_to_dict(payload.get("output_tokens_details"))
    prompt_tokens = int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or payload.get("output_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": int(payload.get("reasoning_tokens") or output_details.get("reasoning_tokens") or 0),
        "cached_input_tokens": int(payload.get("cached_input_tokens") or input_details.get("cached_tokens") or 0),
        "cache_write_tokens": int(payload.get("cache_write_tokens") or input_details.get("cache_write_tokens") or 0),
        "context_tokens": int(payload.get("context_tokens") or 0),
        "total_tokens": total_tokens,
    }


def record_usage(*, request_kind: str, provider: str, model: str, usage: Any) -> Dict[str, Any]:
    record = _extract_usage(usage)
    record["request_kind"] = str(request_kind or "")
    record["provider"] = str(provider or "")
    record["model"] = str(model or "")
    record["timestamp_unix"] = time.time()
    with _USAGE_LOCK:
        _USAGE_RECORDS.append(dict(record))
    return record


def reset_usage_tracker() -> None:
    with _USAGE_LOCK:
        _USAGE_RECORDS.clear()


def get_usage_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "request_count": 0,
        "chat_request_count": 0,
        "embedding_request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
        "turn_count": 0,
        "by_model": [],
    }
    with _USAGE_LOCK:
        records = [dict(x) for x in _USAGE_RECORDS]
    summary["request_count"] = len(records)
    summary["turn_count"] = len(records)
    by_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        request_kind = str(record.get("request_kind") or "")
        if request_kind == "chat":
            summary["chat_request_count"] += 1
        elif request_kind == "embedding":
            summary["embedding_request_count"] += 1
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
        bucket_key = (request_kind, str(record.get("model") or ""))
        bucket = by_model.setdefault(
            bucket_key,
            {
                "request_kind": bucket_key[0],
                "model": bucket_key[1],
                "request_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        bucket["request_count"] += 1
        bucket["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(record.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(record.get("total_tokens") or 0)
    summary["by_model"] = list(by_model.values())
    return summary

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
                if os.getenv("AZURE_OPENAI_API_KEY") or "azure" in base_url.lower():
                    client_kwargs["default_headers"] = {"api-key": api_key}
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            # max_tokens=1000
        )
        record_usage(
            request_kind="chat",
            provider="openai",
            model=self.model,
            usage=getattr(response, "usage", None),
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        # Allow exceptions (like ConnectionError) to bubble up for better debugging
        from litellm import completion

        response = completion(
            model="ollama_chat/{}".format(self.model),
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
        )
        return response.choices[0].message.content

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
