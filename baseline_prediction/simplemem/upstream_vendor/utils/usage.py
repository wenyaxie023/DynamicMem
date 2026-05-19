import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from baseline_prediction.common.provider_config import resolve_openai_compatible_credentials


def _usage_to_dict(raw_usage: Any) -> Dict[str, Any]:
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
    elif hasattr(raw_usage, "to_dict"):
        raw_usage = raw_usage.to_dict()
    return raw_usage if isinstance(raw_usage, dict) else {}


class UsageTracker:
    def __init__(self, request_kind: str, provider: str, model: str):
        self.request_kind = str(request_kind or "")
        self.provider = str(provider or "")
        self.model = str(model or "")
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

    def record(self, usage_payload: Dict[str, Any]) -> None:
        row = {
            "request_kind": self.request_kind,
            "provider": self.provider,
            "model": self.model,
            "request_count": 1,
            "prompt_tokens": int(usage_payload.get("prompt_tokens") or usage_payload.get("input_tokens") or 0),
            "completion_tokens": int(usage_payload.get("completion_tokens") or usage_payload.get("output_tokens") or 0),
            "reasoning_tokens": int(usage_payload.get("reasoning_tokens") or 0),
            "cached_input_tokens": int(usage_payload.get("cached_input_tokens") or 0),
            "cache_write_tokens": int(usage_payload.get("cache_write_tokens") or 0),
            "context_tokens": int(usage_payload.get("context_tokens") or 0),
            "total_tokens": int(
                usage_payload.get("total_tokens")
                or (
                    int(usage_payload.get("prompt_tokens") or usage_payload.get("input_tokens") or 0)
                    + int(usage_payload.get("completion_tokens") or usage_payload.get("output_tokens") or 0)
                )
            ),
            "timestamp_unix": time.time(),
        }
        with self._lock:
            self._records.append(row)

    def summary(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
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
        with self._lock:
            records = [dict(x) for x in self._records]
        merged["request_count"] = len(records)
        merged["turn_count"] = len(records)
        if self.request_kind == "embedding":
            merged["embedding_request_count"] = len(records)
        else:
            merged["chat_request_count"] = len(records)
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
                merged[key] += int(record.get(key) or 0)
        if records:
            merged["by_model"] = [
                {
                    "request_kind": self.request_kind,
                    "model": self.model,
                    "request_count": len(records),
                    "prompt_tokens": merged["prompt_tokens"],
                    "completion_tokens": merged["completion_tokens"],
                    "total_tokens": merged["total_tokens"],
                }
            ]
        return merged


def merge_usage_summary(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
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
    by_model: Dict[tuple, Dict[str, Any]] = {}
    for source in (base, delta):
        if not isinstance(source, dict):
            continue
        for key in (
            "request_count",
            "chat_request_count",
            "embedding_request_count",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "context_tokens",
            "total_tokens",
            "turn_count",
        ):
            merged[key] += int(source.get(key) or 0)
        for bucket in source.get("by_model") or []:
            if not isinstance(bucket, dict):
                continue
            key = (str(bucket.get("request_kind") or ""), str(bucket.get("model") or ""))
            dst = by_model.setdefault(
                key,
                {
                    "request_kind": key[0],
                    "model": key[1],
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for field in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens"):
                dst[field] += int(bucket.get(field) or 0)
    merged["by_model"] = list(by_model.values())
    return merged


@dataclass
class EmbeddingClient:
    provider: str
    model: str
    batch_size: int = 64
    tracker: Optional[UsageTracker] = None

    def __post_init__(self) -> None:
        api_key, base_url = resolve_openai_compatible_credentials(self.provider, require_api_key=True)
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            if str(self.provider or "").strip().lower() == "azure":
                client_kwargs["default_headers"] = {"api-key": api_key}
        self._client = OpenAI(**client_kwargs)

    def _record(self, response: Any) -> None:
        if self.tracker is None:
            return
        self.tracker.record(_usage_to_dict(getattr(response, "usage", None)))

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        outputs: List[List[float]] = []
        step = max(1, int(self.batch_size or 1))
        for offset in range(0, len(texts), step):
            batch = texts[offset : offset + step]
            response = self._client.embeddings.create(model=self.model, input=batch)
            self._record(response)
            outputs.extend([list(item.embedding) for item in response.data])
        return outputs

    def encode_single(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model=self.model, input=[text])
        self._record(response)
        return list(response.data[0].embedding)
