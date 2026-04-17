import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def empty_usage_summary() -> Dict[str, Any]:
    return {
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


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def merge_usage_summary(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged = empty_usage_summary()
    by_model: Dict[tuple[str, str], Dict[str, Any]] = {}
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
            bucket_key = (str(bucket.get("request_kind") or ""), str(bucket.get("model") or ""))
            dst = by_model.setdefault(
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
            for field in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens"):
                dst[field] += int(bucket.get(field) or 0)
    merged["by_model"] = list(by_model.values())
    return merged


def build_usage_sidecar_paths(ckpt_dir: Path) -> tuple[Path, Path]:
    return ckpt_dir / "usage_cost.json", ckpt_dir / "usage_cost_live.json"


def load_usage_sidecar(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_build_usage_sidecar(
    *,
    sidecar_path: Path,
    state_path: Path,
    checkpoint_path: Path,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    embedding_model_name: str,
    build_duration_s: float,
    build_memory_usage: Dict[str, Any],
    is_live: bool,
) -> None:
    payload = {
        "prediction_path": str(state_path),
        "checkpoint_json_path": str(checkpoint_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retriever_provider": retriever_provider,
        "embedding_model": embedding_model_name,
        "usage": {
            "build_memory": build_memory_usage if isinstance(build_memory_usage, dict) else {},
            "retrieval": {},
            "answer_llm": {},
        },
        "timing": {
            "build_memory_duration_s": float(build_duration_s),
            "generation_duration_s": 0.0,
            "total_duration_s": float(build_duration_s),
        },
        "generated_at": datetime.now().isoformat(),
    }
    if is_live:
        payload["is_live"] = True
    atomic_write_json(sidecar_path, payload)
