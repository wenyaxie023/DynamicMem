#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
HIPPORAG2_ROOT_DIR = SCRIPT_DIR.parent
REPO_ROOT_DIR = HIPPORAG2_ROOT_DIR.parent.parent

# The `hipporag` package is an external dependency installed via pip;
# see baseline_prediction/HippoRAG2/requirements.txt.
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.append(str(REPO_ROOT_DIR))


def _patch_transformers_torch_safety() -> None:
    try:
        import transformers.utils.import_utils as import_utils  # type: ignore

        import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
    except Exception:
        pass


def _best_effort_load_local_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(HIPPORAG2_ROOT_DIR / ".env", override=False)


_patch_transformers_torch_safety()

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

from baseline_prediction.common.provider_config import (
    apply_openai_compat_env,
    load_repo_dotenv,
    resolve_openai_compatible_credentials,
)
from baseline_prediction.tce_safety import ensure_destructive_rebuild_allowed
from baseline_prediction.rag.client import LLMClient
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import normalize_app_logs, run_pipeline, to_log_text


def _detect_openie_mode(llm_model: str) -> str:
    model_name = str(llm_model or "").strip().lower()
    if any(token in model_name for token in ("gpt-", "o3-", "openai/")):
        return "online"
    return "offline"


def _configure_hipporag_openai_env(
    *,
    llm_provider: str,
    retriever_provider: str,
) -> Dict[str, Optional[str]]:
    load_repo_dotenv(REPO_ROOT_DIR)
    _best_effort_load_local_dotenv()

    llm_api_key, llm_base_url = resolve_openai_compatible_credentials(llm_provider)
    retriever_api_key, retriever_base_url = resolve_openai_compatible_credentials(retriever_provider)

    apply_openai_compat_env(llm_api_key, llm_base_url)
    if retriever_api_key:
        os.environ["OPENAI_API_KEY_EMBEDDING"] = retriever_api_key
    else:
        os.environ.pop("OPENAI_API_KEY_EMBEDDING", None)

    return {
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "retriever_api_key": retriever_api_key,
        "retriever_base_url": retriever_base_url,
    }


def _usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost.json"


def _live_usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost_live.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(value or "")).strip("._-")
    return cleaned or "default"


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _replace_dir_from(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _builder_root(data_storage_root: Path) -> Path:
    return Path(data_storage_root) / "builder"


ARTIFACT_VERSION = 2


def _builder_workspace_dir(data_storage_root: Path) -> Path:
    return _builder_root(data_storage_root) / "workspace"


def _builder_progress_path(data_storage_root: Path) -> Path:
    return _builder_root(data_storage_root) / "progress.json"


def _periodic_snapshots_root(snapshot_root: Path) -> Path:
    return Path(snapshot_root) / "periodic"


def _checkpoints_root(snapshot_root: Path) -> Path:
    return Path(snapshot_root) / "checkpoints"


def _checkpoint_manifest_path(snapshot_root: Path) -> Path:
    return Path(snapshot_root) / "manifest.json"


def _snapshot_meta_path(snapshot_dir: Path) -> Path:
    return Path(snapshot_dir) / "snapshot_meta.json"


def _write_builder_progress(
    data_storage_root: Path,
    *,
    config_fingerprint: str,
    preprocess_fingerprint: str = "",
    confirmed_last_log_index: int,
    confirmed_last_app_log_id: Optional[str],
    active_workspace_path: Path,
    status: str,
) -> Dict[str, Any]:
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config_fingerprint": str(config_fingerprint or ""),
        "preprocess_fingerprint": str(preprocess_fingerprint or ""),
        "confirmed_last_log_index": int(confirmed_last_log_index),
        "confirmed_last_app_log_id": str(confirmed_last_app_log_id or "").strip(),
        "status": str(status or ""),
        "active_workspace_path": str(Path(active_workspace_path).resolve()),
        "updated_at": datetime.now().isoformat(),
    }
    _atomic_write_json(_builder_progress_path(data_storage_root), payload)
    return payload


def _load_checkpoint_manifest_entries(snapshot_root: Path) -> List[Dict[str, Any]]:
    payload = _load_json_object(_checkpoint_manifest_path(snapshot_root))
    entries = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_checkpoint_manifest_entries(snapshot_root: Path, entries: List[Dict[str, Any]]) -> None:
    _atomic_write_json(
        _checkpoint_manifest_path(snapshot_root),
        {"artifact_version": ARTIFACT_VERSION, "checkpoints": entries},
    )


def _load_snapshot_meta(snapshot_dir: Path) -> Dict[str, Any]:
    return _load_json_object(_snapshot_meta_path(snapshot_dir))


def _write_snapshot_meta(
    snapshot_dir: Path,
    *,
    snapshot_id: str,
    trigger: str,
    last_event_idx: int,
    last_app_log_id: Optional[str],
    checkpoint_id: Optional[str],
    checkpoint_app_log_id: Optional[str],
    config_fingerprint: str,
) -> Dict[str, Any]:
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "snapshot_id": str(snapshot_id or ""),
        "trigger": str(trigger or ""),
        "last_event_idx": int(last_event_idx),
        "last_app_log_id": str(last_app_log_id or "").strip(),
        "checkpoint_id": str(checkpoint_id or "").strip(),
        "checkpoint_app_log_id": str(checkpoint_app_log_id or "").strip(),
        "config_fingerprint": str(config_fingerprint or ""),
        "updated_at": datetime.now().isoformat(),
    }
    _atomic_write_json(_snapshot_meta_path(snapshot_dir), payload)
    return payload


def _checkpoint_cut_index(cp: Dict[str, Any], app_logs: List[Dict[str, Any]]) -> int:
    as_of = cp.get("as_of")
    as_of = as_of if isinstance(as_of, dict) else {}
    log_index = as_of.get("log_index")
    if isinstance(log_index, int) and log_index >= 0:
        return min(log_index, max(len(app_logs) - 1, 0))
    checkpoint_app_log_id = str(as_of.get("app_log_id") or "").strip()
    if checkpoint_app_log_id:
        for idx, log in enumerate(app_logs):
            if str(log.get("app_log_id") or "").strip() == checkpoint_app_log_id:
                return idx
    raise ValueError(
        "Checkpoint must provide as_of.log_index or as_of.app_log_id for HippoRAG2 snapshot materialization: {}".format(
            cp.get("checkpoint_id")
        )
    )


def _build_builder_fingerprint(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    checkpoint_specs: Optional[List[Dict[str, Any]]] = None,
    preprocess_fingerprint: str = "",
    llm_model: str = "",
    llm_base_url: Optional[str] = None,
    embedding_model: str = "",
    embedding_base_url: Optional[str] = None,
    batch_size: int = 0,
    openie_mode: str = "",
) -> str:
    source_checkpoints = []
    for spec in checkpoint_specs or []:
        if not isinstance(spec, dict):
            continue
        source_checkpoints.append(
            {
                "checkpoint_id": str(spec.get("checkpoint_id") or ""),
                "checkpoint_app_log_id": str(spec.get("checkpoint_app_log_id") or ""),
                "cut_index": int(spec.get("cut_index", -1)),
            }
        )
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "app_logs_path": str(Path(app_logs_path).resolve()),
        "checkpoint_specs": source_checkpoints,
        "preprocess_fingerprint": str(preprocess_fingerprint or ""),
        "llm_model": str(llm_model or ""),
        "llm_base_url": str(llm_base_url or ""),
        "embedding_model": str(embedding_model or ""),
        "embedding_base_url": str(embedding_base_url or ""),
        "batch_size": int(batch_size),
        "openie_mode": str(openie_mode or ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _entry_matches_source_checkpoint(entry: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    if str(entry.get("checkpoint_id") or "").strip() != str(spec.get("checkpoint_id") or "").strip():
        return False
    if str(entry.get("checkpoint_app_log_id") or "").strip() != str(spec.get("checkpoint_app_log_id") or "").strip():
        return False
    last_event_idx = entry.get("last_event_idx")
    return isinstance(last_event_idx, int) and int(last_event_idx) == int(spec.get("cut_index", -1))


def _reset_save_root(data_storage_root: Path, snapshot_root: Path, *, allow_destructive_rebuild: bool) -> None:
    ensure_destructive_rebuild_allowed(
        allow_destructive_rebuild=allow_destructive_rebuild,
        operation="clear existing HippoRAG2 builder artifacts",
        paths=[data_storage_root, snapshot_root],
    )
    for root in (data_storage_root, snapshot_root):
        if root.exists():
            shutil.rmtree(root)


def _reset_materialization_root(data_storage_root: Path, snapshot_root: Path, *, allow_destructive_rebuild: bool) -> None:
    ensure_destructive_rebuild_allowed(
        allow_destructive_rebuild=allow_destructive_rebuild,
        operation="clear existing HippoRAG2 materialized artifacts",
        paths=[_builder_root(data_storage_root), Path(snapshot_root)],
    )
    for target in (_builder_root(data_storage_root), Path(snapshot_root)):
        if target.exists():
            shutil.rmtree(target)


def _checkpoint_snapshot_dir(snapshot_root: Path, checkpoint_id: str) -> Path:
    return _checkpoints_root(snapshot_root) / _safe_name(checkpoint_id)


def _persist_workspace_snapshot(
    *,
    workspace_dir: Path,
    snapshot_dir: Path,
    snapshot_id: str,
    trigger: str,
    last_event_idx: int,
    last_app_log_id: Optional[str],
    checkpoint_id: Optional[str] = None,
    checkpoint_app_log_id: Optional[str] = None,
    config_fingerprint: str,
) -> Dict[str, Any]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    _replace_dir_from(workspace_dir, snapshot_dir)
    return _write_snapshot_meta(
        snapshot_dir,
        snapshot_id=snapshot_id,
        trigger=trigger,
        last_event_idx=last_event_idx,
        last_app_log_id=last_app_log_id,
        checkpoint_id=checkpoint_id,
        checkpoint_app_log_id=checkpoint_app_log_id,
        config_fingerprint=config_fingerprint,
    )


def _nearest_resume_snapshot(
    snapshot_root: Path,
    *,
    config_fingerprint: str,
    confirmed_last_log_index: int,
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    periodic_root = _periodic_snapshots_root(snapshot_root)
    if periodic_root.exists():
        for path in periodic_root.iterdir():
            if not path.is_dir():
                continue
            meta = _load_snapshot_meta(path)
            if not meta:
                continue
            meta = dict(meta)
            meta["snapshot_dir"] = str(path)
            candidates.append(meta)

    for entry in _load_checkpoint_manifest_entries(snapshot_root):
        snapshot_path_raw = str(entry.get("snapshot_path") or "").strip()
        if not snapshot_path_raw:
            continue
        snapshot_dir = Path(snapshot_root) / snapshot_path_raw
        if not snapshot_dir.exists():
            continue
        candidate = dict(entry)
        candidate["snapshot_dir"] = str(snapshot_dir)
        candidates.append(candidate)

    valid = []
    for candidate in candidates:
        if str(candidate.get("config_fingerprint") or "") != str(config_fingerprint or ""):
            continue
        last_event_idx = candidate.get("last_event_idx")
        if not isinstance(last_event_idx, int):
            continue
        if last_event_idx > int(confirmed_last_log_index):
            continue
        valid.append(candidate)

    if not valid:
        return None
    return max(valid, key=lambda item: (int(item.get("last_event_idx", -1)), str(item.get("snapshot_id", ""))))


def _empty_usage_summary() -> Dict[str, Any]:
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


def _usage_to_dict(raw_usage: Any) -> Dict[str, Any]:
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
    elif hasattr(raw_usage, "to_dict"):
        raw_usage = raw_usage.to_dict()
    if isinstance(raw_usage, dict):
        return raw_usage
    return {}


def _single_request_usage_summary(
    *,
    request_kind: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    context_tokens: int = 0,
    total_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    summary = _empty_usage_summary()
    total = int(total_tokens or 0)
    if total <= 0:
        total = int(prompt_tokens) + int(completion_tokens)
    summary["request_count"] = 1
    summary["turn_count"] = 1
    if str(request_kind or "") == "embedding":
        summary["embedding_request_count"] = 1
    else:
        summary["chat_request_count"] = 1
    summary["prompt_tokens"] = int(prompt_tokens or 0)
    summary["completion_tokens"] = int(completion_tokens or 0)
    summary["reasoning_tokens"] = int(reasoning_tokens or 0)
    summary["cached_input_tokens"] = int(cached_input_tokens or 0)
    summary["cache_write_tokens"] = int(cache_write_tokens or 0)
    summary["context_tokens"] = int(context_tokens or 0)
    summary["total_tokens"] = total
    summary["by_model"] = [
        {
            "request_kind": str(request_kind or ""),
            "model": str(model or ""),
            "request_count": 1,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": total,
        }
    ]
    return summary


def _usage_summary_from_response(
    response: Any,
    *,
    request_kind: str,
    model: str,
) -> Dict[str, Any]:
    raw_usage = _usage_to_dict(getattr(response, "usage", None))
    input_details = _usage_to_dict(raw_usage.get("input_tokens_details"))
    output_details = _usage_to_dict(raw_usage.get("output_tokens_details"))
    prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
    total_tokens = int(raw_usage.get("total_tokens") or 0)
    return _single_request_usage_summary(
        request_kind=request_kind,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=int(raw_usage.get("reasoning_tokens") or output_details.get("reasoning_tokens") or 0),
        cached_input_tokens=int(raw_usage.get("cached_input_tokens") or input_details.get("cached_tokens") or 0),
        cache_write_tokens=int(raw_usage.get("cache_write_tokens") or input_details.get("cache_write_tokens") or 0),
        context_tokens=int(raw_usage.get("context_tokens") or 0),
        total_tokens=total_tokens,
    )


def _usage_summary_from_openie_metadata(
    metadata: Any,
    *,
    model: str,
) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return _empty_usage_summary()
    prompt_tokens = int(metadata.get("prompt_tokens") or metadata.get("input_tokens") or 0)
    completion_tokens = int(metadata.get("completion_tokens") or metadata.get("output_tokens") or 0)
    total_tokens = int(metadata.get("total_tokens") or 0)
    return _single_request_usage_summary(
        request_kind="response",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _merge_usage_summary(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged = _empty_usage_summary()
    by_model: Dict[Any, Dict[str, Any]] = {}
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


def _write_usage_cost_sidecar(
    *,
    sidecar_path: Path,
    output_path: Path,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    embedding_model_name: str,
    build_duration_s: float,
    generation_duration_s: float,
    total_duration_s: float,
    build_memory_usage: Dict[str, Any],
    retrieval_usage: Dict[str, Any],
    answer_llm_usage: Dict[str, Any],
) -> None:
    existing_payload: Dict[str, Any] = {}
    if sidecar_path.exists():
        try:
            raw_existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            raw_existing = {}
        if isinstance(raw_existing, dict):
            existing_payload = raw_existing

    existing_timing = existing_payload.get("timing") if isinstance(existing_payload.get("timing"), dict) else {}
    timing = {
        "build_memory_duration_s": float(existing_timing.get("build_memory_duration_s") or 0.0) + float(build_duration_s),
        "generation_duration_s": float(existing_timing.get("generation_duration_s") or 0.0) + float(generation_duration_s),
        "total_duration_s": float(existing_timing.get("total_duration_s") or 0.0) + float(total_duration_s),
    }
    existing_usage = existing_payload.get("usage") if isinstance(existing_payload.get("usage"), dict) else {}
    usage = {
        "build_memory": _merge_usage_summary(
            existing_usage.get("build_memory") if isinstance(existing_usage.get("build_memory"), dict) else {},
            build_memory_usage if isinstance(build_memory_usage, dict) else {},
        ),
        "retrieval": _merge_usage_summary(
            existing_usage.get("retrieval") if isinstance(existing_usage.get("retrieval"), dict) else {},
            retrieval_usage if isinstance(retrieval_usage, dict) else {},
        ),
        "answer_llm": _merge_usage_summary(
            existing_usage.get("answer_llm") if isinstance(existing_usage.get("answer_llm"), dict) else {},
            answer_llm_usage if isinstance(answer_llm_usage, dict) else {},
        ),
    }
    payload = {
        "prediction_path": str(output_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retriever_provider": retriever_provider,
        "embedding_model": embedding_model_name,
        "usage": usage,
        "timing": timing,
        "generated_at": datetime.now().isoformat(),
    }
    _atomic_write_json(sidecar_path, payload)


def _write_live_usage_cost_sidecar(
    *,
    sidecar_path: Path,
    output_path: Path,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    embedding_model_name: str,
    build_duration_s: float,
    generation_duration_s: float,
    total_duration_s: float,
    build_memory_usage: Dict[str, Any],
    retrieval_usage: Dict[str, Any],
    answer_llm_usage: Dict[str, Any],
) -> None:
    payload = {
        "prediction_path": str(output_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retriever_provider": retriever_provider,
        "embedding_model": embedding_model_name,
        "usage": {
            "build_memory": build_memory_usage if isinstance(build_memory_usage, dict) else {},
            "retrieval": retrieval_usage if isinstance(retrieval_usage, dict) else {},
            "answer_llm": answer_llm_usage if isinstance(answer_llm_usage, dict) else {},
        },
        "timing": {
            "build_memory_duration_s": float(build_duration_s),
            "generation_duration_s": float(generation_duration_s),
            "total_duration_s": float(total_duration_s),
        },
        "generated_at": datetime.now().isoformat(),
        "is_live": True,
    }
    _atomic_write_json(sidecar_path, payload)


class HippoRAG2UsageTracker:
    def __init__(
        self,
        *,
        output_path: Path,
        llm_provider: str,
        llm_model: str,
        retriever_provider: str,
        embedding_model_name: str,
    ):
        self.output_path = Path(output_path)
        self.llm_provider = str(llm_provider)
        self.llm_model = str(llm_model)
        self.retriever_provider = str(retriever_provider)
        self.embedding_model_name = str(embedding_model_name)
        self._lock = threading.Lock()
        self._phase_local = threading.local()
        self._build_memory_usage = _empty_usage_summary()
        self._retrieval_usage = _empty_usage_summary()
        self._answer_llm_usage = _empty_usage_summary()
        self._build_duration_s = 0.0
        self._generation_duration_s = 0.0
        self._started_at = time.time()
        self._live_sidecar_path = _live_usage_cost_sidecar_path(self.output_path)
        self._final_sidecar_path = _usage_cost_sidecar_path(self.output_path)
        self.write_live()

    @contextmanager
    def phase(self, name: str):
        prev = getattr(self._phase_local, "name", "")
        self._phase_local.name = str(name or "")
        try:
            yield
        finally:
            self._phase_local.name = prev

    def current_phase(self) -> str:
        return str(getattr(self._phase_local, "name", "") or "")

    def _write_live_locked(self) -> None:
        _write_live_usage_cost_sidecar(
            sidecar_path=self._live_sidecar_path,
            output_path=self.output_path,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            retriever_provider=self.retriever_provider,
            embedding_model_name=self.embedding_model_name,
            build_duration_s=self._build_duration_s,
            generation_duration_s=self._generation_duration_s,
            total_duration_s=time.time() - self._started_at,
            build_memory_usage=self._build_memory_usage,
            retrieval_usage=self._retrieval_usage,
            answer_llm_usage=self._answer_llm_usage,
        )

    def write_live(self) -> None:
        with self._lock:
            self._write_live_locked()

    def record_openie_results(self, ner_results: Any, triple_results: Any) -> None:
        delta = _empty_usage_summary()
        for result_map in (ner_results, triple_results):
            if not isinstance(result_map, dict):
                continue
            for result in result_map.values():
                delta = _merge_usage_summary(
                    delta,
                    _usage_summary_from_openie_metadata(getattr(result, "metadata", None), model=self.llm_model),
                )
        with self._lock:
            self._build_memory_usage = _merge_usage_summary(self._build_memory_usage, delta)
            self._write_live_locked()

    def record_embedding_response(self, response: Any, *, phase: str) -> None:
        target_phase = str(phase or "").strip().lower()
        delta = _usage_summary_from_response(response, request_kind="embedding", model=self.embedding_model_name)
        with self._lock:
            if target_phase == "retrieval":
                self._retrieval_usage = _merge_usage_summary(self._retrieval_usage, delta)
            else:
                self._build_memory_usage = _merge_usage_summary(self._build_memory_usage, delta)
            self._write_live_locked()

    def add_build_duration(self, seconds: float) -> None:
        with self._lock:
            self._build_duration_s += max(0.0, float(seconds or 0.0))
            self._write_live_locked()

    def set_build_duration(self, seconds: float) -> None:
        with self._lock:
            self._build_duration_s = max(0.0, float(seconds or 0.0))
            self._write_live_locked()

    def add_generation_duration(self, seconds: float) -> None:
        with self._lock:
            self._generation_duration_s += max(0.0, float(seconds or 0.0))
            self._write_live_locked()

    def set_answer_usage(self, usage_summary: Dict[str, Any]) -> None:
        with self._lock:
            self._answer_llm_usage = _merge_usage_summary({}, usage_summary if isinstance(usage_summary, dict) else {})
            self._write_live_locked()

    def finalize(self) -> None:
        with self._lock:
            _write_usage_cost_sidecar(
                sidecar_path=self._final_sidecar_path,
                output_path=self.output_path,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                retriever_provider=self.retriever_provider,
                embedding_model_name=self.embedding_model_name,
                build_duration_s=self._build_duration_s,
                generation_duration_s=self._generation_duration_s,
                total_duration_s=time.time() - self._started_at,
                build_memory_usage=self._build_memory_usage,
                retrieval_usage=self._retrieval_usage,
                answer_llm_usage=self._answer_llm_usage,
            )
            self._write_live_locked()


class HippoRAG2Runner:
    def __init__(
        self,
        *,
        data_storage_root: Path,
        snapshot_root: Path,
        all_logs: List[Dict[str, Any]],
        llm_model: str,
        llm_base_url: Optional[str],
        embedding_model: str,
        embedding_base_url: Optional[str],
        batch_size: int,
        retrieval_top_k: int,
        openie_mode: str,
        allow_destructive_rebuild: bool = False,
        hipporag_factory: Optional[Callable[[Path], Any]] = None,
        usage_tracker: Optional[HippoRAG2UsageTracker] = None,
    ):
        self.save_root = Path(data_storage_root)
        self.snapshot_root = Path(snapshot_root)
        self.all_logs = list(all_logs)
        self.llm_model = str(llm_model)
        self.llm_base_url = llm_base_url
        self.embedding_model = str(embedding_model)
        self.embedding_base_url = embedding_base_url
        self.batch_size = max(1, int(batch_size))
        self.retrieval_top_k = int(retrieval_top_k)
        self.openie_mode = str(openie_mode)
        self.allow_destructive_rebuild = bool(allow_destructive_rebuild)
        self._hipporag_factory = hipporag_factory
        self._usage_tracker = usage_tracker
        self._prepared_states: Dict[str, Dict[str, Any]] = {}
        self._prepared_states_lock = threading.Lock()

    def _checkpoint_dir(self, checkpoint_id: str) -> Path:
        return _checkpoint_snapshot_dir(self.snapshot_root, checkpoint_id)

    def _create_hipporag(self, save_dir: Path, *, force_index_from_scratch: bool):
        if self._hipporag_factory is not None:
            hipporag = self._hipporag_factory(save_dir)
            self._instrument_hipporag(hipporag)
            return hipporag

        save_dir.mkdir(parents=True, exist_ok=True)
        config = BaseConfig(
            temperature=0,
            llm_base_url=self.llm_base_url,
            embedding_base_url=self.embedding_base_url,
            embedding_batch_size=self.batch_size,
            retrieval_top_k=max(1, self.retrieval_top_k),
            qa_top_k=max(1, self.retrieval_top_k),
            force_index_from_scratch=bool(force_index_from_scratch),
            openie_mode=self.openie_mode,
        )
        hipporag = HippoRAG(
            global_config=config,
            save_dir=str(save_dir),
            llm_model_name=self.llm_model,
            embedding_model_name=self.embedding_model,
            llm_base_url=self.llm_base_url,
            embedding_base_url=self.embedding_base_url,
        )
        self._instrument_hipporag(hipporag)
        return hipporag

    def _instrument_hipporag(self, hipporag: Any) -> None:
        if self._usage_tracker is None or hipporag is None:
            return

        openie = getattr(hipporag, "openie", None)
        batch_openie = getattr(openie, "batch_openie", None)
        if callable(batch_openie) and not getattr(openie, "_dynamicmem_usage_wrapped", False):
            def _wrapped_batch_openie(*args, **kwargs):
                ner_results, triple_results = batch_openie(*args, **kwargs)
                self._usage_tracker.record_openie_results(ner_results, triple_results)
                return ner_results, triple_results

            openie.batch_openie = _wrapped_batch_openie
            openie._dynamicmem_usage_wrapped = True

        embedding_model = getattr(hipporag, "embedding_model", None)
        client = getattr(embedding_model, "client", None)
        embeddings_api = getattr(client, "embeddings", None)
        create_fn = getattr(embeddings_api, "create", None)
        if callable(create_fn) and not getattr(embeddings_api, "_dynamicmem_usage_wrapped", False):
            def _wrapped_embeddings_create(*args, **kwargs):
                response = create_fn(*args, **kwargs)
                self._usage_tracker.record_embedding_response(
                    response,
                    phase=self._usage_tracker.current_phase(),
                )
                return response

            embeddings_api.create = _wrapped_embeddings_create
            embeddings_api._dynamicmem_usage_wrapped = True

    def materialize_checkpoint_snapshots(
        self,
        *,
        benchmark_path: Path,
        app_logs_path: Path,
        resume: bool,
        max_checkpoints: Optional[int],
        builder_save_every_logs: int,
        target_log_index: Optional[int] = None,
    ) -> Path:
        build_started_at = time.time()
        benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        checkpoints = [cp for cp in benchmark.get("checkpoints", []) if isinstance(cp, dict)]
        if max_checkpoints is not None:
            checkpoints = checkpoints[: max(0, int(max_checkpoints))]

        replay_target_idx = len(self.all_logs) - 1
        if target_log_index is not None:
            replay_target_idx = min(replay_target_idx, max(-1, int(target_log_index)))

        checkpoint_specs: List[Dict[str, Any]] = []
        checkpoint_specs_by_id: Dict[str, Dict[str, Any]] = {}
        checkpoint_by_cut_index: Dict[int, List[Dict[str, Any]]] = {}
        required_checkpoint_ids: Set[str] = set()
        for cp in checkpoints:
            checkpoint_id = str(cp.get("checkpoint_id") or "").strip()
            if not checkpoint_id:
                continue
            as_of = cp.get("as_of")
            as_of = as_of if isinstance(as_of, dict) else {}
            checkpoint_app_log_id = str(as_of.get("app_log_id") or "").strip()
            cut_index = _checkpoint_cut_index(cp, self.all_logs)
            spec = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_app_log_id": checkpoint_app_log_id,
                "cut_index": int(cut_index),
            }
            checkpoint_specs.append(spec)
            checkpoint_specs_by_id[checkpoint_id] = spec
            checkpoint_by_cut_index.setdefault(int(cut_index), []).append(spec)
            required_checkpoint_ids.add(checkpoint_id)

        builder_fingerprint = _build_builder_fingerprint(
            benchmark_path=benchmark_path,
            app_logs_path=app_logs_path,
            checkpoint_specs=checkpoint_specs,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            batch_size=self.batch_size,
            openie_mode=self.openie_mode,
        )
        workspace_dir = _builder_workspace_dir(self.save_root)
        periodic_root = _periodic_snapshots_root(self.snapshot_root)
        checkpoint_root = _checkpoints_root(self.snapshot_root)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        existing_entries_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in _load_checkpoint_manifest_entries(self.snapshot_root):
            checkpoint_id = str(entry.get("checkpoint_id") or "").strip()
            snapshot_path_raw = str(entry.get("snapshot_path") or "").strip()
            if not checkpoint_id or not snapshot_path_raw:
                continue
            snapshot_dir = self.snapshot_root / snapshot_path_raw
            model_dir = snapshot_dir / "{}_{}".format(self.llm_model, self.embedding_model)
            fingerprint_matches = str(entry.get("config_fingerprint") or "") == builder_fingerprint
            source_checkpoint_matches = _entry_matches_source_checkpoint(
                entry,
                checkpoint_specs_by_id.get(checkpoint_id, {}),
            )
            if snapshot_dir.exists() and model_dir.exists() and (fingerprint_matches or source_checkpoint_matches):
                existing_entries_by_id[checkpoint_id] = dict(entry)

        last_confirmed_idx = max(
            (int(entry.get("last_event_idx", -1)) for entry in existing_entries_by_id.values()),
            default=-1,
        )
        local_runtime_ready = workspace_dir.exists() and _builder_progress_path(self.save_root).exists()

        if (
            resume
            and existing_entries_by_id
            and last_confirmed_idx >= replay_target_idx
            and required_checkpoint_ids.issubset(existing_entries_by_id.keys())
        ):
            if not local_runtime_ready:
                latest_entry = max(
                    existing_entries_by_id.values(),
                    key=lambda item: (int(item.get("last_event_idx", -1)), str(item.get("snapshot_id", ""))),
                )
                _write_builder_progress(
                    self.save_root,
                    config_fingerprint=builder_fingerprint,
                    confirmed_last_log_index=int(latest_entry.get("last_event_idx", last_confirmed_idx)),
                    confirmed_last_app_log_id=str(latest_entry.get("last_app_log_id") or "").strip() or None,
                    active_workspace_path=workspace_dir,
                    status="complete",
                )
            return self.save_root

        if not resume:
            _reset_save_root(
                self.save_root,
                self.snapshot_root,
                allow_destructive_rebuild=self.allow_destructive_rebuild,
            )
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            existing_entries_by_id = {}
            last_confirmed_idx = -1

        saved_checkpoint_ids: Set[str] = set(existing_entries_by_id.keys())

        if resume:
            snapshot = None
            if existing_entries_by_id:
                snapshot = dict(
                    max(
                        existing_entries_by_id.values(),
                        key=lambda item: (int(item.get("last_event_idx", -1)), str(item.get("snapshot_id", ""))),
                    )
                )
                snapshot_path_raw = str(snapshot.get("snapshot_path") or "").strip()
                if snapshot_path_raw:
                    snapshot["snapshot_dir"] = str(self.snapshot_root / snapshot_path_raw)
                else:
                    snapshot = None
            if snapshot is None:
                snapshot = _nearest_resume_snapshot(
                    self.snapshot_root,
                    config_fingerprint=builder_fingerprint,
                    confirmed_last_log_index=max(last_confirmed_idx, replay_target_idx),
                )
            if snapshot is None:
                _reset_save_root(
                    self.save_root,
                    self.snapshot_root,
                    allow_destructive_rebuild=self.allow_destructive_rebuild,
                )
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                existing_entries_by_id = {}
                saved_checkpoint_ids = set()
                last_confirmed_idx = -1
                workspace_dir.mkdir(parents=True, exist_ok=True)
                hipporag = self._create_hipporag(workspace_dir, force_index_from_scratch=True)
                start_idx = 0
            else:
                snapshot_dir = Path(str(snapshot.get("snapshot_dir") or ""))
                _replace_dir_from(snapshot_dir, workspace_dir)
                restored_idx = int(snapshot.get("last_event_idx", -1))
                restored_app_log_id = str(snapshot.get("last_app_log_id") or "").strip() or None
                _write_builder_progress(
                    self.save_root,
                    config_fingerprint=builder_fingerprint,
                    confirmed_last_log_index=restored_idx,
                    confirmed_last_app_log_id=restored_app_log_id,
                    active_workspace_path=workspace_dir,
                    status="restored_from_snapshot",
                )
                last_confirmed_idx = restored_idx
                hipporag = self._create_hipporag(workspace_dir, force_index_from_scratch=False)
                start_idx = max(0, restored_idx + 1)
        else:
            workspace_dir.mkdir(parents=True, exist_ok=True)
            hipporag = self._create_hipporag(workspace_dir, force_index_from_scratch=True)
            start_idx = 0
            _write_builder_progress(
                self.save_root,
                config_fingerprint=builder_fingerprint,
                confirmed_last_log_index=-1,
                confirmed_last_app_log_id=None,
                active_workspace_path=workspace_dir,
                status="initialized",
            )

        try:
            while start_idx <= replay_target_idx:
                next_periodic_boundary: Optional[int] = None
                if builder_save_every_logs > 0:
                    next_periodic_boundary = ((start_idx // builder_save_every_logs) + 1) * builder_save_every_logs - 1
                    next_periodic_boundary = min(next_periodic_boundary, replay_target_idx)

                next_checkpoint_boundary: Optional[int] = None
                for spec in checkpoint_specs:
                    if spec["checkpoint_id"] in saved_checkpoint_ids:
                        continue
                    cut_index = int(spec["cut_index"])
                    if cut_index < start_idx:
                        continue
                    if next_checkpoint_boundary is None or cut_index < next_checkpoint_boundary:
                        next_checkpoint_boundary = cut_index

                batch_end = min(start_idx + self.batch_size - 1, replay_target_idx)
                if next_periodic_boundary is not None:
                    batch_end = min(batch_end, next_periodic_boundary)
                if next_checkpoint_boundary is not None:
                    batch_end = min(batch_end, next_checkpoint_boundary)

                docs = [json.dumps(log, ensure_ascii=False) for log in self.all_logs[start_idx : batch_end + 1]]
                if docs:
                    if self._usage_tracker is None:
                        hipporag.index(docs=docs)
                    else:
                        with self._usage_tracker.phase("build_memory"):
                            hipporag.index(docs=docs)

                last_confirmed_idx = batch_end
                confirmed_log = self.all_logs[last_confirmed_idx] if 0 <= last_confirmed_idx < len(self.all_logs) else {}
                confirmed_app_log_id = str(confirmed_log.get("app_log_id") or "").strip() or None
                _write_builder_progress(
                    self.save_root,
                    config_fingerprint=builder_fingerprint,
                    confirmed_last_log_index=last_confirmed_idx,
                    confirmed_last_app_log_id=confirmed_app_log_id,
                    active_workspace_path=workspace_dir,
                    status="confirmed",
                )
                if self._usage_tracker is not None:
                    self._usage_tracker.set_build_duration(time.time() - build_started_at)

                if builder_save_every_logs > 0 and (last_confirmed_idx + 1) % builder_save_every_logs == 0:
                    periodic_id = "periodic_{:08d}".format(last_confirmed_idx + 1)
                    periodic_dir = periodic_root / periodic_id
                    _persist_workspace_snapshot(
                        workspace_dir=workspace_dir,
                        snapshot_dir=periodic_dir,
                        snapshot_id=periodic_id,
                        trigger="periodic",
                        last_event_idx=last_confirmed_idx,
                        last_app_log_id=confirmed_app_log_id,
                        config_fingerprint=builder_fingerprint,
                    )

                for spec in checkpoint_by_cut_index.get(last_confirmed_idx, []):
                    checkpoint_id = str(spec.get("checkpoint_id") or "").strip()
                    if not checkpoint_id or checkpoint_id in saved_checkpoint_ids:
                        continue
                    checkpoint_dir = self._checkpoint_dir(checkpoint_id)
                    _persist_workspace_snapshot(
                        workspace_dir=workspace_dir,
                        snapshot_dir=checkpoint_dir,
                        snapshot_id=checkpoint_id,
                        trigger="checkpoint",
                        last_event_idx=last_confirmed_idx,
                        last_app_log_id=confirmed_app_log_id,
                        checkpoint_id=checkpoint_id,
                        checkpoint_app_log_id=str(spec.get("checkpoint_app_log_id") or "").strip() or None,
                        config_fingerprint=builder_fingerprint,
                    )
                    existing_entries_by_id[checkpoint_id] = {
                        "snapshot_id": checkpoint_id,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_app_log_id": str(spec.get("checkpoint_app_log_id") or "").strip(),
                        "snapshot_path": str(checkpoint_dir.relative_to(self.snapshot_root)),
                        "last_event_idx": int(last_confirmed_idx),
                        "config_fingerprint": builder_fingerprint,
                        "updated_at": datetime.now().isoformat(),
                    }
                    saved_checkpoint_ids.add(checkpoint_id)
                    _write_checkpoint_manifest_entries(
                        self.snapshot_root,
                        sorted(
                            existing_entries_by_id.values(),
                            key=lambda item: (int(item.get("last_event_idx", -1)), str(item.get("checkpoint_id") or "")),
                        ),
                    )

                start_idx = last_confirmed_idx + 1
        except Exception:
            interrupted_app_log_id = None
            if 0 <= last_confirmed_idx < len(self.all_logs):
                interrupted_app_log_id = str(self.all_logs[last_confirmed_idx].get("app_log_id") or "").strip() or None
            _write_builder_progress(
                self.save_root,
                config_fingerprint=builder_fingerprint,
                confirmed_last_log_index=last_confirmed_idx,
                confirmed_last_app_log_id=interrupted_app_log_id,
                active_workspace_path=workspace_dir,
                status="interrupted",
            )
            if self._usage_tracker is not None:
                self._usage_tracker.set_build_duration(time.time() - build_started_at)
            raise

        final_app_log_id = None
        if 0 <= last_confirmed_idx < len(self.all_logs):
            final_app_log_id = str(self.all_logs[last_confirmed_idx].get("app_log_id") or "").strip() or None
        if last_confirmed_idx >= replay_target_idx:
            final_snapshot_dir = periodic_root / "final_{:08d}".format(last_confirmed_idx + 1)
            _persist_workspace_snapshot(
                workspace_dir=workspace_dir,
                snapshot_dir=final_snapshot_dir,
                snapshot_id=final_snapshot_dir.name,
                trigger="final",
                last_event_idx=last_confirmed_idx,
                last_app_log_id=final_app_log_id,
                config_fingerprint=builder_fingerprint,
            )

        missing = sorted(required_checkpoint_ids - saved_checkpoint_ids)
        if missing:
            raise RuntimeError(
                "Failed to materialize HippoRAG2 checkpoint snapshots. Missing checkpoint ids: {}".format(
                    ", ".join(missing[:10])
                )
            )

        final_status = "complete" if last_confirmed_idx >= replay_target_idx else "confirmed"
        _write_builder_progress(
            self.save_root,
            config_fingerprint=builder_fingerprint,
            confirmed_last_log_index=last_confirmed_idx,
            confirmed_last_app_log_id=final_app_log_id,
            active_workspace_path=workspace_dir,
            status=final_status,
        )
        if self._usage_tracker is not None:
            self._usage_tracker.set_build_duration(time.time() - build_started_at)
        _write_checkpoint_manifest_entries(
            self.snapshot_root,
            sorted(
                existing_entries_by_id.values(),
                key=lambda item: (int(item.get("last_event_idx", -1)), str(item.get("checkpoint_id") or "")),
            ),
        )
        return self.save_root

    def _resolve_manifest_entry(self, cp: Dict[str, Any]) -> Dict[str, Any]:
        entries = _load_checkpoint_manifest_entries(self.snapshot_root)
        if not entries:
            raise FileNotFoundError(
                "HippoRAG2 checkpoint manifest not found or empty: {}".format(_checkpoint_manifest_path(self.snapshot_root))
            )

        checkpoint_id = str(cp.get("checkpoint_id") or "").strip()
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        checkpoint_app_log_id = str(as_of.get("app_log_id") or "").strip()
        for entry in entries:
            if checkpoint_id and str(entry.get("checkpoint_id") or "").strip() == checkpoint_id:
                return entry
        for entry in entries:
            if checkpoint_app_log_id and str(entry.get("checkpoint_app_log_id") or "").strip() == checkpoint_app_log_id:
                return entry
        raise FileNotFoundError(
            "HippoRAG2 checkpoint snapshot not found for checkpoint_id={} checkpoint_app_log_id={}".format(
                checkpoint_id or "<unknown>",
                checkpoint_app_log_id or "<unknown>",
            )
        )

    def _load_checkpoint_state(self, cp: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._resolve_manifest_entry(cp)
        snapshot_path_raw = str(entry.get("snapshot_path") or "").strip()
        if not snapshot_path_raw:
            raise FileNotFoundError("HippoRAG2 manifest entry missing snapshot_path for checkpoint {}".format(cp.get("checkpoint_id")))
        snapshot_dir = self.snapshot_root / snapshot_path_raw
        if not snapshot_dir.exists():
            raise FileNotFoundError("HippoRAG2 checkpoint snapshot not found: {}".format(snapshot_dir))
        hipporag = self._create_hipporag(snapshot_dir, force_index_from_scratch=False)
        cutoff_idx = entry.get("last_event_idx")
        if not isinstance(cutoff_idx, int):
            cutoff_idx = _checkpoint_cut_index(cp, self.all_logs)
        return {
            "checkpoint": cp,
            "checkpoint_dir": str(snapshot_dir),
            "hipporag": hipporag,
            "retrieval_lock": threading.Lock(),
            "cutoff_idx": int(cutoff_idx),
            "indexed_log_count": max(0, int(cutoff_idx) + 1),
            "manifest_entry": entry,
        }

    def prepare_checkpoint_state(
        self,
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
    ) -> CheckpointHandle:
        checkpoint_id = str(cp.get("checkpoint_id") or "")
        with self._prepared_states_lock:
            state = self._prepared_states.get(checkpoint_id)
            if state is None:
                state = self._load_checkpoint_state(cp)
                self._prepared_states[checkpoint_id] = state

        entry = state.get("manifest_entry") if isinstance(state.get("manifest_entry"), dict) else {}
        return CheckpointHandle(
            checkpoint_id=checkpoint_id,
            state_kind="hipporag2_checkpoint_snapshot",
            state_ref=state,
            metadata={
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp") or ""),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
                "checkpoint_dir": state.get("checkpoint_dir"),
                "indexed_log_count": state.get("indexed_log_count"),
                "snapshot_id": entry.get("snapshot_id"),
                "snapshot_path": entry.get("snapshot_path"),
            },
        )

    def retrieve_context_for_query(
        self,
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        query = str(query_spec.retrieval_query_text or "").strip()
        if not query:
            raise ValueError("HippoRAG2 requires shared QuerySpec.retrieval_query_text.")

        state = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        hipporag = state.get("hipporag")
        retrieval_lock = state.get("retrieval_lock")
        top_k_for_call = self.retrieval_top_k
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            top_k_for_call = int(top_k_override)

        retrieval_start = time.time()
        if hipporag is None:
            solutions = []
        elif retrieval_lock is None:
            if self._usage_tracker is None:
                solutions = hipporag.retrieve(queries=[query])
            else:
                with self._usage_tracker.phase("retrieval"):
                    solutions = hipporag.retrieve(queries=[query])
        else:
            with retrieval_lock:
                if self._usage_tracker is None:
                    solutions = hipporag.retrieve(queries=[query])
                else:
                    with self._usage_tracker.phase("retrieval"):
                        solutions = hipporag.retrieve(queries=[query])
        retrieval_duration = time.time() - retrieval_start
        if self._usage_tracker is not None:
            self._usage_tracker.write_live()

        visible_ids = {
            str(log.get("app_log_id") or "").strip()
            for log in memory_pool
            if str(log.get("app_log_id") or "").strip()
        }
        retrieved_logs: List[Dict[str, Any]] = []
        if solutions:
            raw_docs = list(getattr(solutions[0], "docs", []) or [])
            for doc_str in raw_docs:
                try:
                    log = json.loads(doc_str)
                except Exception:
                    continue
                app_log_id = str(log.get("app_log_id") or "").strip()
                if visible_ids and app_log_id and app_log_id not in visible_ids:
                    continue
                retrieved_logs.append(log)
                if top_k_for_call > 0 and len(retrieved_logs) >= top_k_for_call:
                    break

        retrieved_app_log_ids = [
            str(log.get("app_log_id") or "").strip()
            for log in retrieved_logs
            if str(log.get("app_log_id") or "").strip()
        ]

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in retrieved_logs],
            debug_metadata={
                "retrieval_mode": "hipporag2_checkpoint_snapshot",
                "retrieval_duration_s": retrieval_duration,
                "retrieval_query": query,
                "retrieval_top_k": top_k_for_call,
                "retriever_model": self.embedding_model,
                "retrieved_app_log_ids": retrieved_app_log_ids,
                "num_input_context_logs": len(retrieved_logs),
                "input_context_app_log_ids": retrieved_app_log_ids,
                "checkpoint_dir": state.get("checkpoint_dir"),
                "cutoff_idx": state.get("cutoff_idx"),
                "indexed_log_count": state.get("indexed_log_count"),
            },
        )

    def finalize_checkpoint_state(self, checkpoint_handle: CheckpointHandle) -> None:
        checkpoint_id = str(getattr(checkpoint_handle, "checkpoint_id", "") or "")
        if checkpoint_id:
            with self._prepared_states_lock:
                self._prepared_states.pop(checkpoint_id, None)
        return None


def run_generation(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    snapshot_dir: Path,
    data_storage_path: Path,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    retriever_provider: str,
    retriever_model: str,
    retriever_batch_size: int,
    retrieval_top_k: int,
    builder_save_every_logs: int = 5,
    interleave_build_and_test: bool = False,
    build_only: bool = False,
    predict_from_prebuilt: bool = False,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    task_selection: str = "all",
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
    allow_destructive_rebuild: bool = False,
) -> Dict[str, Any]:
    benchmark_path = benchmark_path.expanduser().resolve()
    app_logs_path = app_logs_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    data_storage_path = data_storage_path.expanduser().resolve()
    if build_only and predict_from_prebuilt:
        raise ValueError("HippoRAG2 build_only and predict_from_prebuilt cannot both be true.")
    provider_env = _configure_hipporag_openai_env(
        llm_provider=llm_provider,
        retriever_provider=retriever_provider,
    )
    usage_tracker = HippoRAG2UsageTracker(
        output_path=output_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        retriever_provider=retriever_provider,
        embedding_model_name=retriever_model,
    )

    client = None

    def _ask_json(prompt: str) -> Any:
        if client is None:
            raise RuntimeError("HippoRAG2 answer client is not initialized.")
        started_at = time.time()
        response = client.ask(prompt, response_type="json")
        usage_tracker.add_generation_duration(time.time() - started_at)
        usage_tracker.set_answer_usage(client.usage_summary())
        return response

    def _ask_structured(prompt: str, fmt: Any) -> Any:
        if client is None:
            raise RuntimeError("HippoRAG2 answer client is not initialized.")
        started_at = time.time()
        response = client.ask_structured(prompt, text_format=fmt)
        usage_tracker.add_generation_duration(time.time() - started_at)
        usage_tracker.set_answer_usage(client.usage_summary())
        return response

    def _current_answer_usage() -> Dict[str, Any]:
        if client is None:
            return {}
        return client.usage_summary()

    all_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    requested_checkpoints = [cp for cp in benchmark_payload.get("checkpoints", []) if isinstance(cp, dict)]
    if max_checkpoints is not None:
        requested_checkpoints = requested_checkpoints[: max(0, int(max_checkpoints))]
    build_only_target_idx: Optional[int] = None
    if build_only and requested_checkpoints:
        build_only_target_idx = _checkpoint_cut_index(requested_checkpoints[-1], all_logs)

    runner = HippoRAG2Runner(
        data_storage_root=Path(data_storage_path),
        snapshot_root=Path(snapshot_dir),
        all_logs=all_logs,
        llm_model=llm_model,
        llm_base_url=provider_env.get("llm_base_url"),
        embedding_model=retriever_model,
        embedding_base_url=provider_env.get("retriever_base_url"),
        batch_size=retriever_batch_size,
        retrieval_top_k=retrieval_top_k,
        openie_mode=_detect_openie_mode(llm_model),
        allow_destructive_rebuild=allow_destructive_rebuild,
        usage_tracker=usage_tracker,
    )

    def _run_pipeline_once(
        *,
        pipeline_resume: bool,
        pipeline_max_checkpoints: Optional[int],
        pipeline_enable_final_qa: bool,
    ) -> Dict[str, Any]:
        return run_pipeline(
            benchmark_path=benchmark_path,
            app_logs_path=app_logs_path,
            output_path=output_path,
            max_visible_logs=max_visible_logs,
            ask_json=_ask_json,
            ask_structured=_ask_structured,
            use_structured_response=client.supports_structured_response(),
            close=lambda: None,
            prepare_checkpoint_state=runner.prepare_checkpoint_state,
            retrieve_context_for_query=runner.retrieve_context_for_query,
            finalize_checkpoint_state=runner.finalize_checkpoint_state,
            baseline_name="hipporag2",
            memory_prompt_mode="inline_memory",
            resume=pipeline_resume,
            max_checkpoints=pipeline_max_checkpoints,
            debug=debug,
            debug_dir=debug_dir,
            save_prompt_and_raw=save_prompt_and_raw,
            enable_change_reasoning=enable_change_reasoning,
            enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
            rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
            task_selection=task_selection,
            rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
            retrieval_options_backend={
                "retriever_provider": retriever_provider,
                "retriever_model": retriever_model,
                "retriever_batch_size": retriever_batch_size,
                "snapshot_dir": str(snapshot_dir),
                "data_storage_path": str(data_storage_path),
                "builder_save_every_logs": max(1, int(builder_save_every_logs)),
                "interleave_build_and_test": bool(interleave_build_and_test),
                "build_only": bool(build_only),
                "predict_from_prebuilt": bool(predict_from_prebuilt),
            },
            checkpoint_workers=checkpoint_workers,
            within_checkpoint_workers=within_checkpoint_workers,
            save_every_generation_keys=save_every_generation_keys,
            enable_final_qa=pipeline_enable_final_qa,
            final_qa_path=final_qa_path,
            final_qa_output_path=final_qa_output_path,
            final_qa_retrieval_top_k=final_qa_retrieval_top_k,
            final_qa_save_prompt_and_raw=final_qa_save_prompt_and_raw,
        )

    try:
        try:
            if build_only:
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=resume,
                    max_checkpoints=max_checkpoints,
                    builder_save_every_logs=max(1, int(builder_save_every_logs)),
                    target_log_index=build_only_target_idx,
                )
                result = {
                    "predictions": [],
                    "build_only": {
                        "enabled": True,
                        "snapshot_dir": str(snapshot_dir),
                        "data_storage_path": str(data_storage_path),
                        "target_log_index": build_only_target_idx,
                        "requested_checkpoint_ids": [
                            str(cp.get("checkpoint_id") or "").strip()
                            for cp in requested_checkpoints
                            if str(cp.get("checkpoint_id") or "").strip()
                        ],
                    },
                }
                usage_tracker.set_answer_usage(_current_answer_usage())
                usage_tracker.finalize()
                return result
            elif predict_from_prebuilt:
                client = LLMClient(
                    provider=llm_provider,
                    model_name=llm_model,
                    max_workers=llm_max_workers,
                    temperature=answer_temperature,
                    top_p=answer_top_p,
                    top_k=answer_top_k,
                )
                result = _run_pipeline_once(
                    pipeline_resume=resume,
                    pipeline_max_checkpoints=max_checkpoints,
                    pipeline_enable_final_qa=enable_final_qa,
                )
            elif interleave_build_and_test and requested_checkpoints:
                client = LLMClient(
                    provider=llm_provider,
                    model_name=llm_model,
                    max_workers=llm_max_workers,
                    temperature=answer_temperature,
                    top_p=answer_top_p,
                    top_k=answer_top_k,
                )
                result: Dict[str, Any] = {"predictions": []}
                for idx, checkpoint in enumerate(requested_checkpoints):
                    checkpoint_limit = idx + 1
                    target_log_index = _checkpoint_cut_index(checkpoint, all_logs)
                    runner.materialize_checkpoint_snapshots(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        resume=resume if idx == 0 else True,
                        max_checkpoints=checkpoint_limit,
                        builder_save_every_logs=max(1, int(builder_save_every_logs)),
                        target_log_index=target_log_index,
                    )
                    result = _run_pipeline_once(
                        pipeline_resume=resume if idx == 0 else True,
                        pipeline_max_checkpoints=checkpoint_limit,
                        pipeline_enable_final_qa=bool(enable_final_qa and idx == len(requested_checkpoints) - 1),
                    )
            else:
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=resume,
                    max_checkpoints=max_checkpoints,
                    builder_save_every_logs=max(1, int(builder_save_every_logs)),
                )
                client = LLMClient(
                    provider=llm_provider,
                    model_name=llm_model,
                    max_workers=llm_max_workers,
                    temperature=answer_temperature,
                    top_p=answer_top_p,
                    top_k=answer_top_k,
                )
                result = _run_pipeline_once(
                    pipeline_resume=resume,
                    pipeline_max_checkpoints=max_checkpoints,
                    pipeline_enable_final_qa=enable_final_qa,
                )
        except Exception:
            usage_tracker.set_answer_usage(_current_answer_usage())
            usage_tracker.finalize()
            raise

        usage_tracker.set_answer_usage(_current_answer_usage())
        usage_tracker.finalize()
        return result
    finally:
        if client is not None:
            client.close()


run_online_generation = run_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-aligned HippoRAG2 TCE runtime")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--data-storage-path", type=Path, required=True)
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-top-k", type=int, default=None)
    parser.add_argument("--retriever-provider", type=str, default="openai")
    parser.add_argument("--retriever-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--retriever-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--builder-save-every-logs", type=int, default=5)
    parser.add_argument("--interleave-build-and-test", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--predict-from-prebuilt", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--enable-change-reasoning", action="store_true")
    parser.add_argument("--enable-rq3-apply-service-qa", action="store_true")
    parser.add_argument(
        "--rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_true",
    )
    parser.add_argument(
        "--no-rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_false",
    )
    parser.set_defaults(rq3_apply_save_prompt_and_raw=True)
    parser.add_argument("--rq3-apply-top-k", type=int, default=None)
    parser.add_argument("--checkpoint-workers", type=int, default=1)
    parser.add_argument("--within-checkpoint-workers", type=int, default=1)
    parser.add_argument("--save-every-generation-keys", type=int, default=1)
    parser.add_argument("--enable-final-qa", action="store_true")
    parser.add_argument("--final-qa-path", type=str, default=None)
    parser.add_argument("--final-qa-output-path", type=str, default=None)
    parser.add_argument("--final-qa-top-k", type=int, default=None)
    parser.add_argument("--final-qa-save-prompt-and-raw", action="store_true")
    args = parser.parse_args()

    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        snapshot_dir=args.snapshot_dir,
        data_storage_path=args.data_storage_path,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
        retrieval_top_k=args.retrieval_top_k,
        builder_save_every_logs=args.builder_save_every_logs,
        interleave_build_and_test=args.interleave_build_and_test,
        build_only=args.build_only,
        predict_from_prebuilt=args.predict_from_prebuilt,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        enable_change_reasoning=args.enable_change_reasoning,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=args.rq3_apply_top_k,
        checkpoint_workers=args.checkpoint_workers,
        within_checkpoint_workers=args.within_checkpoint_workers,
        save_every_generation_keys=args.save_every_generation_keys,
        enable_final_qa=args.enable_final_qa,
        final_qa_path=args.final_qa_path,
        final_qa_output_path=args.final_qa_output_path,
        final_qa_retrieval_top_k=args.final_qa_top_k,
        final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
    )


if __name__ == "__main__":
    main()
