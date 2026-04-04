#!/usr/bin/env python3
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs):
        return False

from generation.common.llm_client import LLMClient
from generation.common.provider_config import resolve_openai_compatible_credentials
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import normalize_app_logs, run_pipeline, to_log_text


ASSISTANT_ID = "assistant"
BUILDER_SAVE_EVERY_LOGS = 5


def _module_dir() -> Path:
    return Path(__file__).resolve().parent / "memoryos-pypi"


load_dotenv(_module_dir() / ".env")


def _load_memoryos_class():
    module_dir = _module_dir()
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    from memoryos import Memoryos  # type: ignore

    return Memoryos


def _load_memoryos_utils_module():
    module_dir = _module_dir()
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    return importlib.import_module("utils")


def _reset_memoryos_usage_tracker() -> None:
    try:
        mod = _load_memoryos_utils_module()
        reset_fn = getattr(mod, "reset_usage_tracker", None)
        if callable(reset_fn):
            reset_fn()
    except Exception:
        return None


def _memoryos_usage_summary() -> Dict[str, Any]:
    try:
        mod = _load_memoryos_utils_module()
        summary_fn = getattr(mod, "get_usage_summary", None)
        if callable(summary_fn):
            summary = summary_fn()
            return summary if isinstance(summary, dict) else {}
    except Exception:
        return {}
    return {}


def _usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost.json"


def _load_usage_cost_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_usage_summary(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
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
    by_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
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


def _write_usage_cost_sidecar(
    *,
    sidecar_path: Path,
    output_path: Path,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    embedding_model_name: str,
    build_duration_s: Optional[float] = None,
    generation_duration_s: Optional[float] = None,
    total_duration_s: Optional[float] = None,
    build_memory_usage: Optional[Dict[str, Any]] = None,
    retrieval_usage: Optional[Dict[str, Any]] = None,
    answer_llm_usage: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    base_payload: Optional[Dict[str, Any]] = None,
) -> None:
    existing_payload: Dict[str, Any] = dict(base_payload or {})

    existing_timing = existing_payload.get("timing")
    if not isinstance(existing_timing, dict):
        existing_timing = {}
    timing = {
        "build_memory_duration_s": float(existing_timing.get("build_memory_duration_s") or 0.0),
        "generation_duration_s": float(existing_timing.get("generation_duration_s") or 0.0),
        "total_duration_s": float(existing_timing.get("total_duration_s") or 0.0),
    }
    if build_duration_s is not None:
        timing["build_memory_duration_s"] += float(build_duration_s)
    if generation_duration_s is not None:
        timing["generation_duration_s"] += float(generation_duration_s)
    if total_duration_s is not None:
        timing["total_duration_s"] += float(total_duration_s)

    existing_usage = existing_payload.get("usage")
    if not isinstance(existing_usage, dict):
        existing_usage = {}
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


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _memory_user_id(user_id: str, size: str) -> str:
    return f"{user_id}_{size}"


def _runtime_data_root(data_storage_path: Optional[str]) -> Path:
    if data_storage_path:
        return Path(data_storage_path).expanduser().resolve()
    return Path(__file__).resolve().parent / "runtime_data"


def _snapshot_root(snapshot_dir: Optional[str], memory_user_id: str) -> Path:
    base = Path(snapshot_dir).expanduser().resolve() if snapshot_dir else Path(__file__).resolve().parent / "snapshots"
    return base / memory_user_id


def _load_manifest_entries(snapshot_root: Path) -> List[Dict[str, Any]]:
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots", []) if isinstance(manifest, dict) else []
    return [x for x in snapshots if isinstance(x, dict)]


def _existing_checkpoint_ids(snapshot_root: Path) -> Set[str]:
    out: Set[str] = set()
    for entry in _load_manifest_entries(snapshot_root):
        checkpoint_id = str(entry.get("checkpoint_id", "")).strip()
        if checkpoint_id:
            out.add(checkpoint_id)
    return out


def _snapshot_sort_key(entry: Dict[str, Any]) -> Tuple[int, str, str]:
    return (
        int(entry.get("last_event_idx", -1)),
        str(entry.get("created_at", "")),
        str(entry.get("snapshot_id", "")),
    )


def _latest_snapshot_entry(snapshot_root: Path) -> Optional[Dict[str, Any]]:
    snapshots = _load_manifest_entries(snapshot_root)
    if not snapshots:
        return None
    snapshots.sort(key=_snapshot_sort_key, reverse=True)
    return snapshots[0]


def _snapshot_checkpoint_payload(snapshot_root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = str(entry.get("snapshot_id") or "")
    checkpoint_path = snapshot_root / entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json"))
    if not checkpoint_path.exists():
        return {}
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_snapshot_bundle(
    *,
    memory_system: Any,
    snapshot_root: Path,
    last_event_idx: int,
    checkpoint_id: Optional[str],
    checkpoint_app_log_id: Optional[str],
    trigger: str = "checkpoint",
) -> Dict[str, Any]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat()
    snapshot_id = f"snap_evt{last_event_idx:07d}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    snapshot_dir = snapshot_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    short_term_path = snapshot_dir / "short_term.json"
    mid_term_path = snapshot_dir / "mid_term.json"
    long_term_path = snapshot_dir / "long_term_user.json"
    assistant_long_term_path = snapshot_dir / "long_term_assistant.json"
    checkpoint_path = snapshot_dir / "checkpoint.json"

    _atomic_write_json(short_term_path, list(memory_system.short_term_memory.memory))
    _atomic_write_json(
        mid_term_path,
        {
            "sessions": memory_system.mid_term_memory.sessions,
            "access_frequency": dict(memory_system.mid_term_memory.access_frequency),
        },
    )
    _atomic_write_json(
        long_term_path,
        {
            "user_profiles": memory_system.user_long_term_memory.user_profiles,
            "knowledge_base": list(memory_system.user_long_term_memory.knowledge_base),
            "assistant_knowledge": list(memory_system.user_long_term_memory.assistant_knowledge),
        },
    )
    _atomic_write_json(
        assistant_long_term_path,
        {
            "user_profiles": memory_system.assistant_long_term_memory.user_profiles,
            "knowledge_base": list(memory_system.assistant_long_term_memory.knowledge_base),
            "assistant_knowledge": list(memory_system.assistant_long_term_memory.assistant_knowledge),
        },
    )
    _atomic_write_json(
        checkpoint_path,
        {
            "last_event_idx": int(last_event_idx),
            "saved_at": created_at,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": checkpoint_app_log_id,
            "trigger": str(trigger or "checkpoint"),
            "builder_state": {
                "last_evicted_page_for_continuity": getattr(
                    getattr(memory_system, "updater", None),
                    "last_evicted_page_for_continuity",
                    None,
                ),
            },
        },
    )

    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"snapshots": []}
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []
    entry = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "last_event_idx": int(last_event_idx),
        "checkpoint_id": checkpoint_id,
        "checkpoint_app_log_id": checkpoint_app_log_id,
        "trigger": str(trigger or "checkpoint"),
        "short_term_path": str(Path(snapshot_id) / "short_term.json"),
        "mid_term_path": str(Path(snapshot_id) / "mid_term.json"),
        "long_term_path": str(Path(snapshot_id) / "long_term_user.json"),
        "assistant_long_term_path": str(Path(snapshot_id) / "long_term_assistant.json"),
        "checkpoint_path": str(Path(snapshot_id) / "checkpoint.json"),
    }
    snapshots.append(entry)
    manifest["snapshots"] = snapshots
    manifest["latest_snapshot"] = snapshot_id
    _atomic_write_json(manifest_path, manifest)
    return entry


def _build_checkpoint_trigger_maps(
    checkpoints: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], List[str]]:
    checkpoint_by_app_log_id: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_by_log_index: Dict[int, List[Dict[str, Any]]] = {}
    ordered_ids: List[str] = []
    for cp in checkpoints:
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        if checkpoint_id:
            ordered_ids.append(checkpoint_id)
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        app_log_id_raw = as_of.get("app_log_id")
        if app_log_id_raw is not None:
            checkpoint_by_app_log_id.setdefault(str(app_log_id_raw), []).append(cp)
        log_index_raw = as_of.get("log_index")
        if isinstance(log_index_raw, int) and log_index_raw >= 0:
            checkpoint_by_log_index.setdefault(log_index_raw, []).append(cp)
    return checkpoint_by_app_log_id, checkpoint_by_log_index, ordered_ids


def _build_memoryos_instance(
    *,
    memory_user_id: str,
    data_storage_root: Path,
    llm_controller_model: str,
    embedding_model_name: str,
    assistant_id: str,
    retriever_provider: str = "openai",
) -> Any:
    Memoryos = _load_memoryos_class()
    provider = str(retriever_provider or "").strip().lower()
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_api_base = os.getenv("EMBEDDING_API_BASE_URL")
    if not embedding_api_key or not embedding_api_base:
        resolved_key, resolved_base = resolve_openai_compatible_credentials(provider or "openai", require_api_key=False)
        embedding_api_key = embedding_api_key or resolved_key
        embedding_api_base = embedding_api_base or resolved_base
    embedding_backend = "openai" if provider in {"", "openai", "azure"} else provider
    return Memoryos(
        user_id=memory_user_id,
        openai_api_key=os.getenv("LLM_CONTROLLER_API_KEY"),
        openai_base_url=os.getenv("LLM_CONTROLLER_API_BASE_URL"),
        data_storage_path=str(data_storage_root),
        llm_model=llm_controller_model,
        assistant_id=assistant_id,
        short_term_capacity=7,
        mid_term_heat_threshold=5,
        retrieval_queue_capacity=10,
        long_term_knowledge_capacity=100,
        mid_term_similarity_threshold=0.6,
        embedding_model_name=embedding_model_name,
        embedding_model_kwargs={
            "embedding_backend": embedding_backend,
            "api_key": embedding_api_key,
            "api_base": embedding_api_base,
            "max_input_tokens": 8000,
            "truncate_from": "end",
        },
    )


def _ensure_snapshots_for_benchmark(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    user_id: str,
    size: str,
    snapshot_dir: Optional[str],
    data_storage_path: Optional[str],
    llm_controller_model: str,
    embedding_model_name: str,
    assistant_id: str,
    max_checkpoints: Optional[int],
    retriever_provider: str = "openai",
    resume: bool = False,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Path:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checkpoints = [cp for cp in benchmark.get("checkpoints", []) if isinstance(cp, dict)]
    if max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, int(max_checkpoints))]

    memory_user_id = _memory_user_id(user_id, size)
    snapshot_root = _snapshot_root(snapshot_dir, memory_user_id)
    required_checkpoint_ids = {
        str(cp.get("checkpoint_id", "")).strip()
        for cp in checkpoints
        if str(cp.get("checkpoint_id", "")).strip()
    }
    if required_checkpoint_ids and required_checkpoint_ids.issubset(_existing_checkpoint_ids(snapshot_root)):
        return snapshot_root

    runtime_root = _runtime_data_root(data_storage_path)
    builder_data_root = runtime_root / memory_user_id / "builder"
    checkpoint_by_app_log_id, checkpoint_by_log_index, _ordered_ids = _build_checkpoint_trigger_maps(checkpoints)
    app_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    latest_entry = _latest_snapshot_entry(snapshot_root) if resume else None
    latest_payload = _snapshot_checkpoint_payload(snapshot_root, latest_entry) if latest_entry else {}

    start_idx = 0
    last_saved_event_idx = -1
    saved_checkpoint_ids: Set[str] = set()

    if latest_entry is not None:
        if builder_data_root.exists():
            shutil.rmtree(builder_data_root)
        memo = _build_memoryos_instance(
            memory_user_id=memory_user_id,
            data_storage_root=builder_data_root,
            llm_controller_model=llm_controller_model,
            retriever_provider=retriever_provider,
            embedding_model_name=embedding_model_name,
            assistant_id=assistant_id,
        )
        shutil.copyfile(Path(snapshot_root) / latest_entry["short_term_path"], Path(memo.short_term_memory.file_path))
        shutil.copyfile(Path(snapshot_root) / latest_entry["mid_term_path"], Path(memo.mid_term_memory.file_path))
        shutil.copyfile(Path(snapshot_root) / latest_entry["long_term_path"], Path(memo.user_long_term_memory.file_path))
        shutil.copyfile(
            Path(snapshot_root) / latest_entry["assistant_long_term_path"],
            Path(memo.assistant_long_term_memory.file_path),
        )
        memo.short_term_memory.load()
        memo.mid_term_memory.load()
        memo.user_long_term_memory.load()
        memo.assistant_long_term_memory.load()
        builder_state = latest_payload.get("builder_state")
        if isinstance(builder_state, dict):
            memo.updater.last_evicted_page_for_continuity = builder_state.get("last_evicted_page_for_continuity")
        last_saved_event_idx = int(latest_payload.get("last_event_idx", latest_entry.get("last_event_idx", -1)))
        start_idx = max(0, last_saved_event_idx + 1)
        saved_checkpoint_ids = _existing_checkpoint_ids(snapshot_root)
    else:
        if builder_data_root.exists():
            shutil.rmtree(builder_data_root)
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        memo = _build_memoryos_instance(
            memory_user_id=memory_user_id,
            data_storage_root=builder_data_root,
            llm_controller_model=llm_controller_model,
            retriever_provider=retriever_provider,
            embedding_model_name=embedding_model_name,
            assistant_id=assistant_id,
        )

    last_processed_idx = start_idx - 1
    last_processed_log_id: Optional[str] = None
    try:
        for idx in range(start_idx, len(app_logs)):
            log = app_logs[idx]
            memo.add_memory(
                user_input=to_log_text(log),
                agent_response="[ingested]",
                timestamp=str(log.get("timestamp") or ""),
            )
            last_processed_idx = idx
            log_id = str(log.get("app_log_id", "")).strip()
            last_processed_log_id = log_id or None
            if callable(progress_callback):
                progress_callback()
            if BUILDER_SAVE_EVERY_LOGS > 0 and (idx + 1) % BUILDER_SAVE_EVERY_LOGS == 0 and idx > last_saved_event_idx:
                _write_snapshot_bundle(
                    memory_system=memo,
                    snapshot_root=snapshot_root,
                    last_event_idx=idx,
                    checkpoint_id=None,
                    checkpoint_app_log_id=log_id or None,
                    trigger="periodic",
                )
                last_saved_event_idx = idx
                if callable(progress_callback):
                    progress_callback()
            triggered = []
            triggered.extend(checkpoint_by_log_index.get(idx, []))
            if log_id:
                triggered.extend(checkpoint_by_app_log_id.get(log_id, []))
            for cp in triggered:
                checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
                if not checkpoint_id or checkpoint_id in saved_checkpoint_ids:
                    continue
                _write_snapshot_bundle(
                    memory_system=memo,
                    snapshot_root=snapshot_root,
                    last_event_idx=idx,
                    checkpoint_id=checkpoint_id,
                    checkpoint_app_log_id=log_id or None,
                    trigger="checkpoint",
                )
                saved_checkpoint_ids.add(checkpoint_id)
                last_saved_event_idx = max(last_saved_event_idx, idx)
                if callable(progress_callback):
                    progress_callback()
    except BaseException:
        if last_processed_idx > last_saved_event_idx:
            _write_snapshot_bundle(
                memory_system=memo,
                snapshot_root=snapshot_root,
                last_event_idx=last_processed_idx,
                checkpoint_id=None,
                checkpoint_app_log_id=last_processed_log_id,
                trigger="interrupt",
            )
            if callable(progress_callback):
                progress_callback()
        raise

    if last_processed_idx > last_saved_event_idx:
        _write_snapshot_bundle(
            memory_system=memo,
            snapshot_root=snapshot_root,
            last_event_idx=last_processed_idx,
            checkpoint_id=None,
            checkpoint_app_log_id=last_processed_log_id,
            trigger="final",
        )
        if callable(progress_callback):
            progress_callback()

    missing = sorted(required_checkpoint_ids - saved_checkpoint_ids)
    if missing:
        raise RuntimeError(
            "Failed to materialize MemoryOS checkpoint snapshots for benchmark. Missing checkpoint ids: {}".format(
                ", ".join(missing[:10])
            )
        )
    return snapshot_root


def _load_snapshot_bundle(snapshot_root: Path, cp: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = _load_manifest_entries(snapshot_root)
    if not snapshots:
        raise FileNotFoundError(f"MemoryOS snapshot manifest not found or empty: {snapshot_root / 'manifest.json'}")

    checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
    as_of = cp.get("as_of")
    as_of = as_of if isinstance(as_of, dict) else {}
    checkpoint_app_log_id = (
        str(as_of.get("app_log_id")).strip() if as_of.get("app_log_id") is not None else ""
    )

    candidates: List[Dict[str, Any]] = []
    for entry in snapshots:
        if checkpoint_id and str(entry.get("checkpoint_id", "")).strip() == checkpoint_id:
            candidates.append(entry)
    if not candidates and checkpoint_app_log_id:
        for entry in snapshots:
            if str(entry.get("checkpoint_app_log_id", "")).strip() == checkpoint_app_log_id:
                candidates.append(entry)
    if not candidates:
        candidates = snapshots

    candidates.sort(
        key=lambda x: (int(x.get("last_event_idx", -1)), str(x.get("created_at", ""))),
        reverse=True,
    )
    latest = candidates[0]
    snapshot_id = str(latest.get("snapshot_id"))
    return {
        "snapshot_id": snapshot_id,
        "checkpoint_id": latest.get("checkpoint_id"),
        "checkpoint_app_log_id": latest.get("checkpoint_app_log_id"),
        "short_term_path": snapshot_root / latest.get("short_term_path", str(Path(snapshot_id) / "short_term.json")),
        "mid_term_path": snapshot_root / latest.get("mid_term_path", str(Path(snapshot_id) / "mid_term.json")),
        "long_term_path": snapshot_root / latest.get("long_term_path", str(Path(snapshot_id) / "long_term_user.json")),
        "assistant_long_term_path": snapshot_root / latest.get(
            "assistant_long_term_path",
            str(Path(snapshot_id) / "long_term_assistant.json"),
        ),
    }


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    user_id: str,
    size: str,
    snapshot_dir: Optional[str],
    data_storage_path: Optional[str],
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    embedding_model_name: str,
    llm_controller_model: str,
    assistant_id: str,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    retriever_provider: str = "openai",
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
) -> Dict[str, Any]:
    memory_user_id = _memory_user_id(user_id, size)
    sidecar_path = _usage_cost_sidecar_path(output_path)
    base_sidecar_payload = _load_usage_cost_sidecar(sidecar_path) if resume else {}
    if not resume and sidecar_path.exists():
        try:
            sidecar_path.unlink()
        except Exception:
            pass
    overall_start = time.perf_counter()
    last_progress_write_s = 0.0

    def _should_write_progress(force: bool = False) -> bool:
        nonlocal last_progress_write_s
        now = time.perf_counter()
        if not force and (now - last_progress_write_s) < 5.0:
            return False
        last_progress_write_s = now
        return True

    _reset_memoryos_usage_tracker()
    build_start = time.perf_counter()
    build_duration_s = 0.0
    build_memory_usage: Dict[str, Any] = {}

    def _persist_build_progress(force: bool = False) -> None:
        if not _should_write_progress(force=force):
            return
        _write_usage_cost_sidecar(
            sidecar_path=sidecar_path,
            output_path=output_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            retriever_provider=retriever_provider,
            embedding_model_name=embedding_model_name,
            build_duration_s=time.perf_counter() - build_start,
            build_memory_usage=_memoryos_usage_summary(),
            total_duration_s=time.perf_counter() - overall_start,
            resume=resume,
            base_payload=base_sidecar_payload,
        )

    build_writer_stop = threading.Event()

    def _build_progress_writer() -> None:
        while not build_writer_stop.wait(5.0):
            try:
                _persist_build_progress(force=True)
            except Exception:
                continue

    build_writer = threading.Thread(
        target=_build_progress_writer,
        name="memoryos-build-progress-writer",
        daemon=True,
    )
    build_writer.start()
    try:
        _persist_build_progress(force=True)
        snapshot_root = _ensure_snapshots_for_benchmark(
            benchmark_path=benchmark_path,
            app_logs_path=app_logs_path,
            user_id=user_id,
            size=size,
            snapshot_dir=snapshot_dir,
            data_storage_path=data_storage_path,
            llm_controller_model=llm_controller_model,
            retriever_provider=retriever_provider,
            embedding_model_name=embedding_model_name,
            assistant_id=assistant_id,
            max_checkpoints=max_checkpoints,
            resume=resume,
            progress_callback=_persist_build_progress,
        )
    finally:
        build_writer_stop.set()
        build_writer.join(timeout=1.0)
    build_duration_s = time.perf_counter() - build_start
    build_memory_usage = _memoryos_usage_summary()
    _write_usage_cost_sidecar(
        sidecar_path=sidecar_path,
        output_path=output_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        retriever_provider=retriever_provider,
        embedding_model_name=embedding_model_name,
        build_duration_s=build_duration_s,
        build_memory_usage=build_memory_usage,
        total_duration_s=build_duration_s,
        resume=resume,
        base_payload=base_sidecar_payload,
    )
    base_sidecar_payload = _load_usage_cost_sidecar(sidecar_path)
    last_progress_write_s = 0.0

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    def ask_json(prompt: str) -> Any:
        response = client.ask(prompt, response_type="json")
        _persist_generation_progress()
        return response

    def ask_structured(prompt: str, text_format: Any) -> Any:
        response = client.ask_structured(prompt, text_format=text_format)
        _persist_generation_progress()
        return response

    def close() -> None:
        client.close()

    _reset_memoryos_usage_tracker()
    generation_start = time.perf_counter()

    def _persist_generation_progress(force: bool = False) -> None:
        if not _should_write_progress(force=force):
            return
        answer_llm_usage = {}
        usage_summary_fn = getattr(client, "usage_summary", None)
        if callable(usage_summary_fn):
            raw_answer_usage = usage_summary_fn()
            if isinstance(raw_answer_usage, dict):
                answer_llm_usage = raw_answer_usage
        _write_usage_cost_sidecar(
            sidecar_path=sidecar_path,
            output_path=output_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            retriever_provider=retriever_provider,
            embedding_model_name=embedding_model_name,
            generation_duration_s=time.perf_counter() - generation_start,
            total_duration_s=time.perf_counter() - overall_start,
            retrieval_usage=_memoryos_usage_summary(),
            answer_llm_usage=answer_llm_usage,
            resume=resume,
            base_payload=base_sidecar_payload,
        )

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        bundle = _load_snapshot_bundle(snapshot_root, cp)
        return CheckpointHandle(
            checkpoint_id=str(cp.get("checkpoint_id") or ""),
            state_kind="snapshot_bundle",
            state_ref=bundle,
            metadata={
                "retrieval_mode": "memoryos_snapshot",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        bundle = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        retrieval_query = str(query_spec.retrieval_query_text or "").strip()
        if not retrieval_query:
            raise ValueError("MemoryOS requires shared QuerySpec.retrieval_query_text.")
        top_k_for_call = retrieval_top_k
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            try:
                top_k_for_call = int(top_k_override)
            except Exception:
                top_k_for_call = retrieval_top_k
        k = len(memory_pool) if top_k_for_call <= 0 else min(top_k_for_call, len(memory_pool))

        if k <= 0:
            result = RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "memoryos_snapshot",
                    "snapshot_id": bundle.get("snapshot_id"),
                    "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                    "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "retriever_provider": retriever_provider,
                    "embedding_model_name": embedding_model_name,
                    "retrieval_query": retrieval_query,
                },
            )
            _persist_generation_progress()
            return result

        memory_pool_by_id: Dict[str, Dict[str, Any]] = {}
        for i, log in enumerate(memory_pool):
            app_log_id = log.get("app_log_id")
            key = str(app_log_id).strip() if app_log_id is not None else f"pool_{i}"
            if key and key not in memory_pool_by_id:
                memory_pool_by_id[key] = log

        with tempfile.TemporaryDirectory(prefix="memoryos_tce_") as td:
            temp_data_root = Path(td) / "data"
            memo = _build_memoryos_instance(
                memory_user_id=memory_user_id,
                data_storage_root=temp_data_root,
                llm_controller_model=llm_controller_model,
                retriever_provider=retriever_provider,
                embedding_model_name=embedding_model_name,
                assistant_id=assistant_id,
            )
            shutil.copyfile(bundle["short_term_path"], memo.short_term_memory.file_path)
            shutil.copyfile(bundle["mid_term_path"], memo.mid_term_memory.file_path)
            shutil.copyfile(bundle["long_term_path"], memo.user_long_term_memory.file_path)
            shutil.copyfile(bundle["assistant_long_term_path"], memo.assistant_long_term_memory.file_path)
            memo.short_term_memory.load()
            memo.mid_term_memory.load()
            memo.user_long_term_memory.load()
            memo.assistant_long_term_memory.load()
            retrieval = memo.retriever.retrieve_context(
                user_query=retrieval_query,
                user_id=memo.user_id,
            )

        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids: Set[str] = set()
        for page in retrieval.get("retrieved_pages", []):
            user_input = page.get("user_input", "")
            for part in str(user_input).split("[APP_LOG] "):
                chunk = part.strip()
                if not chunk:
                    continue
                first_line = chunk.splitlines()[0].strip()
                try:
                    payload = json.loads(first_line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                app_log_id = payload.get("app_log_id")
                if app_log_id is None:
                    continue
                app_log_id = str(app_log_id).strip()
                if not app_log_id or app_log_id in seen_ids:
                    continue
                log = memory_pool_by_id.get(app_log_id)
                if log is None:
                    continue
                seen_ids.add(app_log_id)
                selected_ids.append(app_log_id)
                selected_logs.append(log)

        selected_logs = selected_logs[:k]
        selected_ids = selected_ids[:k]
        result = RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in selected_logs],
            debug_metadata={
                "retrieval_mode": "memoryos_snapshot",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "retriever_provider": retriever_provider,
                "embedding_model_name": embedding_model_name,
                "retrieval_query": retrieval_query,
            },
        )
        _persist_generation_progress()
        return result

    _reset_memoryos_usage_tracker()
    generation_start = time.perf_counter()
    try:
        return run_pipeline(
            benchmark_path=benchmark_path,
            app_logs_path=app_logs_path,
            output_path=output_path,
            max_visible_logs=max_visible_logs,
            ask_json=ask_json,
            ask_structured=ask_structured,
            use_structured_response=client.supports_structured_response(),
            close=close,
            prepare_checkpoint_state=prepare_checkpoint_state,
            retrieve_context_for_query=retrieve_context_for_query,
            baseline_name="memoryos",
            memory_prompt_mode="inline_memory",
            resume=resume,
            max_checkpoints=max_checkpoints,
            debug=debug,
            debug_dir=debug_dir,
            save_prompt_and_raw=save_prompt_and_raw,
            enable_change_reasoning=enable_change_reasoning,
            enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
            rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
            rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
            retrieval_options_backend={
                "retriever_provider": retriever_provider,
                "embedding_model_name": embedding_model_name,
                "llm_controller_model": llm_controller_model,
                "assistant_id": assistant_id,
            },
            checkpoint_workers=checkpoint_workers,
            within_checkpoint_workers=within_checkpoint_workers,
            save_every_generation_keys=save_every_generation_keys,
            enable_final_qa=enable_final_qa,
            final_qa_path=final_qa_path,
            final_qa_output_path=final_qa_output_path,
            final_qa_retrieval_top_k=final_qa_retrieval_top_k,
            final_qa_save_prompt_and_raw=final_qa_save_prompt_and_raw,
        )
    finally:
        _persist_generation_progress(force=True)
