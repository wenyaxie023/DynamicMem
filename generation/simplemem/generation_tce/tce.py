#!/usr/bin/env python3
"""SimpleMem TCE baseline aligned to the upstream write path and shared TCE protocol."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from generation.common.llm_client import LLMClient as SharedLLMClient
from generation.common.provider_config import load_repo_dotenv
from generation.simplemem.upstream_vendor import (
    Dialogue,
    MemoryEntry,
    SimpleMemSystem,
    UsageTracker,
    merge_usage_summary,
)
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import normalize_app_logs, observed_logs_for_checkpoint, run_pipeline, to_log_text


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIR))


_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(value or "")).strip("._-")
    return cleaned or "default"


def _resolved_path_candidates(raw_path: Any) -> set[str]:
    raw = str(raw_path or "").strip()
    if not raw:
        return set()
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT_DIR / path)
    resolved: set[str] = {str(candidate) for candidate in candidates}
    for candidate in candidates:
        try:
            resolved.add(str(candidate.resolve()))
        except OSError:
            resolved.add(str(candidate.absolute()))
    return resolved


def _paths_maybe_same_file(left: Any, right: Any) -> bool:
    left_candidates = _resolved_path_candidates(left)
    right_candidates = _resolved_path_candidates(right)
    return bool(left_candidates and right_candidates and left_candidates.intersection(right_candidates))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_timing(base: Dict[str, Any], build_duration_s: float, generation_duration_s: float) -> Dict[str, Any]:
    existing = base if isinstance(base, dict) else {}
    prior_build = float(existing.get("build_memory_duration_s") or 0.0)
    prior_generation = float(existing.get("generation_duration_s") or 0.0)
    prior_total = float(existing.get("total_duration_s") or 0.0)
    current_total = float(build_duration_s) + float(generation_duration_s)
    return {
        "build_memory_duration_s": prior_build + float(build_duration_s),
        "generation_duration_s": prior_generation + float(generation_duration_s),
        "total_duration_s": max(prior_total + current_total, prior_build + prior_generation + current_total),
    }


def _usage_sidecar_path(output_path: Path, *, live: bool) -> Path:
    return output_path.parent / ("usage_cost_live.json" if live else "usage_cost.json")


def _write_usage_sidecar(
    *,
    output_path: Path,
    live: bool,
    llm_provider: str,
    llm_model: str,
    retriever_provider: str,
    retriever_model: str,
    build_usage: Dict[str, Any],
    retrieval_usage: Dict[str, Any],
    answer_usage: Dict[str, Any],
    build_duration_s: float,
    generation_duration_s: float,
    base_payload: Optional[Dict[str, Any]] = None,
) -> None:
    base = dict(base_payload or {})
    existing_usage = base.get("usage") if isinstance(base.get("usage"), dict) else {}
    timing = _merge_timing(base.get("timing") if isinstance(base.get("timing"), dict) else {}, build_duration_s, generation_duration_s)
    payload = {
        "prediction_path": str(output_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retriever_provider": retriever_provider,
        "embedding_model": retriever_model,
        "usage": {
            "build_memory": merge_usage_summary(
                existing_usage.get("build_memory") if isinstance(existing_usage.get("build_memory"), dict) else {},
                build_usage,
            ),
            "retrieval": merge_usage_summary(
                existing_usage.get("retrieval") if isinstance(existing_usage.get("retrieval"), dict) else {},
                retrieval_usage,
            ),
            "answer_llm": merge_usage_summary(
                existing_usage.get("answer_llm") if isinstance(existing_usage.get("answer_llm"), dict) else {},
                answer_usage,
            ),
        },
        "timing": timing,
        "generated_at": _now_iso(),
    }
    _atomic_write_json(_usage_sidecar_path(output_path, live=live), payload)


def _normalize_timestamp(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = raw.replace(" ", "T")
    if candidate.endswith("Z"):
        return candidate
    try:
        datetime.fromisoformat(candidate)
        return candidate
    except Exception:
        return raw


def _dialogues_from_log(log: Dict[str, Any], *, dialogue_start_id: int) -> List[Dialogue]:
    app_log_id = str(log.get("app_log_id") or "")
    timestamp = _normalize_timestamp(log.get("timestamp"))
    raw_json = to_log_text(log)
    return [
        Dialogue(
            dialogue_id=int(dialogue_start_id),
            speaker="User",
            content=raw_json,
            timestamp=timestamp,
            source_log_id=app_log_id,
        ),
    ]


def _manifest_path(snapshot_root: Path) -> Path:
    return snapshot_root / "manifest.json"


def _lineage_path(db_root: Path) -> Path:
    return db_root / "entry_lineage.json"


def _normalize_lineage_ids(values: Sequence[Any]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _window_source_log_ids(dialogues: Sequence[Dialogue]) -> List[str]:
    return _normalize_lineage_ids(
        getattr(dialogue, "source_log_id", None)
        for dialogue in dialogues
        if getattr(dialogue, "source_log_id", None)
    )


def _load_lineage_mapping(db_root: Path) -> Dict[str, List[str]]:
    payload = _load_json_dict(_lineage_path(db_root))
    raw = payload.get("entry_lineage") if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for entry_id, source_ids in raw.items():
        key = str(entry_id or "").strip()
        if not key:
            continue
        normalized[key] = _normalize_lineage_ids(source_ids if isinstance(source_ids, list) else [])
    return normalized


def _write_lineage_mapping(db_root: Path, mapping: Mapping[str, Sequence[Any]]) -> None:
    normalized: Dict[str, List[str]] = {}
    for entry_id, source_ids in mapping.items():
        key = str(entry_id or "").strip()
        if not key:
            continue
        normalized[key] = _normalize_lineage_ids(source_ids)
    _atomic_write_json(
        _lineage_path(db_root),
        {
            "version": "simplemem_entry_lineage_v1",
            "generated_at": _now_iso(),
            "entry_lineage": normalized,
        },
    )


def _load_manifest_entries(snapshot_root: Path) -> List[Dict[str, Any]]:
    path = _manifest_path(snapshot_root)
    payload = _load_json_dict(path)
    entries = payload.get("snapshots") if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _snapshot_sort_key(entry: Mapping[str, Any]) -> Tuple[int, str, str]:
    last_event_idx = entry.get("last_event_idx", -1)
    if not isinstance(last_event_idx, int):
        last_event_idx = -1
    return (
        int(last_event_idx),
        str(entry.get("created_at") or ""),
        str(entry.get("snapshot_id") or ""),
    )


def _latest_snapshot_entry(snapshot_root: Path) -> Optional[Dict[str, Any]]:
    entries = [entry for entry in _load_manifest_entries(snapshot_root) if _checkpoint_snapshot_exists(snapshot_root, entry)]
    if not entries:
        return None
    entries.sort(key=_snapshot_sort_key, reverse=True)
    return dict(entries[0])


def _write_manifest_entries(snapshot_root: Path, entries: Sequence[Dict[str, Any]]) -> None:
    _atomic_write_json(
        _manifest_path(snapshot_root),
        {
            "version": "simplemem_snapshot_manifest_v1",
            "generated_at": _now_iso(),
            "snapshots": list(entries),
        },
    )


def _snapshot_entry_paths(snapshot_root: Path, snapshot_id: str) -> Dict[str, Path]:
    root = snapshot_root / "snapshots" / snapshot_id
    return {
        "root": root,
        "db_dir": root / "db",
        "checkpoint_path": root / "checkpoint.json",
        "meta_path": root / "meta.json",
    }


def _benchmark_checkpoints(payload: Any, max_checkpoints: Optional[int]) -> List[Dict[str, Any]]:
    checkpoints = payload.get("checkpoints") if isinstance(payload, dict) else []
    if not isinstance(checkpoints, list):
        return []
    selected = checkpoints[: int(max_checkpoints)] if max_checkpoints is not None else checkpoints
    return [checkpoint for checkpoint in selected if isinstance(checkpoint, dict)]


def _checkpoint_source_descriptor(checkpoint_targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "checkpoint_id": str(item.get("checkpoint_id") or ""),
            "checkpoint_app_log_id": str(item.get("checkpoint_app_log_id") or ""),
            "target_log_idx": int(item.get("target_log_idx") or -1),
        }
        for item in checkpoint_targets
    ]


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_source_data_fingerprint(
    *,
    app_logs: Sequence[Mapping[str, Any]],
    checkpoint_targets: Sequence[Mapping[str, Any]],
) -> str:
    return _stable_sha256(
        {
            "app_logs": list(app_logs),
            "checkpoint_targets": _checkpoint_source_descriptor(checkpoint_targets),
        }
    )


def _build_evaluation_pack_fingerprint(
    benchmark_payload: Any,
    *,
    max_checkpoints: Optional[int],
) -> str:
    checkpoints = _benchmark_checkpoints(benchmark_payload, max_checkpoints)
    return _stable_sha256(
        {
            "task_contract_version": benchmark_payload.get("task_contract_version") if isinstance(benchmark_payload, dict) else None,
            "research_frame_version": benchmark_payload.get("research_frame_version") if isinstance(benchmark_payload, dict) else None,
            "canonical_research_doc": benchmark_payload.get("canonical_research_doc") if isinstance(benchmark_payload, dict) else None,
            "checkpoints": [
                {
                    "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                    "state_completion_pack": checkpoint.get("state_completion_pack"),
                    "change_tracking_pack": checkpoint.get("change_tracking_pack"),
                    "rq3_apply_service_qa": checkpoint.get("rq3_apply_service_qa"),
                }
                for checkpoint in checkpoints
            ],
        }
    )


def _build_checkpoint_targets(
    benchmark_payload: Any,
    app_logs: List[Dict[str, Any]],
    max_checkpoints: Optional[int],
) -> List[Dict[str, Any]]:
    checkpoints = _benchmark_checkpoints(benchmark_payload, max_checkpoints)
    targets: List[Dict[str, Any]] = []
    for ordinal, checkpoint in enumerate(checkpoints):
        observed_logs, _cp_dt, _cp_ts = observed_logs_for_checkpoint(checkpoint, app_logs)
        as_of = checkpoint.get("as_of") if isinstance(checkpoint.get("as_of"), dict) else {}
        targets.append(
            {
                "ordinal": ordinal,
                "checkpoint": checkpoint,
                "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                "checkpoint_app_log_id": str(as_of.get("app_log_id") or ""),
                "target_log_idx": len(observed_logs) - 1,
            }
        )
    return targets


def _load_checkpoint_targets(
    benchmark_path: Path,
    app_logs: List[Dict[str, Any]],
    max_checkpoints: Optional[int],
) -> List[Dict[str, Any]]:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return _build_checkpoint_targets(payload, app_logs, max_checkpoints)


def _load_snapshot_bundle(snapshot_root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = str(entry.get("snapshot_id") or "").strip()
    if not snapshot_id:
        raise ValueError("Invalid SimpleMem manifest entry: missing snapshot_id")
    checkpoint_path = snapshot_root / str(entry.get("checkpoint_path") or "")
    meta_path = snapshot_root / str(entry.get("meta_path") or "")
    if not checkpoint_path.exists() or not meta_path.exists():
        raise FileNotFoundError("SimpleMem snapshot bundle is incomplete for {}".format(snapshot_id))
    checkpoint_payload = _load_json_dict(checkpoint_path)
    meta_payload = _load_json_dict(meta_path)
    bundle = dict(entry)
    bundle["checkpoint_payload"] = checkpoint_payload
    bundle["meta_payload"] = meta_payload
    bundle["db_dir"] = snapshot_root / str(entry.get("db_dir") or "")
    bundle["checkpoint_path_abs"] = checkpoint_path
    bundle["meta_path_abs"] = meta_path
    return bundle


def _checkpoint_snapshot_exists(snapshot_root: Path, entry: Dict[str, Any]) -> bool:
    try:
        bundle = _load_snapshot_bundle(snapshot_root, entry)
    except Exception:
        return False
    return Path(bundle.get("db_dir")).exists()


def _required_checkpoint_snapshots_match(
    snapshot_root: Path,
    checkpoint_targets: Sequence[Mapping[str, Any]],
) -> bool:
    manifest_by_checkpoint = {
        str(entry.get("checkpoint_id") or ""): entry
        for entry in _load_manifest_entries(snapshot_root)
        if isinstance(entry, dict) and str(entry.get("checkpoint_id") or "").strip()
    }
    for target in checkpoint_targets:
        checkpoint_id = str(target.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            continue
        entry = manifest_by_checkpoint.get(checkpoint_id)
        if entry is None:
            return False
        try:
            bundle = _load_snapshot_bundle(snapshot_root, entry)
        except Exception:
            return False
        checkpoint_payload = bundle.get("checkpoint_payload") if isinstance(bundle.get("checkpoint_payload"), dict) else {}
        expected_app_log_id = str(target.get("checkpoint_app_log_id") or "")
        stored_app_log_id = str(
            checkpoint_payload.get("checkpoint_app_log_id")
            or bundle.get("checkpoint_app_log_id")
            or entry.get("checkpoint_app_log_id")
            or ""
        )
        if expected_app_log_id and stored_app_log_id and stored_app_log_id != expected_app_log_id:
            return False
        expected_log_idx = int(target.get("target_log_idx") or -1)
        stored_log_idx = checkpoint_payload.get("checkpoint_log_index")
        if stored_log_idx is None:
            stored_log_idx = bundle.get("checkpoint_log_index", entry.get("checkpoint_log_index"))
        if stored_log_idx is not None and int(stored_log_idx) != expected_log_idx:
            return False
    return True


def _checkpoint_snapshot_matches_target(
    snapshot_root: Path,
    entry: Dict[str, Any],
    target: Mapping[str, Any],
) -> bool:
    try:
        bundle = _load_snapshot_bundle(snapshot_root, entry)
    except Exception:
        return False
    checkpoint_payload = bundle.get("checkpoint_payload") if isinstance(bundle.get("checkpoint_payload"), dict) else {}
    expected_app_log_id = str(target.get("checkpoint_app_log_id") or "")
    stored_app_log_id = str(
        checkpoint_payload.get("checkpoint_app_log_id")
        or bundle.get("checkpoint_app_log_id")
        or entry.get("checkpoint_app_log_id")
        or ""
    )
    if expected_app_log_id and stored_app_log_id and stored_app_log_id != expected_app_log_id:
        return False
    expected_log_idx = int(target.get("target_log_idx") or -1)
    stored_log_idx = checkpoint_payload.get("checkpoint_log_index")
    if stored_log_idx is None:
        stored_log_idx = bundle.get("checkpoint_log_index", entry.get("checkpoint_log_index"))
    return stored_log_idx is None or int(stored_log_idx) == expected_log_idx


def _matching_checkpoint_prefix_entries(
    snapshot_root: Path,
    checkpoint_targets: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    manifest_by_checkpoint = {
        str(entry.get("checkpoint_id") or ""): entry
        for entry in _load_manifest_entries(snapshot_root)
        if isinstance(entry, dict) and str(entry.get("checkpoint_id") or "").strip()
    }
    prefix: List[Dict[str, Any]] = []
    for target in checkpoint_targets:
        checkpoint_id = str(target.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            continue
        entry = manifest_by_checkpoint.get(checkpoint_id)
        if entry is None or not _checkpoint_snapshot_exists(snapshot_root, entry):
            break
        if not _checkpoint_snapshot_matches_target(snapshot_root, entry, target):
            return []
        prefix.append(dict(entry))
    return prefix


@dataclass
class SimpleMemBaselineConfig:
    output_root: Path
    data_storage_root: Path
    live_db_root: Path
    snapshot_root: Path
    progress_path: Path
    llm_provider: str
    llm_model: str
    llm_max_workers: int
    builder_llm_provider: str
    builder_llm_model: str
    builder_llm_max_workers: int
    builder_llm_temperature: Optional[float]
    answer_temperature: Optional[float]
    answer_top_p: Optional[float]
    answer_top_k: Optional[int]
    retriever_provider: str
    retriever_model: str
    retriever_batch_size: int
    embedding_dim: int
    retrieval_top_k: int
    semantic_top_k: int
    keyword_top_k: int
    structured_top_k: int
    window_size: int
    overlap_size: int
    save_every_logs: int
    enable_planning: bool
    enable_reflection: bool
    max_reflection_rounds: int
    enable_parallel_processing: bool
    max_parallel_workers: int
    enable_parallel_retrieval: bool
    max_retrieval_workers: int


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    snapshot_dir: Optional[str],
    data_storage_path: Optional[str],
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    retriever_provider: str,
    retriever_model: str,
    retriever_batch_size: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    retrieval_top_k: int = 10,
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
    embedding_dim: Optional[int] = None,
    builder_llm_provider: Optional[str] = None,
    builder_llm_model: Optional[str] = None,
    builder_llm_max_workers: Optional[int] = None,
    builder_llm_temperature: Optional[float] = 0.1,
    build_only: bool = False,
    predict_from_prebuilt: bool = False,
    allow_destructive_rebuild: bool = False,
    window_size: int = 5,
    overlap_size: int = 1,
    save_every_logs: int = 5,
    semantic_top_k: Optional[int] = None,
    keyword_top_k: Optional[int] = None,
    structured_top_k: Optional[int] = None,
    enable_parallel_processing: bool = False,
    max_parallel_workers: int = 1,
    enable_planning: bool = True,
    enable_reflection: bool = True,
    max_reflection_rounds: int = 2,
    enable_parallel_retrieval: bool = False,
    max_retrieval_workers: int = 1,
) -> Dict[str, Any]:

    if build_only and predict_from_prebuilt:
        raise ValueError("SimpleMem build_only and predict_from_prebuilt cannot both be true.")

    load_repo_dotenv(REPO_ROOT_DIR)
    benchmark_path = benchmark_path.expanduser().resolve()
    app_logs_path = app_logs_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not snapshot_dir:
        raise ValueError("SimpleMem requires baseline_params.snapshot_dir; no fallback is supported.")
    if not data_storage_path:
        raise ValueError("SimpleMem requires baseline_params.data_storage_path; no fallback is supported.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = output_path.parent
    data_storage_root = Path(data_storage_path).expanduser().resolve()
    live_db_root = data_storage_root / "simplemem_live_db"
    snapshot_root = Path(snapshot_dir).expanduser().resolve()
    progress_path = data_storage_root / "builder_progress.json"

    if embedding_dim is None:
        embedding_dim = 3072 if "large" in str(retriever_model or "").lower() else 1536

    config = SimpleMemBaselineConfig(
        output_root=output_root,
        data_storage_root=data_storage_root,
        live_db_root=live_db_root,
        snapshot_root=snapshot_root,
        progress_path=progress_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=int(llm_max_workers),
        builder_llm_provider=str(builder_llm_provider or llm_provider),
        builder_llm_model=str(builder_llm_model or llm_model),
        builder_llm_max_workers=int(builder_llm_max_workers or 1),
        builder_llm_temperature=builder_llm_temperature,
        answer_temperature=answer_temperature,
        answer_top_p=answer_top_p,
        answer_top_k=answer_top_k,
        retriever_provider=retriever_provider,
        retriever_model=retriever_model,
        retriever_batch_size=int(retriever_batch_size),
        embedding_dim=int(embedding_dim),
        retrieval_top_k=int(retrieval_top_k),
        semantic_top_k=int(semantic_top_k or retrieval_top_k or 10),
        keyword_top_k=int(keyword_top_k or retrieval_top_k or 10),
        structured_top_k=int(structured_top_k or retrieval_top_k or 10),
        window_size=max(1, int(window_size)),
        overlap_size=max(0, int(overlap_size)),
        save_every_logs=max(1, int(save_every_logs)),
        enable_planning=bool(enable_planning),
        enable_reflection=bool(enable_reflection),
        max_reflection_rounds=max(1, int(max_reflection_rounds)),
        enable_parallel_processing=bool(enable_parallel_processing),
        max_parallel_workers=max(1, int(max_parallel_workers)),
        enable_parallel_retrieval=bool(enable_parallel_retrieval),
        max_retrieval_workers=max(1, int(max_retrieval_workers)),
    )

    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    all_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    checkpoint_targets = _build_checkpoint_targets(benchmark_payload, all_logs, max_checkpoints=max_checkpoints)
    source_data_fingerprint = _build_source_data_fingerprint(
        app_logs=all_logs,
        checkpoint_targets=checkpoint_targets,
    )
    evaluation_pack_fingerprint = _build_evaluation_pack_fingerprint(
        benchmark_payload,
        max_checkpoints=max_checkpoints,
    )

    build_embedding_tracker = UsageTracker("embedding", config.retriever_provider, config.retriever_model)
    retrieval_embedding_tracker = UsageTracker("embedding", config.retriever_provider, config.retriever_model)
    build_aux_llm_usage: Dict[str, Any] = {}
    snapshot_cache: Dict[str, SimpleMemSystem] = {}

    def _create_builder_system(*, db_path: Path, clear_db: bool) -> SimpleMemSystem:
        return SimpleMemSystem(
            llm_provider=config.builder_llm_provider,
            model=config.builder_llm_model,
            llm_max_workers=config.builder_llm_max_workers,
            llm_temperature=config.builder_llm_temperature,
            embedding_provider=config.retriever_provider,
            embedding_model_name=config.retriever_model,
            embedding_batch_size=config.retriever_batch_size,
            embedding_dim=config.embedding_dim,
            embedding_tracker=build_embedding_tracker,
            db_path=str(db_path),
            table_name="memory_entries",
            clear_db=clear_db,
            enable_parallel_processing=config.enable_parallel_processing,
            max_parallel_workers=config.max_parallel_workers,
            window_size=config.window_size,
            overlap_size=config.overlap_size,
            enable_planning=False,
            enable_reflection=False,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
            semantic_top_k=config.semantic_top_k,
            keyword_top_k=config.keyword_top_k,
            structured_top_k=config.structured_top_k,
        )

    def _create_query_system(*, db_path: Path) -> SimpleMemSystem:
        return SimpleMemSystem(
            llm_provider=config.llm_provider,
            model=config.llm_model,
            llm_max_workers=config.llm_max_workers,
            llm_temperature=config.answer_temperature,
            llm_top_p=config.answer_top_p,
            llm_top_k=config.answer_top_k,
            embedding_provider=config.retriever_provider,
            embedding_model_name=config.retriever_model,
            embedding_batch_size=config.retriever_batch_size,
            embedding_dim=config.embedding_dim,
            embedding_tracker=retrieval_embedding_tracker,
            db_path=str(db_path),
            table_name="memory_entries",
            clear_db=False,
            enable_parallel_processing=False,
            max_parallel_workers=1,
            window_size=config.window_size,
            overlap_size=config.overlap_size,
            enable_planning=config.enable_planning,
            enable_reflection=config.enable_reflection,
            max_reflection_rounds=config.max_reflection_rounds,
            enable_parallel_retrieval=config.enable_parallel_retrieval,
            max_retrieval_workers=config.max_retrieval_workers,
            semantic_top_k=config.semantic_top_k,
            keyword_top_k=config.keyword_top_k,
            structured_top_k=config.structured_top_k,
        )

    answer_client = None
    build_system = None
    live_store = None
    builder = None

    base_live_sidecar = _load_json_dict(_usage_sidecar_path(output_path, live=True)) if resume else {}
    base_final_sidecar = _load_json_dict(_usage_sidecar_path(output_path, live=False)) if resume else {}
    build_started_at = time.monotonic()
    generation_started_at: Optional[float] = None
    build_completed_duration_s = 0.0
    builder_progress: Dict[str, Any] = {}
    live_lineage_mapping: Dict[str, List[str]] = {}

    def _has_existing_builder_artifacts() -> bool:
        for root in (config.data_storage_root, config.snapshot_root):
            if not root.exists():
                continue
            try:
                next(root.iterdir())
                return True
            except StopIteration:
                continue
        return False

    def _destructive_rebuild_error(reason: str) -> RuntimeError:
        return RuntimeError(
            "SimpleMem refused to delete existing memory artifacts at snapshot_dir={} data_storage_path={} "
            "while handling {}. Resume/build reuse is expected to be non-destructive by default. "
            "If you intentionally want to discard and rebuild these artifacts, set "
            "baseline_params.allow_destructive_rebuild: true or use a new artifact path.".format(
                config.snapshot_root,
                config.data_storage_root,
                reason,
            )
        )

    def _reopen_builder_system(*, clear_db: bool) -> None:
        nonlocal build_system, live_store, builder
        if build_system is not None:
            build_system.close()
        build_system = _create_builder_system(db_path=config.live_db_root, clear_db=clear_db)
        live_store = build_system.vector_store
        builder = build_system.memory_builder

    def _current_build_usage() -> Dict[str, Any]:
        build_llm_usage = merge_usage_summary(
            build_system.llm_usage_summary(phase="build_memory"),
            build_aux_llm_usage,
        )
        return merge_usage_summary(build_llm_usage, build_embedding_tracker.summary())

    def _current_retrieval_usage() -> Dict[str, Any]:
        usage = retrieval_embedding_tracker.summary()
        for system in snapshot_cache.values():
            usage = merge_usage_summary(usage, system.llm_usage_summary(phase="retrieval"))
        return usage

    def _current_answer_usage() -> Dict[str, Any]:
        if answer_client is None:
            return {}
        return answer_client.usage_summary()

    def _write_live_usage() -> None:
        generation_elapsed = 0.0 if generation_started_at is None else max(0.0, time.monotonic() - generation_started_at)
        build_elapsed = build_completed_duration_s or max(0.0, time.monotonic() - build_started_at)
        _write_usage_sidecar(
            output_path=output_path,
            live=True,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            retriever_provider=config.retriever_provider,
            retriever_model=config.retriever_model,
            build_usage=_current_build_usage(),
            retrieval_usage=_current_retrieval_usage(),
            answer_usage=_current_answer_usage(),
            build_duration_s=build_elapsed,
            generation_duration_s=generation_elapsed,
            base_payload=base_live_sidecar,
        )

    def _write_final_usage() -> None:
        generation_elapsed = 0.0 if generation_started_at is None else max(0.0, time.monotonic() - generation_started_at)
        build_elapsed = build_completed_duration_s or max(0.0, time.monotonic() - build_started_at)
        _write_usage_sidecar(
            output_path=output_path,
            live=False,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            retriever_provider=config.retriever_provider,
            retriever_model=config.retriever_model,
            build_usage=_current_build_usage(),
            retrieval_usage=_current_retrieval_usage(),
            answer_usage=_current_answer_usage(),
            build_duration_s=build_elapsed,
            generation_duration_s=generation_elapsed,
            base_payload=base_final_sidecar,
        )

    def _persist_builder_progress(
        *,
        confirmed_last_log_idx: int,
        status: str,
        checkpoint_id: str = "",
        last_error: str = "",
        completed_snapshot_ids: Optional[Sequence[str]] = None,
    ) -> None:
        confirmed_app_log_id = ""
        if 0 <= confirmed_last_log_idx < len(all_logs):
            confirmed_app_log_id = str(all_logs[confirmed_last_log_idx].get("app_log_id") or "")
        state_payload = builder.export_state()
        if completed_snapshot_ids is None:
            completed_snapshot_ids = builder_progress.get("completed_snapshot_ids") or []
        builder_progress.update(
            {
                "version": "simplemem_builder_progress_v3",
                "app_logs_path": str(app_logs_path),
                "benchmark_path": str(benchmark_path),
                "source_data_fingerprint": source_data_fingerprint,
                "evaluation_pack_fingerprint": evaluation_pack_fingerprint,
                "live_db_path": str(config.live_db_root),
                "snapshot_root": str(config.snapshot_root),
                "resume_source": "snapshot_bundle",
                "confirmed_last_log_idx": int(confirmed_last_log_idx),
                "confirmed_app_log_id": confirmed_app_log_id,
                "indexed_log_count": int(live_store.count_rows()) if config.live_db_root.exists() else 0,
                "status": str(status or ""),
                "checkpoint_id": str(checkpoint_id or ""),
                "last_error": str(last_error or ""),
                "updated_at": _now_iso(),
                "pending_dialogue_buffer": state_payload.get("dialogue_buffer") or [],
                "previous_entries": state_payload.get("previous_entries") or [],
                "processed_count": int(state_payload.get("processed_count") or 0),
                "completed_snapshot_ids": [str(x) for x in completed_snapshot_ids if str(x).strip()],
            }
        )
        _atomic_write_json(config.progress_path, builder_progress)

    def _record_lineage(
        *,
        db_root: Path,
        lineage_mapping: Dict[str, List[str]],
        entries: Sequence[MemoryEntry],
        source_log_ids: Sequence[Any],
    ) -> Dict[str, List[str]]:
        normalized_source_ids = _normalize_lineage_ids(source_log_ids)
        updated = dict(lineage_mapping)
        for entry in entries:
            entry_id = str(getattr(entry, "entry_id", "") or "").strip()
            if not entry_id:
                continue
            updated[entry_id] = list(normalized_source_ids)
        _write_lineage_mapping(db_root, updated)
        return updated

    def _reset_builder_state(*, reason: str) -> None:
        if _has_existing_builder_artifacts() and not allow_destructive_rebuild:
            raise _destructive_rebuild_error(reason)
        shutil.rmtree(config.data_storage_root, ignore_errors=True)
        shutil.rmtree(config.snapshot_root, ignore_errors=True)
        config.data_storage_root.mkdir(parents=True, exist_ok=True)
        config.snapshot_root.mkdir(parents=True, exist_ok=True)
        _reopen_builder_system(clear_db=True)
        builder_progress.clear()
        _persist_builder_progress(confirmed_last_log_idx=-1, status="initialized")

    def _restore_builder_state_from_snapshot() -> bool:
        prefix_entries = _matching_checkpoint_prefix_entries(config.snapshot_root, checkpoint_targets)
        if not prefix_entries:
            return False
        latest_entry = prefix_entries[-1]
        bundle = _load_snapshot_bundle(config.snapshot_root, latest_entry)
        checkpoint_payload = bundle.get("checkpoint_payload") if isinstance(bundle.get("checkpoint_payload"), dict) else {}
        stored_source_fingerprint = str(checkpoint_payload.get("source_data_fingerprint") or "").strip()
        prefix_targets = checkpoint_targets[: len(prefix_entries)]
        prefix_source_data_fingerprint = _build_source_data_fingerprint(
            app_logs=all_logs,
            checkpoint_targets=prefix_targets,
        )
        if stored_source_fingerprint:
            if stored_source_fingerprint not in {source_data_fingerprint, prefix_source_data_fingerprint}:
                return False
        elif str(checkpoint_payload.get("app_logs_path") or "") and not _paths_maybe_same_file(
            checkpoint_payload.get("app_logs_path"),
            app_logs_path,
        ):
            return False
        builder_state = checkpoint_payload.get("builder_state")
        if not isinstance(builder_state, dict):
            return False
        snapshot_db_dir = Path(bundle["db_dir"])
        if not snapshot_db_dir.exists():
            return False
        shutil.rmtree(config.data_storage_root, ignore_errors=True)
        config.data_storage_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(snapshot_db_dir), str(config.live_db_root))
        _reopen_builder_system(clear_db=False)
        build_system.import_builder_state(builder_state)
        restored_snapshot_ids = [str(entry.get("checkpoint_id") or "") for entry in prefix_entries]
        builder_progress.clear()
        _persist_builder_progress(
            confirmed_last_log_idx=int(checkpoint_payload.get("confirmed_last_log_idx") or latest_entry.get("last_event_idx") or -1),
            status="resumed_from_snapshot",
            checkpoint_id=str(bundle.get("checkpoint_id") or ""),
            completed_snapshot_ids=restored_snapshot_ids,
        )
        return True

    if predict_from_prebuilt:
        if not _required_checkpoint_snapshots_match(config.snapshot_root, checkpoint_targets):
            raise FileNotFoundError(
                "SimpleMem memory_action=predict_from_prebuilt requires prebuilt checkpoint snapshots "
                "matching all requested checkpoints under {}.".format(config.snapshot_root)
            )
        if not _restore_builder_state_from_snapshot():
            raise RuntimeError(
                "SimpleMem memory_action=predict_from_prebuilt found checkpoint snapshots under {} "
                "but could not restore snapshot state without rebuilding.".format(config.snapshot_root)
            )
    elif resume:
        if not _restore_builder_state_from_snapshot():
            _reset_builder_state(reason="resume restore fallback")
    else:
        _reset_builder_state(reason="fresh rebuild request")
    live_lineage_mapping = _load_lineage_mapping(config.live_db_root)

    completed_snapshot_ids = [str(x) for x in (builder_progress.get("completed_snapshot_ids") or []) if str(x).strip()]
    manifest_entries = _load_manifest_entries(config.snapshot_root)
    manifest_by_checkpoint = {
        str(entry.get("checkpoint_id") or ""): dict(entry)
        for entry in manifest_entries
        if isinstance(entry, dict) and str(entry.get("checkpoint_id") or "").strip()
    }

    def _snapshot_builder_for_checkpoint(checkpoint_target: Dict[str, Any], confirmed_last_log_idx: int) -> None:
        nonlocal build_aux_llm_usage
        checkpoint = checkpoint_target["checkpoint"]
        checkpoint_id = str(checkpoint_target.get("checkpoint_id") or "")
        snapshot_id = "{}__{}".format(_safe_name(checkpoint_id), int(checkpoint_target.get("target_log_idx") or -1))
        paths = _snapshot_entry_paths(config.snapshot_root, snapshot_id)
        shutil.rmtree(paths["root"], ignore_errors=True)
        paths["root"].mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(config.live_db_root), str(paths["db_dir"]))

        snapshot_build_system = _create_builder_system(
            db_path=paths["db_dir"],
            clear_db=False,
        )
        snapshot_build_system.import_builder_state(build_system.export_builder_state())
        snapshot_lineage_mapping = _load_lineage_mapping(paths["db_dir"])
        remaining_source_log_ids = _window_source_log_ids(snapshot_build_system.memory_builder.dialogue_buffer)
        flushed_entries = snapshot_build_system.finalize()
        snapshot_lineage_mapping = _record_lineage(
            db_root=paths["db_dir"],
            lineage_mapping=snapshot_lineage_mapping,
            entries=flushed_entries,
            source_log_ids=remaining_source_log_ids,
        )
        snapshot_store = snapshot_build_system.vector_store
        build_aux_llm_usage = merge_usage_summary(
            build_aux_llm_usage,
            snapshot_build_system.llm_usage_summary(phase="build_memory"),
        )
        snapshot_build_system.close()

        checkpoint_payload = {
            "snapshot_id": snapshot_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": str(checkpoint_target.get("checkpoint_app_log_id") or ""),
            "checkpoint_log_index": int(checkpoint_target.get("target_log_idx") or -1),
            "confirmed_last_log_idx": int(confirmed_last_log_idx),
            "builder_state": build_system.export_builder_state(),
            "app_logs_path": str(app_logs_path),
            "benchmark_path": str(benchmark_path),
            "source_data_fingerprint": source_data_fingerprint,
            "evaluation_pack_fingerprint": evaluation_pack_fingerprint,
            "created_at": _now_iso(),
            "pending_buffer_flushed_entries": len(flushed_entries),
        }
        meta_payload = {
            "table_name": "memory_entries",
            "retrieval_mode": "simplemem_memory_entry_hybrid",
            "entry_count": snapshot_store.count_rows(),
            "window_size": config.window_size,
            "overlap_size": config.overlap_size,
            "embedding_provider": config.retriever_provider,
            "embedding_model": config.retriever_model,
            "builder_llm_provider": config.builder_llm_provider,
            "builder_llm_model": config.builder_llm_model,
            "created_at": _now_iso(),
        }
        _atomic_write_json(paths["checkpoint_path"], checkpoint_payload)
        _atomic_write_json(paths["meta_path"], meta_payload)

        entry = {
            "snapshot_id": snapshot_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": str(checkpoint_target.get("checkpoint_app_log_id") or ""),
            "checkpoint_log_index": int(checkpoint_target.get("target_log_idx") or -1),
            "last_event_idx": int(confirmed_last_log_idx),
            "created_at": _now_iso(),
            "db_dir": str(paths["db_dir"].relative_to(config.snapshot_root)),
            "checkpoint_path": str(paths["checkpoint_path"].relative_to(config.snapshot_root)),
            "meta_path": str(paths["meta_path"].relative_to(config.snapshot_root)),
        }
        manifest_by_checkpoint[checkpoint_id] = entry
        ordered = [manifest_by_checkpoint[str(item["checkpoint_id"])] for item in checkpoint_targets if str(item["checkpoint_id"]) in manifest_by_checkpoint]
        _write_manifest_entries(config.snapshot_root, ordered)
        if checkpoint_id not in completed_snapshot_ids:
            completed_snapshot_ids.append(checkpoint_id)
        _persist_builder_progress(
            confirmed_last_log_idx=confirmed_last_log_idx,
            status="snapshot_ready",
            checkpoint_id=checkpoint_id,
            completed_snapshot_ids=completed_snapshot_ids,
        )

    confirmed_last_log_idx = int(builder_progress.get("confirmed_last_log_idx") or -1)
    try:
        for target in checkpoint_targets:
            checkpoint_id = str(target.get("checkpoint_id") or "")
            target_log_idx = int(target.get("target_log_idx") or -1)
            if checkpoint_id in completed_snapshot_ids:
                continue
            while confirmed_last_log_idx < target_log_idx:
                next_idx = confirmed_last_log_idx + 1
                log = all_logs[next_idx]
                builder.add_dialogues(
                    _dialogues_from_log(log, dialogue_start_id=next_idx + 1),
                    auto_process=False,
                )
                while len(builder.dialogue_buffer) >= builder.window_size:
                    window_source_log_ids = _window_source_log_ids(builder.dialogue_buffer[: builder.window_size])
                    processed_entries = builder.process_window() or []
                    live_lineage_mapping = _record_lineage(
                        db_root=config.live_db_root,
                        lineage_mapping=live_lineage_mapping,
                        entries=processed_entries,
                        source_log_ids=window_source_log_ids,
                    )
                confirmed_last_log_idx = next_idx
                _persist_builder_progress(
                    confirmed_last_log_idx=confirmed_last_log_idx,
                    status="building",
                    checkpoint_id=checkpoint_id,
                    completed_snapshot_ids=completed_snapshot_ids,
                )
                if (confirmed_last_log_idx + 1) % config.save_every_logs == 0:
                    _write_live_usage()
            _snapshot_builder_for_checkpoint(target, confirmed_last_log_idx=confirmed_last_log_idx)
            _write_live_usage()
        build_completed_duration_s = max(0.0, time.monotonic() - build_started_at)
        _persist_builder_progress(
            confirmed_last_log_idx=confirmed_last_log_idx,
            status="build_completed",
            checkpoint_id=str(checkpoint_targets[-1]["checkpoint_id"]) if checkpoint_targets else "",
            completed_snapshot_ids=completed_snapshot_ids,
        )
        _write_live_usage()
    except Exception as exc:
        _persist_builder_progress(
            confirmed_last_log_idx=confirmed_last_log_idx,
            status="build_failed",
            checkpoint_id=str(builder_progress.get("checkpoint_id") or ""),
            last_error=str(exc),
            completed_snapshot_ids=completed_snapshot_ids,
        )
        _write_live_usage()
        build_system.close()
        raise

    if build_only:
        _write_final_usage()
        builder_usage = _current_build_usage()
        build_system.close()
        return {
            "predictions": [],
            "build_only": {
                "snapshot_dir": str(config.snapshot_root),
                "data_storage_path": str(config.data_storage_root),
                "completed_snapshot_ids": list(completed_snapshot_ids),
                "confirmed_last_log_idx": int(confirmed_last_log_idx),
            },
            "builder_usage": builder_usage,
            "answer_llm_usage": {},
            "retrieval_usage": {},
        }

    answer_client = SharedLLMClient(
        provider=config.llm_provider,
        model_name=config.llm_model,
        max_workers=config.llm_max_workers,
        temperature=config.answer_temperature,
        top_p=config.answer_top_p,
        top_k=config.answer_top_k,
    )

    manifest_entries = _load_manifest_entries(config.snapshot_root)
    bundle_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    bundle_by_app_log_id: Dict[str, Dict[str, Any]] = {}
    for entry in manifest_entries:
        if not isinstance(entry, dict):
            continue
        bundle = _load_snapshot_bundle(config.snapshot_root, entry)
        checkpoint_id = str(bundle.get("checkpoint_id") or "")
        checkpoint_app_log_id = str(bundle.get("checkpoint_app_log_id") or "")
        if checkpoint_id:
            bundle_by_checkpoint_id[checkpoint_id] = bundle
        if checkpoint_app_log_id:
            bundle_by_app_log_id[checkpoint_app_log_id] = bundle

    generation_started_at = time.monotonic()
    snapshot_cache.clear()

    def _get_snapshot_system(bundle: Dict[str, Any]) -> SimpleMemSystem:
        db_dir = Path(bundle["db_dir"]).resolve()
        key = str(db_dir)
        if key not in snapshot_cache:
            if not db_dir.exists():
                raise FileNotFoundError("SimpleMem snapshot DB missing: {}".format(db_dir))
            snapshot_cache[key] = _create_query_system(db_path=db_dir)
        return snapshot_cache[key]

    def ask_json(prompt: str) -> Any:
        response = answer_client.ask(prompt, response_type="json")
        _write_live_usage()
        return response

    def ask_structured(prompt: str, text_format: Any) -> Any:
        response = answer_client.ask_structured(prompt, text_format=text_format)
        _write_live_usage()
        return response

    def close() -> None:
        try:
            _write_live_usage()
            _write_final_usage()
        finally:
            build_system.close()
            for system in snapshot_cache.values():
                system.close()
            if answer_client is not None:
                answer_client.close()

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        checkpoint_id = str(cp.get("checkpoint_id") or "")
        as_of = cp.get("as_of") if isinstance(cp.get("as_of"), dict) else {}
        checkpoint_app_log_id = str(as_of.get("app_log_id") or "")
        bundle = bundle_by_checkpoint_id.get(checkpoint_id) or bundle_by_app_log_id.get(checkpoint_app_log_id)
        if bundle is None:
            raise FileNotFoundError(
                "SimpleMem checkpoint snapshot not found for checkpoint_id={}, checkpoint_app_log_id={}".format(
                    checkpoint_id or "<unknown>",
                    checkpoint_app_log_id or "<unknown>",
                )
            )
        return CheckpointHandle(
            checkpoint_id=checkpoint_id,
            state_kind="simplemem_snapshot_bundle",
            state_ref=bundle,
            metadata={
                "retrieval_mode": "simplemem_memory_entry_hybrid",
                "checkpoint_timestamp": str(as_of.get("timestamp") or ""),
                "checkpoint_app_log_id": checkpoint_app_log_id,
                "snapshot_id": str(bundle.get("snapshot_id") or ""),
                "snapshot_checkpoint_id": str(bundle.get("checkpoint_id") or ""),
                "snapshot_checkpoint_app_log_id": str(bundle.get("checkpoint_app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        query_text = str(query_spec.retrieval_query_text or "").strip()
        if not query_text:
            raise ValueError("SimpleMem requires shared QuerySpec.retrieval_query_text.")
        bundle = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        top_k_for_call = int(config.retrieval_top_k)
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            top_k_for_call = int(top_k_override)
        if top_k_for_call == 0:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "simplemem_memory_entry_hybrid",
                    "retrieval_query": query_text,
                    "retrieval_top_k": top_k_for_call,
                    "retrieved_memory_entry_ids": [],
                    "retrieved_app_log_ids": [],
                    "retrieved_memory_entries": [],
                },
            )

        system = _get_snapshot_system(bundle)
        lineage_mapping = _load_lineage_mapping(Path(bundle["db_dir"]).resolve())
        retrieved_entries = system.retrieve(query_text)
        selected_entries: List[MemoryEntry] = []
        selected_ids: List[str] = []
        lineage_debug: List[Dict[str, Any]] = []
        retrieved_entry_ids: List[str] = []
        seen_log_ids = set()
        for entry_idx, entry in enumerate(retrieved_entries):
            if not isinstance(entry, MemoryEntry):
                continue
            if top_k_for_call > 0 and entry_idx >= top_k_for_call:
                break
            selected_entries.append(entry)
            entry_id = str(entry.entry_id)
            retrieved_entry_ids.append(entry_id)
            source_log_ids = list(lineage_mapping.get(entry_id) or [])
            lineage_debug.append(
                {
                    "entry_id": entry_id,
                    "source_log_ids": source_log_ids,
                    "lossless_restatement": entry.lossless_restatement,
                }
            )
            for source_log_id in source_log_ids:
                if source_log_id in seen_log_ids:
                    continue
                seen_log_ids.add(source_log_id)
                selected_ids.append(source_log_id)

        visible_logs_by_id = {
            str(log.get("app_log_id") or "").strip(): log
            for log in memory_pool
            if isinstance(log, dict) and str(log.get("app_log_id") or "").strip()
        }
        inline_memory_blocks = [
            to_log_text(visible_logs_by_id[source_log_id])
            for source_log_id in selected_ids
            if source_log_id in visible_logs_by_id
        ]
        _write_live_usage()
        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=inline_memory_blocks,
            debug_metadata={
                "retrieval_mode": "simplemem_memory_entry_hybrid",
                "retrieval_query": query_text,
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_ids),
                "num_retrieved_memory_entries": len(selected_entries),
                "retrieved_memory_entry_ids": retrieved_entry_ids,
                "retrieved_app_log_ids": selected_ids,
                "retrieved_memory_lineage": lineage_debug,
                "retrieved_memory_entries": [entry.to_dict() for entry in selected_entries],
                "retriever_provider": config.retriever_provider,
                "retriever_model": config.retriever_model,
                "snapshot_id": str(bundle.get("snapshot_id") or ""),
                "checkpoint_state_kind": checkpoint_handle.state_kind,
            },
        )

    result = run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=False,
        close=close,
        prepare_checkpoint_state=prepare_checkpoint_state,
        retrieve_context_for_query=retrieve_context_for_query,
        baseline_name="simplemem",
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
            "embedding_provider": config.retriever_provider,
            "embedding_model": config.retriever_model,
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
    if isinstance(result, dict):
        result["builder_usage"] = _current_build_usage()
        result["answer_llm_usage"] = _current_answer_usage()
        result["retrieval_usage"] = _current_retrieval_usage()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleMem baseline for TCE")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=str, required=True)
    parser.add_argument("--data-storage-path", type=str, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--retriever-provider", type=str, default="openai")
    parser.add_argument("--retriever-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--retriever-batch-size", type=int, default=64)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--builder-llm-provider", type=str, default=None)
    parser.add_argument("--builder-llm-model", type=str, default=None)
    parser.add_argument("--builder-llm-max-workers", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--overlap-size", type=int, default=1)
    parser.add_argument("--save-every-logs", type=int, default=5)
    parser.add_argument("--enable-planning", action="store_true")
    parser.add_argument("--enable-reflection", action="store_true")
    parser.add_argument("--max-reflection-rounds", type=int, default=2)
    parser.add_argument("--enable-parallel-processing", action="store_true")
    parser.add_argument("--max-parallel-workers", type=int, default=1)
    parser.add_argument("--enable-parallel-retrieval", action="store_true")
    parser.add_argument("--max-retrieval-workers", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--predict-from-prebuilt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    args = parser.parse_args()

    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        snapshot_dir=args.snapshot_dir,
        data_storage_path=args.data_storage_path,
        max_visible_logs=None,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        builder_llm_provider=args.builder_llm_provider,
        builder_llm_model=args.builder_llm_model,
        builder_llm_max_workers=args.builder_llm_max_workers,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=None,
        save_prompt_and_raw=args.save_prompt_and_raw,
        retrieval_top_k=args.retrieval_top_k,
        build_only=args.build_only,
        predict_from_prebuilt=args.predict_from_prebuilt,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        save_every_logs=args.save_every_logs,
        enable_planning=args.enable_planning,
        enable_reflection=args.enable_reflection,
        max_reflection_rounds=args.max_reflection_rounds,
        enable_parallel_processing=args.enable_parallel_processing,
        max_parallel_workers=args.max_parallel_workers,
        enable_parallel_retrieval=args.enable_parallel_retrieval,
        max_retrieval_workers=args.max_retrieval_workers,
    )


if __name__ == "__main__":
    main()
