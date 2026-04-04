#!/usr/bin/env python3
import hashlib
import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs):
        return False

from tce_core.final_checkpoint_qa import build_final_qa_prompt_with_agent_memory
from tce_core.orchestrator_protocol import (
    AnswerExecutionResult,
    CheckpointHandle,
    RetrievalOptions,
    RetrievalResult,
)
from tce_core.pipeline import (
    _build_apply_prompt_from_queryspec,
    _build_change_prompt_from_queryspec,
    build_state_completion_prompt,
    run_pipeline,
)
from generation.letta.agent_loop import LettaAgentLoop, estimate_usage_cost_usd, sort_logs

load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_logs(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    elif isinstance(payload, list):
        logs = payload
    else:
        logs = []
    return sort_logs([x for x in logs if isinstance(x, dict)])


def _load_checkpoint_targets(benchmark_path: Path) -> List[Tuple[int, str]]:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    out: List[Tuple[int, str]] = []
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        log_index = (cp.get("as_of") or {}).get("log_index")
        if checkpoint_id and isinstance(log_index, int) and log_index >= 0:
            out.append((log_index, checkpoint_id))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _clamp_log_index(idx: Any, total_logs: int) -> int:
    try:
        value = int(idx)
    except Exception:
        return -1
    if total_logs <= 0:
        return -1
    return max(-1, min(value, total_logs - 1))


def _log_signature(log: Dict[str, Any]) -> str:
    normalized = json.dumps(log or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _confirmed_progress_fields(all_logs: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    clamped_idx = _clamp_log_index(idx, len(all_logs))
    last_app_log_id = None
    signature = None
    if 0 <= clamped_idx < len(all_logs):
        last_app_log_id = str(all_logs[clamped_idx].get("app_log_id") or "") or None
        signature = _log_signature(all_logs[clamped_idx])
    return {
        "ingested_until_log_index": clamped_idx,
        "ingested_log_count": max(0, clamped_idx + 1),
        "last_ingested_app_log_id": last_app_log_id,
        "confirmed_ingest_signature": signature,
    }


def _ensure_state_shape(state: Dict[str, Any], logs_path: Path, total_logs: int) -> Dict[str, Any]:
    ckpts = state.get("checkpoint_agentfiles")
    if not isinstance(ckpts, dict):
        ckpts = {}
    state["checkpoint_agentfiles"] = ckpts
    state["logs_path"] = str(logs_path)
    state["total_logs"] = int(total_logs)
    state["last_ingested_log_index"] = int(state.get("last_ingested_log_index", -1))
    if "final_agentfile" not in state:
        state["final_agentfile"] = None
    return state


def _ensure_progress_shape(progress: Dict[str, Any], logs_path: Path, total_logs: int) -> Dict[str, Any]:
    out = progress if isinstance(progress, dict) else {}
    out["logs_path"] = str(logs_path)
    out["total_logs"] = int(total_logs)
    out["builder_agent_id"] = str(out.get("builder_agent_id", "") or "") or None
    out["builder_agent_name"] = str(out.get("builder_agent_name", "") or "") or None
    out["checkpoint_id_in_progress"] = str(out.get("checkpoint_id_in_progress", "") or "") or None
    out["checkpoint_snapshot_path"] = str(out.get("checkpoint_snapshot_path", "") or "") or None
    out["checkpoint_snapshot_checkpoint_id"] = (
        str(out.get("checkpoint_snapshot_checkpoint_id", "") or "") or None
    )
    out["ingested_until_log_index"] = int(out.get("ingested_until_log_index", -1))
    out["last_ingested_app_log_id"] = str(out.get("last_ingested_app_log_id", "") or "") or None
    out["confirmed_ingest_signature"] = str(out.get("confirmed_ingest_signature", "") or "") or None
    out["ingested_log_count"] = int(out.get("ingested_log_count", 0))
    uncertain_idx = _optional_int(out.get("uncertain_log_index"))
    out["uncertain_log_index"] = uncertain_idx
    out["uncertain_app_log_id"] = str(out.get("uncertain_app_log_id", "") or "") or None
    out["uncertain_error"] = str(out.get("uncertain_error", "") or "") or None
    out["status"] = str(out.get("status", "initialized") or "initialized")
    out["requested_letta_mode"] = str(out.get("requested_letta_mode", "") or "") or None
    out["effective_letta_backend"] = str(out.get("effective_letta_backend", "") or "") or None
    out["builder_session_token"] = str(out.get("builder_session_token", "") or "") or None
    out["owner_pid"] = _optional_int(out.get("owner_pid"))
    out["owner_hostname"] = str(out.get("owner_hostname", "") or "") or None
    out["owner_started_at"] = str(out.get("owner_started_at", "") or "") or None
    out["lock_path"] = str(out.get("lock_path", "") or "") or None
    out["last_error"] = str(out.get("last_error", "") or "") or None
    out["updated_at"] = str(out.get("updated_at", "") or "") or None
    return out


def _set_confirmed_progress(progress: Dict[str, Any], all_logs: List[Dict[str, Any]], idx: int) -> None:
    progress.update(_confirmed_progress_fields(all_logs, idx))


def _index_from_agentfile_name(
    file_name: str,
    checkpoint_id_to_idx: Dict[str, int],
    total_logs: int,
) -> Optional[int]:
    stem = Path(file_name).stem
    if stem.startswith("final_"):
        raw = stem[len("final_") :]
        if raw.isdigit():
            return max(-1, min(int(raw) - 1, total_logs - 1))
    if stem.startswith("log_"):
        raw = stem[len("log_") :]
        if raw.isdigit():
            return max(-1, min(int(raw) - 1, total_logs - 1))
    if stem in checkpoint_id_to_idx:
        return max(-1, min(checkpoint_id_to_idx[stem], total_logs - 1))
    return None


def _resolve_latest_agentfile(
    *,
    output_dir: Path,
    state: Dict[str, Any],
    checkpoint_id_to_idx: Dict[str, int],
    total_logs: int,
) -> Optional[Tuple[Path, int]]:
    candidates: List[Path] = []
    final_af = state.get("final_agentfile")
    if isinstance(final_af, str) and final_af.strip():
        candidates.append(Path(final_af))
    ckpt_map = state.get("checkpoint_agentfiles")
    if isinstance(ckpt_map, dict):
        for value in ckpt_map.values():
            if isinstance(value, str) and value.strip():
                candidates.append(Path(value))
    if output_dir.exists():
        candidates.extend(sorted(output_dir.glob("*.af")))

    best_path: Optional[Path] = None
    best_idx = -1
    best_mtime = -1.0
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            continue
        idx = _index_from_agentfile_name(path.name, checkpoint_id_to_idx, total_logs)
        if idx is None:
            continue
        mtime = path.stat().st_mtime
        if idx > best_idx or (idx == best_idx and mtime > best_mtime):
            best_path = path
            best_idx = idx
            best_mtime = mtime
    if best_path is None:
        return None
    return best_path, best_idx


def _default_checkpoint_agents_dir(app_logs_path: Path) -> Path:
    return _repo_root() / "generation" / "letta" / "agents" / "tce" / app_logs_path.parent.name


def _usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost.json"


def _pricing_for_model(llm_provider: str, llm_model: str) -> Optional[Dict[str, Any]]:
    provider = str(llm_provider or "").strip().lower()
    model = str(llm_model or "").strip().lower()
    if provider == "azure" and (model == "gpt-5-mini" or model.endswith("/gpt-5-mini")):
        return {
            "provider": "azure",
            "model": llm_model,
            "prompt_cost_per_1m": 0.275,
            "completion_cost_per_1m": 2.2,
            "cached_input_cost_per_1m": None,
            "cache_write_cost_per_1m": None,
            "pricing_note": (
                "Cached-input and cache-write rates are not configured; "
                "estimated_cost_usd excludes any separate cached-token billing. "
                "Embedding cost is not included."
            ),
        }
    return None


def _merge_usage_summaries(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "turn_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
        "step_count": 0,
        "run_ids": [],
    }
    seen_run_ids = set()
    for source in (base, delta):
        if not isinstance(source, dict):
            continue
        for key in (
            "turn_count",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "context_tokens",
            "total_tokens",
            "step_count",
        ):
            merged[key] += int(source.get(key) or 0)
        for run_id in source.get("run_ids") or []:
            run_id = str(run_id or "").strip()
            if not run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            merged["run_ids"].append(run_id)
    return merged


def _usage_summary_by_phase(records: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "turn_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "context_tokens": 0,
        "total_tokens": 0,
        "step_count": 0,
        "run_ids": [],
    }
    seen_run_ids = set()
    target_phase = str(phase or "").strip()
    for record in records:
        if str(record.get("phase") or "").strip() != target_phase:
            continue
        summary["turn_count"] += 1
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "context_tokens",
            "total_tokens",
            "step_count",
        ):
            summary[key] += int(record.get(key) or 0)
        for run_id in record.get("run_ids") or []:
            run_id = str(run_id or "").strip()
            if not run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            summary["run_ids"].append(run_id)
    return summary


def _write_usage_cost_sidecar(
    *,
    sidecar_path: Path,
    output_path: Path,
    llm_provider: str,
    llm_model: str,
    letta_embedding: Optional[str],
    effective_backend: str,
    checkpoint_agents_dir: Path,
    current_summary: Dict[str, Any],
    build_memory_summary: Optional[Dict[str, Any]] = None,
    answer_summary: Optional[Dict[str, Any]] = None,
    build_memory_duration_s: Optional[float] = None,
    answer_duration_s: Optional[float] = None,
    total_duration_s: Optional[float] = None,
    resume: bool,
) -> None:
    existing_summary: Dict[str, Any] = {}
    existing_usage: Dict[str, Any] = {}
    existing_timing: Dict[str, Any] = {}
    if resume and sidecar_path.exists():
        try:
            existing_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            existing_payload = {}
        if isinstance(existing_payload, dict):
            raw_existing_summary = existing_payload.get("usage_summary")
            if isinstance(raw_existing_summary, dict):
                existing_summary = raw_existing_summary
            raw_existing_usage = existing_payload.get("usage")
            if isinstance(raw_existing_usage, dict):
                existing_usage = raw_existing_usage
            raw_existing_timing = existing_payload.get("timing")
            if isinstance(raw_existing_timing, dict):
                existing_timing = raw_existing_timing
    merged_summary = _merge_usage_summaries(existing_summary, current_summary)
    usage = {
        "build_memory": _merge_usage_summaries(
            existing_usage.get("build_memory") if isinstance(existing_usage.get("build_memory"), dict) else {},
            build_memory_summary if isinstance(build_memory_summary, dict) else {},
        ),
        "answer": _merge_usage_summaries(
            existing_usage.get("answer") if isinstance(existing_usage.get("answer"), dict) else {},
            answer_summary if isinstance(answer_summary, dict) else {},
        ),
    }
    timing = {
        "build_memory_duration_s": float(existing_timing.get("build_memory_duration_s") or 0.0),
        "answer_duration_s": float(existing_timing.get("answer_duration_s") or 0.0),
        "total_duration_s": float(existing_timing.get("total_duration_s") or 0.0),
    }
    if build_memory_duration_s is not None:
        timing["build_memory_duration_s"] += float(build_memory_duration_s)
    if answer_duration_s is not None:
        timing["answer_duration_s"] += float(answer_duration_s)
    if total_duration_s is not None:
        timing["total_duration_s"] += float(total_duration_s)
    pricing = _pricing_for_model(llm_provider=llm_provider, llm_model=llm_model)
    estimated_cost = None
    cost_breakdown = None
    if pricing is not None:
        estimated_cost = estimate_usage_cost_usd(
            merged_summary,
            prompt_cost_per_1m=float(pricing["prompt_cost_per_1m"]),
            completion_cost_per_1m=float(pricing["completion_cost_per_1m"]),
            cached_input_cost_per_1m=pricing["cached_input_cost_per_1m"],
            cache_write_cost_per_1m=pricing["cache_write_cost_per_1m"],
        )
        cost_breakdown = {
            "build_memory": estimate_usage_cost_usd(
                usage["build_memory"],
                prompt_cost_per_1m=float(pricing["prompt_cost_per_1m"]),
                completion_cost_per_1m=float(pricing["completion_cost_per_1m"]),
                cached_input_cost_per_1m=pricing["cached_input_cost_per_1m"],
                cache_write_cost_per_1m=pricing["cache_write_cost_per_1m"],
            ),
            "answer": estimate_usage_cost_usd(
                usage["answer"],
                prompt_cost_per_1m=float(pricing["prompt_cost_per_1m"]),
                completion_cost_per_1m=float(pricing["completion_cost_per_1m"]),
                cached_input_cost_per_1m=pricing["cached_input_cost_per_1m"],
                cache_write_cost_per_1m=pricing["cache_write_cost_per_1m"],
            ),
        }
    payload = {
        "prediction_path": str(output_path),
        "backend": effective_backend,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_model": letta_embedding or None,
        "checkpoint_agents_dir": str(checkpoint_agents_dir),
        "usage_summary": merged_summary,
        "usage": usage,
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_breakdown_usd": cost_breakdown,
        "pricing": pricing,
        "timing": timing,
        "generated_at": _now_iso(),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_is_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_builder_lock(
    *,
    lock_path: Path,
    session_token: str,
    owner_pid: int,
    owner_hostname: str,
    owner_started_at: str,
) -> None:
    payload = {
        "session_token": session_token,
        "owner_pid": int(owner_pid),
        "owner_hostname": owner_hostname,
        "owner_started_at": owner_started_at,
        "updated_at": _now_iso(),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            with open(lock_path, "x", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return
        except FileExistsError:
            existing = _read_state(lock_path)
            existing_session = str(existing.get("session_token") or "").strip()
            if existing_session == session_token:
                return
            existing_host = str(existing.get("owner_hostname") or "").strip()
            existing_pid = _optional_int(existing.get("owner_pid"))
            if existing_host == owner_hostname and not _pid_is_alive(existing_pid):
                tmp_path = lock_path.with_suffix(lock_path.suffix + f".stale.{session_token}.tmp")
                _write_state(tmp_path, payload)
                os.replace(tmp_path, lock_path)
                return
            raise RuntimeError(
                "Letta TCE builder lock already held by pid {} on host {}. "
                "Refusing to reuse the live builder agent concurrently.".format(
                    existing_pid if existing_pid is not None else "<unknown>",
                    existing_host or "<unknown>",
                )
            )


def _release_builder_lock(lock_path: Path, session_token: str) -> None:
    try:
        existing = _read_state(lock_path)
        if str(existing.get("session_token") or "").strip() != str(session_token or "").strip():
            return
        lock_path.unlink(missing_ok=True)
    except Exception:
        return


def _load_or_init_builder_agent(
    *,
    checkpoint_agents_dir: Path,
    checkpoint_state_path: Path,
    benchmark_path: Path,
    logs_path: Path,
    all_logs: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    answer_temperature: Optional[float],
    answer_top_p: Optional[float],
    answer_top_k: Optional[int],
    letta_mode: str,
    allow_local_fallback: bool,
    resume: bool,
    lease_registry_path: Path,
    progress_path: Path,
    letta_embedding: Optional[str] = None,
) -> Tuple[LettaAgentLoop, Dict[str, Any], Dict[str, Any], int]:
    checkpoint_targets = _load_checkpoint_targets(benchmark_path)
    checkpoint_id_to_idx = {checkpoint_id: idx for idx, checkpoint_id in checkpoint_targets}
    raw_state = _read_state(checkpoint_state_path) if resume else {}
    state = _ensure_state_shape(raw_state, logs_path=logs_path, total_logs=len(all_logs))
    raw_progress = _read_state(progress_path) if resume else {}
    progress = _ensure_progress_shape(raw_progress, logs_path=logs_path, total_logs=len(all_logs))
    latest_snapshot = None
    if resume:
        latest_snapshot = _resolve_latest_agentfile(
            output_dir=checkpoint_agents_dir,
            state=state,
            checkpoint_id_to_idx=checkpoint_id_to_idx,
            total_logs=len(all_logs),
        )

    agent = LettaAgentLoop(
        user_namespace=f"tce::{logs_path.parent.name}",
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding=letta_embedding,
        answer_temperature=answer_temperature,
        answer_top_p=answer_top_p,
        answer_top_k=answer_top_k,
        mode=letta_mode,
        allow_local_fallback=allow_local_fallback,
        create_agent=False,
        lease_registry_path=lease_registry_path,
    )

    ingested_until = -1
    snapshot_idx = latest_snapshot[1] if latest_snapshot is not None else -1
    progress_idx = _clamp_log_index(progress.get("ingested_until_log_index", -1), len(all_logs))
    progress_agent_id = str(progress.get("builder_agent_id") or "").strip()
    attached_existing_builder = False
    resumed_from_snapshot = False

    if resume and not getattr(agent, "_fallback", False):
        if progress_agent_id and progress_idx >= 0:
            try:
                agent.attach_agent(progress_agent_id)
                ingested_until = progress_idx
                attached_existing_builder = True
                _set_confirmed_progress(progress, all_logs, ingested_until)
                progress["uncertain_log_index"] = None
                progress["uncertain_app_log_id"] = None
                progress["uncertain_error"] = None
                progress["status"] = "resumed_live"
                progress["last_error"] = None
            except Exception as exc:
                progress["last_error"] = f"attach_existing_builder_failed: {exc}"
                if latest_snapshot is None:
                    raise RuntimeError(
                        "Cannot resume Letta builder: failed to attach recorded builder_agent_id and no checkpoint snapshot is available."
                    )

        if not attached_existing_builder and latest_snapshot is not None:
            latest_path, latest_idx = latest_snapshot
            agent.load_agent_file(latest_path, activate=True)
            ingested_until = latest_idx
            resumed_from_snapshot = True
            _set_confirmed_progress(progress, all_logs, ingested_until)
            progress["uncertain_log_index"] = None
            progress["uncertain_app_log_id"] = None
            progress["uncertain_error"] = None
            progress["checkpoint_snapshot_path"] = str(latest_path)
            progress["checkpoint_snapshot_checkpoint_id"] = latest_path.stem
            progress["status"] = "resumed_snapshot"

    if agent.active_agent_id() is None:
        agent.create_agent()
    if not attached_existing_builder and not resumed_from_snapshot:
        _set_confirmed_progress(progress, all_logs, ingested_until)
        progress["uncertain_log_index"] = None
        progress["uncertain_app_log_id"] = None
        progress["uncertain_error"] = None
        progress["status"] = "initialized"

    state["last_ingested_log_index"] = max(int(state.get("last_ingested_log_index", -1)), ingested_until)
    progress["builder_agent_id"] = agent.active_agent_id()
    progress["builder_agent_name"] = agent.active_agent_name()
    return agent, state, progress, ingested_until


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    *,
    llm_provider: str,
    llm_model: str,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    letta_mode: str,
    allow_local_fallback: bool,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_save_prompt_and_raw: bool = False,
    query_isolation_mode: str = "checkpoint_snapshot",
    checkpoint_agents_dir: Optional[Path] = None,
    letta_embedding: Optional[str] = None,
    baseline_name: str = "letta",
) -> Dict[str, Any]:
    # Load the canonical app-log stream once; every checkpoint is a prefix of this sequence.
    all_logs = _load_logs(app_logs_path)
    # Keep all Letta builder artifacts under one stable directory so resume can find them.
    checkpoint_agents_dir = (checkpoint_agents_dir or _default_checkpoint_agents_dir(app_logs_path)).resolve()
    # Persistent builder state and snapshots live beside the output.
    checkpoint_state_path = checkpoint_agents_dir / "checkpoint_state.json"
    progress_path = checkpoint_agents_dir / "builder_progress.json"
    lease_registry_path = checkpoint_agents_dir / "leased_agent_ids.json"
    lock_path = checkpoint_agents_dir / "builder.lock.json"
    checkpoint_agents_dir.mkdir(parents=True, exist_ok=True)

    # Tag this process as the current builder owner before touching any live agent state.
    session_token = uuid.uuid4().hex
    owner_pid = os.getpid()
    owner_hostname = socket.gethostname()
    owner_started_at = _now_iso()
    _acquire_builder_lock(
        lock_path=lock_path,
        session_token=session_token,
        owner_pid=owner_pid,
        owner_hostname=owner_hostname,
        owner_started_at=owner_started_at,
    )

    agent = None
    effective_backend = ""
    closed = False
    close_error: Optional[Exception] = None

    try:
        overall_start = time.perf_counter()
        build_memory_duration_s = 0.0
        answer_duration_s = 0.0

        # Reuse the locally recorded builder when possible; otherwise recover from the newest snapshot or start fresh.
        agent, checkpoint_state, builder_progress, ingested_until = _load_or_init_builder_agent(
            checkpoint_agents_dir=checkpoint_agents_dir,
            checkpoint_state_path=checkpoint_state_path,
            benchmark_path=benchmark_path,
            logs_path=app_logs_path,
            all_logs=all_logs,
            llm_provider=llm_provider,
            llm_model=llm_model,
            letta_embedding=letta_embedding,
            answer_temperature=answer_temperature,
            answer_top_p=answer_top_p,
            answer_top_k=answer_top_k,
            letta_mode=letta_mode,
            allow_local_fallback=allow_local_fallback,
            resume=resume,
            lease_registry_path=lease_registry_path,
            progress_path=progress_path,
        )

        effective_backend = agent.effective_backend()
        usage_cost_path = _usage_cost_sidecar_path(output_path)

        def _persist_builder_progress(**updates: Any) -> None:
            # Treat confirmed ingest progress as monotonic; never let later writes move it backwards.
            requested_confirmed_idx = updates.pop("ingested_until_log_index", None)
            requested_count = updates.pop("ingested_log_count", None)
            updates.pop("last_ingested_app_log_id", None)
            updates.pop("confirmed_ingest_signature", None)
            if requested_confirmed_idx is not None:
                trusted_idx = max(
                    int(builder_progress.get("ingested_until_log_index", -1)),
                    _clamp_log_index(requested_confirmed_idx, len(all_logs)),
                )
                _set_confirmed_progress(builder_progress, all_logs, trusted_idx)
            else:
                _set_confirmed_progress(builder_progress, all_logs, int(builder_progress.get("ingested_until_log_index", -1)))
            if requested_count is not None:
                builder_progress["ingested_log_count"] = max(
                    int(builder_progress.get("ingested_log_count", 0)),
                    int(requested_count),
                )
            for key, value in updates.items():
                builder_progress[key] = value
            builder_progress["builder_agent_id"] = agent.active_agent_id()
            builder_progress["builder_agent_name"] = agent.active_agent_name()
            builder_progress["requested_letta_mode"] = letta_mode
            builder_progress["effective_letta_backend"] = effective_backend
            builder_progress["builder_session_token"] = session_token
            builder_progress["owner_pid"] = owner_pid
            builder_progress["owner_hostname"] = owner_hostname
            builder_progress["owner_started_at"] = owner_started_at
            builder_progress["lock_path"] = str(lock_path)
            builder_progress["updated_at"] = _now_iso()
            _write_state(progress_path, _ensure_progress_shape(builder_progress, app_logs_path, len(all_logs)))

        def _write_checkpoint_state() -> None:
            # Persist the snapshot manifest that maps checkpoint IDs to .af files.
            checkpoint_state["logs_path"] = str(app_logs_path)
            checkpoint_state["total_logs"] = len(all_logs)
            _write_state(checkpoint_state_path, checkpoint_state)

        # Refresh the progress file once per run so owner/session metadata is always current.
        _persist_builder_progress(
            status=str(builder_progress.get("status") or "initialized"),
            checkpoint_id_in_progress=None,
            last_error=builder_progress.get("last_error"),
        )

        # Formal checkpoint isolation only works against the real Letta SDK, not the local fallback stub.
        if query_isolation_mode == "checkpoint_snapshot" and effective_backend != "sdk":
            raise RuntimeError(
                "checkpoint_snapshot isolation requires real Letta SDK. "
                "Either disable fallback for formal runs or switch to query_isolation_mode=shared_agent for debug."
            )

        def _save_checkpoint_snapshot(checkpoint_id: str, cp_idx: int) -> Path:
            # Reuse an existing snapshot if this checkpoint was already materialized in a previous run.
            existing = checkpoint_state.get("checkpoint_agentfiles", {}).get(checkpoint_id)
            if isinstance(existing, str) and existing.strip():
                existing_path = Path(existing)
                if existing_path.exists():
                    _persist_builder_progress(
                        status="checkpoint_snapshot_saved",
                        checkpoint_id_in_progress=checkpoint_id,
                        checkpoint_snapshot_path=str(existing_path),
                        checkpoint_snapshot_checkpoint_id=checkpoint_id,
                        last_error=None,
                    )
                    return existing_path

            # Otherwise export the current builder state as the authoritative snapshot for this checkpoint.
            snapshot_path = checkpoint_agents_dir / f"{checkpoint_id}.af"
            agent.save_agent_file(snapshot_path)
            checkpoint_state["checkpoint_agentfiles"][checkpoint_id] = str(snapshot_path)
            checkpoint_state["last_ingested_log_index"] = max(int(checkpoint_state.get("last_ingested_log_index", -1)), cp_idx)
            _write_checkpoint_state()
            _persist_builder_progress(
                status="checkpoint_snapshot_saved",
                checkpoint_id_in_progress=checkpoint_id,
                checkpoint_snapshot_path=str(snapshot_path),
                checkpoint_snapshot_checkpoint_id=checkpoint_id,
                ingested_until_log_index=cp_idx,
                last_error=None,
            )
            return snapshot_path

        def ensure_checkpoint_ready(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> Dict[str, Any]:
            nonlocal ingested_until, build_memory_duration_s
            # Resolve the exact app-log cutoff for this checkpoint.
            checkpoint_id = str(cp.get("checkpoint_id", "") or "")
            cp_idx_raw = (cp.get("as_of") or {}).get("log_index")
            cp_idx = cp_idx_raw if isinstance(cp_idx_raw, int) else len(memory_pool) - 1
            cp_idx = _clamp_log_index(cp_idx, len(all_logs))
            if cp_idx > ingested_until:
                # Advance the shared builder strictly in chronological order until it reaches this checkpoint.
                _persist_builder_progress(
                    status="ingesting",
                    checkpoint_id_in_progress=checkpoint_id,
                    checkpoint_snapshot_path=None,
                    checkpoint_snapshot_checkpoint_id=None,
                    last_error=None,
                )
                for i in range(ingested_until + 1, cp_idx + 1):
                    try:
                        # Each log is ingested exactly once per clean run; resume may start from a later prefix.
                        ingest_start = time.perf_counter()
                        agent.ingest_log(all_logs[i])
                        build_memory_duration_s += time.perf_counter() - ingest_start
                    except Exception as exc:
                        # A failed ingest is ambiguous: the remote side may have applied it, so mark progress uncertain.
                        _persist_builder_progress(
                            status="ingest_uncertain",
                            checkpoint_id_in_progress=checkpoint_id,
                            uncertain_log_index=i,
                            uncertain_app_log_id=str(all_logs[i].get("app_log_id") or "") or None,
                            uncertain_error=f"ingest_log_failed at index {i}: {exc}",
                            last_error=f"ingest_log_failed at index {i}: {exc}",
                        )
                        raise
                    ingested_until = i
                    checkpoint_state["last_ingested_log_index"] = max(
                        int(checkpoint_state.get("last_ingested_log_index", -1)),
                        ingested_until,
                    )
                    # Persist confirmed progress after every successful ingest so resume can restart from a verified prefix.
                    _persist_builder_progress(
                        status="ingesting",
                        checkpoint_id_in_progress=checkpoint_id,
                        ingested_until_log_index=i,
                        uncertain_log_index=None,
                        uncertain_app_log_id=None,
                        uncertain_error=None,
                        last_error=None,
                    )
                _write_checkpoint_state()
            else:
                # The builder is already at or beyond this prefix; no additional ingest is required.
                _persist_builder_progress(
                    status="checkpoint_ready",
                    checkpoint_id_in_progress=checkpoint_id,
                    last_error=None,
                )

            snapshot_path = ""
            if query_isolation_mode == "checkpoint_snapshot":
                # Snapshot mode isolates every downstream Task A/B/C query from future mutations.
                snapshot_path = str(_save_checkpoint_snapshot(checkpoint_id, cp_idx))

            # Expose the checkpoint scope to the shared orchestrator for metadata/debugging.
            allowed_ids = [
                str(log.get("app_log_id")).strip()
                for log in memory_pool
                if log.get("app_log_id") is not None
            ]
            metadata = {
                "retrieval_mode": (
                    "letta_checkpoint_snapshot"
                    if query_isolation_mode == "checkpoint_snapshot"
                    else f"letta_agent_loop_{letta_mode}"
                ),
                "num_retrieved_logs": 0,
                "retrieved_app_log_ids": [],
                "ingested_until_log_index": ingested_until,
                "num_ingested_logs": max(0, ingested_until + 1),
                "query_isolation_mode": query_isolation_mode,
                "requested_letta_mode": letta_mode,
                "effective_letta_backend": effective_backend,
                "letta_embedding": letta_embedding or None,
            }
            if snapshot_path:
                metadata["checkpoint_snapshot_path"] = snapshot_path
            return {
                "checkpoint_id": checkpoint_id,
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "available_app_log_ids": allowed_ids,
                "checkpoint_snapshot_path": snapshot_path,
                "metadata": metadata,
            }

        def ask_checkpoint_json(checkpoint_ctx: Dict[str, Any], prompt: str) -> Any:
            nonlocal answer_duration_s
            # Record that the builder has moved from ingesting to answering at this checkpoint.
            _persist_builder_progress(
                status="querying",
                checkpoint_id_in_progress=checkpoint_ctx.get("checkpoint_id") or None,
                last_error=None,
            )
            if query_isolation_mode == "shared_agent":
                # Debug-only path: ask the live builder directly, without per-query isolation.
                query_start = time.perf_counter()
                response = agent.ask_json(prompt)
                answer_duration_s += time.perf_counter() - query_start
                return response

            # Formal path: import the checkpoint snapshot into a temporary agent and query that copy.
            snapshot_path_raw = str(checkpoint_ctx.get("checkpoint_snapshot_path", "") or "").strip()
            if not snapshot_path_raw:
                raise RuntimeError("Missing checkpoint snapshot for Letta checkpoint_snapshot query mode.")
            snapshot_path = Path(snapshot_path_raw)
            temp_agent_id = agent.import_agent_file(snapshot_path.read_bytes(), activate=False)
            try:
                query_start = time.perf_counter()
                response = agent.ask_json_with_agent(temp_agent_id, prompt)
                answer_duration_s += time.perf_counter() - query_start
                return response
            finally:
                # Always tear down the temporary query agent so Task A/B/C queries never pollute shared state.
                agent.delete_agent(temp_agent_id, ignore_missing=True)

        def close(*, run_succeeded: bool, failure_message: Optional[str]) -> None:
            nonlocal closed, close_error
            if closed:
                return
            closed = True
            try:
                failure_like_statuses = {"failed", "ingest_uncertain"}
                current_status = str(builder_progress.get("status") or "")
                if not run_succeeded and current_status not in failure_like_statuses:
                    # If the pipeline aborted outside the explicit ingest-failure path, mark the builder as failed.
                    current_status = "failed"
                    _persist_builder_progress(
                        status="failed",
                        checkpoint_id_in_progress=builder_progress.get("checkpoint_id_in_progress"),
                        last_error=str(failure_message or "pipeline_failed"),
                    )

                if run_succeeded and current_status not in failure_like_statuses:
                    if effective_backend == "sdk" and ingested_until >= 0:
                        # Save one final snapshot for the fully ingested builder state on successful completion.
                        final_path = checkpoint_agents_dir / f"final_{ingested_until + 1}.af"
                        agent.save_agent_file(final_path)
                        checkpoint_state["final_agentfile"] = str(final_path)
                        checkpoint_state["last_ingested_log_index"] = max(
                            int(checkpoint_state.get("last_ingested_log_index", -1)),
                            ingested_until,
                        )
                        _write_checkpoint_state()
                        _persist_builder_progress(
                            status="completed",
                            checkpoint_id_in_progress=None,
                            checkpoint_snapshot_path=str(final_path),
                            checkpoint_snapshot_checkpoint_id=None,
                            ingested_until_log_index=ingested_until,
                            uncertain_log_index=None,
                            uncertain_app_log_id=None,
                            uncertain_error=None,
                            last_error=None,
                        )
                    else:
                        _persist_builder_progress(
                            status="completed",
                            checkpoint_id_in_progress=None,
                            last_error=None,
                        )
                _write_usage_cost_sidecar(
                    # Persist cost/usage accounting independently from prediction JSON.
                    sidecar_path=usage_cost_path,
                    output_path=output_path,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    letta_embedding=letta_embedding,
                    effective_backend=effective_backend,
                    checkpoint_agents_dir=checkpoint_agents_dir,
                    current_summary=agent.usage_summary(),
                    build_memory_summary=_usage_summary_by_phase(agent.usage_records(), "ingest_log"),
                    answer_summary=_usage_summary_by_phase(agent.usage_records(), "ask"),
                    build_memory_duration_s=build_memory_duration_s,
                    answer_duration_s=answer_duration_s,
                    total_duration_s=time.perf_counter() - overall_start,
                    resume=resume,
                )
                agent.close()
            except Exception as exc:
                close_error = exc
            finally:
                # Release the builder lease even if close bookkeeping partially failed.
                _release_builder_lock(lock_path, session_token)

        result: Optional[Dict[str, Any]] = None
        failure_message: Optional[str] = None
        try:
            def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
                # Map Letta's internal checkpoint context into the shared orchestrator handle.
                checkpoint_ctx = ensure_checkpoint_ready(cp, memory_pool)
                metadata = dict(checkpoint_ctx.get("metadata") or {})
                metadata.setdefault("checkpoint_timestamp", str(checkpoint_ctx.get("checkpoint_timestamp") or ""))
                metadata.setdefault("available_app_log_ids", list(checkpoint_ctx.get("available_app_log_ids") or []))
                state_kind = (
                    "agent_snapshot"
                    if str(metadata.get("query_isolation_mode") or "").strip() == "checkpoint_snapshot"
                    else "shared_agent_memory"
                )
                return CheckpointHandle(
                    checkpoint_id=str(checkpoint_ctx.get("checkpoint_id") or cp.get("checkpoint_id") or ""),
                    state_kind=state_kind,
                    state_ref=checkpoint_ctx,
                    metadata=metadata,
                )

            def retrieve_context_for_query(
                checkpoint_handle: CheckpointHandle,
                query_spec,
                retrieval_options: RetrievalOptions,
                memory_pool: List[Dict[str, Any]],
            ) -> RetrievalResult:
                # Letta does not perform explicit inline retrieval here; the agent already holds the checkpoint memory.
                del memory_pool
                metadata = dict(checkpoint_handle.metadata or {})
                metadata.setdefault("retrieval_query", query_spec.retrieval_query_text)
                metadata.setdefault("num_retrieved_logs", 0)
                metadata.setdefault("retrieved_app_log_ids", [])
                return RetrievalResult(
                    mode=str(metadata.get("retrieval_mode") or "agent_memory"),
                    inline_memory_blocks=[],
                    debug_metadata=metadata,
                )

            def answer_query(
                checkpoint_handle: CheckpointHandle,
                query_spec,
                retrieval_result: RetrievalResult,
            ) -> AnswerExecutionResult:
                # Reuse the shared TCE prompt contracts, but target agent memory instead of inline memory blocks.
                checkpoint_ctx = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
                if query_spec.task_name == "Task A":
                    target_value_templates = dict((query_spec.task_payload or {}).get("target_value_templates") or {})
                    prompt = build_state_completion_prompt(
                        checkpoint={"as_of": {"timestamp": query_spec.checkpoint_timestamp}},
                        context_logs=None,
                        target_keys=query_spec.target_keys,
                        target_value_templates=target_value_templates,
                        memory_prompt_mode="agent_memory",
                        inline_memory_blocks=[],
                        task_text_override=query_spec.answer_query_text,
                    )
                elif query_spec.task_name == "Task B":
                    prompt = _build_change_prompt_from_queryspec(
                        memory_prompt_mode="agent_memory",
                        query_spec=query_spec,
                        changed_value_templates=dict((query_spec.task_payload or {}).get("changed_value_templates") or {}),
                        retrieval_result=retrieval_result,
                    )
                elif query_spec.task_name == "Task C":
                    prompt = _build_apply_prompt_from_queryspec(
                        memory_prompt_mode="agent_memory",
                        query_spec=query_spec,
                        retrieval_result=retrieval_result,
                    )
                elif query_spec.task_name == "Final QA":
                    prompt = build_final_qa_prompt_with_agent_memory(question=query_spec.answer_query_text)
                else:
                    raise ValueError("Unsupported task_name for Letta answer_query: {}".format(query_spec.task_name))
                # Execute the prompt against the checkpoint-scoped Letta state prepared above.
                raw_output = ask_checkpoint_json(checkpoint_ctx, prompt)
                return AnswerExecutionResult(
                    raw_output=raw_output,
                    prompt=prompt,
                    debug_metadata={"retrieval_metadata": dict(retrieval_result.debug_metadata or {})},
                )

            # Hand the Letta-specific hooks to the shared TCE orchestrator; Task A/B/C control flow lives there.
            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=max_visible_logs,
                ask_json=lambda _prompt: {},
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                prepare_checkpoint_state=prepare_checkpoint_state,
                retrieve_context_for_query=retrieve_context_for_query,
                answer_query=answer_query,
                finalize_checkpoint_state=lambda _checkpoint_handle: None,
                baseline_name=baseline_name,
                memory_prompt_mode="agent_memory",
                resume=resume,
                max_checkpoints=max_checkpoints,
                debug=debug,
                debug_dir=debug_dir,
                save_prompt_and_raw=save_prompt_and_raw,
                enable_change_reasoning=enable_change_reasoning,
                enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
                rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
                retrieval_options_backend={
                    "query_isolation_mode": query_isolation_mode,
                    "requested_letta_mode": letta_mode,
                    "effective_letta_backend": effective_backend,
                    "letta_embedding": letta_embedding or None,
                },
                checkpoint_workers=checkpoint_workers,
                within_checkpoint_workers=within_checkpoint_workers,
                save_every_generation_keys=save_every_generation_keys,
                enable_final_qa=enable_final_qa,
                final_qa_path=final_qa_path,
                final_qa_output_path=final_qa_output_path,
                final_qa_save_prompt_and_raw=final_qa_save_prompt_and_raw,
            )
            return result
        except Exception as exc:
            # Defer status persistence to close(); just preserve the message for the failure record.
            failure_message = str(exc)
            raise
        finally:
            # Always run the Letta-specific teardown even if the shared orchestrator raised.
            close(run_succeeded=failure_message is None, failure_message=failure_message)
            if close_error is not None:
                raise close_error
    finally:
        if not closed:
            # Double-release guard: if setup failed before close() ran, still free the builder lease.
            _release_builder_lock(lock_path, session_token)
