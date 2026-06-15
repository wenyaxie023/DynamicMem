import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import TceAdapterArgs


def _usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost.json"


def _live_usage_cost_sidecar_path(output_path: Path) -> Path:
    return output_path.parent / "usage_cost_live.json"


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


_MEMORY_ACTIONS = {"build_only", "build_then_predict", "predict_from_prebuilt"}


def _normalize_memory_action(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "build": "build_only",
        "build_only": "build_only",
        "build_then_predict": "build_then_predict",
        "build_and_predict": "build_then_predict",
        "predict": "predict_from_prebuilt",
        "predict_only": "predict_from_prebuilt",
        "prebuilt": "predict_from_prebuilt",
        "prebuilt_only": "predict_from_prebuilt",
        "generation_only": "predict_from_prebuilt",
        "predict_from_prebuilt": "predict_from_prebuilt",
    }
    action = aliases.get(raw)
    if action not in _MEMORY_ACTIONS:
        raise ValueError(
            "Unsupported amem baseline_params.memory_action: {}. "
            "Use one of: build_only, build_then_predict, predict_from_prebuilt.".format(value)
        )
    return action


def _optional_nonnegative_int(extras: Dict[str, Any], key: str) -> Optional[int]:
    raw = extras.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"amem baseline_params.{key} must be a non-negative integer.") from exc
    if value < 0:
        raise ValueError(f"amem baseline_params.{key} must be a non-negative integer.")
    return value


def _resolve_memory_action(extras: Dict[str, Any]) -> str:
    explicit = str(extras.get("memory_action") or "").strip()
    legacy_build_only = _is_true(extras.get("build_only"))
    legacy_prebuilt_only = _is_true(extras.get("skip_build")) or _is_true(extras.get("prebuilt_only"))
    if legacy_build_only and legacy_prebuilt_only:
        raise ValueError(
            "amem adapter cannot combine legacy build_only with skip_build/prebuilt_only. "
            "Use baseline_params.memory_action instead."
        )
    if explicit:
        action = _normalize_memory_action(explicit)
        if legacy_build_only and action != "build_only":
            raise ValueError(
                "Conflicting amem memory controls: build_only=true but memory_action={}.".format(action)
            )
        if legacy_prebuilt_only and action != "predict_from_prebuilt":
            raise ValueError(
                "Conflicting amem memory controls: skip_build/prebuilt_only=true but memory_action={}.".format(action)
            )
        return action
    if legacy_build_only:
        return "build_only"
    if legacy_prebuilt_only:
        return "predict_from_prebuilt"
    return "build_then_predict"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


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


def run(args: TceAdapterArgs):
    legacy_checkpoint_dir = args.extras.get("checkpoint_dir")
    if legacy_checkpoint_dir:
        raise ValueError(
            "amem adapter no longer accepts baseline_params.checkpoint_dir; "
            "use baseline_params.data_storage_path for builder state and baseline_params.snapshot_dir for snapshots."
        )

    user_id = args.user_id
    if not user_id:
        raise ValueError("amem adapter requires shared runtime.user_id")
    memory_action = _resolve_memory_action(args.extras)
    linked_neighbor_top_k = _optional_nonnegative_int(args.extras, "linked_neighbor_top_k")
    runs_builder = memory_action in {"build_only", "build_then_predict"}
    runs_prediction = memory_action in {"build_then_predict", "predict_from_prebuilt"}
    data_storage_path = args.extras.get("data_storage_path")
    if not data_storage_path and runs_builder:
        raise ValueError("amem adapter requires baseline_params.data_storage_path")
    snapshot_dir = args.extras.get("snapshot_dir")
    if not snapshot_dir:
        raise ValueError("amem adapter requires baseline_params.snapshot_dir")

    from baseline_prediction.Amem.agentic_memory.llm_controller import (
        get_usage_summary as amem_usage_summary,
        reset_usage_tracker as reset_amem_usage_tracker,
    )
    from baseline_prediction.Amem.tce import run_generation
    from baseline_prediction.common.provider_config import (
        load_repo_dotenv,
        resolve_openai_compatible_credentials,
    )

    repo_root = Path(__file__).resolve().parents[2]
    load_repo_dotenv(repo_root)

    embedding_api_key = args.extras.get("embedding_api_key")
    embedding_api_base_url = args.extras.get("embedding_api_base_url")
    if not embedding_api_key:
        embedding_api_key, resolved_embedding_base = resolve_openai_compatible_credentials(
            args.retriever_provider,
            require_api_key=True,
        )
        if not embedding_api_base_url:
            embedding_api_base_url = resolved_embedding_base

    build_amem_memory = None
    llm_controller_api_key = None
    llm_controller_api_base_url = None
    if runs_builder:
        from baseline_prediction.Amem.amem import build_amem_memory

        llm_controller_api_key, llm_controller_api_base_url = resolve_openai_compatible_credentials(
            args.llm_provider,
            require_api_key=True,
        )
    final_sidecar_path = _usage_cost_sidecar_path(args.output)
    live_sidecar_path = _live_usage_cost_sidecar_path(args.output)

    def _write_live_snapshot(
        *,
        build_duration_s: float,
        generation_duration_s: float,
        build_memory_usage: Dict[str, Any],
        retrieval_usage: Dict[str, Any],
        answer_llm_usage: Dict[str, Any],
    ) -> None:
        _write_live_usage_cost_sidecar(
            sidecar_path=live_sidecar_path,
            output_path=args.output,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            retriever_provider=args.retriever_provider,
            embedding_model_name=args.retriever_model,
            build_duration_s=build_duration_s,
            generation_duration_s=generation_duration_s,
            total_duration_s=build_duration_s + generation_duration_s,
            build_memory_usage=build_memory_usage,
            retrieval_usage=retrieval_usage,
            answer_llm_usage=answer_llm_usage,
        )

    def _run_prediction_phase(*, build_duration_s: float, build_memory_usage: Dict[str, Any]):
        reset_amem_usage_tracker()
        generation_t0 = time.time()

        def _generation_usage_callback(payload: Dict[str, Any]) -> None:
            retrieval_usage = amem_usage_summary()
            answer_llm_usage = payload if isinstance(payload, dict) else {}
            _write_live_snapshot(
                build_duration_s=build_duration_s,
                generation_duration_s=time.time() - generation_t0,
                build_memory_usage=build_memory_usage,
                retrieval_usage=retrieval_usage,
                answer_llm_usage=answer_llm_usage,
            )

        result = run_generation(
            benchmark_path=args.benchmark,
            app_logs_path=args.app_logs_path,
            output_path=args.output,
            user_id=user_id,
            size=args.extras.get("size", "small"),
            snapshot_dir=snapshot_dir,
            retrieval_top_k=args.retrieval_top_k,
            max_visible_logs=args.max_visible_logs,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_max_workers=args.llm_max_workers,
            answer_temperature=args.llm_temperature,
            answer_top_p=args.llm_top_p,
            answer_top_k=args.llm_top_k,
            embedding_backend="openai" if args.retriever_provider in {"openai", "azure"} else args.retriever_provider,
            embedding_model_name=args.retriever_model,
            embedding_api_key=embedding_api_key,
            embedding_api_base_url=embedding_api_base_url,
            resume=args.resume,
            max_checkpoints=args.max_checkpoints,
            debug=args.debug,
            debug_dir=args.debug_dir,
            save_prompt_and_raw=args.save_prompt_and_raw,
            enable_change_reasoning=args.enable_change_reasoning,
            enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
            rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
            rq3_apply_retrieval_top_k=args.rq3_apply_retrieval_top_k,
            linked_neighbor_top_k=linked_neighbor_top_k,
            checkpoint_workers=args.checkpoint_workers,
            within_checkpoint_workers=args.within_checkpoint_workers,
            save_every_generation_keys=args.save_every_generation_keys,
            enable_final_qa=args.enable_final_qa,
            final_qa_path=args.final_qa_path,
            final_qa_output_path=args.final_qa_output_path,
            final_qa_retrieval_top_k=args.final_qa_retrieval_top_k,
            final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
            usage_callback=_generation_usage_callback,
        )
        generation_duration_s = time.time() - generation_t0
        retrieval_usage = amem_usage_summary()
        answer_llm_usage = {}
        if isinstance(result, dict) and isinstance(result.get("answer_llm_usage"), dict):
            answer_llm_usage = dict(result.get("answer_llm_usage") or {})
        _write_usage_cost_sidecar(
            sidecar_path=final_sidecar_path,
            output_path=args.output,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            retriever_provider=args.retriever_provider,
            embedding_model_name=args.retriever_model,
            build_duration_s=build_duration_s,
            generation_duration_s=generation_duration_s,
            total_duration_s=build_duration_s + generation_duration_s,
            build_memory_usage=build_memory_usage,
            retrieval_usage=retrieval_usage,
            answer_llm_usage=answer_llm_usage,
        )
        _write_live_snapshot(
            build_duration_s=build_duration_s,
            generation_duration_s=generation_duration_s,
            build_memory_usage=build_memory_usage,
            retrieval_usage=retrieval_usage,
            answer_llm_usage=answer_llm_usage,
        )
        return result

    if memory_action == "predict_from_prebuilt":
        snapshot_root = Path(str(snapshot_dir)).expanduser().resolve() / user_id / args.extras.get("size", "small")
        manifest_path = snapshot_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "amem memory_action=predict_from_prebuilt requires prebuilt checkpoint snapshots. "
                f"Missing manifest: {manifest_path}"
            )
        return _run_prediction_phase(build_duration_s=0.0, build_memory_usage={})

    reset_amem_usage_tracker()
    build_t0 = time.time()

    def _builder_progress_callback(_info: Dict[str, Any]) -> None:
        current_build_usage = amem_usage_summary()
        _write_live_snapshot(
            build_duration_s=time.time() - build_t0,
            generation_duration_s=0.0,
            build_memory_usage=current_build_usage,
            retrieval_usage={},
            answer_llm_usage={},
        )

    build_amem_memory(
        user_id=user_id,
        app_log_path=args.app_logs_path,
        benchmark_path=str(args.benchmark),
        size=args.extras.get("size", "small"),
        embedding_model_name=args.retriever_model,
        embedding_backend=(
            "openai" if args.retriever_provider in {"openai", "azure"} else args.retriever_provider
        ),
        llm_controller_backend=(
            "openai" if args.llm_provider in {"openai", "azure"} else args.llm_provider
        ),
        llm_controller_model_name=args.llm_model,
        llm_controller_api_key=llm_controller_api_key,
        llm_controller_api_base_url=llm_controller_api_base_url,
        resume=args.resume,
        allow_destructive_rebuild=(
            bool(args.allow_destructive_rebuild)
            or str(args.extras.get("allow_destructive_rebuild", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
        ),
        data_storage_path=data_storage_path,
        save_every=int(args.extras.get("save_every", 50)),
        snapshot_dir=snapshot_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_base_url=embedding_api_base_url,
        progress_callback=_builder_progress_callback,
    )
    build_duration_s = time.time() - build_t0
    build_memory_usage = amem_usage_summary()
    _write_live_snapshot(
        build_duration_s=build_duration_s,
        generation_duration_s=0.0,
        build_memory_usage=build_memory_usage,
        retrieval_usage={},
        answer_llm_usage={},
    )

    if memory_action == "build_only":
        _write_usage_cost_sidecar(
            sidecar_path=final_sidecar_path,
            output_path=args.output,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            retriever_provider=args.retriever_provider,
            embedding_model_name=args.retriever_model,
            build_duration_s=build_duration_s,
            generation_duration_s=0.0,
            total_duration_s=build_duration_s,
            build_memory_usage=build_memory_usage,
            retrieval_usage={},
            answer_llm_usage={},
        )
        return {
            "predictions": [],
            "build_only": {
                "enabled": True,
                "data_storage_path": str(data_storage_path),
                "snapshot_dir": str(snapshot_dir or ""),
                "user_id": user_id,
                "size": args.extras.get("size", "small"),
            },
        }

    if not runs_prediction:
        return {"predictions": []}

    return _run_prediction_phase(build_duration_s=build_duration_s, build_memory_usage=build_memory_usage)
