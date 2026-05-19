import os
from pathlib import Path

from .base import TceAdapterArgs


_MEMORY_ACTIONS = {"build_only", "build_then_predict", "predict_from_prebuilt"}


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_memory_action(value) -> str:
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
            "Unsupported hipporag2 baseline_params.memory_action: {}. "
            "Use one of: build_only, build_then_predict, predict_from_prebuilt.".format(value)
        )
    return action


def _resolve_memory_action(extras) -> str:
    explicit = str(extras.get("memory_action") or "").strip()
    legacy_build_only = _as_bool(extras.get("build_only"), default=False)
    legacy_prebuilt_only = _as_bool(extras.get("skip_build"), default=False) or _as_bool(
        extras.get("prebuilt_only"),
        default=False,
    )
    if legacy_build_only and legacy_prebuilt_only:
        raise ValueError(
            "hipporag2 adapter cannot combine legacy build_only with skip_build/prebuilt_only. "
            "Use baseline_params.memory_action instead."
        )
    if explicit:
        action = _normalize_memory_action(explicit)
        if legacy_build_only and action != "build_only":
            raise ValueError(
                "Conflicting hipporag2 memory controls: build_only=true but memory_action={}.".format(action)
            )
        if legacy_prebuilt_only and action != "predict_from_prebuilt":
            raise ValueError(
                "Conflicting hipporag2 memory controls: skip_build/prebuilt_only=true but memory_action={}.".format(
                    action
                )
            )
        return action
    if legacy_build_only:
        return "build_only"
    if legacy_prebuilt_only:
        return "predict_from_prebuilt"
    return "build_then_predict"


def run(args: TceAdapterArgs):
    from baseline_prediction.HippoRAG2.generation_tce.online_tce import run_generation

    snapshot_dir = args.extras.get("snapshot_dir")
    if not snapshot_dir:
        raise ValueError("hipporag2 adapter requires baseline_params.snapshot_dir")
    data_storage_path = args.extras.get("data_storage_path")
    if not data_storage_path:
        raise ValueError("hipporag2 adapter requires baseline_params.data_storage_path")
    builder_save_every_logs_raw = args.extras.get("builder_save_every_logs", 5)
    try:
        builder_save_every_logs = int(builder_save_every_logs_raw)
    except Exception:
        builder_save_every_logs = 5
    interleave_build_and_test = _as_bool(args.extras.get("interleave_build_and_test"), default=False)
    memory_action = _resolve_memory_action(args.extras)
    build_only = memory_action == "build_only"
    predict_from_prebuilt = memory_action == "predict_from_prebuilt"
    task_selection = str(args.extras.get("__task_selection__", "all")).strip().lower() or "all"
    openie_max_workers_raw = str(args.extras.get("openie_max_workers", "")).strip()
    prev_openie_max_workers = os.environ.get("OPENIE_MAX_WORKERS")
    if openie_max_workers_raw:
        os.environ["OPENIE_MAX_WORKERS"] = openie_max_workers_raw

    try:
        return run_generation(
            benchmark_path=args.benchmark,
            app_logs_path=args.app_logs_path,
            output_path=args.output,
            snapshot_dir=Path(snapshot_dir),
            data_storage_path=Path(data_storage_path),
            max_visible_logs=args.max_visible_logs,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_max_workers=args.llm_max_workers,
            answer_temperature=args.llm_temperature,
            answer_top_p=args.llm_top_p,
            answer_top_k=args.llm_top_k,
            retriever_provider=args.retriever_provider,
            retriever_model=args.retriever_model,
            retriever_batch_size=args.retriever_batch_size,
            retrieval_top_k=args.retrieval_top_k,
            builder_save_every_logs=builder_save_every_logs,
            interleave_build_and_test=interleave_build_and_test,
            build_only=build_only,
            predict_from_prebuilt=predict_from_prebuilt,
            resume=args.resume,
            max_checkpoints=args.max_checkpoints,
            debug=args.debug,
            debug_dir=args.debug_dir,
            save_prompt_and_raw=args.save_prompt_and_raw,
            enable_change_reasoning=args.enable_change_reasoning,
            enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
            rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
            task_selection=task_selection,
            rq3_apply_retrieval_top_k=args.rq3_apply_retrieval_top_k,
            checkpoint_workers=args.checkpoint_workers,
            within_checkpoint_workers=args.within_checkpoint_workers,
            save_every_generation_keys=args.save_every_generation_keys,
            enable_final_qa=args.enable_final_qa,
            final_qa_path=args.final_qa_path,
            final_qa_output_path=args.final_qa_output_path,
            final_qa_retrieval_top_k=args.final_qa_retrieval_top_k,
            final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
            allow_destructive_rebuild=(
                bool(args.allow_destructive_rebuild) or _as_bool(args.extras.get("allow_destructive_rebuild"))
            ),
        )
    finally:
        if openie_max_workers_raw:
            if prev_openie_max_workers is None:
                os.environ.pop("OPENIE_MAX_WORKERS", None)
            else:
                os.environ["OPENIE_MAX_WORKERS"] = prev_openie_max_workers
