from .base import TceAdapterArgs


_MEMORY_ACTIONS = {"build_only", "build_then_predict", "predict_from_prebuilt"}


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_memory_action(value) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
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
    action = aliases.get(raw, raw)
    if action not in _MEMORY_ACTIONS:
        raise ValueError(
            "Unsupported simplemem baseline_params.memory_action: {}. "
            "Use one of: build_only, build_then_predict, predict_from_prebuilt.".format(value)
        )
    return action


def _resolve_memory_action(extras) -> str:
    explicit = str(extras.get("memory_action") or "").strip()
    legacy_build_only = _is_true(extras.get("build_only"))
    if explicit:
        action = _normalize_memory_action(explicit)
        if legacy_build_only and action != "build_only":
            raise ValueError(
                "Conflicting simplemem memory controls: build_only=true but memory_action={}.".format(action)
            )
        return action
    if legacy_build_only:
        return "build_only"
    return "build_then_predict"


def run(args: TceAdapterArgs):
    from baseline_prediction.simplemem.generation_tce.tce import run_generation

    snapshot_dir = args.extras.get("snapshot_dir")
    if not snapshot_dir:
        raise ValueError("simplemem adapter requires baseline_params.snapshot_dir")
    data_storage_path = args.extras.get("data_storage_path")
    if not data_storage_path:
        raise ValueError("simplemem adapter requires baseline_params.data_storage_path")
    memory_action = _resolve_memory_action(args.extras)

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        snapshot_dir=snapshot_dir,
        data_storage_path=data_storage_path,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        retrieval_top_k=args.retrieval_top_k,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        enable_change_reasoning=args.enable_change_reasoning,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=args.rq3_apply_retrieval_top_k,
        checkpoint_workers=args.checkpoint_workers,
        within_checkpoint_workers=args.within_checkpoint_workers,
        save_every_generation_keys=args.save_every_generation_keys,
        enable_final_qa=args.enable_final_qa,
        final_qa_path=args.final_qa_path,
        final_qa_output_path=args.final_qa_output_path,
        final_qa_retrieval_top_k=args.final_qa_retrieval_top_k,
        final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
        embedding_dim=(
            int(args.extras.get("embedding_dim"))
            if str(args.extras.get("embedding_dim", "")).strip()
            else None
        ),
        builder_llm_provider=(
            str(args.extras.get("builder_llm_provider")).strip()
            if str(args.extras.get("builder_llm_provider", "")).strip()
            else None
        ),
        builder_llm_model=(
            str(args.extras.get("builder_llm_model")).strip()
            if str(args.extras.get("builder_llm_model", "")).strip()
            else None
        ),
        builder_llm_max_workers=(
            int(args.extras.get("builder_llm_max_workers"))
            if str(args.extras.get("builder_llm_max_workers", "")).strip()
            else None
        ),
        builder_llm_temperature=(
            float(args.extras.get("builder_llm_temperature"))
            if str(args.extras.get("builder_llm_temperature", "")).strip()
            else 0.1
        ),
        build_only=memory_action == "build_only",
        predict_from_prebuilt=memory_action == "predict_from_prebuilt",
        allow_destructive_rebuild=(
            bool(args.allow_destructive_rebuild) or _is_true(args.extras.get("allow_destructive_rebuild"))
        ),
        window_size=(
            int(args.extras.get("window_size"))
            if str(args.extras.get("window_size", "")).strip()
            else 5
        ),
        overlap_size=(
            int(args.extras.get("overlap_size"))
            if str(args.extras.get("overlap_size", "")).strip()
            else 1
        ),
        save_every_logs=(
            int(args.extras.get("save_every_logs"))
            if str(args.extras.get("save_every_logs", "")).strip()
            else 5
        ),
        semantic_top_k=(
            int(args.extras.get("semantic_top_k"))
            if str(args.extras.get("semantic_top_k", "")).strip()
            else None
        ),
        keyword_top_k=(
            int(args.extras.get("keyword_top_k"))
            if str(args.extras.get("keyword_top_k", "")).strip()
            else None
        ),
        structured_top_k=(
            int(args.extras.get("structured_top_k"))
            if str(args.extras.get("structured_top_k", "")).strip()
            else None
        ),
        enable_parallel_processing=(
            str(args.extras.get("enable_parallel_processing", "")).strip().lower() in {"1", "true", "yes", "on"}
        ),
        max_parallel_workers=(
            int(args.extras.get("max_parallel_workers"))
            if str(args.extras.get("max_parallel_workers", "")).strip()
            else 1
        ),
        enable_planning=(
            str(args.extras.get("enable_planning", "true")).strip().lower() in {"1", "true", "yes", "on"}
        ),
        enable_reflection=(
            str(args.extras.get("enable_reflection", "true")).strip().lower() in {"1", "true", "yes", "on"}
        ),
        max_reflection_rounds=(
            int(args.extras.get("max_reflection_rounds"))
            if str(args.extras.get("max_reflection_rounds", "")).strip()
            else 2
        ),
        enable_parallel_retrieval=(
            str(args.extras.get("enable_parallel_retrieval", "")).strip().lower() in {"1", "true", "yes", "on"}
        ),
        max_retrieval_workers=(
            int(args.extras.get("max_retrieval_workers"))
            if str(args.extras.get("max_retrieval_workers", "")).strip()
            else 1
        ),
    )
