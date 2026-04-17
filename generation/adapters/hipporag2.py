from pathlib import Path

from .base import TceAdapterArgs


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


def run(args: TceAdapterArgs):
    from generation.HippoRAG2.generation_tce.online_tce import run_generation

    save_dir = args.extras.get("save_dir") or args.extras.get("hipporag_dir")
    if not save_dir:
        raise ValueError("hipporag2 adapter requires baseline_params.save_dir or baseline_params.hipporag_dir")
    builder_save_every_logs_raw = args.extras.get("builder_save_every_logs", 5)
    try:
        builder_save_every_logs = int(builder_save_every_logs_raw)
    except Exception:
        builder_save_every_logs = 5
    interleave_build_and_test = _as_bool(args.extras.get("interleave_build_and_test"), default=False)
    stop_after_build = _as_bool(args.extras.get("stop_after_build"), default=False)

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        save_dir=Path(save_dir),
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
        stop_after_build=stop_after_build,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
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
    )
