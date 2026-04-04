from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.oracle.tce import run_generation

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
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
    )
