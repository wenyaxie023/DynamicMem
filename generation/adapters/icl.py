from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.icl.tce import run_generation

    rq3_apply_items_per_key = int(args.extras.get("rq3_apply_items_per_key", "1"))
    rq3_apply_retrieval_top_k_raw = str(args.extras.get("rq3_apply_retrieval_top_k", "")).strip()
    rq3_apply_retrieval_top_k = int(rq3_apply_retrieval_top_k_raw) if rq3_apply_retrieval_top_k_raw else None
    checkpoint_workers = int(args.extras.get("checkpoint_workers", "1"))
    within_checkpoint_workers = int(args.extras.get("within_checkpoint_workers", "1"))
    save_every_generation_keys = int(args.extras.get("save_every_generation_keys", "1"))

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_fail_on_missing_pack=args.rq3_apply_fail_on_missing_pack,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        rq3_apply_items_per_key=rq3_apply_items_per_key,
        rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
    )
