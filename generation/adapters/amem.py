from .base import DspAdapterArgs


def run(args: DspAdapterArgs):
    from generation.Amem.amem_dynamic_state_prediction import run_generation

    user_id = args.extras.get("user_id")
    if not user_id:
        raise ValueError("amem adapter requires baseline_params.user_id")

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        user_id=user_id,
        size=args.extras.get("size", "small"),
        snapshot_dir=args.extras.get("snapshot_dir"),
        checkpoint_dir=args.extras.get("checkpoint_dir"),
        retrieval_top_k=int(args.extras.get("retrieval_top_k", "5")),
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        embedding_backend=args.extras.get("embedding_backend", "openai"),
        embedding_model_name=args.extras.get("embedding_model_name", "openrouter/openai/text-embedding-3-large"),
        embedding_api_key=args.extras.get("embedding_api_key"),
        embedding_api_base_url=args.extras.get("embedding_api_base_url"),
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )
