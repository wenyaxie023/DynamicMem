from .base import DspAdapterArgs


def run(args: DspAdapterArgs):
    from generation.letta.dynamic_state_prediction import run_generation

    allow_local_fallback = args.extras.get("allow_local_fallback", "true").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        retrieval_top_k=int(args.extras.get("retrieval_top_k", "10")),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        letta_mode=args.extras.get("letta_mode", "sdk"),
        allow_local_fallback=allow_local_fallback,
        checkpoint_state_path=args.extras.get("checkpoint_state_path"),
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )
