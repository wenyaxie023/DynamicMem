from .base import DspAdapterArgs


def run(args: DspAdapterArgs):
    from generation.rag.rag_dynamic_state_prediction import run_generation

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        retrieval_top_k=int(args.extras.get("retrieval_top_k", "5")),
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        retriever_provider=args.extras.get("retriever_provider", "openai"),
        retriever_model=args.extras.get("retriever_model", "text-embedding-3-large"),
        retriever_batch_size=int(args.extras.get("retriever_batch_size", "64")),
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )
