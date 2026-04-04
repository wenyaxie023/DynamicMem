from pathlib import Path

from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.HippoRAG2.generation_tce.tce import run_generation

    hipporag_dir = args.extras.get("hipporag_dir")
    if not hipporag_dir:
        raise ValueError("hipporag2 adapter requires --hipporag-dir")

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        hipporag_dir=Path(hipporag_dir),
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        checkpoint_workers=args.checkpoint_workers,
        within_checkpoint_workers=args.within_checkpoint_workers,
        save_every_generation_keys=args.save_every_generation_keys,
    )
