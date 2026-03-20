from pathlib import Path

from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.HippoRAG2.generation_tce.tce import run_generation

    hipporag_dir = args.extras.get("hipporag_dir")
    if not hipporag_dir:
        raise ValueError("hipporag2 adapter requires --hipporag-dir")
    checkpoint_workers = int(args.extras.get("checkpoint_workers", "1"))
    within_checkpoint_workers = int(args.extras.get("within_checkpoint_workers", "1"))
    save_every_generation_keys = int(args.extras.get("save_every_generation_keys", "1"))

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        hipporag_dir=Path(hipporag_dir),
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
    )
