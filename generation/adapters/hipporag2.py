from pathlib import Path

from .base import DspAdapterArgs


def run(args: DspAdapterArgs):
    from generation.HippoRAG2.generation_dsp.dynamic_state_prediction import run_generation

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
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )
