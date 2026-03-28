from pathlib import Path
from typing import Any, Dict

from .base import TceAdapterArgs


def run(args: TceAdapterArgs) -> Dict[str, Any]:
    from generation.simplemem.generation_tce.tce import run_generation
    
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
        retrieval_top_k=int(args.extras.get("retrieval_top_k", "20")),
        embedding_model=args.extras.get("embedding_model", "text-embedding-3-large"),
        batch_size=int(args.extras.get("batch_size", "64")),
        checkpoint_workers=1,
        within_checkpoint_workers=1,
    )