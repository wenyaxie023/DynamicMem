from pathlib import Path
from typing import Any, Dict

from .base import TceAdapterArgs


def run(args: TceAdapterArgs) -> Dict[str, Any]:
    from generation.zep.generation_tce.tce import run_generation
    
    embedding_model = args.extras.get("embedding_model")
    if not embedding_model:
        raise ValueError("embedding_model must be specified in baseline_params")
    
    batch_size = args.extras.get("batch_size")
    if batch_size is None:
        raise ValueError("batch_size must be specified in baseline_params")
    batch_size = int(batch_size)
    
    max_coroutines = args.extras.get("max_coroutines")
    if max_coroutines is None:
        raise ValueError("max_coroutines must be specified in baseline_params")
    max_coroutines = int(max_coroutines)
    
    checkpoint_workers = args.extras.get("checkpoint_workers")
    if checkpoint_workers is None:
        raise ValueError("checkpoint_workers must be specified in baseline_params")
    checkpoint_workers = int(checkpoint_workers)
    
    within_checkpoint_workers = args.extras.get("within_checkpoint_workers")
    if within_checkpoint_workers is None:
        raise ValueError("within_checkpoint_workers must be specified in baseline_params")
    within_checkpoint_workers = int(within_checkpoint_workers)
    
    save_every_generation_keys = args.extras.get("save_every_generation_keys")
    if save_every_generation_keys is None:
        raise ValueError("save_every_generation_keys must be specified in baseline_params")
    save_every_generation_keys = int(save_every_generation_keys)
    
    print(f"[Zep Adapter] Using embedding_model: {embedding_model}")
    
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
        embedding_model=embedding_model,
        batch_size=batch_size,
        max_coroutines=max_coroutines,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
    )
