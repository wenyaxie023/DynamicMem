from pathlib import Path
from typing import Any, Dict

from .base import TceAdapterArgs


def run(args: TceAdapterArgs) -> Dict[str, Any]:
    from generation.zep.generation_tce.tce import run_generation
    
    # 必须从配置文件读取，不使用默认值
    embedding_model = args.extras.get("embedding_model")
    if not embedding_model:
        raise ValueError("embedding_model must be specified in baseline_params")
    
    batch_size = int(args.extras.get("batch_size", "10"))
    max_coroutines = int(args.extras.get("max_coroutines", "5"))
    checkpoint_workers = int(args.extras.get("checkpoint_workers", "1"))
    within_checkpoint_workers = int(args.extras.get("within_checkpoint_workers", "1"))
    save_every_generation_keys = int(args.extras.get("save_every_generation_keys", "1"))
    
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
