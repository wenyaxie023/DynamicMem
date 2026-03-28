from pathlib import Path

from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.HippoRAG2.generation_tce.tce import run_generation

    hipporag_dir = args.extras.get("hipporag_dir")
    if not hipporag_dir:
        raise ValueError("hipporag2 adapter requires --hipporag-dir")
    
    # REQUIRED: embedding_model must be specified in config
    embedding_model = args.extras.get("embedding_model")
    if not embedding_model:
        raise ValueError("hipporag2 adapter requires 'embedding_model' in baseline_params")
    
    online = args.extras.get("online", "true").lower() in ("true", "1", "yes")
    openie_cache_path = args.extras.get("openie_cache_path")
    checkpoint_workers = int(args.extras.get("checkpoint_workers", "1"))
    within_checkpoint_workers = int(args.extras.get("within_checkpoint_workers", "1"))
    save_every_generation_keys = int(args.extras.get("save_every_generation_keys", "1"))
    enable_rq3_apply = args.extras.get("enable_rq3_apply_service_qa", "true").lower() in ("true", "1", "yes")
    rq3_apply_items = int(args.extras.get("rq3_apply_items_per_key", "1"))
    enable_change_reasoning = args.extras.get("enable_change_reasoning", "false").lower() in ("true", "1", "yes")
    retrieval_top_k = int(args.extras.get("retrieval_top_k", "5"))

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
        online=online,
        embedding_model=embedding_model,
        openie_cache_path=Path(openie_cache_path) if openie_cache_path else None,
        retrieval_top_k=retrieval_top_k,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
        enable_rq3_apply_service_qa=enable_rq3_apply,
        rq3_apply_items_per_key=rq3_apply_items,
        enable_change_reasoning=enable_change_reasoning,
    )
