from pathlib import Path

from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.letta.tce import run_generation

    allow_local_fallback = args.extras.get("allow_local_fallback", "true").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    checkpoint_agents_dir_raw = str(args.extras.get("checkpoint_agents_dir", "")).strip()

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        letta_embedding=args.extras.get("embedding"),
        context_window_limit=args.extras.get("context_window_limit"),
        human_block_limit_chars=args.extras.get("human_block_limit_chars"),
        client_timeout_seconds=args.extras.get("client_timeout_seconds"),
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        letta_mode=args.extras.get("letta_mode", "sdk"),
        allow_local_fallback=allow_local_fallback,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        enable_change_reasoning=args.enable_change_reasoning,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        checkpoint_workers=args.checkpoint_workers,
        within_checkpoint_workers=args.within_checkpoint_workers,
        save_every_generation_keys=args.save_every_generation_keys,
        enable_final_qa=args.enable_final_qa,
        final_qa_path=args.final_qa_path,
        final_qa_output_path=args.final_qa_output_path,
        final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
        query_isolation_mode=args.extras.get("query_isolation_mode", "checkpoint_snapshot"),
        checkpoint_agents_dir=Path(checkpoint_agents_dir_raw) if checkpoint_agents_dir_raw else None,
        baseline_name=args.baseline,
    )
