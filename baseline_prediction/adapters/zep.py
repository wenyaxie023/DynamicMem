from typing import Any, Dict

from .base import TceAdapterArgs


def run(args: TceAdapterArgs) -> Dict[str, Any]:
    from generation.zep.generation_tce.tce import run_generation

    max_coroutines = int(args.extras.get("max_coroutines", "5"))
    graphiti_llm_provider = str(args.extras.get("graphiti_llm_provider") or args.llm_provider).strip()
    graphiti_llm_model = str(args.extras.get("graphiti_llm_model") or args.llm_model).strip()
    graphiti_max_tokens_raw = args.extras.get("graphiti_max_tokens")
    graphiti_max_tokens = int(graphiti_max_tokens_raw) if graphiti_max_tokens_raw not in {None, ""} else 16384

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        max_coroutines=max_coroutines,
        graphiti_llm_provider=graphiti_llm_provider,
        graphiti_llm_model=graphiti_llm_model,
        graphiti_max_tokens=graphiti_max_tokens,
        enable_change_reasoning=args.enable_change_reasoning,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=args.rq3_apply_retrieval_top_k,
        checkpoint_workers=args.checkpoint_workers,
        within_checkpoint_workers=args.within_checkpoint_workers,
        save_every_generation_keys=args.save_every_generation_keys,
        enable_final_qa=args.enable_final_qa,
        final_qa_path=args.final_qa_path,
        final_qa_output_path=args.final_qa_output_path,
        final_qa_retrieval_top_k=args.final_qa_retrieval_top_k,
        final_qa_save_prompt_and_raw=args.final_qa_save_prompt_and_raw,
    )
