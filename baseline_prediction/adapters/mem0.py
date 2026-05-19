from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from baseline_prediction.mem0.tce import run_generation

    user_id = args.user_id
    if not user_id:
        raise ValueError("mem0 adapter requires shared runtime.user_id")

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        user_id=user_id,
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
        collection_name=args.extras.get("collection_name", f"membench_mem0_{user_id}"),
        qdrant_host=args.extras.get("qdrant_host", "localhost"),
        qdrant_port=int(args.extras.get("qdrant_port", "6333")),
        config_path=args.extras.get("config_path"),
        checkpoint_dir=args.extras.get("checkpoint_dir"),
        embedder_api_key=args.extras.get("embedder_api_key"),
        embedder_api_base=args.extras.get("embedder_api_base"),
        llm_api_key=args.extras.get("llm_api_key"),
        llm_api_base=args.extras.get("llm_api_base"),
        embedder_model=args.retriever_model,
        answer_llm_model=args.llm_model,
        embedder_azure_deployment=args.extras.get("embedder_azure_deployment"),
        embedder_azure_api_version=args.extras.get("embedder_azure_api_version"),
        llm_azure_deployment=args.extras.get("llm_azure_deployment"),
        llm_azure_api_version=args.extras.get("llm_azure_api_version"),
        reset_collections=str(args.extras.get("reset_collections", "false")).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        allow_destructive_rebuild=(
            bool(args.allow_destructive_rebuild)
            or str(args.extras.get("allow_destructive_rebuild", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
        ),
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
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
