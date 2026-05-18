from .base import TceAdapterArgs


def run(args: TceAdapterArgs):
    from generation.rag.rag_tce import run_generation
    predict_per_key = args.extras.get("predict_per_key", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not predict_per_key:
        raise ValueError(
            "Current TCE protocol requires baseline_params.predict_per_key=true for RAG Task A generation. "
            "Checkpoint-level combined Task A retrieval/prompting is no longer supported."
        )
    exposure_anchors_raw = str(args.extras.get("exposure_anchors", "")).strip()
    exposure_anchors = [
        int(x.strip())
        for x in exposure_anchors_raw.split(",")
        if x.strip()
    ]
    calendar_anchor_freq = str(args.extras.get("calendar_anchor_freq", "")).strip()
    exposure_tokenizer_model = str(args.extras.get("exposure_tokenizer_model", "gpt-4o-mini")).strip()

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
        predict_per_key=predict_per_key,
        exposure_anchors=exposure_anchors,
        calendar_anchor_freq=calendar_anchor_freq or None,
        exposure_tokenizer_model=exposure_tokenizer_model or "gpt-4o-mini",
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
