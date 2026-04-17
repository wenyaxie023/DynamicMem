from .base import TceAdapterArgs


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(args: TceAdapterArgs):
    import os
    from pathlib import Path

    from generation.MemoryOS.tce_adapter import run_generation
    from generation.common.provider_config import (
        load_repo_dotenv,
        resolve_openai_compatible_credentials,
    )

    user_id = args.user_id
    if not user_id:
        raise ValueError("memoryos adapter requires shared runtime.user_id")
    repo_root = Path(__file__).resolve().parents[2]
    load_repo_dotenv(repo_root)

    llm_controller_api_key = os.getenv("LLM_CONTROLLER_API_KEY")
    llm_controller_api_base_url = os.getenv("LLM_CONTROLLER_API_BASE_URL")
    if not llm_controller_api_key or not llm_controller_api_base_url:
        resolved_key, resolved_base = resolve_openai_compatible_credentials(
            args.llm_provider,
            require_api_key=True,
        )
        if not llm_controller_api_key and resolved_key:
            os.environ["LLM_CONTROLLER_API_KEY"] = resolved_key
        if not llm_controller_api_base_url and resolved_base:
            os.environ["LLM_CONTROLLER_API_BASE_URL"] = resolved_base

    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_api_base_url = os.getenv("EMBEDDING_API_BASE_URL")
    if not embedding_api_key or not embedding_api_base_url:
        resolved_key, resolved_base = resolve_openai_compatible_credentials(
            args.retriever_provider,
            require_api_key=True,
        )
        if not embedding_api_key and resolved_key:
            os.environ["EMBEDDING_API_KEY"] = resolved_key
        if not embedding_api_base_url and resolved_base:
            os.environ["EMBEDDING_API_BASE_URL"] = resolved_base

    return run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        user_id=user_id,
        size=args.extras.get("size", "large"),
        snapshot_dir=args.extras.get("snapshot_dir"),
        data_storage_path=args.extras.get("data_storage_path"),
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        answer_temperature=args.llm_temperature,
        answer_top_p=args.llm_top_p,
        answer_top_k=args.llm_top_k,
        retriever_provider=args.retriever_provider,
        embedding_model_name=args.retriever_model,
        llm_controller_model=args.extras.get("llm_controller_model", args.llm_model),
        assistant_id=args.extras.get("assistant_id", "assistant"),
        build_only=_is_true(args.extras.get("build_only")),
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
