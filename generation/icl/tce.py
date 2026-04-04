#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from generation.icl.client import LLMClient
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import run_pipeline, to_log_text

load_dotenv()


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
) -> Dict[str, Any]:
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def close() -> None:
        client.close()

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        return CheckpointHandle(
            checkpoint_id=str(cp.get("checkpoint_id") or ""),
            state_kind="prefix_logs",
            state_ref=cp,
            metadata={
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        del checkpoint_handle, query_spec, retrieval_options
        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in memory_pool],
            debug_metadata={
                "retrieval_mode": "observed_logs",
                "num_retrieved_logs": len(memory_pool),
                "retrieved_app_log_ids": [log.get("app_log_id") for log in memory_pool],
            },
        )

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        prepare_checkpoint_state=prepare_checkpoint_state,
        retrieve_context_for_query=retrieve_context_for_query,
        baseline_name="icl",
        memory_prompt_mode="inline_memory",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        enable_change_reasoning=enable_change_reasoning,
        enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
        retrieval_options_backend={},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ICL baseline generation for TCE.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to tce_benchmark.json")
    parser.add_argument(
        "--app-logs-path",
        type=Path,
        required=True,
        help="Path to raw app logs (recommended: app_log_large.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for tce_results.json",
    )
    parser.add_argument(
        "--max-visible-logs",
        type=int,
        default=None,
        help="Optional tail truncation for debugging. Default uses full history until checkpoint.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="openai|azure|aimlapi|gemini|vllm")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--llm-max-workers", type=int, default=1, help="Max workers for LLM client")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug artifacts (save prompt/raw outputs per checkpoint).")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug directory path.")
    parser.add_argument("--save-prompt-and-raw", action="store_true", help="Save prompt and raw model output into prediction metadata.")
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Only run the first N checkpoints (for quick debugging).",
    )
    args = parser.parse_args()

    result = run_generation(
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
    )

    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
