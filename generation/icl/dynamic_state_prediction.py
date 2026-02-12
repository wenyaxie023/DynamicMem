#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from generation.icl.client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline

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
) -> Dict[str, Any]:
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def close() -> None:
        client.close()

    def retrieve_context(
        _cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        _target_keys: List[str],
    ) -> Dict[str, Any]:
        return {
            "context_logs": memory_pool,
            "context_note": f"Observed app logs up to this checkpoint ({len(memory_pool)} logs)",
            "metadata": {},
        }

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="icl",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ICL baseline generation for dynamic state prediction.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to dynamic_state_prediction_benchmark.json")
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
        help="Output path for dynamic_state_prediction_results.json",
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
