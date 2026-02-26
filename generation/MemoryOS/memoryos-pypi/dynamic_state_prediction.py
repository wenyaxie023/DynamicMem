#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline
from dynamic_state_prediction_core.retrieval_query import build_retrieval_query
from memoryos import Memoryos

load_dotenv(Path(__file__).resolve().parent / ".env")

ASSISTANT_ID = "assistant"
DATA_STORAGE_PATH = ""
LLM_CONTROLLER_API_KEY = os.getenv("LLM_CONTROLLER_API_KEY")
LLM_CONTROLLER_API_BASE_URL = os.getenv("LLM_CONTROLLER_API_BASE_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL")


def _load_snapshot_bundle(snapshot_root: Path, cp: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = snapshot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots", []) if isinstance(manifest, dict) else []
    snapshots = [x for x in snapshots if isinstance(x, dict)]

    checkpoint_id = str(cp.get("checkpoint_id", ""))
    as_of = cp.get("as_of")
    as_of = as_of if isinstance(as_of, dict) else {}
    app_log_id_raw = as_of.get("app_log_id")
    checkpoint_app_log_id = str(app_log_id_raw) if app_log_id_raw is not None else ""

    candidates: List[Dict[str, Any]] = []
    for entry in snapshots:
        if checkpoint_id and str(entry.get("checkpoint_id", "")) == checkpoint_id:
            candidates.append(entry)
    if not candidates and checkpoint_app_log_id:
        for entry in snapshots:
            if str(entry.get("checkpoint_app_log_id", "")) == checkpoint_app_log_id:
                candidates.append(entry)
    if not candidates:
        candidates = snapshots

    candidates.sort(
        key=lambda x: (int(x.get("last_event_idx", -1)), str(x.get("created_at", ""))),
        reverse=True,
    )
    latest = candidates[0]
    snapshot_id = str(latest.get("snapshot_id"))

    return {
        "snapshot_id": snapshot_id,
        "checkpoint_id": latest.get("checkpoint_id"),
        "checkpoint_app_log_id": latest.get("checkpoint_app_log_id"),
        "short_term_path": snapshot_root / latest.get("short_term_path", str(Path(snapshot_id) / "short_term.json")),
        "mid_term_path": snapshot_root / latest.get("mid_term_path", str(Path(snapshot_id) / "mid_term.json")),
        "long_term_path": snapshot_root / latest.get("long_term_path", str(Path(snapshot_id) / "long_term_user.json")),
        "assistant_long_term_path": snapshot_root / latest.get(
            "assistant_long_term_path",
            str(Path(snapshot_id) / "long_term_assistant.json"),
        ),
    }


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    user_id: str,
    snapshot_dir: Optional[str],
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    embedding_model_name: str,
    llm_controller_model: str,
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

    memory_user_id = f"{user_id}_large"
    data_storage_root = Path(os.path.abspath(DATA_STORAGE_PATH or ""))
    memo = Memoryos(
        user_id=memory_user_id,
        openai_api_key=LLM_CONTROLLER_API_KEY,
        openai_base_url=LLM_CONTROLLER_API_BASE_URL,
        data_storage_path=str(data_storage_root),
        llm_model=llm_controller_model,
        assistant_id=ASSISTANT_ID,
        short_term_capacity=7,
        mid_term_heat_threshold=5,
        retrieval_queue_capacity=10,
        long_term_knowledge_capacity=100,
        mid_term_similarity_threshold=0.6,
        embedding_model_name=embedding_model_name,
        embedding_model_kwargs={
            "embedding_backend": "openai",
            "api_key": EMBEDDING_API_KEY,
            "api_base": EMBEDDING_API_BASE_URL,
            "max_input_tokens": 8000,
            "truncate_from": "end",
        },
    )

    snapshot_root = (
        Path(snapshot_dir) / memory_user_id
        if snapshot_dir
        else (Path(__file__).resolve().parent / "snapshots" / memory_user_id)
    )

    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        bundle = _load_snapshot_bundle(snapshot_root, cp)
        shutil.copyfile(bundle["short_term_path"], memo.short_term_memory.file_path)
        shutil.copyfile(bundle["mid_term_path"], memo.mid_term_memory.file_path)
        shutil.copyfile(bundle["long_term_path"], memo.user_long_term_memory.file_path)
        shutil.copyfile(bundle["assistant_long_term_path"], memo.assistant_long_term_memory.file_path)
        memo.short_term_memory.load()
        memo.mid_term_memory.load()
        memo.user_long_term_memory.load()
        memo.assistant_long_term_memory.load()

        retrieval_query = build_retrieval_query(cp, target_keys)
        retrieval = memo.retriever.retrieve_context(
            user_query=retrieval_query,
            user_id=memo.user_id,
        )

        memory_pool_by_id: Dict[str, Dict[str, Any]] = {}
        for i, log in enumerate(memory_pool):
            app_log_id = log.get("app_log_id")
            key = str(app_log_id).strip() if app_log_id is not None else f"pool_{i}"
            if key and key not in memory_pool_by_id:
                memory_pool_by_id[key] = log

        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids = set()
        for page in retrieval.get("retrieved_pages", []):
            user_input = page.get("user_input", "")
            for part in str(user_input).split("[APP_LOG] "):
                chunk = part.strip()
                if not chunk:
                    continue
                payload = json.loads(chunk.splitlines()[0].strip())
                if not isinstance(payload, dict):
                    continue
                app_log_id = payload.get("app_log_id")
                if app_log_id is None:
                    continue
                app_log_id = str(app_log_id).strip()
                if not app_log_id or app_log_id in seen_ids:
                    continue
                log = memory_pool_by_id.get(app_log_id)
                if log is None:
                    continue
                seen_ids.add(app_log_id)
                selected_ids.append(app_log_id)
                selected_logs.append(log)

        k = len(memory_pool) if retrieval_top_k <= 0 else min(retrieval_top_k, len(memory_pool))
        selected_logs = selected_logs[:k]
        selected_ids = selected_ids[:k]

        return {
            "context_logs": selected_logs,
            "context_note": f"Retrieved app logs from MemoryOS snapshot (top-{k})",
            "retrieval_query": retrieval_query,
            "metadata": {
                "retrieval_mode": "memoryos_snapshot",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "retrieval_top_k": retrieval_top_k,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "embedding_model_name": embedding_model_name,
            },
        }

    def ask_json(prompt: str) -> Any:

        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
       
        return client.ask_structured(prompt, text_format=text_format)
       
    def close() -> None:
        client.close()

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="memoryos",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemoryOS baseline generation for dynamic state prediction.")
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
    parser.add_argument("--user-id", type=str, required=True, help="User identifier (e.g., 001_user_001)")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "snapshots"),
        help="Root directory containing per-user snapshot folders.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Top-k logs retrieved from MemoryOS snapshot per checkpoint.",
    )
    parser.add_argument(
        "--max-visible-logs",
        type=int,
        default=None,
        help="Optional tail truncation for pipeline memory_pool; default uses full observed history.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="openai|azure|aimlapi|gemini|vllm")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--llm-max-workers", type=int, default=1, help="Max workers for LLM client")
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model name used by MemoryOS retriever.",
    )
    parser.add_argument(
        "--llm-controller-model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Controller LLM model used inside MemoryOS.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug artifacts")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug artifact directory")
    parser.add_argument(
        "--save-prompt-and-raw",
        action="store_true",
        help="Save prompt and raw model output into prediction metadata.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Only run first N checkpoints for debugging.",
    )
    args = parser.parse_args()

    result = run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        user_id=args.user_id,
        snapshot_dir=args.snapshot_dir,
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        embedding_model_name=args.embedding_model_name,
        llm_controller_model=args.llm_controller_model,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )
    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))
