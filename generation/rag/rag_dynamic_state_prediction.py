#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from generation.rag.client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline

load_dotenv()


def _build_openai_client(provider: str) -> OpenAI:
    if provider not in {"openai", "azure"}:
        raise ValueError(f"Unsupported retriever provider for embeddings: {provider}")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        if os.getenv("AZURE_OPENAI_API_KEY") or "azure" in base_url:
            client_kwargs["default_headers"] = {"api-key": api_key}
    return OpenAI(**client_kwargs)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _log_search_text(log: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(log.get("app_name", "")),
            str(log.get("api_name", "")),
            json.dumps(log.get("request", {}), ensure_ascii=False),
            json.dumps(log.get("response", {}), ensure_ascii=False),
        ]
    )


def _embed_texts(
    client: OpenAI,
    model: str,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    chunks: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        arr = np.array([d.embedding for d in resp.data], dtype="float32")
        chunks.append(arr)
    emb = np.vstack(chunks)
    return _normalize_rows(emb)


def _build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    as_of = checkpoint.get("as_of", {})
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])
    return (
        f"Predict values for provided state keys at checkpoint time {ts}. "
        f"Target keys include: {keys_hint}"
    )


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    retriever_provider: str,
    retriever_model: str,
    retriever_batch_size: int,
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

    embed_client = _build_openai_client(retriever_provider)
    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(all_logs_payload, dict):
        all_logs = all_logs_payload.get("app_logs", [])
    elif isinstance(all_logs_payload, list):
        all_logs = all_logs_payload
    else:
        all_logs = []
    all_logs = [x for x in all_logs if isinstance(x, dict)]

    # Build embeddings for all logs once; pipeline passes prefix/tail slices preserving order.
    all_texts = [_log_search_text(log) for log in all_logs]
    all_embeddings = _embed_texts(
        embed_client,
        retriever_model,
        all_texts,
        batch_size=retriever_batch_size,
    )

    # Map app_log_id -> embedding row for O(1) lookup when pipeline gives memory_pool logs.
    emb_by_log_id = {}
    for idx, log in enumerate(all_logs):
        emb_by_log_id[str(log.get("app_log_id", f"idx_{idx}"))] = idx

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def close() -> None:
        client.close()

    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        if not memory_pool:
            return {
                "context_logs": [],
                "context_note": "Retrieved app logs (top-0)",
                "retrieval_query": _build_retrieval_query(cp, target_keys),
                "metadata": {
                    "retrieval_top_k": retrieval_top_k,
                    "num_retrieved_logs": 0,
                    "retriever_provider": retriever_provider,
                    "retriever_model": retriever_model,
                    "retrieved_app_log_ids": [],
                },
            }

        indices = []
        for i, log in enumerate(memory_pool):
            key = str(log.get("app_log_id", f"pool_{i}"))
            idx = emb_by_log_id.get(key)
            if idx is not None:
                indices.append(idx)
            else:
                indices.append(-1)

        # fallback for logs not found in map
        memory_texts = [_log_search_text(log) for log in memory_pool]
        missing_positions = [i for i, idx in enumerate(indices) if idx == -1]
        missing_emb = None
        if missing_positions:
            missing_emb = _embed_texts(
                embed_client,
                retriever_model,
                [memory_texts[i] for i in missing_positions],
                batch_size=retriever_batch_size,
            )

        mem_emb_rows = []
        miss_ptr = 0
        for pos, idx in enumerate(indices):
            if idx == -1:
                mem_emb_rows.append(missing_emb[miss_ptr])
                miss_ptr += 1
            else:
                mem_emb_rows.append(all_embeddings[idx])
        mem_emb = np.vstack(mem_emb_rows).astype("float32")
        mem_emb = _normalize_rows(mem_emb)

        retrieval_query = _build_retrieval_query(cp, target_keys)
        query_emb = _embed_texts(embed_client, retriever_model, [retrieval_query], batch_size=1)[0]

        k = len(memory_pool) if retrieval_top_k <= 0 else min(retrieval_top_k, len(memory_pool))
        scores = mem_emb @ query_emb
        top_idx = np.argsort(scores)[::-1][:k]
        retrieved = [memory_pool[i] for i in top_idx]

        return {
            "context_logs": retrieved,
            "context_note": f"Retrieved app logs (top-{k} from memory pool)",
            "retrieval_query": retrieval_query,
            "metadata": {
                "retrieval_top_k": retrieval_top_k,
                "num_retrieved_logs": len(retrieved),
                "retriever_provider": retriever_provider,
                "retriever_model": retriever_model,
                "retrieved_app_log_ids": [x.get("app_log_id") for x in retrieved],
            },
        }

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="rag",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG baseline generation for dynamic state prediction.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to dynamic_state_prediction_benchmark.json",
    )
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
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Top-k logs retrieved from memory pool for each checkpoint.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="openai|azure|aimlapi|gemini|vllm")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--llm-max-workers", type=int, default=1, help="Max workers for LLM client")
    parser.add_argument("--retriever-provider", type=str, default="openai", help="Embedding provider (openai|azure)")
    parser.add_argument("--retriever-model", type=str, default="text-embedding-3-large", help="Embedding model name")
    parser.add_argument("--retriever-batch-size", type=int, default=64, help="Batch size for embedding requests")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug artifacts (save prompt/raw outputs per checkpoint).")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug directory path.")
    parser.add_argument(
        "--save-prompt-and-raw",
        action="store_true",
        help="Save prompt and raw model output into prediction metadata.",
    )
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
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        retriever_provider=args.retriever_provider,
        retriever_model=args.retriever_model,
        retriever_batch_size=args.retriever_batch_size,
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
