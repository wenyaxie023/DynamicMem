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
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import run_pipeline, to_log_text

load_dotenv()


def _build_openai_client(provider: str) -> OpenAI:
    if provider not in {"openai", "azure"}:
        raise ValueError(f"Unsupported retriever provider for embeddings: {provider}")

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        if provider == "azure":
            client_kwargs["default_headers"] = {"api-key": api_key}
    return OpenAI(**client_kwargs)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _log_search_text(log: Dict[str, Any]) -> str:
    return to_log_text(log)


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
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    predict_per_key: bool = True,
    exposure_anchors: Optional[List[int]] = None,
    calendar_anchor_freq: Optional[str] = None,
    exposure_tokenizer_model: str = "gpt-4o-mini",
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
) -> Dict[str, Any]:
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
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
        query_text = str(query_spec.retrieval_query_text or "").strip()
        if not query_text:
            raise ValueError("RAG requires shared QuerySpec.retrieval_query_text.")
        top_k_for_call = retrieval_top_k
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            try:
                top_k_for_call = int(top_k_override)
            except Exception:
                top_k_for_call = retrieval_top_k
        if not memory_pool:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "rag_embedding",
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retriever_provider": retriever_provider,
                    "retriever_model": retriever_model,
                    "retrieved_app_log_ids": [],
                    "top20_similarity": [],
                    "retrieval_query": query_text,
                },
            )

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

        retrieval_query = query_text
        query_emb = _embed_texts(embed_client, retriever_model, [retrieval_query], batch_size=1)[0]

        k = len(memory_pool) if top_k_for_call <= 0 else min(top_k_for_call, len(memory_pool))
        scores = mem_emb @ query_emb
        top_idx = np.argsort(scores)[::-1][:k]
        retrieved = [memory_pool[i] for i in top_idx]
        top20_idx = np.argsort(scores)[::-1][: min(20, len(memory_pool))]
        top20_similarity = [
            {
                "rank": rank + 1,
                "app_log_id": memory_pool[i].get("app_log_id"),
                "similarity": float(scores[i]),
            }
            for rank, i in enumerate(top20_idx)
        ]

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in retrieved],
            debug_metadata={
                "retrieval_mode": "rag_embedding",
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(retrieved),
                "retriever_provider": retriever_provider,
                "retriever_model": retriever_model,
                "retrieved_app_log_ids": [x.get("app_log_id") for x in retrieved],
                "top20_similarity": top20_similarity,
                "retrieval_query": retrieval_query,
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
        baseline_name="rag",
        memory_prompt_mode="inline_memory",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        predict_per_key=predict_per_key,
        exposure_anchors=exposure_anchors,
        calendar_anchor_freq=calendar_anchor_freq,
        exposure_tokenizer_model=exposure_tokenizer_model,
        enable_change_reasoning=enable_change_reasoning,
        enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
        retrieval_options_backend={
            "retriever_provider": retriever_provider,
            "retriever_model": retriever_model,
            "retriever_batch_size": retriever_batch_size,
        },
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
        enable_final_qa=enable_final_qa,
        final_qa_path=final_qa_path,
        final_qa_output_path=final_qa_output_path,
        final_qa_retrieval_top_k=final_qa_retrieval_top_k,
        final_qa_save_prompt_and_raw=final_qa_save_prompt_and_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG baseline generation for TCE.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to tce_benchmark.json",
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
        help="Output path for tce_results.json",
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
    parser.add_argument(
        "--predict-per-key",
        dest="predict_per_key",
        action="store_true",
        help="Predict each Task A target key in an isolated call. Required by the current TCE protocol.",
    )
    parser.add_argument(
        "--no-predict-per-key",
        dest="predict_per_key",
        action="store_false",
        help="Deprecated. Combined Task A retrieval/prompting is no longer supported by the current TCE protocol.",
    )
    parser.set_defaults(predict_per_key=True)
    parser.add_argument(
        "--exposure-anchors",
        type=str,
        default="",
        help="Comma-separated exposure percentages over full app log, e.g. 10,20,30",
    )
    parser.add_argument(
        "--calendar-anchor-freq",
        type=str,
        default="",
        help="Calendar anchor frequency: weekly|monthly|quarterly",
    )
    parser.add_argument(
        "--exposure-tokenizer-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI tokenizer model used for token-based exposure anchors.",
    )
    parser.add_argument(
        "--enable-change-reasoning",
        action="store_true",
        help="Also ask change-reasoning questions for state items changed since previous anchor.",
    )
    parser.add_argument("--enable-rq3-apply-service-qa", action="store_true")
    parser.add_argument("--rq3-apply-fail-on-missing-pack", action="store_true")
    parser.add_argument(
        "--rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_true",
        help="Save rq3 apply prompt/raw metadata.",
    )
    parser.add_argument(
        "--no-rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_false",
        help="Disable saving rq3 apply prompt/raw metadata.",
    )
    parser.set_defaults(rq3_apply_save_prompt_and_raw=True)
    parser.add_argument("--rq3-apply-retrieval-top-k", type=int, default=None)
    args = parser.parse_args()
    exposure_anchors = [
        int(x.strip())
        for x in str(args.exposure_anchors).split(",")
        if x.strip()
    ]

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
        predict_per_key=args.predict_per_key,
        exposure_anchors=exposure_anchors,
        calendar_anchor_freq=(args.calendar_anchor_freq or None),
        exposure_tokenizer_model=args.exposure_tokenizer_model,
        enable_change_reasoning=args.enable_change_reasoning,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=args.rq3_apply_retrieval_top_k,
    )

    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
