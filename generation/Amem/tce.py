#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs):
        return False

from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import run_pipeline, to_log_text
from generation.Amem.amem import (
    _ensure_nltk,
    _load_manifest,
    _snapshot_root,
    setup_logger,
)
from generation.Amem.client import LLMClient

load_dotenv(Path(__file__).resolve().parent / ".env")


def _load_snapshot_bundle(snapshot_root: Path, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = manifest_entry.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Invalid snapshot manifest entry: missing non-empty 'snapshot_id'.")

    checkpoint_path = Path(
        str(manifest_entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json")))
    )
    meta_path = Path(str(manifest_entry.get("meta_path", str(Path(snapshot_id) / "meta.json"))))
    chroma_dir = Path(str(manifest_entry.get("chroma_dir", str(Path(snapshot_id) / "chroma"))))

    if not checkpoint_path.is_absolute():
        checkpoint_path = snapshot_root / checkpoint_path
    if not meta_path.is_absolute():
        meta_path = snapshot_root / meta_path
    if not chroma_dir.is_absolute():
        chroma_dir = snapshot_root / chroma_dir

    with checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint_payload = json.load(f)

    with meta_path.open("r", encoding="utf-8") as f:
        meta_payload = json.load(f)

    collection_name = meta_payload.get("collection_name") or manifest_entry.get("collection_name")
    last_event_idx = int(checkpoint_payload.get("last_event_idx", manifest_entry.get("last_event_idx", -1)))
    created_at_raw = manifest_entry.get("created_at")
    created_at = (
        datetime.fromisoformat(created_at_raw)
        if isinstance(created_at_raw, str) and created_at_raw
        else datetime.min
    )
    checkpoint_id = checkpoint_payload.get("checkpoint_id", manifest_entry.get("checkpoint_id"))
    checkpoint_app_log_id = checkpoint_payload.get(
        "checkpoint_app_log_id",
        manifest_entry.get("checkpoint_app_log_id"),
    )

    return {
        "snapshot_id": snapshot_id,
        "chroma_dir": chroma_dir,
        "checkpoint_path": checkpoint_path,
        "meta_path": meta_path,
        "collection_name": collection_name,
        "last_event_idx": last_event_idx,
        "created_at": created_at,
        "created_at_raw": created_at_raw,
        "checkpoint_id": str(checkpoint_id).strip() if checkpoint_id is not None else "",
        "checkpoint_app_log_id": (
            str(checkpoint_app_log_id).strip() if checkpoint_app_log_id is not None else ""
        ),
    }


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    user_id: str,
    size: str,
    snapshot_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    retrieval_top_k: int = 5,
    max_visible_logs: Optional[int] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-5-mini",
    llm_max_workers: int = 1,
    embedding_backend: str = "openai",
    embedding_model_name: str = "text-embedding-3-large",
    embedding_api_key: Optional[str] = None,
    embedding_api_base_url: Optional[str] = None,
    resume: bool = False,
    max_checkpoints: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    save_prompt_and_raw: bool = False,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    snapshot_root: Optional[str] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _ensure_nltk()
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    default_checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).resolve().parent / "checkpoints"
    resolved_snapshot_root = (
        Path(snapshot_root)
        if snapshot_root
        else _snapshot_root(snapshot_dir, default_checkpoint_root, user_id, size)
    )
    manifest_path = resolved_snapshot_root / "manifest.json"
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / f"amem_tce_{user_id}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log")

    bundle_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    bundle_by_app_log_id: Dict[str, Dict[str, Any]] = {}

    def _register_snapshot_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        bundle = _load_snapshot_bundle(resolved_snapshot_root, entry)
        checkpoint_id = str(bundle.get("checkpoint_id", "")).strip()
        checkpoint_app_log_id = str(bundle.get("checkpoint_app_log_id", "")).strip()
        if checkpoint_id:
            bundle_by_checkpoint_id[checkpoint_id] = bundle
        if checkpoint_app_log_id:
            bundle_by_app_log_id[checkpoint_app_log_id] = bundle
        return bundle

    def _refresh_snapshot_index() -> None:
        bundle_by_checkpoint_id.clear()
        bundle_by_app_log_id.clear()
        if not manifest_path.exists():
            return
        manifest = _load_manifest(manifest_path)
        entries = [x for x in manifest.get("snapshots", []) if isinstance(x, dict)]
        entries.sort(
            key=lambda entry: (
                int(entry.get("last_event_idx", -1)) if isinstance(entry.get("last_event_idx", -1), int) else -1,
                str(entry.get("created_at", "")),
            ),
            reverse=True,
            )
        for entry in entries:
            _register_snapshot_entry(entry)

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Amem test phase requires prebuilt checkpoint snapshots. "
            f"Missing manifest: {manifest_path}"
        )
    _refresh_snapshot_index()
    retriever_cache: Dict[Tuple[str, str], Any] = {}
    answer_llm_usage: Dict[str, Any] = {}

    def _emit_usage_update() -> None:
        if not callable(usage_callback):
            return
        usage_summary_fn = getattr(client, "usage_summary", None)
        answer_usage: Dict[str, Any] = {}
        if callable(usage_summary_fn):
            raw_usage = usage_summary_fn()
            if isinstance(raw_usage, dict):
                answer_usage = raw_usage
        usage_callback(answer_usage)

    def ask_json(prompt: str) -> Any:
        result = client.ask(prompt, response_type="json")
        _emit_usage_update()
        return result

    def ask_structured(prompt: str, text_format: Any) -> Any:
        result = client.ask_structured(prompt, text_format=text_format)
        _emit_usage_update()
        return result

    def close() -> None:
        nonlocal answer_llm_usage
        usage_summary_fn = getattr(client, "usage_summary", None)
        if callable(usage_summary_fn):
            raw_usage = usage_summary_fn()
            answer_llm_usage = raw_usage if isinstance(raw_usage, dict) else {}
        _emit_usage_update()
        client.close()

    def _get_retriever(bundle: Dict[str, Any]) -> Any:
        cache_key = (str(bundle["chroma_dir"]), str(bundle["collection_name"]))
        if cache_key not in retriever_cache:
            from generation.Amem.agentic_memory.retrievers import PersistentChromaRetriever

            retriever_cache[cache_key] = PersistentChromaRetriever(
                directory=cache_key[0],
                collection_name=cache_key[1],
                model_name=embedding_model_name,
                embedding_backend=embedding_backend,
                openai_api_key=embedding_api_key,
                openai_api_base=embedding_api_base_url,
                extend=True,
            )
        return retriever_cache[cache_key]

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        _refresh_snapshot_index()
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        checkpoint_app_log_id = (
            str(as_of.get("app_log_id")).strip() if as_of.get("app_log_id") is not None else ""
        )
        bundle = bundle_by_checkpoint_id.get(checkpoint_id) or bundle_by_app_log_id.get(checkpoint_app_log_id)
        if bundle is None:
            raise FileNotFoundError(
                "Amem checkpoint snapshot not found for "
                f"checkpoint_id={checkpoint_id or '<unknown>'}, "
                f"checkpoint_app_log_id={checkpoint_app_log_id or '<unknown>'}. "
                "Run the Amem builder phase first."
            )

        return CheckpointHandle(
            checkpoint_id=checkpoint_id,
            state_kind="snapshot_bundle",
            state_ref=bundle,
            metadata={
                "retrieval_mode": "amem_snapshot_chroma",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": checkpoint_app_log_id,
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        bundle = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        retrieval_query = str(query_spec.retrieval_query_text or "").strip()
        if not retrieval_query:
            raise ValueError("Amem requires shared QuerySpec.retrieval_query_text.")
        top_k_for_call = retrieval_top_k
        top_k_override = retrieval_options.common.get("top_k")
        if isinstance(top_k_override, int):
            try:
                top_k_for_call = int(top_k_override)
            except Exception:
                top_k_for_call = retrieval_top_k
        k = len(memory_pool) if top_k_for_call <= 0 else min(top_k_for_call, len(memory_pool))
        if k <= 0:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "amem_snapshot_chroma",
                    "snapshot_id": bundle.get("snapshot_id"),
                    "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                    "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "embedding_backend": embedding_backend,
                    "embedding_model_name": embedding_model_name,
                    "retrieval_query": retrieval_query,
                },
            )

        retriever = _get_retriever(bundle)
        results = retriever.search(retrieval_query, k=k)
        _emit_usage_update()

        memory_pool_by_id: Dict[str, Dict[str, Any]] = {}
        for i, log in enumerate(memory_pool):
            app_log_id = log.get("app_log_id")
            key = str(app_log_id).strip() if app_log_id is not None else f"pool_{i}"
            if key and key not in memory_pool_by_id:
                memory_pool_by_id[key] = log

        raw_documents = results.get("documents", [])
        documents = raw_documents[0] if raw_documents and isinstance(raw_documents[0], list) else []
        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids = set()
        for doc in documents:
            payload = json.loads(str(doc))
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

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in selected_logs],
            debug_metadata={
                "retrieval_mode": "amem_snapshot_chroma",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "embedding_backend": embedding_backend,
                "embedding_model_name": embedding_model_name,
                "retrieval_query": retrieval_query,
            },
        )

    result = run_pipeline(
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
        baseline_name="amem",
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
        retrieval_options_backend={
            "embedding_backend": embedding_backend,
            "embedding_model_name": embedding_model_name,
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
    if isinstance(result, dict):
        result["answer_llm_usage"] = dict(answer_llm_usage)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amem baseline generation for TCE."
    )
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to tce_benchmark.json")
    parser.add_argument(
        "--app-logs-path",
        type=Path,
        required=True,
        help="Path to raw app logs JSON (recommended: app_log_large.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for tce_results.json",
    )
    parser.add_argument("--user-id", type=str, required=True, help="User identifier (e.g., 001_user_001)")
    parser.add_argument("--size", type=str, default="small", help="Snapshot size label: small|medium|large")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Root directory for snapshot bundles. Defaults to <checkpoint-dir>/chroma_snapshots.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory used to derive default snapshot root.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=None,
        help="Exact path to the snapshot root directory.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Top-k memories retrieved from snapshot Chroma per checkpoint.",
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
        "--embedding-backend",
        type=str,
        default="openai",
        help="Embedding backend for snapshot Chroma retrieval (openai|sentence-transformers).",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model name used for snapshot retrieval.",
    )
    parser.add_argument("--embedding-api-key", type=str, default=None, help="Embedding API key")
    parser.add_argument(
        "--embedding-api-base-url",
        type=str,
        default=None,
        help="Embedding API base URL",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if available")
    parser.add_argument("--debug", action="store_true", help="Write per-checkpoint debug artifacts")
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
        size=args.size,
        snapshot_dir=args.snapshot_dir,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_root=args.snapshot_root,
        retrieval_top_k=args.retrieval_top_k,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        embedding_backend=args.embedding_backend,
        embedding_model_name=args.embedding_model_name,
        embedding_api_key=args.embedding_api_key,
        embedding_api_base_url=args.embedding_api_base_url,
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
