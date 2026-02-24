#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from dynamic_state_prediction_core.pipeline import run_pipeline
from generation.Amem.agentic_memory.retrievers import PersistentChromaRetriever
from generation.Amem.client import LLMClient

load_dotenv(Path(__file__).resolve().parent / ".env")


def _load_snapshot_bundle(snapshot_root: Path, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = manifest_entry.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Invalid snapshot manifest entry: missing non-empty 'snapshot_id'.")

    checkpoint_path = Path(
        str(manifest_entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json")))
    )
    meta_path = Path(str(manifest_entry.get("meta_path", str(Path(snapshot_id) / "meta.json")))
    )
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

    last_event_idx_raw = checkpoint_payload.get("last_event_idx", manifest_entry.get("last_event_idx", -1))
    last_event_idx = int(last_event_idx_raw)

    created_at_raw = manifest_entry.get("created_at")
    created_at = datetime.fromisoformat(created_at_raw) if isinstance(created_at_raw, str) and created_at_raw else datetime.min
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
    user_id: str,
    size: str,
    snapshot_dir: Optional[str],
    checkpoint_dir: Optional[str],
    retrieval_top_k: int,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    embedding_backend: str,
    embedding_model_name: str,
    embedding_api_key: Optional[str],
    embedding_api_base_url: Optional[str],
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    snapshot_root: Optional[str] = None,
) -> Dict[str, Any]:
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )

    default_checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).resolve().parent / "checkpoints"
    resolved_snapshot_root = (
        Path(snapshot_root)
        if snapshot_root
        else (Path(snapshot_dir) if snapshot_dir else default_checkpoint_root / "chroma_snapshots") / user_id / size
    )
    manifest_path = resolved_snapshot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [x for x in manifest.get("snapshots", []) if isinstance(x, dict)]
    bundles = [_load_snapshot_bundle(resolved_snapshot_root, entry) for entry in entries]
    bundles.sort(
        key=lambda x: (int(x.get("last_event_idx", -1)), x.get("created_at", datetime.min)),
        reverse=True,
    )
    bundle_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    bundle_by_app_log_id: Dict[str, Dict[str, Any]] = {}
    for bundle in bundles:
        checkpoint_id = str(bundle.get("checkpoint_id", "")).strip()
        checkpoint_app_log_id = str(bundle.get("checkpoint_app_log_id", "")).strip()
        if checkpoint_id and checkpoint_id not in bundle_by_checkpoint_id:
            bundle_by_checkpoint_id[checkpoint_id] = bundle
        if checkpoint_app_log_id and checkpoint_app_log_id not in bundle_by_app_log_id:
            bundle_by_app_log_id[checkpoint_app_log_id] = bundle

    retriever_cache: Dict[Tuple[str, str], PersistentChromaRetriever] = {}

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def close() -> None:
        client.close()

    def _get_retriever(bundle: Dict[str, Any]) -> PersistentChromaRetriever:
        cache_key = (str(bundle["chroma_dir"]), str(bundle["collection_name"]))
        if cache_key not in retriever_cache:
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

    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        checkpoint_app_log_id = (
            str(as_of.get("app_log_id")).strip() if as_of.get("app_log_id") is not None else ""
        )
        bundle = bundle_by_checkpoint_id.get(checkpoint_id) or bundle_by_app_log_id.get(checkpoint_app_log_id)
        if bundle is None and bundles:
            bundle = bundles[0]

        if bundle is None:
            raise FileNotFoundError(
                "No pre-resolved snapshot bundle found while retrieving context. "
                f"checkpoint_id={checkpoint_id}, app_log_id={as_of.get('app_log_id')}, "
                f"manifest={manifest_path}."
            )

        retrieval_query = _build_retrieval_query(cp, target_keys)

        k = len(memory_pool) if retrieval_top_k <= 0 else min(retrieval_top_k, len(memory_pool))
        if k <= 0:
            return {
                "context_logs": [],
                "context_note": "Retrieved app logs from snapshot Chroma (top-0)",
                "retrieval_query": retrieval_query,
                "metadata": {
                    "retrieval_mode": "amem_snapshot_chroma",
                    "snapshot_id": bundle.get("snapshot_id"),
                    "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                    "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                    "retrieval_top_k": retrieval_top_k,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "embedding_backend": embedding_backend,
                    "embedding_model_name": embedding_model_name,
                },
            }

        retriever = _get_retriever(bundle)
        results = retriever.search(retrieval_query, k=k)

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

        return {
            "context_logs": selected_logs,
            "context_note": f"Retrieved app logs from snapshot Chroma (top-{k})",
            "retrieval_query": retrieval_query,
            "metadata": {
                "retrieval_mode": "amem_snapshot_chroma",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "retrieval_top_k": retrieval_top_k,
                "num_retrieved_logs": len(selected_logs),
                "retrieved_app_log_ids": selected_ids,
                "embedding_backend": embedding_backend,
                "embedding_model_name": embedding_model_name,
            },
        }

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
        baseline_name="amem_baseline",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot-backed Amem baseline generation for dynamic state prediction."
    )
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to dynamic_state_prediction_benchmark.json")
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
        help="Output path for dynamic_state_prediction_results.json",
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
        help="Exact path to the snapshot root directory (contains manifest.json).",
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
        default="openrouter/openai/text-embedding-3-large",
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
