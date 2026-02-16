#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from generation.Amem.agentic_memory.retrievers import PersistentChromaRetriever
from generation.Amem.client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline

load_dotenv(Path(__file__).resolve().parent / ".env")


def _snapshot_root(
    snapshot_dir: Optional[str],
    checkpoint_dir: Path,
    user_id: str,
    size: str,
) -> Path:
    base = Path(snapshot_dir) if snapshot_dir else (checkpoint_dir / "chroma_snapshots")
    return base / user_id / size


def _snapshot_manifest_path(snapshot_root: Path) -> Path:
    return snapshot_root / "manifest.json"


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid snapshot manifest format at {path}: expected top-level object.")

    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError(
            f"Invalid snapshot manifest format at {path}: expected 'snapshots' to be a list."
        )
    return payload


def _manifest_entry_path(snapshot_root: Path, path_value: Any) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return snapshot_root / p


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min
    return datetime.min


def _entry_sort_key(entry: Dict[str, Any]) -> Tuple[int, datetime]:
    last_event_idx_raw = entry.get("last_event_idx", -1)
    try:
        last_event_idx = int(last_event_idx_raw)
    except (TypeError, ValueError):
        last_event_idx = -1

    created_at = entry.get("created_at")
    if isinstance(created_at, datetime):
        created_at_dt = created_at
    else:
        created_at_dt = _parse_created_at(created_at)
    return last_event_idx, created_at_dt


def _load_snapshot_bundle(snapshot_root: Path, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = manifest_entry.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Invalid snapshot manifest entry: missing non-empty 'snapshot_id'.")

    checkpoint_path = _manifest_entry_path(
        snapshot_root,
        manifest_entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json")),
    )
    meta_path = _manifest_entry_path(
        snapshot_root,
        manifest_entry.get("meta_path", str(Path(snapshot_id) / "meta.json")),
    )
    chroma_dir = _manifest_entry_path(
        snapshot_root,
        manifest_entry.get("chroma_dir", str(Path(snapshot_id) / "chroma")),
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Snapshot artifact missing for '{snapshot_id}': checkpoint.json not found at {checkpoint_path}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Snapshot artifact missing for '{snapshot_id}': meta.json not found at {meta_path}"
        )
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Snapshot artifact missing for '{snapshot_id}': chroma directory not found at {chroma_dir}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint_payload = json.load(f)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}: expected object.")

    with meta_path.open("r", encoding="utf-8") as f:
        meta_payload = json.load(f)
    if not isinstance(meta_payload, dict):
        raise ValueError(f"Invalid meta payload at {meta_path}: expected object.")

    collection_name = meta_payload.get("collection_name") or manifest_entry.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError(
            f"Missing collection name for snapshot '{snapshot_id}'. "
            f"Checked {meta_path} and manifest entry."
        )

    last_event_idx_raw = checkpoint_payload.get("last_event_idx", manifest_entry.get("last_event_idx", -1))
    try:
        last_event_idx = int(last_event_idx_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid last_event_idx for snapshot '{snapshot_id}' in {checkpoint_path}: {last_event_idx_raw}"
        ) from e

    created_at = _parse_created_at(manifest_entry.get("created_at"))
    checkpoint_id = checkpoint_payload.get("checkpoint_id", manifest_entry.get("checkpoint_id"))
    checkpoint_app_log_id = checkpoint_payload.get(
        "checkpoint_app_log_id",
        manifest_entry.get("checkpoint_app_log_id"),
    )

    return {
        "snapshot_id": snapshot_id,
        "chroma_dir": str(chroma_dir),
        "checkpoint_path": str(checkpoint_path),
        "meta_path": str(meta_path),
        "collection_name": collection_name,
        "last_event_idx": last_event_idx,
        "created_at": created_at,
        "created_at_raw": manifest_entry.get("created_at"),
        "checkpoint_id": str(checkpoint_id).strip() if checkpoint_id is not None else "",
        "checkpoint_app_log_id": (
            str(checkpoint_app_log_id).strip() if checkpoint_app_log_id is not None else ""
        ),
    }


def _build_snapshot_indices(
    snapshot_root: Path,
) -> Tuple[Path, Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    manifest_path = _snapshot_manifest_path(snapshot_root)
    manifest = _load_manifest(manifest_path)
    raw_entries = manifest.get("snapshots", [])

    if not raw_entries:
        raise FileNotFoundError(
            f"No snapshots listed in manifest: {manifest_path}. "
            "Run generation/Amem/membench_amem.py to produce snapshots first."
        )

    bundles: List[Dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid snapshot entry in manifest {manifest_path}: expected object, got {type(entry)}."
            )
        bundles.append(_load_snapshot_bundle(snapshot_root, entry))

    by_checkpoint_id: Dict[str, List[Dict[str, Any]]] = {}
    by_checkpoint_app_log_id: Dict[str, List[Dict[str, Any]]] = {}
    for bundle in bundles:
        checkpoint_id = bundle.get("checkpoint_id", "")
        checkpoint_app_log_id = bundle.get("checkpoint_app_log_id", "")
        if checkpoint_id:
            by_checkpoint_id.setdefault(checkpoint_id, []).append(bundle)
        if checkpoint_app_log_id:
            by_checkpoint_app_log_id.setdefault(checkpoint_app_log_id, []).append(bundle)

    for bucket in by_checkpoint_id.values():
        bucket.sort(key=_entry_sort_key, reverse=True)
    for bucket in by_checkpoint_app_log_id.values():
        bucket.sort(key=_entry_sort_key, reverse=True)

    return manifest_path, by_checkpoint_id, by_checkpoint_app_log_id


def _load_app_logs(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    elif isinstance(payload, list):
        logs = payload
    else:
        logs = []
    return [x for x in logs if isinstance(x, dict)]


def _build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    as_of = checkpoint.get("as_of", {})
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])
    return (
        f"Predict values for provided state keys at checkpoint time {ts}. "
        f"Target keys include: {keys_hint}"
    )


def _parse_json_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_app_log_id_from_result_item(metadata: Any, document: Any) -> Optional[str]:
    if isinstance(metadata, dict):
        direct = metadata.get("app_log_id")
        if direct is not None:
            direct_str = str(direct).strip()
            if direct_str:
                return direct_str
        content_payload = _parse_json_object(metadata.get("content"))
        if isinstance(content_payload, dict):
            nested = content_payload.get("app_log_id")
            if nested is not None:
                nested_str = str(nested).strip()
                if nested_str:
                    return nested_str

    doc_payload = _parse_json_object(document)
    if isinstance(doc_payload, dict):
        nested = doc_payload.get("app_log_id")
        if nested is not None:
            nested_str = str(nested).strip()
            if nested_str:
                return nested_str
    return None


def _extract_app_log_ids_from_results(results: Dict[str, Any]) -> List[str]:
    ids_nested = results.get("ids")
    if not isinstance(ids_nested, list) or (ids_nested and not isinstance(ids_nested[0], list)):
        raise ValueError("Unexpected retriever query result format: expected nested list in 'ids'.")

    metadatas_nested = results.get("metadatas")
    documents_nested = results.get("documents")

    ids = ids_nested[0] if ids_nested else []
    metadatas = (
        metadatas_nested[0]
        if isinstance(metadatas_nested, list) and metadatas_nested and isinstance(metadatas_nested[0], list)
        else []
    )
    documents = (
        documents_nested[0]
        if isinstance(documents_nested, list) and documents_nested and isinstance(documents_nested[0], list)
        else []
    )

    out: List[str] = []
    for idx in range(len(ids)):
        metadata = metadatas[idx] if idx < len(metadatas) else None
        document = documents[idx] if idx < len(documents) else None
        app_log_id = _extract_app_log_id_from_result_item(metadata, document)
        if app_log_id:
            out.append(app_log_id)
    return out


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
) -> Dict[str, Any]:
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (Path(__file__).resolve().parent / "checkpoints")
    snapshot_root = _snapshot_root(snapshot_dir, ckpt_dir, user_id, size)
    manifest_path, by_checkpoint_id, by_checkpoint_app_log_id = _build_snapshot_indices(snapshot_root)

    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_checkpoints = benchmark_payload.get("checkpoints", []) if isinstance(benchmark_payload, dict) else []
    if not isinstance(benchmark_checkpoints, list):
        raise ValueError(
            f"Invalid benchmark format at {benchmark_path}: expected top-level 'checkpoints' list."
        )
    if max_checkpoints is not None:
        benchmark_checkpoints = benchmark_checkpoints[:max_checkpoints]

    resolved_bundle_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    for cp in benchmark_checkpoints:
        if not isinstance(cp, dict):
            continue
        checkpoint_id = str(cp.get("checkpoint_id"))
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        app_log_id_raw = as_of.get("app_log_id")
        app_log_id = str(app_log_id_raw).strip() if app_log_id_raw is not None else ""

        candidates = by_checkpoint_id.get(checkpoint_id) or []
        if not candidates and app_log_id:
            candidates = by_checkpoint_app_log_id.get(app_log_id) or []

        if not candidates:
            raise FileNotFoundError(
                "No snapshot bundle could be resolved for benchmark checkpoint. "
                f"checkpoint_id={checkpoint_id}, app_log_id={app_log_id or '<none>'}, "
                f"snapshot_root={snapshot_root}, manifest={manifest_path}. "
                "Generate snapshots first with generation/Amem/membench_amem.py."
            )

        resolved_bundle_by_checkpoint_id[checkpoint_id] = candidates[0]

    all_logs = _load_app_logs(app_logs_path)
    all_logs_by_id: Dict[str, Dict[str, Any]] = {}
    for idx, log in enumerate(all_logs):
        app_log_id = log.get("app_log_id")
        key = str(app_log_id).strip() if app_log_id is not None else f"idx_{idx}"
        if key and key not in all_logs_by_id:
            all_logs_by_id[key] = log

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
        checkpoint_id = str(cp.get("checkpoint_id"))
        bundle = resolved_bundle_by_checkpoint_id.get(checkpoint_id)
        if bundle is None:
            as_of = cp.get("as_of")
            as_of = as_of if isinstance(as_of, dict) else {}
            raise FileNotFoundError(
                "No pre-resolved snapshot bundle found while retrieving context. "
                f"checkpoint_id={checkpoint_id}, app_log_id={as_of.get('app_log_id')}, "
                f"manifest={manifest_path}."
            )

        retrieval_query = _build_retrieval_query(cp, target_keys)
        if not memory_pool:
            return {
                "context_logs": [],
                "context_note": "Retrieved app logs from snapshot (top-0)",
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

        k = len(memory_pool) if retrieval_top_k <= 0 else min(retrieval_top_k, len(memory_pool))
        if k <= 0:
            return {
                "context_logs": [],
                "context_note": "Retrieved app logs from snapshot (top-0)",
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
        retrieved_ids = _extract_app_log_ids_from_results(results)

        selected_logs: List[Dict[str, Any]] = []
        selected_ids: List[str] = []
        seen_ids = set()
        for app_log_id in retrieved_ids:
            if app_log_id in seen_ids:
                continue
            log = all_logs_by_id.get(app_log_id)
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
