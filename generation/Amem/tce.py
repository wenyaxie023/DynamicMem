#!/usr/bin/env python3
import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs):
        return False

from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import run_pipeline
from generation.Amem.amem import (
    _ensure_nltk,
    _load_manifest,
    _snapshot_root,
    setup_logger,
)
from generation.Amem.client import LLMClient

load_dotenv(Path(__file__).resolve().parent / ".env")


_KEYWORD_SPLIT_RE = re.compile(r"(?:\bcosmos\b|[\n,]+)", re.IGNORECASE)
_APP_LOG_ID_RE = re.compile(r'"app_log_id"\s*:\s*"([^"]+)"')


def _build_keyword_generation_prompt(question: str) -> str:
    return (
        "Given the following question, generate several keywords, using 'cosmos' as the separator.\n\n"
        f"Question: {question}\n\n"
        'Format your response as a JSON object with a "keywords" field containing the selected text.\n\n'
        'Example response format:\n{"keywords": "keyword1 cosmos keyword2 cosmos keyword3"}'
    )


def _parse_generated_keywords(raw_text: Any, fallback: str) -> str:
    if isinstance(raw_text, dict):
        kw_value = raw_text.get("keywords")
        if isinstance(kw_value, str):
            candidate = kw_value
        else:
            return fallback
        pieces = [piece.strip() for piece in _KEYWORD_SPLIT_RE.split(candidate) if piece.strip()]
        return ", ".join(pieces) if pieces else fallback
    if not isinstance(raw_text, str):
        return fallback
    candidate = str(raw_text).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            kw_value = parsed.get("keywords")
            if isinstance(kw_value, str):
                candidate = kw_value
    except Exception:
        pass
    pieces = [piece.strip() for piece in _KEYWORD_SPLIT_RE.split(candidate) if piece.strip()]
    return ", ".join(pieces) if pieces else fallback


def _extract_app_log_ids_from_native_context(raw_context: Any) -> List[str]:
    if not isinstance(raw_context, str):
        return []
    seen = set()
    out: List[str] = []
    for value in _APP_LOG_ID_RE.findall(raw_context):
        app_log_id = str(value or "").strip()
        if not app_log_id or app_log_id in seen:
            continue
        seen.add(app_log_id)
        out.append(app_log_id)
    return out


class _NativeAnswerPathRetriever:
    """Thin adapter that reuses the upstream/native retrieval path only."""

    def __init__(self, *, memory_system: Any, ask_json_fn: Callable[[str], Any], retrieve_k: int):
        self.memory_system = memory_system
        self.ask_json = ask_json_fn
        self.retrieve_k = int(retrieve_k)

    def retrieve_memory(self, content: str, k: int = 10) -> str:
        return self.memory_system.find_related_memories_raw(content, k=k)

    def generate_query_llm(self, question: str) -> str:
        prompt = _build_keyword_generation_prompt(question)
        response = self.ask_json(prompt)
        return _parse_generated_keywords(response, question)

    def retrieve_context(self, question: str, *, k: Optional[int] = None) -> Tuple[str, str]:
        effective_k = self.retrieve_k if k is None else int(k)
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=effective_k)
        return keywords, raw_context


def _load_ordered_snapshot_memories(state_path: Path) -> Tuple[List[str], List[Any]]:
    with state_path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        ordered_ids = [f"note_{idx}" for idx in range(1, len(payload["documents"]) + 1)]
        ordered_memories = [
            SimpleNamespace(
                id=memory_id,
                content=str(document),
                links=[],
                timestamp="",
                context="General",
                keywords=[],
                tags=[],
                category="Uncategorized",
            )
            for memory_id, document in zip(ordered_ids, payload["documents"])
        ]
        return ordered_ids, ordered_memories
    if isinstance(payload, dict):
        ordered_ids = list(payload.keys())
        ordered_memories = [payload[memory_id] for memory_id in ordered_ids]
        return ordered_ids, ordered_memories
    return [], []


def _load_snapshot_bundle(snapshot_root: Path, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = manifest_entry.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Invalid snapshot manifest entry: missing non-empty 'snapshot_id'.")

    checkpoint_path = Path(
        str(manifest_entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json")))
    )
    state_path = Path(str(manifest_entry.get("state_path", str(Path(snapshot_id) / "state.pkl"))))
    meta_path = Path(str(manifest_entry.get("meta_path", str(Path(snapshot_id) / "meta.json"))))
    chroma_dir = Path(str(manifest_entry.get("chroma_dir", str(Path(snapshot_id) / "chroma"))))

    if not checkpoint_path.is_absolute():
        checkpoint_path = snapshot_root / checkpoint_path
    if not state_path.is_absolute():
        state_path = snapshot_root / state_path
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
        "state_path": state_path,
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

    if snapshot_root:
        resolved_snapshot_root = Path(snapshot_root)
    else:
        if not snapshot_dir:
            raise ValueError(
                "Amem standalone generation requires --snapshot-dir or --snapshot-root; no fallback is supported."
            )
        resolved_snapshot_root = _snapshot_root(snapshot_dir, Path("."), user_id, size)
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
    snapshot_state_cache: Dict[str, Tuple[List[str], List[Any]]] = {}
    native_memory_system_cache: Dict[str, Any] = {}
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
        cache_key = (str(bundle["chroma_dir"]), str(bundle.get("collection_name") or "memories"))
        if cache_key not in retriever_cache:
            from generation.Amem.agentic_memory.retrievers import SimpleEmbeddingRetriever

            retriever_cache[cache_key] = SimpleEmbeddingRetriever(
                directory=cache_key[0],
                collection_name=cache_key[1],
                model_name=embedding_model_name,
                embedding_backend=embedding_backend,
                openai_api_key=embedding_api_key,
                openai_api_base=embedding_api_base_url,
                extend=True,
            )
        return retriever_cache[cache_key]

    def _get_snapshot_memories(bundle: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        cache_key = str(bundle.get("state_path") or "")
        if cache_key not in snapshot_state_cache:
            state_path = bundle.get("state_path")
            if not isinstance(state_path, Path):
                raise ValueError("Amem snapshot bundle missing state_path.")
            snapshot_state_cache[cache_key] = _load_ordered_snapshot_memories(state_path)
        return snapshot_state_cache[cache_key]

    def _get_native_memory_system(bundle: Dict[str, Any]) -> Any:
        cache_key = str(bundle.get("state_path") or "")
        if cache_key not in native_memory_system_cache:
            from generation.Amem.agentic_memory.memory_system import AgenticMemorySystem

            ordered_ids, ordered_memories = _get_snapshot_memories(bundle)
            memory_system = AgenticMemorySystem.__new__(AgenticMemorySystem)
            memory_system.memories = {
                memory_id: memory
                for memory_id, memory in zip(ordered_ids, ordered_memories)
            }
            memory_system.retriever = _get_retriever(bundle)
            native_memory_system_cache[cache_key] = memory_system
        return native_memory_system_cache[cache_key]

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
                "retrieval_mode": "amem_snapshot_native",
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
        del memory_pool
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
        memory_system = _get_native_memory_system(bundle)
        k = len(getattr(memory_system, "memories", {}) or {}) if top_k_for_call <= 0 else int(top_k_for_call)
        if k <= 0:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "amem_snapshot_native",
                    "snapshot_id": bundle.get("snapshot_id"),
                    "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                    "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                    "retrieval_top_k": top_k_for_call,
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": [],
                    "generated_keywords": "",
                    "primary_memory_indices": [],
                    "neighbor_memory_indices": [],
                    "retrieved_memory_indices": [],
                    "embedding_backend": embedding_backend,
                    "embedding_model_name": embedding_model_name,
                    "retrieval_query": retrieval_query,
                },
            )

        native_retriever = _NativeAnswerPathRetriever(
            memory_system=memory_system,
            ask_json_fn=ask_json,
            retrieve_k=k,
        )
        generated_keywords, raw_context = native_retriever.retrieve_context(retrieval_query, k=k)
        _emit_usage_update()
        selected_ids = _extract_app_log_ids_from_native_context(raw_context)
        inline_memory_blocks = [raw_context] if str(raw_context or "").strip() else []

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=inline_memory_blocks,
            debug_metadata={
                "retrieval_mode": "amem_snapshot_native",
                "snapshot_id": bundle.get("snapshot_id"),
                "snapshot_checkpoint_id": bundle.get("checkpoint_id"),
                "snapshot_checkpoint_app_log_id": bundle.get("checkpoint_app_log_id"),
                "retrieval_top_k": top_k_for_call,
                "num_retrieved_logs": len(selected_ids),
                "generated_keywords": generated_keywords,
                "retrieved_app_log_ids": selected_ids,
                "embedding_backend": embedding_backend,
                "embedding_model_name": embedding_model_name,
                "retrieval_query": retrieval_query,
                "native_raw_context": str(raw_context or ""),
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
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to the pack-first TCE benchmark JSON (for example tce_benchmark_vnext_*task_packs*.json).",
    )
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
        help="Output path for TCE predictions JSON (for example tce_results_v14_taskabc.json).",
    )
    parser.add_argument("--user-id", type=str, required=True, help="User identifier (e.g., 001_user_001)")
    parser.add_argument("--size", type=str, default="small", help="Snapshot size label: small|medium|large")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Root directory for snapshot bundles. Resolved root is <snapshot-dir>/<user-id>/<size>.",
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
