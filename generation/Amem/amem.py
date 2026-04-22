import os
import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

try:
    import nltk
except Exception:  # pragma: no cover - optional in stripped test environments
    nltk = None  # type: ignore[assignment]
from dotenv import load_dotenv

try:
    from generation.Amem.agentic_memory.memory_system import AgenticMemorySystem
except Exception:  # pragma: no cover - optional in test environments without chromadb
    AgenticMemorySystem = None  # type: ignore[assignment]
from generation.Amem.usage_sidecar import (
    build_usage_sidecar_paths,
    empty_usage_summary,
    load_usage_sidecar,
    merge_usage_summary,
    write_build_usage_sidecar,
)

try:
    from generation.Amem.agentic_memory.llm_controller import (
        get_usage_summary as amem_usage_summary,
        reset_usage_tracker as reset_amem_usage_tracker,
    )
except Exception:  # pragma: no cover - optional in stripped test environments
    def amem_usage_summary() -> Dict[str, Any]:
        return empty_usage_summary()

    def reset_amem_usage_tracker() -> None:
        return None

load_dotenv(Path(__file__).resolve().parent / ".env")

GENERATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GENERATION_DIR / "data"
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from generation.load_dataset import build_membench_memory_from_event, load_membench_dataset


def _env_value(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        value = value.strip()
        if not value:
            continue
        if value.endswith(","):
            value = value[:-1].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            return value
    return None
def _ensure_nltk() -> None:
    if nltk is None:
        raise RuntimeError("Missing nltk package. Install it to run Amem.")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        raise RuntimeError(
            "Missing NLTK data 'punkt'. Install it with: python -m nltk.downloader punkt"
        ) from e


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("membench_amem")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _load_checkpoint(path: Path, logger: logging.Logger) -> tuple[int, int]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
        last_event_idx = int(payload.get("last_event_idx", -1))
        events_processed = int(payload.get("events_processed", last_event_idx + 1))
        return last_event_idx, events_processed
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return -1, 0


def _write_checkpoint(path: Path, last_event_idx: int, processed: int) -> None:
    payload = {
        "last_event_idx": last_event_idx,
        "events_processed": processed,
        "saved_at": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _checkpoint_paths(ckpt_dir: Path, user_id: str, size_label: str) -> tuple[Path, Path]:
    state_path = ckpt_dir / f"membench_amem_{user_id}_{size_label}.pkl"
    checkpoint_path = ckpt_dir / f"membench_amem_{user_id}_{size_label}.json"
    return state_path, checkpoint_path


def _snapshot_root(
    snapshot_dir: Optional[str],
    ckpt_dir: Path,
    user_id: str,
    size_label: str,
) -> Path:
    base = Path(snapshot_dir).expanduser().resolve() if snapshot_dir else (ckpt_dir / "chroma_snapshots").resolve()
    return base / user_id / size_label


def _snapshot_manifest_path(snapshot_root: Path) -> Path:
    return snapshot_root / "manifest.json"


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"latest_snapshot": None, "snapshots": []}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest format at {path}: expected object.")
    snapshots = payload.get("snapshots")
    if snapshots is None:
        payload["snapshots"] = []
    elif not isinstance(snapshots, list):
        raise ValueError(f"Invalid manifest format at {path}: 'snapshots' must be a list.")
    if "latest_snapshot" not in payload:
        payload["latest_snapshot"] = None
    return payload


def _snapshot_sort_key(entry: Dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(entry.get("last_event_idx", -1)) if isinstance(entry.get("last_event_idx", -1), int) else -1,
        str(entry.get("created_at", "")),
        str(entry.get("snapshot_id", "")),
    )


def _latest_snapshot_entry(snapshot_root: Path) -> Optional[Dict[str, Any]]:
    manifest = _load_manifest(_snapshot_manifest_path(snapshot_root))
    entries = [x for x in manifest.get("snapshots", []) if isinstance(x, dict)]
    if not entries:
        return None
    entries.sort(key=_snapshot_sort_key, reverse=True)
    return entries[0]


def _load_snapshot_checkpoint_payload(snapshot_root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_id = str(entry.get("snapshot_id") or "")
    checkpoint_path = snapshot_root / entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json"))
    if not checkpoint_path.exists():
        return {}
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_snapshot_state_path(snapshot_root: Path, entry: Dict[str, Any]) -> Path:
    snapshot_id = str(entry.get("snapshot_id") or "")
    return snapshot_root / entry.get("state_path", str(Path(snapshot_id) / "state.pkl"))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_checkpoints_from_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    with benchmark_path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    if not isinstance(checkpoints, list):
        raise ValueError(
            f"Invalid checkpoint format in {benchmark_path}: expected top-level 'checkpoints' list."
        )
    return checkpoints


def _write_snapshot_bundle(
    memory_system: AgenticMemorySystem,
    snapshot_root: Path,
    collection_name: str,
    last_event_idx: int,
    processed: int,
    logger: logging.Logger,
    *,
    trigger: str,
    checkpoint_id: Optional[str],
    checkpoint_app_log_id: Optional[str],
) -> Dict[str, Any]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    base_snapshot_id = f"snap_evt{last_event_idx:07d}_{timestamp}"
    if checkpoint_id:
        safe_ckpt = checkpoint_id.replace("/", "_")
        base_snapshot_id = f"{base_snapshot_id}_{safe_ckpt}"

    snapshot_id = base_snapshot_id
    suffix = 1
    while (snapshot_root / snapshot_id).exists() or (snapshot_root / f".{snapshot_id}.tmp").exists():
        snapshot_id = f"{base_snapshot_id}_{suffix}"
        suffix += 1

    temp_dir = snapshot_root / f".{snapshot_id}.tmp"
    final_dir = snapshot_root / snapshot_id
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        state_path = temp_dir / "state.pkl"
        checkpoint_path = temp_dir / "checkpoint.json"
        meta_path = temp_dir / "meta.json"
        chroma_dir = temp_dir / "chroma"

        memory_system.save_state(state_path)

        checkpoint_payload = {
            "last_event_idx": int(last_event_idx),
            "events_processed": int(processed),
            "saved_at": created_at,
            "trigger": trigger,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": checkpoint_app_log_id,
        }
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_payload, f, ensure_ascii=False, indent=2)

        clone_meta = memory_system.retriever.clone_collection_to_directory(
            dest_directory=chroma_dir,
            dest_collection_name=collection_name,
            overwrite=False,
            batch_size=100,
        )

        meta_payload = {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "collection_name": collection_name,
            "trigger": trigger,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": checkpoint_app_log_id,
            "clone_meta": clone_meta,
            "last_event_idx": int(last_event_idx),
            "events_processed": int(processed),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        os.replace(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    manifest_path = _snapshot_manifest_path(snapshot_root)
    manifest = _load_manifest(manifest_path)
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []
        manifest["snapshots"] = snapshots

    entry = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "last_event_idx": int(last_event_idx),
        "events_processed": int(processed),
        "collection_name": collection_name,
        "state_path": str(Path(snapshot_id) / "state.pkl"),
        "checkpoint_path": str(Path(snapshot_id) / "checkpoint.json"),
        "chroma_dir": str(Path(snapshot_id) / "chroma"),
        "meta_path": str(Path(snapshot_id) / "meta.json"),
        "trigger": trigger,
        "checkpoint_id": checkpoint_id,
        "checkpoint_app_log_id": checkpoint_app_log_id,
    }
    snapshots.append(entry)
    manifest["latest_snapshot"] = snapshot_id
    _atomic_write_json(manifest_path, manifest)
    logger.info(
        "Saved snapshot %s (trigger=%s, checkpoint_id=%s, last_event_idx=%d)",
        snapshot_id,
        trigger,
        checkpoint_id,
        last_event_idx,
    )
    return entry


def evaluate_membench(
    user_id: str,
    *,
    app_log_path: Optional[str] = None,
    benchmark_path: str,
    size: str = "small",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_backend: Optional[str] = None,
    collection_name: str = "memories",
    llm_controller_backend: str = "openai",
    llm_controller_model_name: str = "gpt-4o-mini",
    llm_controller_api_key: Optional[str] = None,
    llm_controller_api_base_url: Optional[str] = None,
    resume: bool = False,
    data_storage_path: Optional[str] = None,
    save_every: int = 50,
    snapshot_dir: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_api_base_url: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> dict:
    _ensure_nltk()
    if AgenticMemorySystem is None:
        raise RuntimeError("Amem builder requires AgenticMemorySystem and its retrieval dependencies.")

    reset_amem_usage_tracker()
    build_t0 = time.time()

    embedding_api_key = embedding_api_key or _env_value("embedding_api_key", "EMBEDDING_API_KEY")
    embedding_api_base_url = embedding_api_base_url or _env_value(
        "embedding_api_base_url",
        "EMBEDDING_API_BASE_URL",
        "embedding_api_base",
        "EMBEDDING_API_BASE",
    )
    llm_controller_api_key = llm_controller_api_key or _env_value(
        "llm_controller_api_key",
        "LLM_CONTROLLER_API_KEY",
    )
    llm_controller_api_base_url = llm_controller_api_base_url or _env_value(
        "llm_controller_api_base_url",
        "LLM_CONTROLLER_API_BASE_URL",
        "llm_controller_base_url",
        "LLM_CONTROLLER_BASE_URL",
    )

    resolved_app_log_path = Path(app_log_path).expanduser().resolve() if app_log_path else DATA_DIR.resolve()
    benchmark_path = str(Path(benchmark_path).expanduser().resolve())

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger = setup_logger(log_dir / f"membench_amem_{user_id}_{timestamp}.log")

    if not data_storage_path:
        raise ValueError("Amem builder requires baseline_params.data_storage_path; no fallback is supported.")
    if not snapshot_dir:
        raise ValueError("Amem builder requires baseline_params.snapshot_dir; no fallback is supported.")
    builder_state_dir = Path(data_storage_path).expanduser().resolve()
    state_path, checkpoint_path = _checkpoint_paths(builder_state_dir, user_id, size)
    final_usage_sidecar_path, live_usage_sidecar_path = build_usage_sidecar_paths(state_path)
    accumulated_duration_s = 0.0
    accumulated_usage: Dict[str, Any] = {}
    if resume:
        existing_usage_payload = load_usage_sidecar(final_usage_sidecar_path)
        if not existing_usage_payload:
            existing_usage_payload = load_usage_sidecar(live_usage_sidecar_path)
        existing_timing = (
            existing_usage_payload.get("timing")
            if isinstance(existing_usage_payload.get("timing"), dict)
            else {}
        )
        existing_usage = (
            existing_usage_payload.get("usage")
            if isinstance(existing_usage_payload.get("usage"), dict)
            else {}
        )
        accumulated_duration_s = float(existing_timing.get("build_memory_duration_s") or 0.0)
        accumulated_usage = (
            existing_usage.get("build_memory")
            if isinstance(existing_usage.get("build_memory"), dict)
            else {}
        )

    snapshot_root = _snapshot_root(snapshot_dir, builder_state_dir, user_id, size)
    resume_enabled = False
    start_index = 0
    processed = 0
    resume_snapshot_state_path: Optional[Path] = None
    if resume:
        latest_entry = _latest_snapshot_entry(snapshot_root)
        if latest_entry is not None:
            latest_payload = _load_snapshot_checkpoint_payload(snapshot_root, latest_entry)
            resume_snapshot_state_path = _load_snapshot_state_path(snapshot_root, latest_entry)
            if resume_snapshot_state_path.exists():
                last_event_idx = int(latest_payload.get("last_event_idx", latest_entry.get("last_event_idx", -1)))
                processed = int(latest_payload.get("events_processed", last_event_idx + 1))
                start_index = max(0, last_event_idx + 1)
                resume_enabled = True

    logger.info(
        "Embedding config: backend=%s model=%s collection=%s",
        embedding_backend,
        embedding_model_name,
        collection_name,
    )

    memory_system = AgenticMemorySystem(
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        embedding_backend=embedding_backend,
        embedding_api_key=embedding_api_key,
        embedding_api_base_url=embedding_api_base_url,
        llm_controller_backend=llm_controller_backend,
        llm_controller_model_name=llm_controller_model_name,
        llm_controller_api_key=llm_controller_api_key,
        llm_controller_api_base_url=llm_controller_api_base_url,
        retriever_directory=None,
        reset_collection=not resume_enabled,
    )

    llm_provider_label = str(llm_controller_backend or "openai")
    retriever_provider_label = str(embedding_backend or "openai")

    def _current_build_usage_snapshot() -> Dict[str, Any]:
        current_usage = amem_usage_summary()
        return merge_usage_summary(accumulated_usage, current_usage)

    def _current_build_duration_s() -> float:
        return accumulated_duration_s + max(time.time() - build_t0, 0.0)

    def _write_live_usage_snapshot() -> None:
        write_build_usage_sidecar(
            sidecar_path=live_usage_sidecar_path,
            state_path=state_path,
            checkpoint_path=checkpoint_path,
            llm_provider=llm_provider_label,
            llm_model=llm_controller_model_name,
            retriever_provider=retriever_provider_label,
            embedding_model_name=embedding_model_name,
            build_duration_s=_current_build_duration_s(),
            build_memory_usage=_current_build_usage_snapshot(),
            is_live=True,
        )

    if resume_enabled:
        memory_system.load_state(resume_snapshot_state_path)
        memory_system.rebuild_retriever()
        logger.info("Resuming from snapshot: start_index=%d", start_index)

    if processed <= 0:
        processed = start_index

    samples = load_membench_dataset(
        resolved_app_log_path,
        user_id=user_id,
        size=size,
        load_ckpts=False,
    )
    if not samples:
        raise ValueError(f"User id not found in dataset: {user_id}")
    sample = samples[0]
    checkpoints = _load_checkpoints_from_benchmark(Path(benchmark_path))

    checkpoint_by_app_log_id: Dict[str, List[Dict[str, Any]]] = {}
    expected_checkpoint_ids: Set[str] = set()
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
        as_of = cp.get("as_of")
        if not isinstance(as_of, dict):
            as_of = {}
        app_log_id_raw = as_of.get("app_log_id")
        if app_log_id_raw is not None:
            checkpoint_by_app_log_id.setdefault(str(app_log_id_raw), []).append(cp)

        checkpoint_id_raw = cp.get("checkpoint_id")
        if checkpoint_id_raw is None:
            continue
        checkpoint_id = str(checkpoint_id_raw)
        if not checkpoint_id:
            continue
        log_index_raw = as_of.get("log_index")
        if isinstance(log_index_raw, int):
            if log_index_raw >= start_index:
                expected_checkpoint_ids.add(checkpoint_id)
        else:
            expected_checkpoint_ids.add(checkpoint_id)

    seen_checkpoint_ids: Set[str] = set()
    snapshots_written = 0

    logger.info(
        "Loaded MemBench sample_id=%s events=%d checkpoints=%d",
        sample.sample_id,
        len(sample.app_logs),
        len(checkpoints),
    )
    last_event_idx = start_index - 1
    interrupted_exc = None
    try:
        for idx, event in enumerate(sample.app_logs):
            if idx < start_index:
                continue
            raw_payload_text, time_str = build_membench_memory_from_event(event)
            memory_system.add_note(
                raw_payload_text,
                time=time_str,
            )
            processed += 1
            last_event_idx = idx
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {processed} processed")
            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "phase": "build_memory",
                            "events_processed": processed,
                            "last_event_idx": last_event_idx,
                        }
                    )
                except Exception as e:
                    logger.warning("Amem progress callback failed after event %d: %s", last_event_idx, e)

            if save_every > 0 and processed % save_every == 0:
                memory_system.save_state(state_path)
                _write_checkpoint(checkpoint_path, last_event_idx, processed)
                _write_live_usage_snapshot()

            event_id = str(event.event_id)
            for cp in checkpoint_by_app_log_id.get(event_id, []):
                cp_id_raw = cp.get("checkpoint_id")
                cp_id = str(cp_id_raw) if cp_id_raw is not None else ""
                if cp_id and cp_id in seen_checkpoint_ids:
                    continue
                _write_snapshot_bundle(
                    memory_system=memory_system,
                    snapshot_root=snapshot_root,
                    collection_name=memory_system.collection_name,
                    last_event_idx=last_event_idx,
                    processed=processed,
                    logger=logger,
                    trigger="checkpoint",
                    checkpoint_id=cp_id or None,
                    checkpoint_app_log_id=event_id,
                )
                snapshots_written += 1
                if cp_id:
                    seen_checkpoint_ids.add(cp_id)
    except BaseException as exc:
        interrupted_exc = exc
        logger.warning("Run interrupted (%s). Saving checkpoint.", type(exc).__name__)
    finally:
        if last_event_idx >= 0:
            memory_system.save_state(state_path)
            _write_checkpoint(checkpoint_path, last_event_idx, processed)
        _write_live_usage_snapshot()
    if interrupted_exc is not None:
        raise interrupted_exc

    missing_checkpoint_ids = sorted(expected_checkpoint_ids - seen_checkpoint_ids)
    if missing_checkpoint_ids:
        preview = ", ".join(missing_checkpoint_ids[:15])
        logger.warning(
            "Unmatched benchmark checkpoints for this run: %d. First missing IDs: %s",
            len(missing_checkpoint_ids),
            preview,
        )

    _write_snapshot_bundle(
        memory_system=memory_system,
        snapshot_root=snapshot_root,
        collection_name=memory_system.collection_name,
        last_event_idx=last_event_idx,
        processed=processed,
        logger=logger,
        trigger="final",
        checkpoint_id=None,
        checkpoint_app_log_id=None,
    )
    snapshots_written += 1

    write_build_usage_sidecar(
        sidecar_path=final_usage_sidecar_path,
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        llm_provider=llm_provider_label,
        llm_model=llm_controller_model_name,
        retriever_provider=retriever_provider_label,
        embedding_model_name=embedding_model_name,
        build_duration_s=_current_build_duration_s(),
        build_memory_usage=_current_build_usage_snapshot(),
        is_live=False,
    )

    return {
        "user_id": user_id,
        "events_processed": processed,
        "snapshots_written": snapshots_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest MemBench app logs into AgenticMemorySystem."
    )
    parser.add_argument("--user-id", required=True, help="User/sample id to ingest")
    parser.add_argument("--app-log", type=str, default=str(DATA_DIR), help="App log path")
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Benchmark path providing the checkpoint list for snapshot triggers.",
    )
    parser.add_argument("--size", type=str, default="small", help="Dataset size: small|medium|large")
    parser.add_argument("--embedding-model-name", "--model-name", dest="embedding_model_name", type=str, default="text-embedding-3-large", help="Embedding model")
    parser.add_argument("--embedding-backend", type=str, default="openai", help="Embedding backend")
    parser.add_argument("--collection-name", type=str, default="memories", help="ChromaDB collection name")
    parser.add_argument("--llm-controller-backend", "--llm-backend", dest="llm_controller_backend", type=str, default="openai", help="LLM backend")
    parser.add_argument("--llm-controller-model-name", "--llm-model", dest="llm_controller_model_name", type=str, default="openrouter/openai/gpt-5-mini", help="LLM model")
    parser.add_argument("--llm-controller-api-key", type=str, default=None, help="LLM API key")
    parser.add_argument(
        "--llm-controller-api-base-url",
        "--llm-controller-base-url",
        dest="llm_controller_api_base_url",
        type=str,
        default=None,
        help="LLM API base URL",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument(
        "--data-storage-path",
        dest="data_storage_path",
        type=str,
        default=None,
        help="Runtime data directory for builder state and usage sidecars.",
    )
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N events")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Root directory for durable snapshot bundles")
    parser.add_argument("--embedding-api-key", type=str, default=None, help="Embedding API key")
    parser.add_argument(
        "--embedding-api-base-url",
        "--embedding-api-base",
        dest="embedding_api_base_url",
        type=str,
        default=None,
        help="Embedding API base URL",
    )
    args = parser.parse_args()

    summary = evaluate_membench(
        user_id=args.user_id,
        app_log_path=args.app_log,
        benchmark_path=args.benchmark_path,
        size=args.size,
        embedding_model_name=args.embedding_model_name,
        embedding_backend=args.embedding_backend or None,
        collection_name=args.collection_name,
        llm_controller_backend=args.llm_controller_backend,
        llm_controller_model_name=args.llm_controller_model_name,
        llm_controller_api_key=args.llm_controller_api_key,
        llm_controller_api_base_url=args.llm_controller_api_base_url,
        resume=args.resume,
        data_storage_path=args.data_storage_path,
        save_every=args.save_every,
        snapshot_dir=args.snapshot_dir,
        embedding_api_key=args.embedding_api_key,
        embedding_api_base_url=args.embedding_api_base_url,
    )
    print(f"Done (events={summary['events_processed']})")


if __name__ == "__main__":
    main()
