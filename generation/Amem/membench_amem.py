import os
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import nltk
from dotenv import load_dotenv

from agentic_memory.memory_system import AgenticMemorySystem

load_dotenv(Path(__file__).resolve().parent / ".env")

GENERATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GENERATION_DIR / "data"
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import build_membench_memory_from_event, load_membench_dataset  # type: ignore


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
    base = Path(snapshot_dir) if snapshot_dir else (ckpt_dir / "chroma_snapshots")
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


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _snapshot_entry_sort_key(entry: Dict[str, Any]) -> tuple[int, datetime]:
    last_event_idx_raw = entry.get("last_event_idx", -1)
    try:
        last_event_idx = int(last_event_idx_raw)
    except (TypeError, ValueError):
        last_event_idx = -1
    created_at_raw = entry.get("created_at")
    created_at_dt = datetime.min
    if isinstance(created_at_raw, str) and created_at_raw:
        try:
            created_at_dt = datetime.fromisoformat(created_at_raw)
        except ValueError:
            created_at_dt = datetime.min
    return last_event_idx, created_at_dt


def _manifest_entry_path(snapshot_root: Path, path_value: Any) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return snapshot_root / p


def _resolve_snapshot_resume_bundle(snapshot_root: Path) -> Dict[str, Any]:
    manifest_path = _snapshot_manifest_path(snapshot_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    entries = [x for x in manifest.get("snapshots", []) if isinstance(x, dict)]
    if not entries:
        raise FileNotFoundError(f"No snapshots recorded in manifest: {manifest_path}")

    for entry in sorted(entries, key=_snapshot_entry_sort_key, reverse=True):
        snapshot_id = entry.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            continue
        state_path = _manifest_entry_path(
            snapshot_root, entry.get("state_path", str(Path(snapshot_id) / "state.pkl"))
        )
        checkpoint_path = _manifest_entry_path(
            snapshot_root, entry.get("checkpoint_path", str(Path(snapshot_id) / "checkpoint.json"))
        )
        chroma_dir = _manifest_entry_path(
            snapshot_root, entry.get("chroma_dir", str(Path(snapshot_id) / "chroma"))
        )
        meta_path = _manifest_entry_path(
            snapshot_root, entry.get("meta_path", str(Path(snapshot_id) / "meta.json"))
        )

        if not state_path.exists():
            continue
        if not checkpoint_path.exists():
            continue
        if not chroma_dir.exists():
            continue
        if not meta_path.exists():
            continue

        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint_payload = json.load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        last_event_idx = int(checkpoint_payload.get("last_event_idx", -1))
        events_processed = int(checkpoint_payload.get("events_processed", last_event_idx + 1))
        collection_name = (
            meta_payload.get("collection_name")
            or entry.get("collection_name")
        )
        if not isinstance(collection_name, str) or not collection_name:
            continue

        return {
            "snapshot_id": snapshot_id,
            "state_path": state_path,
            "checkpoint_path": checkpoint_path,
            "chroma_dir": chroma_dir,
            "meta_path": meta_path,
            "collection_name": collection_name,
            "start_index": max(0, last_event_idx + 1),
            "events_processed": max(0, events_processed),
            "last_event_idx": last_event_idx,
        }

    raise FileNotFoundError(
        f"No valid snapshot bundle found under {snapshot_root}. "
        "Manifest exists but entries are missing required files."
    )


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
    size: str = "small",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_backend: Optional[str] = None,
    collection_name: str = "memories",
    llm_controller_backend: str = "openai",
    llm_controller_model_name: str = "gpt-4o-mini",
    llm_controller_api_key: Optional[str] = None,
    llm_controller_api_base_url: Optional[str] = None,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
    resume: bool = False,
    resume_from_size: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 50,
    snapshot_dir: Optional[str] = None,
    snapshot_every: Optional[int] = None,
    resume_from_snapshot: bool = True,
    embedding_api_key: Optional[str] = None,
    embedding_api_base_url: Optional[str] = None,
) -> dict:
    _ensure_nltk()

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

    resolved_app_log_path = Path(app_log_path) if app_log_path else DATA_DIR

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger = setup_logger(log_dir / f"membench_amem_{user_id}_{timestamp}.log")

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (Path(__file__).resolve().parent / "checkpoints")
    state_path, checkpoint_path = _checkpoint_paths(ckpt_dir, user_id, size)
    resume_state_path = state_path
    resume_checkpoint_path = checkpoint_path
    resume_label = size

    if resume_from_size:
        resume_state_path, resume_checkpoint_path = _checkpoint_paths(
            ckpt_dir, user_id, resume_from_size
        )
        resume_label = resume_from_size

    snapshot_root = _snapshot_root(snapshot_dir, ckpt_dir, user_id, size)
    if snapshot_every is not None and snapshot_every > 0:
        logger.info(
            "snapshot_every=%d is ignored in checkpoint-only snapshot mode.",
            snapshot_every,
        )

    resume_requested = resume or resume_from_size is not None
    resume_enabled = False
    resume_snapshot_bundle: Optional[Dict[str, Any]] = None
    start_index = 0
    processed = 0
    if resume_requested and resume_from_snapshot:
        resume_snapshot_root = _snapshot_root(snapshot_dir, ckpt_dir, user_id, resume_label)
        try:
            resume_snapshot_bundle = _resolve_snapshot_resume_bundle(resume_snapshot_root)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e} Set --no-resume-from-snapshot to use legacy checkpoint resume."
            ) from e
        resume_enabled = True
        start_index = int(resume_snapshot_bundle["start_index"])
        processed = int(resume_snapshot_bundle["events_processed"])
    else:
        if resume_from_size:
            if not resume_state_path.exists() or not resume_checkpoint_path.exists():
                missing = []
                if not resume_state_path.exists():
                    missing.append(str(resume_state_path))
                if not resume_checkpoint_path.exists():
                    missing.append(str(resume_checkpoint_path))
                raise FileNotFoundError(
                    "Resume checkpoint not found for size "
                    f"'{resume_from_size}': {', '.join(missing)}"
                )
            resume_enabled = True
        elif resume and resume_state_path.exists() and resume_checkpoint_path.exists():
            resume_enabled = True

        if resume_enabled:
            last_event_idx, processed = _load_checkpoint(resume_checkpoint_path, logger)
            start_index = last_event_idx + 1
            if start_index < 0:
                start_index = 0

    retriever_directory = (
        str(resume_snapshot_bundle["chroma_dir"])
        if resume_snapshot_bundle is not None
        else None
    )
    active_collection_name = (
        str(resume_snapshot_bundle["collection_name"])
        if resume_snapshot_bundle is not None
        else collection_name
    )
    resume_snapshot_id = (
        str(resume_snapshot_bundle["snapshot_id"])
        if resume_snapshot_bundle is not None
        else None
    )
    logger.info(
        "Embedding config: backend=%s model=%s active_collection=%s snapshot_id=%s",
        embedding_backend,
        embedding_model_name,
        active_collection_name,
        resume_snapshot_id,
    )

    memory_system = AgenticMemorySystem(
        collection_name=active_collection_name,
        embedding_model_name=embedding_model_name,
        embedding_backend=embedding_backend,
        embedding_api_key=embedding_api_key,
        embedding_api_base_url=embedding_api_base_url,
        llm_controller_backend=llm_controller_backend,
        llm_controller_model_name=llm_controller_model_name,
        llm_controller_api_key=llm_controller_api_key,
        llm_controller_api_base_url=llm_controller_api_base_url,
        retriever_directory=retriever_directory,
        reset_collection=not resume_enabled,
    )

    if resume_enabled:
        if resume_snapshot_bundle is not None:
            memory_system.load_state(Path(resume_snapshot_bundle["state_path"]))
            logger.info(
                "Resuming from snapshot: id=%s start_index=%d",
                resume_snapshot_bundle["snapshot_id"],
                start_index,
            )
        else:
            memory_system.load_state(resume_state_path)
            memory_system.rebuild_retriever()
            if resume_label != size:
                logger.info(
                    "Resuming from checkpoint size=%s -> target size=%s: start_index=%d",
                    resume_label,
                    size,
                    start_index,
                )
            else:
                logger.info("Resuming from checkpoint: start_index=%d", start_index)
    elif resume_label != size:
        logger.info(
            "Starting fresh target size=%s (resume label %s not active).",
            size,
            resume_label,
        )

    if processed <= 0:
        processed = start_index

    loaded = load_membench_dataset(
        resolved_app_log_path,
        user_id=user_id,
        size=size,
        load_ckpts=True,
    )
    samples, checkpoints = loaded
    if not samples:
        raise ValueError(f"User id not found in dataset: {user_id}")
    sample = samples[0]

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
            content, time_str = build_membench_memory_from_event(event)
            memory_system.add_note(content, time=time_str)
            processed += 1
            last_event_idx = idx
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {processed} processed")

            if save_every > 0 and processed % save_every == 0:
                memory_system.save_state(state_path)
                _write_checkpoint(checkpoint_path, last_event_idx, processed)

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
        if resume_requested and last_event_idx >= 0:
            memory_system.save_state(state_path)
            _write_checkpoint(checkpoint_path, last_event_idx, processed)
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
    parser.add_argument("--size", type=str, default="small", help="Dataset size: small|medium|large")
    parser.add_argument("--embedding-model-name", "--model-name", dest="embedding_model_name", type=str, default="openrouter/openai/text-embedding-3-large", help="Embedding model")
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
    parser.add_argument("--sglang-host", type=str, default="http://localhost", help="SGLang host")
    parser.add_argument("--sglang-port", type=int, default=30000, help="SGLang port")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument(
        "--resume-from-size",
        type=str,
        default=None,
        help="Resume from a checkpoint generated with a different dataset size (e.g. small)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N events")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Root directory for durable snapshot bundles")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help="Compatibility arg (ignored in checkpoint-only snapshot mode).",
    )
    parser.add_argument(
        "--resume-from-snapshot",
        dest="resume_from_snapshot",
        action="store_true",
        help="Resume from latest durable snapshot bundle when --resume is set (default).",
    )
    parser.add_argument(
        "--no-resume-from-snapshot",
        dest="resume_from_snapshot",
        action="store_false",
        help="Use legacy checkpoint resume (pickle + rebuild) when --resume is set.",
    )
    parser.set_defaults(resume_from_snapshot=True)
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
        size=args.size,
        embedding_model_name=args.embedding_model_name,
        embedding_backend=args.embedding_backend or None,
        collection_name=args.collection_name,
        llm_controller_backend=args.llm_controller_backend,
        llm_controller_model_name=args.llm_controller_model_name,
        llm_controller_api_key=args.llm_controller_api_key,
        llm_controller_api_base_url=args.llm_controller_api_base_url,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        resume=args.resume,
        resume_from_size=args.resume_from_size,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        snapshot_dir=args.snapshot_dir,
        snapshot_every=args.snapshot_every,
        resume_from_snapshot=args.resume_from_snapshot,
        embedding_api_key=args.embedding_api_key,
        embedding_api_base_url=args.embedding_api_base_url,
    )
    print(f"Done (events={summary['events_processed']})")


if __name__ == "__main__":
    main()
