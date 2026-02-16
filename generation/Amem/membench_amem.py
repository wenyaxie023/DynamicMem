import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional
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

    resume_requested = resume or resume_from_size is not None
    resume_enabled = False
    start_index = 0
    processed = 0
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
        reset_collection=not resume_enabled,
    )

    if resume_enabled:
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

    if processed <= 0:
        processed = start_index

    samples = load_membench_dataset(
        resolved_app_log_path,
        user_id=user_id,
        size=size,
    )
    if not samples:
        raise ValueError(f"User id not found in dataset: {user_id}")
    sample = samples[0]

    logger.info(
        "Loaded MemBench sample_id=%s events=%d",
        sample.sample_id,
        len(sample.app_logs),
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
    except BaseException as exc:
        interrupted_exc = exc
        logger.warning("Run interrupted (%s). Saving checkpoint.", type(exc).__name__)
    finally:
        if resume_requested and last_event_idx >= 0:
            memory_system.save_state(state_path)
            _write_checkpoint(checkpoint_path, last_event_idx, processed)
    if interrupted_exc is not None:
        raise interrupted_exc

    return {
        "user_id": user_id,
        "events_processed": processed,
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
        embedding_api_key=args.embedding_api_key,
        embedding_api_base_url=args.embedding_api_base_url,
    )
    print(f"Done (events={summary['events_processed']})")


if __name__ == "__main__":
    main()
