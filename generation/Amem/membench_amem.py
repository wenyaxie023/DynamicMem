import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import nltk

from agentic_memory.memory_system import AgenticMemorySystem

GENERATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GENERATION_DIR / "data"
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import build_membench_memory_from_event, load_membench_dataset  # type: ignore


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


def _load_checkpoint(path: Path, logger: logging.Logger) -> int:
    try:
        with path.open("r") as f:
            payload = json.load(f)
        return int(payload.get("last_event_idx", -1))
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return -1


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


def evaluate_membench(
    user_id: str,
    *,
    app_log_path: Optional[str] = None,
    size: str = "small",
    model_name: str = "all-MiniLM-L6-v2",
    embedding_backend: Optional[str] = None,
    collection_name: str = "memories",
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o-mini",
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
    resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 50,
) -> dict:
    _ensure_nltk()

    resolved_app_log_path = Path(app_log_path) if app_log_path else DATA_DIR

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger = setup_logger(log_dir / f"membench_amem_{user_id}_{timestamp}.log")

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (Path(__file__).resolve().parent / "checkpoints")
    state_path = ckpt_dir / f"membench_amem_{user_id}_{size}.pkl"
    checkpoint_path = ckpt_dir / f"membench_amem_{user_id}_{size}.json"

    resume_enabled = resume and state_path.exists() and checkpoint_path.exists()
    start_index = 0
    if resume_enabled:
        start_index = _load_checkpoint(checkpoint_path, logger) + 1
        if start_index < 0:
            start_index = 0

    memory_system = AgenticMemorySystem(
        collection_name=collection_name,
        model_name=model_name,
        embedding_backend=embedding_backend,
        llm_backend=llm_backend,
        llm_model=llm_model,
        api_key=os.environ.get("AIML_API_KEY"),
        openai_api_base=os.environ.get("API_BASE_URL"),
        llm_base_url=os.environ.get("API_BASE_URL"),
        reset_collection=not resume_enabled,
    )

    if resume_enabled:
        memory_system.load_state(state_path)
        memory_system.rebuild_retriever()
        logger.info("Resuming from checkpoint: start_index=%d", start_index)

    processed = start_index
    found = False
    for sample in load_membench_dataset(resolved_app_log_path, size=size):
        if sample.sample_id != user_id:
            continue
        found = True
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
                if resume and save_every > 0 and processed % save_every == 0:
                    memory_system.save_state(state_path)
                    _write_checkpoint(checkpoint_path, last_event_idx, processed)
        except BaseException as exc:
            interrupted_exc = exc
            logger.warning("Run interrupted (%s). Saving checkpoint.", type(exc).__name__)
        finally:
            if resume and last_event_idx >= 0:
                memory_system.save_state(state_path)
                _write_checkpoint(checkpoint_path, last_event_idx, processed)
        if interrupted_exc is not None:
            raise interrupted_exc

    if not found:
        raise ValueError(f"User id not found in dataset: {user_id}")

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
    parser.add_argument("--model-name", type=str, default="text-embedding-3-large", help="Embedding model")
    parser.add_argument("--embedding-backend", type=str, default="openai", help="Embedding backend")
    parser.add_argument("--collection-name", type=str, default="memories", help="ChromaDB collection name")
    parser.add_argument("--llm-backend", type=str, default="openai", help="LLM backend")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--sglang-host", type=str, default="http://localhost", help="SGLang host")
    parser.add_argument("--sglang-port", type=int, default=30000, help="SGLang port")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N events")
    args = parser.parse_args()

    summary = evaluate_membench(
        user_id=args.user_id,
        app_log_path=args.app_log,
        size=args.size,
        model_name=args.model_name,
        embedding_backend=args.embedding_backend or None,
        collection_name=args.collection_name,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
    )
    print(f"Done (events={summary['events_processed']})")


if __name__ == "__main__":
    main()
