
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from memoryos import Memoryos

GENERATION_DIR = Path(__file__).resolve().parents[2]
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import (  # type: ignore  # noqa: E402
    build_membench_memory_from_event,
    load_membench_dataset,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Basic Configuration ---
ASSISTANT_ID = "assistant"
# TODO: Load api keys and api base from .env
LLM_CONTROLLER_API_KEY = os.getenv("LLM_CONTROLLER_API_KEY")
LLM_CONTROLLER_API_BASE_URL = os.getenv("LLM_CONTROLLER_API_BASE_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL")
DATA_STORAGE_PATH = ""
DEFAULT_LLM_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
DATA_ROOT = GENERATION_DIR / "data"

def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def resolve_snapshot_resume_bundle(snapshot_root: Path) -> Optional[dict]:
    # given the snapshot root path, tell us which one  to continue from regarding the manifest.json file
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots", []) if isinstance(manifest, dict) else []
    snapshots = [x for x in snapshots if isinstance(x, dict)]
    if not snapshots:
        return None

    snapshots.sort(
        key=lambda x: (
            int(x.get("last_event_idx", -1)),
            str(x.get("created_at", "")),
        ),
        reverse=True,
    )
    latest = snapshots[0]
    snapshot_id = str(latest.get("snapshot_id"))
    checkpoint_path = snapshot_root / latest.get(
        "checkpoint_path",
        str(Path(snapshot_id) / "checkpoint.json"),
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    last_event_idx = int(checkpoint.get("last_event_idx", latest.get("last_event_idx", -1)))

    return {
        "snapshot_id": snapshot_id,
        "start_index": max(0, last_event_idx + 1),
        "last_event_idx": last_event_idx,
        "short_term_path": snapshot_root / latest.get("short_term_path", str(Path(snapshot_id) / "short_term.json")),
        "mid_term_path": snapshot_root / latest.get("mid_term_path", str(Path(snapshot_id) / "mid_term.json")),
        "long_term_path": snapshot_root / latest.get("long_term_path", str(Path(snapshot_id) / "long_term_user.json")),
        "assistant_long_term_path": snapshot_root / latest.get(
            "assistant_long_term_path",
            str(Path(snapshot_id) / "long_term_assistant.json"),
        ),
    }

def _write_snapshot_bundle(
    memory_system: Memoryos,
    snapshot_root: Path,
    last_event_idx: int,
    checkpoint_id: Optional[str] = None,
    checkpoint_app_log_id: Optional[str] = None,
) -> dict:
    # write the current state of the memory system as an snapshot
    # use _atomic_json to save short_term, mid_term and long_term memory
    snapshot_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat()
    snapshot_id = f"snap_evt{last_event_idx:07d}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    snapshot_dir = snapshot_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    short_term_path = snapshot_dir / "short_term.json"
    mid_term_path = snapshot_dir / "mid_term.json"
    long_term_path = snapshot_dir / "long_term_user.json"
    assistant_long_term_path = snapshot_dir / "long_term_assistant.json"
    checkpoint_path = snapshot_dir / "checkpoint.json"

    _atomic_write_json(short_term_path, list(memory_system.short_term_memory.memory))
    _atomic_write_json(
        mid_term_path,
        {
            "sessions": memory_system.mid_term_memory.sessions,
            "access_frequency": dict(memory_system.mid_term_memory.access_frequency),
        },
    )
    _atomic_write_json(
        long_term_path,
        {
            "user_profiles": memory_system.user_long_term_memory.user_profiles,
            "knowledge_base": list(memory_system.user_long_term_memory.knowledge_base),
            "assistant_knowledge": list(memory_system.user_long_term_memory.assistant_knowledge),
        },
    )
    _atomic_write_json(
        assistant_long_term_path,
        {
            "user_profiles": memory_system.assistant_long_term_memory.user_profiles,
            "knowledge_base": list(memory_system.assistant_long_term_memory.knowledge_base),
            "assistant_knowledge": list(memory_system.assistant_long_term_memory.assistant_knowledge),
        },
    )
    _atomic_write_json(
        checkpoint_path,
        {
            "last_event_idx": int(last_event_idx),
            "saved_at": created_at,
            "checkpoint_id": checkpoint_id,
            "checkpoint_app_log_id": checkpoint_app_log_id,
        },
    )

    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"snapshots": []}
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []
    entry = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "last_event_idx": int(last_event_idx),
        "checkpoint_id": checkpoint_id,
        "checkpoint_app_log_id": checkpoint_app_log_id,
        "short_term_path": str(Path(snapshot_id) / "short_term.json"),
        "mid_term_path": str(Path(snapshot_id) / "mid_term.json"),
        "long_term_path": str(Path(snapshot_id) / "long_term_user.json"),
        "assistant_long_term_path": str(Path(snapshot_id) / "long_term_assistant.json"),
        "checkpoint_path": str(Path(snapshot_id) / "checkpoint.json"),
    }
    snapshots.append(entry)
    manifest["snapshots"] = snapshots
    manifest["latest_snapshot"] = snapshot_id
    _atomic_write_json(manifest_path, manifest)
    return entry

def run_evaluation(
    target_sample_id: str,
    llm_model: str,
    embedding_model_name: str,
    snapshot_root: str,
):
    print("MemoryOS Evaluation")
    dataset_size = "large"
    if not target_sample_id:
        raise ValueError("target_sample_id is required.")
    user_id = f"{target_sample_id}_{dataset_size}"
    data_storage_root = Path(os.path.abspath(DATA_STORAGE_PATH or ""))
    
    # 1. Initialize MemoryOS
    print("Initializing MemoryOS...")
    memo = Memoryos(
        user_id=user_id,
        openai_api_key=LLM_CONTROLLER_API_KEY,
        openai_base_url=LLM_CONTROLLER_API_BASE_URL,
        data_storage_path=str(data_storage_root),
        llm_model=llm_model,
        assistant_id=ASSISTANT_ID,
        short_term_capacity=7,  
        mid_term_heat_threshold=5,  
        retrieval_queue_capacity=10,
        long_term_knowledge_capacity=100,
        mid_term_similarity_threshold=0.6,
        embedding_model_name=embedding_model_name, # text-embedding-3-large, Qwen/Qwen3-Embedding-8B
        embedding_model_kwargs={
            "embedding_backend": "openai",
            "api_key": EMBEDDING_API_KEY,
            'api_base': EMBEDDING_API_BASE_URL,
            "max_input_tokens": 8000,
            "truncate_from": "end",
            # "embedding_backend": "sentence-transformers",
            # "device": "mps",  # or "cpu"/"cuda"
        },
    )
    print("MemoryOS initialized successfully!\n")

    # 2. Load MemBench app logs and checkpoints
    print("Loading MemBench dataset and checkpoints...")
    samples, checkpoints = load_membench_dataset(
        DATA_ROOT,
        user_id=target_sample_id,
        size=dataset_size,
        load_ckpts=True,
    )
    if not samples:
        print("No MemBench samples found for the requested filter.")
        return
    sample = samples[0]

    print(
        f"Loaded sample '{sample.sample_id}' with "
        f"{len(sample.app_logs)} events and {len(sample.qa)} QA pairs."
    )

    checkpoint_by_app_log_id: dict[str, list[dict]] = {}
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
        as_of = cp.get("as_of")
        as_of = as_of if isinstance(as_of, dict) else {}
        app_log_id_raw = as_of.get("app_log_id")
        if app_log_id_raw is not None:
            checkpoint_by_app_log_id.setdefault(str(app_log_id_raw), []).append(cp)

    snapshot_root = Path(snapshot_root) / user_id
    resume_bundle = resolve_snapshot_resume_bundle(snapshot_root)
    start_index = 0
    if resume_bundle:
        shutil.copyfile(resume_bundle["short_term_path"], memo.short_term_memory.file_path)
        shutil.copyfile(resume_bundle["mid_term_path"], memo.mid_term_memory.file_path)
        shutil.copyfile(resume_bundle["long_term_path"], memo.user_long_term_memory.file_path)
        shutil.copyfile(resume_bundle["assistant_long_term_path"], memo.assistant_long_term_memory.file_path)
        memo.short_term_memory.load()
        memo.mid_term_memory.load()
        memo.user_long_term_memory.load()
        memo.assistant_long_term_memory.load()
        start_index = int(resume_bundle["start_index"])
        print(
            f"Resuming from snapshot '{resume_bundle['snapshot_id']}' "
            f"at event index {start_index}."
        )

    for idx, event in enumerate(sample.app_logs):
        if idx < start_index:
            continue
        content, time_str = build_membench_memory_from_event(event)
        memo.add_memory(
            user_input=f"[APP_LOG] {content}",
            agent_response="[ingested]",
            timestamp=time_str,
        )
        event_id = str(event.event_id)
        for cp in checkpoint_by_app_log_id.get(event_id, []):
            cp_id_raw = cp.get("checkpoint_id")
            cp_id = str(cp_id_raw) if cp_id_raw is not None else None
            _write_snapshot_bundle(
                memory_system=memo,
                snapshot_root=snapshot_root,
                last_event_idx=idx,
                checkpoint_id=cp_id,
                checkpoint_app_log_id=event_id,
            )
    # end loading checkpoints

    print("Ingestion completed.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemoryOS MemBench evaluation")
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help="Sample id/user id to load",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM model passed to MemoryOS",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="Embedding model name passed to MemoryOS",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "snapshots"),
        help="Snapshot root directory",
    )
    args = parser.parse_args()

    run_evaluation(
        args.sample_id,
        args.llm_model,
        args.embedding_model_name,
        args.snapshot_root,
    )
