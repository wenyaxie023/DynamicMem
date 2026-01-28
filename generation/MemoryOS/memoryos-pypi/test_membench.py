
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from memoryos import Memoryos

GENERATION_DIR = Path(__file__).resolve().parents[2]
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import (  # type: ignore  # noqa: E402
    build_membench_memory_from_event,
    load_membench_dataset,
)

# --- Basic Configuration ---
DEFAULT_DATASET_SIZE = "large"  # small | medium | large
DEFAULT_TARGET_SAMPLE_ID = "004_user_004"
DEFAULT_BATCH_SIZE = 4
ASSISTANT_ID = "assistant"
API_KEY =  os.getenv("AIML_API_KEY")  # Replace with your key
# API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.aimlapi.com"  # Optional: if using a custom OpenAI endpoint
# BASE_URL = ""
DATA_STORAGE_PATH = ""
LLM_MODEL = "gpt-5-mini-2025-08-07"
DATA_ROOT = GENERATION_DIR / "data"
QA_PATH = None  # Per-user QA is loaded automatically by load_membench_dataset.
MAX_EVENTS = None  # Optional: set an int to limit ingested events
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PROMPT = """# Task
Given a question and a list of user app logs, answer the question using ONLY the information in the logs.

# Input
Question:
{{ question }}

User App Logs (each log has: app_log_id, timestamp, app_name, api_name, content):
{{ context }}

# Output (JSON ONLY; no markdown, no extra text)
Return a single JSON object with this schema:
{
  "evidence": [
    {
      "app_log_id": "string",
      "timestamp": "string (as in the log; prefer ISO-8601 if present)",
      "app_name": "string",
      "api_name": "string",
      "anchor": "string" // [required, a short justification grounded in the context.]
    }
  ],
  "answer": "string"
}

# Missing-field policy
- If you don't know the app_log_id / timestamp / app_name / api_name, you should set them to null.
"""
RETRY_EMPTY_RESPONSES = 3
RETRY_SLEEP_SECONDS = 1

_SIZE_ALIASES = {
    "s": "small",
    "small": "small",
    "m": "medium",
    "med": "medium",
    "medium": "medium",
    "l": "large",
    "lg": "large",
    "large": "large",
}

def _normalize_membench_size(size: str) -> str:
    size_key = (size or "small").strip().lower()
    normalized = _SIZE_ALIASES.get(size_key)
    if normalized:
        return normalized
    raise ValueError(f"Invalid size '{size}'. Expected small, medium, or large.")

def _baseline_size(size: str) -> Optional[str]:
    if size == "medium":
        return "small"
    if size == "large":
        return "medium"
    return None

def _resolve_data_storage_root() -> Path:
    return Path(os.path.abspath(DATA_STORAGE_PATH or ""))

def _count_sample_events(size: str, sample_id: str) -> Optional[int]:
    for item in load_membench_dataset(DATA_ROOT, size=size):
        if item.sample_id == sample_id:
            return len(item.app_logs)
    return None

def _maybe_seed_user_data(data_storage_root: Path, seed_user_id: str, target_user_id: str) -> bool:
    users_dir = data_storage_root / "users"
    dest_dir = users_dir / target_user_id
    if dest_dir.exists():
        return False

    candidate_roots = [data_storage_root, Path(__file__).resolve().parent]
    source_dir = None
    for root in candidate_roots:
        candidate = root / "users" / seed_user_id
        if candidate.exists():
            source_dir = candidate
            break

    if source_dir is None:
        return False

    users_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, dest_dir)
    return True

def _extract_app_logs(text: str) -> list[dict]:
    if not text:
        return []
    logs: list[dict] = []
    parts = text.split("[APP_LOG] ")
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        first_line = chunk.splitlines()[0].strip()
        try:
            payload = json.loads(first_line)
        except Exception:
            payload = {"content": chunk}
        logs.append(payload)
    return logs

def _build_context_from_retrieval(retrieval: dict) -> str:
    context_logs = []
    for page in retrieval.get("retrieved_pages", []):
        user_input = page.get("user_input", "")
        for payload in _extract_app_logs(user_input):
            if isinstance(payload, dict):
                context_logs.append(
                    {
                        "app_log_id": payload.get("app_log_id"),
                        "timestamp": payload.get("timestamp") or payload.get("memory_timestamp"),
                        "app_name": payload.get("app_name"),
                        "api_name": payload.get("api_name"),
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    }
                )
            else:
                context_logs.append(
                    {
                        "app_log_id": None,
                        "timestamp": None,
                        "app_name": None,
                        "api_name": None,
                        "content": str(payload),
                    }
                )
    return json.dumps(context_logs, ensure_ascii=False, indent=2)

def _maybe_parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None

def _is_empty_response(response: object) -> bool:
    if response is None:
        return True
    if isinstance(response, str):
        return response.strip() == ""
    return False

def _find_sample_qa_file(sample_dir: Path, sample_id: str) -> Optional[Path]:
    direct = sample_dir / f"qa_human_{sample_id}.json"
    if direct.exists():
        return direct
    digits = [d for d in "".join(c if c.isdigit() else " " for c in sample_id).split() if d]
    if digits:
        candidate = sample_dir / f"qa_human_{digits[-1]}.json"
        if candidate.exists():
            return candidate
    matches = sorted(sample_dir.glob("qa_human_*.json"))
    if len(matches) == 1:
        return matches[0]
    return None

def _load_raw_qa_list(sample_dir: Path, sample_id: str) -> list[dict]:
    qa_file = _find_sample_qa_file(sample_dir, sample_id)
    if qa_file is None or not qa_file.exists():
        return []
    with qa_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "qa" in raw:
        raw = raw["qa"]
    return raw if isinstance(raw, list) else []

def simple_demo(dataset_size: str, target_sample_id: str, batch_size: int):
    print("MemoryOS Simple Demo")
    dataset_size = _normalize_membench_size(dataset_size)
    user_id = f"{target_sample_id}_{dataset_size}" if target_sample_id else f"demo_{dataset_size}"
    data_storage_root = _resolve_data_storage_root()
    skip_events = 0
    seeded = False

    baseline = _baseline_size(dataset_size)
    if baseline and target_sample_id:
        seed_user_id = f"{target_sample_id}_{baseline}"
        if _maybe_seed_user_data(data_storage_root, seed_user_id, user_id):
            seeded = True
            print(f"Seeded user data from '{seed_user_id}' to '{user_id}'.")
        baseline_count = _count_sample_events(baseline, target_sample_id)
        dest_dir = data_storage_root / "users" / user_id
        if baseline_count and (seeded or dest_dir.exists()):
            skip_events = baseline_count
    
    # 1. Initialize MemoryOS
    print("Initializing MemoryOS...")
    try:
        memo = Memoryos(
            user_id=user_id,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=str(data_storage_root),
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,  
            mid_term_heat_threshold=5,  
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=100,
            mid_term_similarity_threshold=0.6,
            embedding_model_name="text-embedding-3-large", # text-embedding-3-large, Qwen/Qwen3-Embedding-8B
            embedding_model_kwargs={
                "embedding_backend": "openai",
                "api_key": API_KEY,
                "api_base": "https://api.aimlapi.com/v1",
                "max_input_tokens": 8000,
                "truncate_from": "end",
                # "embedding_backend": "sentence-transformers",
                # "device": "mps",  # or "cpu"/"cuda"
            },
        )
        print("MemoryOS initialized successfully!\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load MemBench app logs
    print("Loading MemBench dataset...")
    try:
        sample = None
        for item in load_membench_dataset(
            DATA_ROOT,
            size=dataset_size,
        ):
            if target_sample_id and item.sample_id != target_sample_id:
                continue
            sample = item
            break
        if sample is None:
            print("No MemBench samples found for the requested filter.")
            return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(
        f"Loaded sample '{sample.sample_id}' with "
        f"{len(sample.app_logs)} events and {len(sample.qa)} QA pairs."
    )
    if skip_events:
        print(f"Skipping first {skip_events} events based on '{baseline}' data.")

    # 3. Add app logs as memories
    # print("Adding app logs as memories...")
    # added = 0
    # batch_contents = []
    # batch_time = None
    # for idx, event in enumerate(sample.app_logs):
    #     if skip_events and idx < skip_events:
    #         continue
    #     content, time_str = build_membench_memory_from_event(event)
    #     batch_contents.append(f"[APP_LOG] {content}")
    #     batch_time = time_str or batch_time
    #     added += 1
    #     if MAX_EVENTS is not None and added >= MAX_EVENTS:
    #         break
    #     if len(batch_contents) < batch_size:
    #         continue
    #     memo.add_memory(
    #         user_input="\n".join(batch_contents),
    #         agent_response="[ingested_batch]",
    #         timestamp=batch_time,
    #     )
    #     print(f"addeed {added}")
    #     print("-" * 20)
    #     batch_contents = []
    #     batch_time = None

    # if batch_contents:
    #     memo.add_memory(
    #         user_input="\n".join(batch_contents),
    #         agent_response="[ingested_batch]",
    #         timestamp=batch_time,
    #     )

    if not sample.qa:
        print("No QA pairs found; nothing to run.")
        return

    sample_dir = DATA_ROOT / sample.sample_id
    raw_qa_list = _load_raw_qa_list(sample_dir, sample.sample_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = OUTPUT_DIR / f"membench_answers_{sample.sample_id}_{dataset_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing answers to {output_dir}")
    results_path = output_dir / "memoryos_results.json"
    existing_results: list[dict] = []
    if results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing_results = existing
        except Exception:
            existing_results = []
    for idx, qa in enumerate(sample.qa):
        test_query = qa.question
        print(f"[{idx + 1}/{len(sample.qa)}] {test_query}")
        retrieval = memo.retriever.retrieve_context(user_query=test_query, user_id=memo.user_id)
        context = _build_context_from_retrieval(retrieval)
        prompt = PROMPT.replace("{{ question }}", test_query).replace("{{ context }}", context)
        print("prompt:", prompt)
        response = ""
        for attempt in range(1, RETRY_EMPTY_RESPONSES + 1):
            response = memo.client.chat_completion(
                model="gpt-5-mini-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
            )
            if not _is_empty_response(response):
                break
            print(f"Empty response (attempt {attempt}). Retrying...")
            time.sleep(RETRY_SLEEP_SECONDS)
        print("response:", response)
        parsed = _maybe_parse_json(response)
        record = {
            "idx": idx,
            "question": test_query,
            "response": response,
            "parsed_response": parsed,
            "reference": qa.final_answer,
            "evidence": qa.evidence,
        }

        raw_qa = raw_qa_list[idx] if idx < len(raw_qa_list) and isinstance(raw_qa_list[idx], dict) else {}
        prediction = parsed.get("answer") if isinstance(parsed, dict) else None
        predicted_evidence = parsed.get("evidence") if isinstance(parsed, dict) else []
        if not isinstance(predicted_evidence, list):
            predicted_evidence = []
        result_item = {
            "id": raw_qa.get("id"),
            "query": raw_qa.get("query") or test_query,
            "reference": raw_qa.get("reference") or qa.final_answer,
            "prediction": prediction,
            "predicted_evidence": predicted_evidence,
        }
        metadata = raw_qa.get("metadata")
        if isinstance(metadata, dict) and metadata:
            result_item["metadata"] = metadata
        existing_results.append(result_item)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
        print(f"Updated {results_path} ({len(existing_results)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemoryOS MemBench demo")
    parser.add_argument("--size", type=str, default=DEFAULT_DATASET_SIZE, help="Dataset size: small|medium|large")
    parser.add_argument(
        "--sample-id",
        type=str,
        default=DEFAULT_TARGET_SAMPLE_ID,
        help="Sample id to load (empty to take the first sample)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for app log ingest")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    simple_demo(args.size, args.sample_id, args.batch_size)
