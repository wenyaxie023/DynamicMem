#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

# Assuming the class above is saved in generation/letta/agent_loop.py
from generation.letta.agent_loop import LettaAgentLoop

load_dotenv()

QA_PROMPT_TEMPLATE = """You are answering a QA query from previously ingested user app logs.
Return plain natural language answer only.

Question:
{question}

Rules:
1. Use only information grounded in your archival memory (logs).
2. If the logs don't contain the answer, say "Information not found".
3. Use 'archival_memory_search' to find the evidence.
"""

def run_qa(
    *,
    input_user_dir: Path,
    qa_user_dir: Path,
    prediction_dir: Path,
    app_logs_filename: str,
    qa_filename: str,
    persona: Optional[str],
    human: Optional[str],
    resume: bool,
) -> List[Dict[str, Any]]:
    prediction_dir.mkdir(parents=True, exist_ok=True)

    app_logs_path = input_user_dir / app_logs_filename
    qa_path = qa_user_dir / qa_filename
    output_path = prediction_dir / "letta_results.json"

    # Data Loading
    if not app_logs_path.exists():
        raise FileNotFoundError(f"Logs not found: {app_logs_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA not found: {qa_path}")

    app_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    app_logs = (
        app_logs_payload
        if isinstance(app_logs_payload, list)
        else app_logs_payload.get("app_logs", [])
    )
    app_logs = [x for x in app_logs if isinstance(x, dict)]

    qa_list = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa_list, list):
        raise ValueError(f"QA payload must be a list: {qa_path}")

    agent = LettaAgentLoop(
        persona=persona,
        human=human,
    )

    run_name = input_user_dir.name or "letta"
    print(f"Ingesting {len(app_logs)} logs into archival memory for {run_name}...")
    agent.ingest_logs_bulk(app_logs)

    existing_results: List[Dict[str, Any]] = []
    done_keys = set()
    if resume and output_path.exists():
        try:
            existing_results = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(existing_results, list):
                existing_results = []
            for item in existing_results:
                if item.get("id") is not None:
                    done_keys.add(("id", item.get("id")))
                elif item.get("query"):
                    done_keys.add(("query", item.get("query")))
        except Exception:
            existing_results = []

    results = list(existing_results)

    try:
        for item in tqdm(qa_list, desc=f"LETTA-QA {run_name}"):
            query = str(item.get("query", ""))
            key = ("id", item.get("id")) if item.get("id") is not None else ("query", query)

            if key in done_keys:
                continue

            prompt = QA_PROMPT_TEMPLATE.format(question=query)

            t1 = time.time()
            # Agent will autonomously call archival_memory_search during this call
            answer_text = agent.ask(prompt)
            t2 = time.time()

            out = {
                "id": item.get("id"),
                "query": query,
                "reference": item.get("reference", ""),
                "prediction": answer_text,
                "metadata": {
                    **(item.get("metadata") or {}),
                    "response_time": t2 - t1,
                    "mode": "archival_memory_rag",
                },
            }

            results.append(out)
            output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        if hasattr(agent, "close"):
            agent.close()

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Letta baseline for QA.")
    parser.add_argument("--input-user-dir", type=Path, required=True)
    parser.add_argument("--qa-user-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--app-logs-filename", default="app_log_large.json")
    parser.add_argument("--qa-filename", default="qa.json")
    parser.add_argument("--persona", default=None)
    parser.add_argument("--human", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = run_qa(**vars(args))
    print("Total QA predictions:", len(results))

if __name__ == "__main__":
    main()
