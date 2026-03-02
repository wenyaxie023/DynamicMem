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
    output_path: Optional[Path],
    app_logs_filename: str,
    qa_filename: str,
    resume: bool,
    checkpoint_state_path: Optional[Path],
    agentfile_path: Optional[Path],
    lease_registry_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    prediction_dir.mkdir(parents=True, exist_ok=True)

    app_logs_path = input_user_dir / app_logs_filename
    qa_path = qa_user_dir / qa_filename
    if output_path is None:
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

    resolved_agentfile: Optional[Path] = None
    if checkpoint_state_path is not None and checkpoint_state_path.exists():
        state_raw = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
        if isinstance(state_raw, dict):
            final_af = state_raw.get("final_agentfile")
            if isinstance(final_af, str) and final_af.strip():
                candidate = Path(final_af)
                if candidate.exists():
                    resolved_agentfile = candidate
    if resolved_agentfile is None and agentfile_path is not None and agentfile_path.exists():
        resolved_agentfile = agentfile_path

    agent = LettaAgentLoop(
        create_agent=(resolved_agentfile is None),
        lease_registry_path=lease_registry_path,
    )

    run_name = input_user_dir.name or "letta"
    if resolved_agentfile is None:
        print(f"Ingesting {len(app_logs)} logs via dialogue turns for {run_name}...")
        agent.ingest_logs_dialogue(app_logs)
    else:
        print(f"Using checkpoint agentfile for QA base: {resolved_agentfile}")
        base_agentfile_bytes = resolved_agentfile.read_bytes()

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
            if resolved_agentfile is None:
                # Agent will autonomously call archival_memory_search during this call
                answer_text = agent.ask(prompt)
            else:
                # Reload a clean copy from final .af every QA turn.
                temp_agent_id = agent.import_agent_file(base_agentfile_bytes, activate=False)
                try:
                    answer_text = agent.ask_with_agent(temp_agent_id, prompt)
                finally:
                    agent.delete_agent(temp_agent_id, ignore_missing=True)
            t2 = time.time()

            out = {
                "id": item.get("id"),
                "query": query,
                "reference": item.get("reference", ""),
                "prediction": answer_text,
                "metadata": {
                    **(item.get("metadata") or {}),
                    "response_time": t2 - t1,
                    "mode": "letta",
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
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--app-logs-filename", default="app_log_large.json")
    parser.add_argument("--qa-filename", default="qa.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-state-path",
        type=Path,
        default=None,
        help="Optional checkpoint_state.json generated by checkpoint_agent_builder.py.",
    )
    parser.add_argument(
        "--agentfile-path",
        type=Path,
        default=None,
        help="Optional direct .af path used as QA base agent snapshot.",
    )
    parser.add_argument(
        "--lease-registry-path",
        type=Path,
        default=None,
        help="Optional JSON path to track temporary imported agent ids.",
    )
    args = parser.parse_args()

    results = run_qa(**vars(args))
    print("Total QA predictions:", len(results))

if __name__ == "__main__":
    main()
