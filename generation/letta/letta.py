#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from generation.letta.agent_loop import LettaAgentLoop, sort_logs

load_dotenv()

QA_PROMPT_TEMPLATE = """You are answering a QA query from previously ingested user app logs.
Return JSON only with this schema:
{
  "answer": string,
  "evidence": [{"app_log_id": string, "supporting_content": string}]
}

Question:
{question}

Rules:
1. Use only information grounded in ingested logs.
2. Evidence must reference app_log_id strings.
3. If unknown, answer conservatively and use [] evidence.
"""


def _normalize_user_dir(user_idx: str) -> str:
    if user_idx.isdigit():
        idx = int(user_idx)
        return f"{idx:03d}_user_{idx:03d}"
    return user_idx


def _normalize_user_id(user_idx: str) -> str:
    if user_idx.isdigit():
        return f"{int(user_idx):03d}"
    digits = "".join(ch for ch in user_idx if ch.isdigit())
    if len(digits) >= 3:
        return digits[-3:]
    return user_idx


def _select_qa_path(
    user_dir: Path,
    qa_dir: Path,
    user_idx: str,
) -> Path:
    user_id = _normalize_user_id(user_idx)
    qa_path = qa_dir / f"qa_human_{user_id}.json"
    if qa_path.exists():
        return qa_path
    legacy = user_dir / "QA.json"
    if legacy.exists():
        return legacy
    fallback = user_dir / "qa_w0_w4_with_app_logs.json"
    return fallback


def run_qa(
    *,
    user_idx: str,
    input_root_dir: Path,
    output_root_dir: Path,
    qa_dir: Path,
    retrieval_top_k: int,
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    letta_mode: str,
    allow_local_fallback: bool,
    persona: Optional[str],
    human: Optional[str],
    resume: bool,
) -> List[Dict[str, Any]]:
    user_dir_name = _normalize_user_dir(user_idx)
    input_user_dir = input_root_dir / user_dir_name
    output_user_dir = output_root_dir / user_dir_name
    prediction_dir = output_user_dir / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    app_logs_path = input_user_dir / "app_log_large.json"
    qa_path = _select_qa_path(input_user_dir, qa_dir, user_idx)
    output_path = prediction_dir / "letta_results.json"

    if not app_logs_path.exists():
        raise FileNotFoundError(f"app_log_large not found: {app_logs_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    app_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    app_logs = app_logs_payload if isinstance(app_logs_payload, list) else app_logs_payload.get("app_logs", [])
    app_logs = [x for x in app_logs if isinstance(x, dict)]
    app_logs = sort_logs(app_logs)

    qa_list = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa_list, list):
        raise ValueError(f"QA payload must be a list: {qa_path}")

    agent = LettaAgentLoop(
        user_namespace=f"qa::{user_dir_name}",
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        mode=letta_mode,
        allow_local_fallback=allow_local_fallback,
        persona=persona,
        human=human,
    )
    for log in tqdm(app_logs, desc=f"LETTA-INGEST {user_dir_name}"):
        agent.ingest_log(log)

    existing_results: List[Dict[str, Any]] = []
    done_keys = set()
    if resume and output_path.exists():
        try:
            existing_results = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(existing_results, list):
                existing_results = []
        except Exception:
            existing_results = []
        for item in existing_results:
            if item.get("id") is not None:
                done_keys.add(("id", item.get("id")))
            elif item.get("query"):
                done_keys.add(("query", item.get("query")))

    results = list(existing_results)
    try:
        for item in tqdm(qa_list, desc=f"LETTA-QA {user_dir_name}"):
            key = None
            if item.get("id") is not None:
                key = ("id", item.get("id"))
            elif item.get("query"):
                key = ("query", item.get("query"))
            if key and key in done_keys:
                continue

            query = str(item.get("query", ""))
            prompt = QA_PROMPT_TEMPLATE.format(question=query)
            t1 = time.time()
            try:
                parsed = agent.ask_json(prompt)
                raw = parsed
                parse_err = None if parsed else "Agent returned empty or non-JSON output"
            except Exception as exc:
                raw = {"_error": str(exc)}
                parsed = {"answer": "", "evidence": []}
                parse_err = f"LLM call failed: {exc}"
            t2 = time.time()

            evidence = parsed.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            out = {
                "id": item.get("id"),
                "query": query,
                "reference": item.get("reference", ""),
                "prediction": parsed.get("answer", ""),
                "predicted_evidence": evidence,
                "metadata": {
                    **(item.get("metadata") or {}),
                    "retrieval_mode": f"letta_agent_loop_{letta_mode}",
                    "retrieval_top_k": retrieval_top_k,
                    "response_time": t2 - t1,
                    "evidence_prediction": evidence,
                },
            }
            if parse_err:
                out["metadata"]["llm_parse_error"] = parse_err
                out["metadata"]["llm_raw_preview"] = str(raw)[:1000]

            results.append(out)
            done_keys.add(key)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        agent.close()

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Letta baseline for QA.")
    parser.add_argument("--user-idx", required=True, help="User index (1/2/3) or full directory name (001_user_001).")
    parser.add_argument(
        "--input-root-dir",
        type=Path,
        required=True,
        help="Root directory containing user app logs.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        required=True,
        help="Root directory for baseline outputs.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        required=True,
        help="Directory containing QA files like qa_human_001.json.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=10, help="Number of retrieved logs per question.")
    parser.add_argument("--llm-provider", type=str, default="openai", help="openai|azure|aimlapi|gemini|vllm")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=4)
    parser.add_argument(
        "--letta-mode",
        type=str,
        default="sdk",
        choices=["sdk", "local"],
        help="Retrieval backend: sdk tries Letta Python SDK; local uses lexical fallback.",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Fail if Letta SDK is unavailable/incompatible.",
    )
    parser.add_argument("--persona", type=str, default=None, help="Optional Letta core persona override.")
    parser.add_argument("--human", type=str, default=None, help="Optional Letta core human override.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = run_qa(
        user_idx=args.user_idx,
        input_root_dir=args.input_root_dir,
        output_root_dir=args.output_root_dir,
        qa_dir=args.qa_dir,
        retrieval_top_k=args.retrieval_top_k,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        letta_mode=args.letta_mode,
        allow_local_fallback=not args.no_local_fallback,
        persona=args.persona,
        human=args.human,
        resume=args.resume,
    )
    print("Total QA predictions:", len(results))


if __name__ == "__main__":
    main()
