"""Optional post-TCE final-checkpoint QA hook."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def build_final_qa_prompt_with_inline_memory(question: str, context_text: str) -> str:
    return """# Task
Answer the question using only the retrieved app logs from the final TCE checkpoint memory.

Return strict JSON with exactly these keys:
- "answer": string
- "evidence": array of objects with:
  - "app_log_id": string
  - "evidence_content": string

If the context is insufficient, return an empty answer and an empty evidence list.
Fill `app_log_id` when it can be stably mapped to a retrieved log; otherwise use "".

# Question
{question}

# Retrieved Context
{context}
""".format(
        question=question,
        context=context_text,
    )


def build_final_qa_prompt_with_agent_memory(question: str) -> str:
    return """# Task
Answer the question using only the final TCE checkpoint memory already stored in the agent.

Return strict JSON with exactly these keys:
- "answer": string
- "evidence": array of objects with:
  - "app_log_id": string
  - "evidence_content": string

If the memory is insufficient, return an empty answer and an empty evidence list.
Fill `app_log_id` when it can be stably mapped to an agent-stored log; otherwise use "".

# Question
{question}

# Agent memory
The relevant user memory is already stored in the agent and is not pasted inline below.
""".format(
        question=question,
    )


def _normalize_user_id(user_id: str) -> str:
    if str(user_id).isdigit():
        return "{:03d}".format(int(user_id))
    digits = "".join(ch for ch in str(user_id) if ch.isdigit())
    if len(digits) >= 3:
        return digits[-3:]
    return str(user_id)


def _select_qa_path(benchmark_path: Path, user_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)

    user_dir = benchmark_path.parent
    normalized_user_id = _normalize_user_id(user_id)

    candidate = user_dir / "qa_human_{}.json".format(normalized_user_id)
    if candidate.exists():
        return candidate

    candidate = Path(__file__).resolve().parents[1] / "generation" / "qa" / "qa_human_{}.json".format(
        normalized_user_id
    )
    if candidate.exists():
        return candidate

    legacy = user_dir / "QA.json"
    if legacy.exists():
        return legacy

    return user_dir / "qa_w0_w4_with_app_logs.json"


def _load_qa_list(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "qa" in raw:
        raw = raw["qa"]
    if not isinstance(raw, list):
        raise ValueError("QA file must be a list or {'qa': [...]} object: {}".format(path))
    return [item for item in raw if isinstance(item, dict)]


def _final_qa_output_path(output_path: Path, explicit_output_path: Optional[str]) -> Path:
    if explicit_output_path:
        return Path(explicit_output_path)
    return output_path.with_name("{}_final_qa.json".format(output_path.stem))


def _normalize_answer_output(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as exc:
            return {"answer": "", "evidence": []}, "LLM JSON parse failed: {}".format(exc)
    if not isinstance(payload, dict):
        return {"answer": "", "evidence": []}, "Unexpected LLM output type: {}".format(type(payload).__name__)

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    return {
        "answer": str(payload.get("answer", "") or ""),
        "evidence": evidence,
    }, None


def _reference_app_logs(item: Dict[str, Any], app_log_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata = item.get("metadata") or {}
    raw_ids = metadata.get("app_log_ids", [])
    if not isinstance(raw_ids, list):
        return []
    out: List[Dict[str, Any]] = []
    for raw_id in raw_ids:
        if raw_id is None:
            continue
        log = app_log_by_id.get(str(raw_id).strip())
        if log is not None:
            out.append(log)
    return out


def _item_key(item: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    if item.get("id") is not None:
        return ("id", item.get("id"))
    query = item.get("query")
    if query:
        return ("query", query)
    return None
