from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import Future, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from qa_generation.llm.client import LLMClient
from qa_generation.llm.parser import parse_qa_response
from qa_generation.prompts.templates import render_qa_prompt
from shared.config import GenerationConfig

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TaskItem = Dict[str, Any]
QAItem = Dict[str, Any]


def _read_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("QA.generator")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _load_done_uids(path: Path) -> Set[str]:
    done: Set[str] = set()
    for item in _read_list(path):
        uid = item.get("uid")
        if isinstance(uid, str) and uid:
            done.add(uid)
    return done


def _save_state(path: Path, qa_items: List[QAItem]) -> None:
    done_uids = [item["uid"] for item in qa_items if isinstance(item.get("uid"), str)]
    payload = {
        "done_count": len(done_uids),
        "done_uids": done_uids,
    }
    _write_json(path, payload)


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str) and x]


def _build_edges_from_raw(raw_items: List[Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    event_to_logs: Dict[str, Set[str]] = {}
    log_to_events: Dict[str, Set[str]] = {}

    for item in raw_items:
        app_log_ids = _normalize_str_list(item.get("app_log_ids"))
        event_ids = _normalize_str_list(item.get("event_ids"))

        source = item.get("from")
        ctx = item.get("context")
        if not isinstance(ctx, dict):
            continue

        if source == "event":
            event_id = ctx.get("event_id")
            if isinstance(event_id, str) and event_id:
                event_to_logs.setdefault(event_id, set()).update(app_log_ids)
        elif source == "log":
            log_id = ctx.get("app_log_id")
            if isinstance(log_id, str) and log_id:
                log_to_events.setdefault(log_id, set()).update(event_ids)

    return event_to_logs, log_to_events


def _build_prompt(task_item: TaskItem) -> str:
    payload = task_item["payload"]
    prompt = render_qa_prompt(
        task=payload.get("task"),
        context=payload.get("context"),
        qtype=task_item.get("qtype", payload.get("qtype")),
    )
    return prompt


def _to_qa_item(
    task_item: TaskItem,
    raw_response: Any,
    *,
    event_to_logs: Dict[str, Set[str]],
    log_to_events: Dict[str, Set[str]],
) -> QAItem:
    parsed = parse_qa_response(raw_response)
    qa = parsed if isinstance(parsed, dict) else {}

    qtype = task_item["qtype"]
    category = task_item["category"]

    evidence_raw = qa.get("metadata", {}).get("reference_evidence")
    evidence = _normalize_str_list(evidence_raw)

    if qtype in {1, 2, 3, 4}:
        event_ids = evidence
        ref_evidence_set: Set[str] = set()
        for event_id in event_ids:
            ref_evidence_set.update(event_to_logs.get(event_id, set()))
        reference_evidence = sorted(ref_evidence_set)
    else:
        reference_evidence = evidence
        event_ids_set: Set[str] = set()
        for log_id in reference_evidence:
            event_ids_set.update(log_to_events.get(log_id, set()))
        event_ids = sorted(event_ids_set)

    return {
        "uid": task_item.get("uid"),
        "query": qa.get("query", ""),
        "reference": qa.get("reference", ""),
        "prediction": "",
        "metadata": {
            "qtype": qtype,
            "category": category,
            "event_ids": event_ids,
            "reference_evidence": reference_evidence,
            "draft_question": qa.get("metadata", {}).get("draft_question", ""),
            "draft_answer": qa.get("metadata", {}).get("draft_answer", ""),
        },
    }


def run_generation(
    *,
    config: GenerationConfig,
) -> int:
    sampled_path = config.sampled_tasks_path
    qa_path = config.qa_output_path
    state_path = config.run_state_path

    sampled = _read_list(sampled_path)
    existing = _read_list(qa_path)
    done_uids = _load_done_uids(qa_path)
    raw_items = _read_list(config.raw_path)
    event_to_logs, log_to_events = _build_edges_from_raw(raw_items)
    logger = _build_logger(config.log_path)
    logger.info("start generation sampled=%d existing=%d", len(sampled), len(existing))

    pending: List[TaskItem] = []
    for item in sampled:
        uid = item.get("uid")
        payload = item.get("payload")
        if not isinstance(uid, str) or not uid or not isinstance(payload, dict):
            continue
        if uid in done_uids:
            continue
        pending.append(item)

    qa_items: List[QAItem] = list(existing)
    if not pending:
        _save_state(state_path, qa_items)
        logger.info("no pending tasks, output=%s count=%d", qa_path, len(qa_items))
        print(f"No pending tasks. Output: {qa_path} (count={len(qa_items)})")
        return 0

    llm = LLMClient(
        provider=config.provider,
        model_name=config.model_name,
        max_workers=config.max_workers,
        retry_times=config.retry_times,
    )

    pbar = None
    try:
        futures: List[Tuple[TaskItem, Future]] = []
        for task in pending:
            prompt = _build_prompt(task)
            futures.append((task, llm.ask_async(prompt, response_type="json")))

        future_to_task = {fut: task for task, fut in futures}
        flush_count = 0
        fail_count = 0
        if tqdm is not None:
            pbar = tqdm(total=len(future_to_task), desc="Generating QA")

        for fut in as_completed(future_to_task):
            task = future_to_task[fut]
            try:
                raw_resp = fut.result()
                qa_item = _to_qa_item(
                    task,
                    raw_resp,
                    event_to_logs=event_to_logs,
                    log_to_events=log_to_events,
                )
                qa_items.append(qa_item)
                flush_count += 1
                if flush_count >= config.flush_every:
                    _write_json(qa_path, qa_items)
                    _save_state(state_path, qa_items)
                    logger.info("flush checkpoint count=%d", len(qa_items))
                    flush_count = 0
            except Exception as exc:
                fail_count += 1
                uid = task.get("uid")
                logger.warning("task failed uid=%s err=%s", uid, exc)
            finally:
                if pbar is not None:
                    pbar.update(1)

        _write_json(qa_path, qa_items)
        _save_state(state_path, qa_items)
        logger.info("generation done count=%d failed=%d", len(qa_items), fail_count)
    finally:
        if pbar is not None:
            pbar.close()
        llm.close()

    print(f"Saved QA: {qa_path} (count={len(qa_items)})")
    print(f"Saved state: {state_path}")
    print(f"Failed tasks: {fail_count}")
    return 0
