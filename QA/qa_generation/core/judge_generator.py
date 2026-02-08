from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qa_generation.io import read_json_list
from qa_generation.llm.parser import parse_judge_response
from qa_generation.prompts.templates import render_judge_prompt
from shared.config import GenerationConfig

from .base_generator import BaseGenerator, RecordItem, TaskItem
from .records import JudgeRecord


class JudgeRefineGenerator(BaseGenerator):
    def __init__(self, *, config: GenerationConfig) -> None:
        super().__init__(config=config)
        self._source_count = 0
        self._context_by_uid: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def output_path(self) -> Path:
        return self.config.judge_refine_output_path

    @property
    def state_path(self) -> Path:
        return self.config.judge_state_path

    @property
    def output_label(self) -> str:
        return "Judge+Refine"

    @property
    def progress_desc(self) -> str:
        return "Judge & Refine"

    def log_start(self, *, existing_count: int, pending_count: int) -> None:
        self.logger.info(
            "start judge source=%d existing=%d pending=%d",
            self._source_count,
            existing_count,
            pending_count,
        )

    def build_pending_tasks(self, *, existing: List[RecordItem], done_uids: set[str]) -> List[TaskItem]:
        source = read_json_list(self.config.qa_output_path, allow_missing=False)
        self._source_count = len(source)
        self._context_by_uid = _load_qa_context_by_uid(self.config.qa_context_path)
        if not self._context_by_uid:
            raise FileNotFoundError(
                f"Missing or empty QA context cache: {self.config.qa_context_path}. "
                "Run `python -m QA generation context-cache --user-id <id>` first."
            )
        pending: List[TaskItem] = []
        for item in source:
            uid = item.get("uid")
            if not isinstance(uid, str) or not uid:
                continue
            if uid in done_uids:
                continue
            pending.append(item)
        return pending

    def _resolve_context_logs(self, task: TaskItem) -> List[Dict[str, Any]]:
        uid = task.get("uid")
        if not isinstance(uid, str) or not uid:
            return []
        return self._context_by_uid.get(uid, [])

    def build_prompt(self, task: TaskItem) -> str:
        return render_judge_prompt(
            question=task.get("query", ""),
            answer=task.get("reference", ""),
            context=self._resolve_context_logs(task),
        )

    def build_record(self, *, task: TaskItem, raw_response: Any) -> JudgeRecord:
        uid = task.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Task uid is required and must be a non-empty string")

        rationale, judge, refine_info = parse_judge_response(raw_response)
        metadata = task.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Task metadata must be an object")
        out_metadata = dict(metadata)
        refine_payload: Any = "null"
        if isinstance(refine_info, dict):
            refine_check = refine_info.get("refine_check")
            if isinstance(refine_check, str) and refine_check.isdigit():
                refine_check = int(refine_check)
            if not isinstance(refine_check, int):
                refine_check = 0
            refine_payload = {
                "draft": refine_info.get("draft", "") if isinstance(refine_info.get("draft"), str) else "",
                "refine_check": 1 if refine_check == 1 else 0,
                "refine_query": refine_info.get("refine_query", "") if isinstance(refine_info.get("refine_query"), str) else "",
                "refine_answer": refine_info.get("refine_answer", "") if isinstance(refine_info.get("refine_answer"), str) else "",
            }
        query = task.get("query", "") if isinstance(task.get("query"), str) else ""
        reference = task.get("reference", "") if isinstance(task.get("reference"), str) else ""
        history = {"query": "null", "answer": "null"}

        if isinstance(refine_payload, dict) and refine_payload.get("refine_check") == 1:
            refine_query = refine_payload.get("refine_query")
            refine_answer = refine_payload.get("refine_answer")
            if isinstance(refine_query, str) and refine_query and refine_query.lower() != "null":
                history["query"] = query
                query = refine_query
            if isinstance(refine_answer, str) and refine_answer and refine_answer.lower() != "null":
                history["answer"] = reference
                reference = refine_answer
            refine_payload["history"] = history
        elif isinstance(refine_payload, dict):
            refine_payload["history"] = history

        out_metadata["checker_info"] = {"rationale": rationale, "judge": judge, "refine": refine_payload}

        return JudgeRecord(
            uid=uid,
            query=query,
            reference=reference,
            prediction=task.get("prediction", "") if isinstance(task.get("prediction"), str) else "",
            metadata=out_metadata,
        )


def _load_qa_context_by_uid(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_json_list(path, allow_missing=True)
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        uid = row.get("uid")
        context = row.get("context")
        if not isinstance(uid, str) or not uid:
            continue
        if not isinstance(context, list):
            continue
        mapping[uid] = [x for x in context if isinstance(x, dict)]
    return mapping


def run_judge_refine(
    *,
    config: GenerationConfig,
) -> int:
    return JudgeRefineGenerator(config=config).run()
