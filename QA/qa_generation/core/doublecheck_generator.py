from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qa_generation.io import read_json_list
from qa_generation.llm.parser import parse_doublecheck_response
from qa_generation.prompts.templates import render_doublecheck_prompt
from shared.config import GenerationConfig

from .base_generator import BaseGenerator, RecordItem, TaskItem
from .records import DoublecheckRecord


class DoublecheckGenerator(BaseGenerator):
    def __init__(self, *, config: GenerationConfig) -> None:
        super().__init__(config=config)
        self._source_count = 0
        self._context_by_uid: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def output_path(self) -> Path:
        return self.config.doublecheck_output_path

    @property
    def state_path(self) -> Path:
        return self.config.doublecheck_state_path

    @property
    def output_label(self) -> str:
        return "Doublecheck"

    @property
    def progress_desc(self) -> str:
        return "Doublecheck"

    def log_start(self, *, existing_count: int, pending_count: int) -> None:
        self.logger.info(
            "start doublecheck source=%d existing=%d pending=%d",
            self._source_count,
            existing_count,
            pending_count,
        )

    def build_pending_tasks(self, *, existing: List[RecordItem], done_uids: set[str]) -> List[TaskItem]:
        source = read_json_list(self.config.judge_refine_output_path, allow_missing=True)
        if not source:
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
        return render_doublecheck_prompt(
            question=task.get("query", ""),
            answer=task.get("reference", ""),
            context=self._resolve_context_logs(task),
        )

    def build_record(self, *, task: TaskItem, raw_response: Any) -> DoublecheckRecord:
        uid = task.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Task uid is required and must be a non-empty string")
        rationale, judge = parse_doublecheck_response(raw_response)
        metadata = task.get("metadata")
        out_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        out_metadata["doublecheck_info"] = {
            "rationale": rationale,
            "judge": judge,
        }
        return DoublecheckRecord(
            uid=uid,
            query=task.get("query", "") if isinstance(task.get("query"), str) else "",
            reference=task.get("reference", "") if isinstance(task.get("reference"), str) else "",
            prediction=task.get("prediction", "") if isinstance(task.get("prediction"), str) else "",
            metadata=out_metadata,
        )


def run_doublecheck(
    *,
    config: GenerationConfig,
) -> int:
    return DoublecheckGenerator(config=config).run()


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
