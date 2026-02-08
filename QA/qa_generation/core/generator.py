from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qa_generation.io import read_json_list
from qa_generation.llm.parser import parse_qa_response
from qa_generation.prompts.templates import render_qa_prompt
from qa_generation.resolvers import EvidenceResolver
from shared.config import GenerationConfig

from .base_generator import BaseGenerator, RecordItem, TaskItem
from .records import QAMetadata, QARecord


class QAGenerator(BaseGenerator):
    def __init__(self, *, config: GenerationConfig) -> None:
        super().__init__(config=config)
        self._sampled_count = 0
        raw_items = read_json_list(config.raw_path, allow_missing=True)
        self.resolver = EvidenceResolver(raw_items)

    @property
    def output_path(self) -> Path:
        return self.config.qa_output_path

    @property
    def state_path(self) -> Path:
        return self.config.run_state_path

    @property
    def output_label(self) -> str:
        return "QA"

    @property
    def progress_desc(self) -> str:
        return "Generating QA"

    def log_start(self, *, existing_count: int, pending_count: int) -> None:
        self.logger.info(
            "start generation sampled=%d existing=%d",
            self._sampled_count,
            existing_count,
        )

    @staticmethod
    def _normalize_str_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []
        return [x for x in value if isinstance(x, str) and x]

    def build_pending_tasks(self, *, existing: List[RecordItem], done_uids: set[str]) -> List[TaskItem]:
        sampled = read_json_list(self.config.sampled_tasks_path, allow_missing=False)
        self._sampled_count = len(sampled)

        pending: List[TaskItem] = []
        for item in sampled:
            uid = item.get("uid")
            payload = item.get("payload")
            if not isinstance(uid, str) or not uid or not isinstance(payload, dict):
                continue
            if uid in done_uids:
                continue
            pending.append(item)
        return pending

    def build_prompt(self, task: TaskItem) -> str:
        payload = task["payload"]
        return render_qa_prompt(
            task=payload.get("task"),
            context=payload.get("context"),
            qtype=task.get("qtype", payload.get("qtype")),
        )

    def build_record(self, *, task: TaskItem, raw_response: Any) -> QARecord:
        parsed = parse_qa_response(raw_response)
        qa = parsed if isinstance(parsed, dict) else {}

        qtype = task["qtype"]
        category = task["category"]

        evidence_raw = qa.get("metadata", {}).get("reference_evidence")
        evidence = self._normalize_str_list(evidence_raw)

        if qtype in {1, 2, 3, 4}:
            event_ids = evidence
            reference_evidence = self.resolver.map_event_ids_to_log_ids(event_ids)
        else:
            reference_evidence = evidence
            event_ids = self.resolver.map_log_ids_to_event_ids(reference_evidence)

        uid = task.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Task uid is required and must be a non-empty string")

        return QARecord(
            uid=uid,
            query=qa.get("query", ""),
            reference=qa.get("reference", ""),
            prediction="",
            metadata=QAMetadata(
                qtype=qtype,
                category=category,
                event_ids=event_ids,
                reference_evidence=reference_evidence,
                draft_question=qa.get("metadata", {}).get("draft_question", ""),
                draft_answer=qa.get("metadata", {}).get("draft_answer", ""),
            ),
        )


def run_generation(
    *,
    config: GenerationConfig,
) -> int:
    return QAGenerator(config=config).run()
