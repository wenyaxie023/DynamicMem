from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from concurrent.futures import Future, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_generation.io import read_json_list, write_json, write_run_state
from qa_generation.llm.client import LLMClient
from shared.config import GenerationConfig

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TaskItem = Dict[str, Any]
RecordItem = Dict[str, Any]


class BaseGenerator(ABC):
    def __init__(self, *, config: GenerationConfig) -> None:
        self.config = config
        self.logger = self._build_logger(self.log_path, self.logger_name)

    @property
    def logger_name(self) -> str:
        return "QA.generator"

    @property
    @abstractmethod
    def output_path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def state_path(self) -> Path:
        raise NotImplementedError

    @property
    def log_path(self) -> Path:
        return self.config.log_path

    @property
    def output_label(self) -> str:
        return "Output"

    @property
    def progress_desc(self) -> str:
        return "Generating"

    @staticmethod
    def _build_logger(log_path: Path, logger_name: str) -> logging.Logger:
        logger = logging.getLogger(logger_name)
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

    def _build_llm_client(self) -> LLMClient:
        return LLMClient(
            provider=self.config.provider,
            model_name=self.config.model_name,
            max_workers=self.config.max_workers,
            retry_times=self.config.retry_times,
        )

    def _task_uid(self, task: TaskItem) -> Optional[str]:
        uid = task.get("uid")
        return uid if isinstance(uid, str) and uid else None

    def _record_uid(self, record: RecordItem) -> Optional[str]:
        uid = record.get("uid")
        return uid if isinstance(uid, str) and uid else None

    def _to_record_dict(self, record: Any) -> RecordItem:
        if isinstance(record, dict):
            return record
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
            raise TypeError("to_dict() must return dict")
        if is_dataclass(record):
            payload = asdict(record)
            if isinstance(payload, dict):
                return payload
        raise TypeError(f"Unsupported record type: {type(record)!r}")

    def _load_existing_records(self) -> List[RecordItem]:
        return read_json_list(self.output_path, allow_missing=True)

    def _load_done_uids(self, existing: List[RecordItem]) -> set[str]:
        done: set[str] = set()
        for item in existing:
            uid = self._record_uid(item)
            if uid:
                done.add(uid)
        return done

    def _save_state(self, records: List[RecordItem], *, pending_uids: Optional[List[str]] = None) -> None:
        done_uids = [uid for uid in (self._record_uid(item) for item in records) if uid]
        write_run_state(path=self.state_path, done_uids=done_uids, pending_uids=pending_uids)

    @abstractmethod
    def build_pending_tasks(self, *, existing: List[RecordItem], done_uids: set[str]) -> List[TaskItem]:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, task: TaskItem) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_record(self, *, task: TaskItem, raw_response: Any) -> Any:
        raise NotImplementedError

    def log_start(self, *, existing_count: int, pending_count: int) -> None:
        self.logger.info("start generation pending=%d existing=%d", pending_count, existing_count)

    def run(self) -> int:
        existing = self._load_existing_records()
        done_uids = self._load_done_uids(existing)
        pending = self.build_pending_tasks(existing=existing, done_uids=done_uids)
        pending_uid_set = {uid for uid in (self._task_uid(task) for task in pending) if uid}
        self.log_start(existing_count=len(existing), pending_count=len(pending))

        records: List[RecordItem] = list(existing)
        if not pending:
            self._save_state(records, pending_uids=[])
            self.logger.info("no pending tasks, output=%s count=%d", self.output_path, len(records))
            print(f"No pending tasks. Output: {self.output_path} (count={len(records)})")
            return 0

        llm = self._build_llm_client()
        pbar = None
        fail_count = 0
        try:
            future_to_task: Dict[Future, TaskItem] = {}
            for task in pending:
                prompt = self.build_prompt(task)
                fut = llm.ask_async(prompt, response_type="json")
                future_to_task[fut] = task

            flush_count = 0
            if tqdm is not None:
                pbar = tqdm(total=len(future_to_task), desc=self.progress_desc)

            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    raw_resp = fut.result()
                    record = self.build_record(task=task, raw_response=raw_resp)
                    record_dict = self._to_record_dict(record)
                    records.append(record_dict)
                    flush_count += 1
                    if flush_count >= self.config.flush_every:
                        write_json(self.output_path, records)
                        done_now = {uid for uid in (self._record_uid(item) for item in records) if uid}
                        pending_now = sorted(uid for uid in pending_uid_set if uid not in done_now)
                        self._save_state(records, pending_uids=pending_now)
                        self.logger.info("flush checkpoint count=%d", len(records))
                        flush_count = 0
                except Exception as exc:
                    fail_count += 1
                    self.logger.warning("task failed uid=%s err=%s", self._task_uid(task), exc)
                finally:
                    if pbar is not None:
                        pbar.update(1)

            write_json(self.output_path, records)
            done_now = {uid for uid in (self._record_uid(item) for item in records) if uid}
            pending_now = sorted(uid for uid in pending_uid_set if uid not in done_now)
            self._save_state(records, pending_uids=pending_now)
            self.logger.info("generation done count=%d failed=%d", len(records), fail_count)
        finally:
            if pbar is not None:
                pbar.close()
            llm.close()

        print(f"Saved {self.output_label}: {self.output_path} (count={len(records)})")
        print(f"Saved state: {self.state_path}")
        print(f"Failed tasks: {fail_count}")
        return 0
