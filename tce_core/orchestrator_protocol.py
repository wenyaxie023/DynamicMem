from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointHandle:
    checkpoint_id: str
    state_kind: str
    state_ref: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuerySpec:
    task_name: str
    item_key: str
    target_keys: List[str]
    checkpoint_timestamp: str
    task_query_text: str
    retrieval_query_text: str
    answer_query_text: str
    task_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalOptions:
    common: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    mode: str
    inline_memory_blocks: List[str] = field(default_factory=list)
    debug_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerExecutionResult:
    raw_output: Any
    prompt: str = ""
    debug_metadata: Dict[str, Any] = field(default_factory=dict)


def ensure_checkpoint_handle(value: Any, *, checkpoint_id: str = "") -> CheckpointHandle:
    if isinstance(value, CheckpointHandle):
        return value
    if isinstance(value, dict):
        return CheckpointHandle(
            checkpoint_id=str(value.get("checkpoint_id") or checkpoint_id or ""),
            state_kind=str(value.get("state_kind") or ""),
            state_ref=value.get("state_ref"),
            metadata=dict(value.get("metadata") or {}),
        )
    return CheckpointHandle(checkpoint_id=str(checkpoint_id or ""), state_kind="", state_ref=None, metadata={})


def ensure_retrieval_result(value: Any) -> RetrievalResult:
    if isinstance(value, RetrievalResult):
        return value
    if isinstance(value, dict):
        return RetrievalResult(
            mode=str(value.get("mode") or value.get("retrieval_mode") or ""),
            inline_memory_blocks=[
                str(block)
                for block in (value.get("inline_memory_blocks") or [])
                if str(block).strip()
            ],
            debug_metadata=dict(value.get("debug_metadata") or value.get("metadata") or {}),
        )
    return RetrievalResult(mode="", inline_memory_blocks=[], debug_metadata={})


def ensure_answer_execution_result(value: Any) -> AnswerExecutionResult:
    if isinstance(value, AnswerExecutionResult):
        return value
    if isinstance(value, dict):
        return AnswerExecutionResult(
            raw_output=value.get("raw_output"),
            prompt=str(value.get("prompt") or ""),
            debug_metadata=dict(value.get("debug_metadata") or {}),
        )
    return AnswerExecutionResult(raw_output=value, prompt="", debug_metadata={})
