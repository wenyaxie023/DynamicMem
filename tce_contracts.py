"""Shared TCE task-contract metadata helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Mapping, Optional, Set

TASK_CONTRACT_VERSION_V1 = "taskabc_v1"
TASK_CONTRACT_VERSION_V2 = "taskabc_v2"
LEGACY_TASK_CONTRACT_VERSION = TASK_CONTRACT_VERSION_V1
CURRENT_TASK_CONTRACT_VERSION = TASK_CONTRACT_VERSION_V2

RESEARCH_FRAME_VERSION_LEGACY = "rq_legacy_pre_20260413"
RESEARCH_FRAME_VERSION_V2 = "rq_20260413"

CANONICAL_RESEARCH_DOC_V2 = (
    "analysis_tools/tce_research_questions/001_user_001/new_research_question.md"
)
TASK_A_EXCLUDED_VALUE_FIELDS_V2 = frozenset({"priority", "schedule_date", "schedule_dates"})
STATE_VALIDATION_EXCLUDED_FIELD_TAILS_V2 = frozenset(
    {
        "priority",
        "signal",
        "signals",
        "schedule_date",
        "schedule_dates",
        "scheduled_date",
        "scheduled_dates",
    }
)
_TRANSITION_FIELDS = frozenset({"from", "to"})


def _nonempty_str(value: Any) -> str:
    return str(value or "").strip()


def _coerce_contract_version(value: Any) -> str:
    if isinstance(value, Mapping):
        return infer_task_contract_version(value)
    text = _nonempty_str(value)
    return text or LEGACY_TASK_CONTRACT_VERSION


def infer_task_contract_version(payload: Any) -> str:
    if isinstance(payload, Mapping):
        version = _nonempty_str(payload.get("task_contract_version"))
        if version:
            return version
        contract_metadata = payload.get("contract_metadata")
        if isinstance(contract_metadata, Mapping):
            version = _nonempty_str(contract_metadata.get("task_contract_version"))
            if version:
                return version
    return LEGACY_TASK_CONTRACT_VERSION


def task_contract_is_v2(value: Any) -> bool:
    return _coerce_contract_version(value) == TASK_CONTRACT_VERSION_V2


def task_contract_supports_change_tracking(value: Any) -> bool:
    return not task_contract_is_v2(value)


def has_materialized_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def drop_task_a_excluded_fields_v2(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if str(key).strip().lower() in TASK_A_EXCLUDED_VALUE_FIELDS_V2:
                continue
            out[key] = drop_task_a_excluded_fields_v2(child)
        return out
    if isinstance(value, list):
        return [drop_task_a_excluded_fields_v2(child) for child in value]
    return value


def _prune_empty_containers(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            cleaned_child = _prune_empty_containers(child)
            if has_materialized_value(cleaned_child):
                out[key] = cleaned_child
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            cleaned_child = _prune_empty_containers(child)
            if has_materialized_value(cleaned_child):
                out.append(cleaned_child)
        return out
    return value


def field_path_tail(path: Any) -> str:
    parts = [part.strip().lower() for part in str(path or "").split(".") if part.strip()]
    return parts[-1] if parts else ""


def state_validation_excludes_field_path_v2(path: Any) -> bool:
    return field_path_tail(path) in STATE_VALIDATION_EXCLUDED_FIELD_TAILS_V2


def normalize_task_a_current_value(
    value: Any,
    *,
    task_contract_version: Any,
) -> Any:
    candidate = copy.deepcopy(value)
    if isinstance(candidate, dict):
        normalized_keys = {str(key).strip().lower() for key in candidate.keys()}
        if normalized_keys and normalized_keys.issubset(_TRANSITION_FIELDS):
            candidate = copy.deepcopy(candidate.get("to"))
    if not has_materialized_value(candidate):
        return None
    if task_contract_is_v2(task_contract_version):
        candidate = drop_task_a_excluded_fields_v2(candidate)
        candidate = _prune_empty_containers(candidate)
        if not has_materialized_value(candidate):
            return None
    return candidate


def infer_research_frame_version(payload: Any) -> str:
    if isinstance(payload, Mapping):
        version = _nonempty_str(payload.get("research_frame_version"))
        if version:
            return version
        contract_metadata = payload.get("contract_metadata")
        if isinstance(contract_metadata, Mapping):
            version = _nonempty_str(contract_metadata.get("research_frame_version"))
            if version:
                return version
    if infer_task_contract_version(payload) == CURRENT_TASK_CONTRACT_VERSION:
        return RESEARCH_FRAME_VERSION_V2
    return RESEARCH_FRAME_VERSION_LEGACY


def infer_canonical_research_doc(payload: Any) -> Optional[str]:
    if isinstance(payload, Mapping):
        path = _nonempty_str(payload.get("canonical_research_doc"))
        if path:
            return path
        contract_metadata = payload.get("contract_metadata")
        if isinstance(contract_metadata, Mapping):
            path = _nonempty_str(contract_metadata.get("canonical_research_doc"))
            if path:
                return path
    if infer_task_contract_version(payload) == CURRENT_TASK_CONTRACT_VERSION:
        return CANONICAL_RESEARCH_DOC_V2
    return None


def normalized_contract_metadata(payload: Any) -> Dict[str, str]:
    out = {
        "task_contract_version": infer_task_contract_version(payload),
        "research_frame_version": infer_research_frame_version(payload),
    }
    canonical_doc = infer_canonical_research_doc(payload)
    if canonical_doc:
        out["canonical_research_doc"] = canonical_doc
    return out


def normalize_stage2_tasks(
    tasks: Iterable[Any],
    *,
    task_contract_version: Any,
) -> Set[str]:
    requested = {str(task).strip().lower() for task in tasks if str(task or "").strip()}
    aliases = {
        "a": "state_completion",
        "b": "change_tracking",
        "c": "apply",
        "rq3_apply": "apply",
        "rq3_apply_service_qa": "apply",
        "all": "all",
    }
    normalized = {aliases.get(task, task) for task in requested}
    if "all" in normalized:
        normalized = {"state_completion", "apply"} if task_contract_is_v2(task_contract_version) else {
            "state_completion",
            "change_tracking",
            "apply",
        }
    return normalized


def apply_contract_metadata(
    payload: Dict[str, Any],
    *,
    task_contract_version: Optional[str] = None,
    research_frame_version: Optional[str] = None,
    canonical_research_doc: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    contract_version = _nonempty_str(task_contract_version) or CURRENT_TASK_CONTRACT_VERSION
    frame_version = _nonempty_str(research_frame_version)
    if not frame_version:
        if contract_version == CURRENT_TASK_CONTRACT_VERSION:
            frame_version = RESEARCH_FRAME_VERSION_V2
        else:
            frame_version = RESEARCH_FRAME_VERSION_LEGACY
    canonical_doc = _nonempty_str(canonical_research_doc)
    if not canonical_doc and contract_version == CURRENT_TASK_CONTRACT_VERSION:
        canonical_doc = CANONICAL_RESEARCH_DOC_V2

    if overwrite or not _nonempty_str(payload.get("task_contract_version")):
        payload["task_contract_version"] = contract_version
    if overwrite or not _nonempty_str(payload.get("research_frame_version")):
        payload["research_frame_version"] = frame_version
    if canonical_doc and (overwrite or not _nonempty_str(payload.get("canonical_research_doc"))):
        payload["canonical_research_doc"] = canonical_doc
    return payload
