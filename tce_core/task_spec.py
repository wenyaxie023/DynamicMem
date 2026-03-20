"""Shared task-spec builder used by retrieval query and generation prompt."""

import json
from typing import Any, Dict, Iterable, List, Set


def humanize_key(key: str) -> str:
    text = str(key)
    category = ""
    state = text
    if ":" in text:
        category, state = text.split(":", 1)
    category = category.replace("_state", "")
    category = category.replace("_", " ").strip()
    state = state.replace("_", " ").strip()
    if category:
        return f"{category} {state}".strip()
    return state


def flatten_snapshot(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in snapshot.keys()):
        return {str(k): v for k, v in snapshot.items()}
    flat: Dict[str, Any] = {}
    for category, content in snapshot.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def _collect_leaf_field_names(value: Any, out: Set[str], *, limit: int = 24) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            field = str(k).strip().lower()
            if field:
                out.add(field)
            _collect_leaf_field_names(v, out, limit=limit)
            if len(out) >= limit:
                return
        return
    if isinstance(value, list) and value:
        _collect_leaf_field_names(value[0], out, limit=limit)


def key_descriptor(key: str, value_schema: Any) -> str:
    key_text = humanize_key(key)
    fields: Set[str] = set()
    _collect_leaf_field_names(value_schema, fields)
    if fields:
        return f"{key_text} [{', '.join(sorted(fields))}]"
    return key_text


def build_target_descriptors(
    target_keys: Iterable[str],
    value_schemas_by_key: Dict[str, Any],
) -> List[str]:
    return [key_descriptor(k, value_schemas_by_key.get(k)) for k in target_keys]


def _to_blank_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_blank_template(v) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        return [_to_blank_template(value[0])]
    return "<fill the blank>"


def build_prediction_task_instruction() -> str:
    return "Predict the user's current state values for the requested state items using only the provided user memory."


def build_prediction_task_text(cutoff_ts: str, target_template: Dict[str, Any]) -> str:
    template_block = json.dumps(target_template, ensure_ascii=False, sort_keys=True)
    return (
        f"As of {cutoff_ts}, infer the user's current state for all items in this template: "
        f"{template_block}. "
    )


def build_change_reasoning_instruction() -> str:
    return "Infer the latest change details for the requested changed state items using only the provided user memory."


def build_change_reasoning_task_text(
    current_cutoff_ts: str,
    previous_cutoff_ts: str,
    changed_template: Dict[str, Any],
) -> str:
    template_block = json.dumps(changed_template, ensure_ascii=False, sort_keys=True)
    return (
        f"As of {current_cutoff_ts}, for state items that changed since {previous_cutoff_ts}, "
        f"infer the latest change details using this template: {template_block}. "
    )


def build_prediction_task_from_checkpoint(
    checkpoint: Dict[str, Any],
    target_keys: Iterable[str],
) -> str:
    cutoff_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
    expected_snapshot = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
    target_template = {
        str(k): _to_blank_template(expected_snapshot.get(str(k)))
        for k in target_keys
    }
    return build_prediction_task_text(cutoff_ts, target_template)
