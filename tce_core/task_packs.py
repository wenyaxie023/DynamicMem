"""Shared TCE task-pack builders and pack-first extraction helpers."""

import copy
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from .prompts import (
    TASK_C_V2_ACTION_CONFIGURATION_TASK_INSTRUCTION,
    TASK_C_V2_INFORMATION_REQUEST_TASK_INSTRUCTION,
    TASK_C_V2_USER_COMMUNICATION_TASK_INSTRUCTION,
    SCHEDULE_DATE_ENCODING_TEXT,
    build_task_c_task_body,
    build_rq3_apply_question_pack_prompt,
    build_rq3_apply_rewrite_prompt,
    build_rq3_apply_validation_prompt,
    build_task_c_v2_question_pack_prompt,
    build_task_c_v2_rewrite_prompt,
    build_task_c_v2_validation_prompt,
)
from .scoring_points import (
    SCORING_POINTS_VERSION,
    build_validated_apply_answer_scoring_points,
    build_change_reason_scoring_points,
    build_task_c_v2_structured_answer_scoring_points,
    build_task_c_v2_user_communication_answer_scoring_points,
    build_value_scoring_points,
)
from .state_validation import bool_like, value_signature
from .state_validation import collect_evidence_ids as collect_state_evidence_ids
from .task_spec import flatten_snapshot, humanize_key
from tce_contracts import (
    CANONICAL_RESEARCH_DOC_V2,
    CURRENT_TASK_CONTRACT_VERSION,
    LEGACY_TASK_CONTRACT_VERSION,
    RESEARCH_FRAME_VERSION_V2,
    apply_contract_metadata,
    has_materialized_value,
    infer_task_contract_version,
    normalize_task_a_current_value,
    normalize_stage2_tasks,
    task_contract_is_v2,
)

STATE_COMPLETION_PACK_VERSION = "v6"
CHANGE_TRACKING_PACK_VERSION = "v6"
APPLY_PACK_VERSION_V1 = "v6"
APPLY_PACK_VERSION_V2 = "v9"
APPLY_PACK_PROMPT_VERSION_V1 = "apply_pack_prompt_v13"
APPLY_PACK_PROMPT_VERSION_V2 = "apply_pack_prompt_v31_taskc_leaf_core_anchors"
CHANGE_REASON_TEMPLATE = "<fill the blank>"
EVIDENCE_TEMPLATE = [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}]
APPLY_VALIDATION_SEMANTIC_CRITERIA_V1 = [
    "personalization_necessity",
    "service_decision_quality",
    "answer_groundedness",
]
APPLY_VALIDATION_SEMANTIC_CRITERIA_V2 = [
    "answerability",
    "service_realism",
    "full_field_dependency",
    "low_leakage",
    "output_groundedness",
]
_TRANSITION_FIELDS = {"from", "to"}
_REASON_CURRENT_STATE_UNRESOLVED_MISSING_TO = "current_state_unresolved_missing_to"
_REASON_CURRENT_STATE_UNRESOLVED_EMPTY = "current_state_unresolved_empty"
_REASON_BEFORE_STATE_UNRESOLVED = "before_state_unresolved"
_REASON_AFTER_STATE_UNRESOLVED = "after_state_unresolved"
_REASON_REFERENCE_CHANGE_REASON_MISSING = "reference_change_reason_missing"
_REASON_REFERENCE_CHANGE_REASON_INVALID = "reference_change_reason_invalid"
_REASON_TASK_A_EXCLUDED_FIELDS_ONLY = "task_a_value_empty_after_excluded_fields_removed"
_REASON_TASK_C_V2_MISSING_SCENARIO = "missing_scenario"
_REASON_TASK_C_V2_MISSING_TASK_INSTRUCTION = "missing_task_instruction"
_REASON_TASK_C_V2_MISSING_REFERENCE_ANSWER = "missing_reference_answer"
_REASON_TASK_C_V2_MISSING_OUTPUT_TEMPLATE = "missing_output_template"
_REASON_TASK_C_V2_MISSING_REFERENCE_OUTPUT = "missing_reference_output"
_REASON_TASK_C_V2_OUTPUT_TEMPLATE_MISMATCH = "output_template_mismatch"
_REASON_TASK_C_V2_NOT_STRUCTURED_SERVICE_OBJECT = "not_structured_service_object"
_REASON_TASK_C_V2_RAW_STATE_MIRROR = "raw_state_mirror"
_REASON_TASK_C_V2_NO_ACCEPTED_ITEMS = "no_accepted_task_c_items"


def _apply_pack_version(task_contract_version: str) -> str:
    return APPLY_PACK_VERSION_V2 if task_contract_is_v2(task_contract_version) else APPLY_PACK_VERSION_V1


def _apply_pack_prompt_version(task_contract_version: str) -> str:
    return APPLY_PACK_PROMPT_VERSION_V2 if task_contract_is_v2(task_contract_version) else APPLY_PACK_PROMPT_VERSION_V1


def _apply_validation_semantic_criteria(task_contract_version: str, *, service_family: str = "") -> List[str]:
    if task_contract_is_v2(task_contract_version):
        return APPLY_VALIDATION_SEMANTIC_CRITERIA_V2
    return APPLY_VALIDATION_SEMANTIC_CRITERIA_V1
def fill_blank_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: fill_blank_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_blank_template(v) for v in value]
    return "<fill the blank>"


def _infer_apply_state_type(state_key: str) -> str:
    prefix = str(state_key or "").split(":", 1)[0].strip().lower()
    if prefix == "habits_state":
        return "habit"
    if prefix == "preferences_state":
        return "preference"
    return "attribute"


def _infer_task_c_v2_service_family(state_key: str) -> str:
    state_type = _infer_apply_state_type(state_key)
    if state_type == "habit":
        return "user_communication"
    if state_type == "preference":
        return "information_request_construction"
    return "action_configuration"


def _task_c_v2_uses_structured_output(service_family: str) -> bool:
    return str(service_family or "").strip() != "user_communication"


def _task_c_v2_task_instruction(service_family: str, generated_value: Any = "") -> str:
    normalized_family = str(service_family or "").strip()
    if normalized_family == "user_communication":
        return TASK_C_V2_USER_COMMUNICATION_TASK_INSTRUCTION
    if normalized_family == "information_request_construction":
        return TASK_C_V2_INFORMATION_REQUEST_TASK_INSTRUCTION
    if normalized_family == "action_configuration":
        return TASK_C_V2_ACTION_CONFIGURATION_TASK_INSTRUCTION
    return str(generated_value or "").strip()


def _task_c_v2_fill_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _task_c_v2_fill_template(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_task_c_v2_fill_template(child) for child in value]
    return "<fill>"


def _task_c_v2_leaf_paths(value: Any, path: str = "") -> List[Tuple[str, Any]]:
    if isinstance(value, dict):
        out: List[Tuple[str, Any]] = []
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            child_path = key if not path else f"{path}.{key}"
            out.extend(_task_c_v2_leaf_paths(child, child_path))
        return out
    if isinstance(value, list):
        out: List[Tuple[str, Any]] = []
        base_path = path or "current_value"
        for index, child in enumerate(value):
            out.extend(_task_c_v2_leaf_paths(child, f"{base_path}.{index}"))
        return out
    return [(path or "current_value", value)]


def _stable_item_id(prefix: str, identity: Dict[str, Any]) -> str:
    digest = hashlib.sha1(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _validated_snapshot_flat(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return flatten_snapshot(checkpoint.get("validated_snapshot_state") or {})


def _validated_state_keys(checkpoint: Dict[str, Any]) -> List[str]:
    return sorted(_validated_snapshot_flat(checkpoint).keys())


def _transition_field_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, child in value.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key not in _TRANSITION_FIELDS:
            return {}
        normalized[normalized_key] = child
    return normalized if normalized else {}


def _resolve_task_a_ground_truth(validated_value: Any) -> Tuple[bool, Any, List[str]]:
    transition_map = _transition_field_map(validated_value)
    if transition_map:
        projected = transition_map.get("to")
        if not has_materialized_value(projected):
            return False, None, [_REASON_CURRENT_STATE_UNRESOLVED_MISSING_TO]
        return True, copy.deepcopy(projected), []
    if not has_materialized_value(validated_value):
        return False, None, [_REASON_CURRENT_STATE_UNRESOLVED_EMPTY]
    return True, copy.deepcopy(validated_value), []


def _resolve_task_a_candidates_from_flat(
    validated_flat: Dict[str, Any],
    *,
    task_contract_version: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    resolved: Dict[str, Any] = {}
    filtered: Dict[str, Dict[str, Any]] = {}
    for state_key in sorted(validated_flat.keys()):
        is_valid, projected_value, reason_codes = _resolve_task_a_ground_truth(validated_flat.get(state_key))
        if is_valid and task_contract_is_v2(task_contract_version):
            projected_value = normalize_task_a_current_value(
                projected_value,
                task_contract_version=task_contract_version,
            )
            if not has_materialized_value(projected_value):
                is_valid = False
                reason_codes = [_REASON_TASK_A_EXCLUDED_FIELDS_ONLY]
        if is_valid:
            resolved[state_key] = projected_value
        else:
            filtered[state_key] = {
                "reason_codes": list(reason_codes),
            }
    return resolved, filtered


def _resolve_task_b_ground_truth(
    *,
    previous_current_value: Any,
    current_validated_value: Any,
    current_current_value: Any,
) -> Tuple[bool, Any, Any, List[str]]:
    transition_map = _transition_field_map(current_validated_value)
    reason_codes: List[str] = []

    before_value = None
    after_value = None
    if transition_map:
        if has_materialized_value(transition_map.get("from")):
            before_value = copy.deepcopy(transition_map.get("from"))
        elif has_materialized_value(previous_current_value):
            before_value = copy.deepcopy(previous_current_value)
        else:
            reason_codes.append(_REASON_BEFORE_STATE_UNRESOLVED)

        if has_materialized_value(transition_map.get("to")):
            after_value = copy.deepcopy(transition_map.get("to"))
        else:
            reason_codes.append(_REASON_AFTER_STATE_UNRESOLVED)
    else:
        if has_materialized_value(previous_current_value):
            before_value = copy.deepcopy(previous_current_value)
        else:
            reason_codes.append(_REASON_BEFORE_STATE_UNRESOLVED)
        if has_materialized_value(current_current_value):
            after_value = copy.deepcopy(current_current_value)
        else:
            reason_codes.append(_REASON_AFTER_STATE_UNRESOLVED)

    if reason_codes:
        return False, None, None, reason_codes
    return True, before_value, after_value, []


def _extract_reference_change_reason(state_observability: Any) -> str:
    if not isinstance(state_observability, dict):
        return ""
    return str(state_observability.get("last_change_reason", "") or "").strip()


def _reference_change_reason_is_valid(questionability_entry: Any) -> bool:
    if not isinstance(questionability_entry, dict):
        return False
    validation = (
        questionability_entry.get("change_reason_validation")
        if isinstance(questionability_entry.get("change_reason_validation"), dict)
        else None
    )
    if validation is None:
        return False
    return bool_like(validation.get("is_valid"), default=False)


def _require_validated_benchmark(benchmark: Dict[str, Any]) -> None:
    checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("benchmark.checkpoints must be a list")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        if "validated_snapshot_state" not in checkpoint or "state_questionability" not in checkpoint:
            raise ValueError(
                "Validated benchmark required: run data_construction.build_tce_state_validation first."
            )


def _task_label(task_name: str) -> str:
    mapping = {
        "state_completion": "state_completion",
        "change_tracking": "change_tracking",
        "apply": "apply",
    }
    return mapping.get(str(task_name), str(task_name))


def _iter_checkpoint_dicts(benchmark: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return []
    return [checkpoint for checkpoint in checkpoints if isinstance(checkpoint, dict)]


def _state_completion_counts(benchmark: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    rows: List[Tuple[str, int]] = []
    total = 0
    for checkpoint in _iter_checkpoint_dicts(benchmark):
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        count = len(_validated_state_keys(checkpoint))
        rows.append((checkpoint_id, count))
        total += count
    return total, rows


def _change_tracking_counts(benchmark: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    rows: List[Tuple[str, int]] = []
    total = 0
    prev_validated_flat: Optional[Dict[str, Any]] = None
    for checkpoint in _iter_checkpoint_dicts(benchmark):
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        validated_flat = _validated_snapshot_flat(checkpoint)
        count = 0
        if prev_validated_flat is not None:
            for state_key in sorted(set(prev_validated_flat.keys()) & set(validated_flat.keys())):
                if prev_validated_flat.get(state_key) != validated_flat.get(state_key):
                    count += 1
        rows.append((checkpoint_id, count))
        total += count
        prev_validated_flat = validated_flat
    return total, rows


def _apply_candidate_counts(benchmark: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    rows: List[Tuple[str, int]] = []
    total = 0
    for checkpoint in _iter_checkpoint_dicts(benchmark):
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        count = len(_validated_state_keys(checkpoint))
        rows.append((checkpoint_id, count))
        total += count
    return total, rows


def _print_stage2_task_summary(
    *,
    task_name: str,
    total_count: int,
    checkpoint_counts: List[Tuple[str, int]],
) -> None:
    print(f"[Stage2][pre][{_task_label(task_name)}] total_keys={total_count}")
    for checkpoint_id, count in checkpoint_counts:
        print(
            f"[Stage2][pre][{_task_label(task_name)}][checkpoint]"
            f" checkpoint_id={checkpoint_id},"
            f" candidate_keys={count}"
        )


def _state_completion_question_text(
    *,
    state_key: str,
    answer_template: Any,
) -> str:
    key_label = humanize_key(state_key)
    template_block = json.dumps({state_key: answer_template}, ensure_ascii=False, sort_keys=True)
    text = (
        f"Infer the user's current state for {key_label} "
        f"({state_key}) using this template: {template_block}."
    )
    if str(state_key or "").strip().startswith("habits_state:"):
        text += f" Schedule date encoding: {SCHEDULE_DATE_ENCODING_TEXT}"
    return text


def _change_tracking_question_text(
    *,
    state_key: str,
    before_template: Any,
    after_template: Any,
) -> str:
    key_label = humanize_key(state_key)
    template_block = json.dumps(
        {
            state_key: {
                "before": before_template,
                "after": after_template,
                "change_reason": CHANGE_REASON_TEMPLATE,
                "evidence": EVIDENCE_TEMPLATE,
            }
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        f"Infer how the user's {key_label} ({state_key}) changed using this template: "
        f"{template_block}."
    )


def build_state_completion_pack_inplace(
    *,
    benchmark: Dict[str, Any],
    generator_client: Optional[Any] = None,
    validator_client: Optional[Any] = None,
    show_progress: bool = False,
    enable_scoring_points: bool = True,
) -> Dict[str, Any]:
    _require_validated_benchmark(benchmark)
    task_contract_version = infer_task_contract_version(benchmark)
    reuse_cache: Dict[str, Dict[str, Any]] = {}
    checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return benchmark

    total_keys, _ = _state_completion_counts(benchmark)
    progress = tqdm(
        total=total_keys,
        desc="TCE Stage2 state_completion",
        unit="key",
        disable=(not show_progress),
    )

    try:
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
            validated_flat = _validated_snapshot_flat(checkpoint)
            resolved_flat, filtered_keys = _resolve_task_a_candidates_from_flat(
                validated_flat,
                task_contract_version=task_contract_version,
            )
            pack_keys: Dict[str, Any] = {}
            cp_reused = 0
            cp_computed = 0
            for state_key in sorted(validated_flat.keys()):
                if show_progress:
                    progress.set_postfix(checkpoint_id=checkpoint_id, state_key=state_key)
                validated_value = resolved_flat.get(state_key)
                if state_key not in resolved_flat:
                    progress.update(1)
                    continue
                validated_sig = value_signature(validated_value)
                cache_key = f"{state_key}::{validated_sig}"
                cached_item = reuse_cache.get(cache_key)
                if isinstance(cached_item, dict):
                    item = copy.deepcopy(cached_item)
                    item["pack_source"] = "reused"
                    cp_reused += 1
                else:
                    answer_template = fill_blank_template(validated_value)
                    identity = {
                        "state_key": state_key,
                        "validated_state_value_signature": validated_sig,
                        "pack_version": STATE_COMPLETION_PACK_VERSION,
                    }
                    scoring_points = (
                        build_value_scoring_points(
                            state_key=state_key,
                            value=validated_value,
                            generator_client=generator_client,
                            validator_client=validator_client,
                            prefix=f"scp_{state_key.replace(':', '_')}",
                        )
                        if enable_scoring_points
                        else []
                    )
                    item = {
                        "item_id": _stable_item_id("scp", identity),
                        "state_key": state_key,
                        "question_text": _state_completion_question_text(
                            state_key=state_key,
                            answer_template=answer_template,
                        ),
                        "answer_template": answer_template,
                        "retrieval_query": _state_completion_question_text(
                            state_key=state_key,
                            answer_template=answer_template,
                        ),
                        "scoring_points": scoring_points,
                        "pack_source": "computed",
                        "pack_identity": identity,
                    }
                    reuse_cache[cache_key] = copy.deepcopy(item)
                    cp_computed += 1
                pack_keys[state_key] = item
                progress.update(1)

            checkpoint["state_completion_pack"] = {
                "version": STATE_COMPLETION_PACK_VERSION,
                "pack_authoring": "point_based_vnext",
                "scoring_points_version": SCORING_POINTS_VERSION,
                "keys": pack_keys,
            }
            if filtered_keys:
                checkpoint["state_completion_pack"]["filtered_keys"] = filtered_keys
            if show_progress:
                print(
                    "[Stage2][state_completion][checkpoint]"
                    f" checkpoint_id={checkpoint_id},"
                    f" keys={len(pack_keys)},"
                    f" filtered={len(filtered_keys)},"
                    f" reused={cp_reused},"
                    f" computed={cp_computed}"
                )
    finally:
        progress.close()

    return benchmark


def build_change_tracking_pack_inplace(
    *,
    benchmark: Dict[str, Any],
    generator_client: Optional[Any] = None,
    validator_client: Optional[Any] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    _require_validated_benchmark(benchmark)
    task_contract_version = infer_task_contract_version(benchmark)
    reuse_cache: Dict[str, Dict[str, Any]] = {}
    checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return benchmark

    prev_checkpoint_id = ""
    prev_checkpoint_ts = ""
    prev_validated_flat: Optional[Dict[str, Any]] = None
    prev_resolved_flat: Optional[Dict[str, Any]] = None
    total_keys, _ = _change_tracking_counts(benchmark)
    progress = tqdm(
        total=total_keys,
        desc="TCE Stage2 change_tracking",
        unit="key",
        disable=(not show_progress),
    )

    try:
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
            checkpoint_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
            validated_flat = _validated_snapshot_flat(checkpoint)
            obs_flat = flatten_snapshot(checkpoint.get("state_observability") or {})
            qmap = checkpoint.get("state_questionability") if isinstance(checkpoint.get("state_questionability"), dict) else {}
            resolved_flat, _ = _resolve_task_a_candidates_from_flat(
                validated_flat,
                task_contract_version=task_contract_version,
            )
            pack_keys: Dict[str, Any] = {}
            filtered_keys: Dict[str, Dict[str, Any]] = {}
            cp_reused = 0
            cp_computed = 0

            if prev_validated_flat is not None and prev_checkpoint_ts:
                for state_key in sorted(set(prev_validated_flat.keys()) & set(validated_flat.keys())):
                    if show_progress:
                        progress.set_postfix(checkpoint_id=checkpoint_id, state_key=state_key)
                    prev_current_value = (prev_resolved_flat or {}).get(state_key)
                    current_validated_value = validated_flat.get(state_key)
                    current_current_value = resolved_flat.get(state_key)
                    is_valid, before_value, after_value, reason_codes = _resolve_task_b_ground_truth(
                        previous_current_value=prev_current_value,
                        current_validated_value=current_validated_value,
                        current_current_value=current_current_value,
                    )
                    if not is_valid:
                        filtered_keys[state_key] = {"reason_codes": list(reason_codes)}
                        progress.update(1)
                        continue
                    if before_value == after_value:
                        progress.update(1)
                        continue
                    state_obs = obs_flat.get(state_key) if isinstance(obs_flat.get(state_key), dict) else {}
                    state_qmeta = qmap.get(state_key) if isinstance(qmap.get(state_key), dict) else {}
                    reference_change_reason = _extract_reference_change_reason(state_obs)
                    if not reference_change_reason:
                        filtered_keys[state_key] = {
                            "reason_codes": [_REASON_REFERENCE_CHANGE_REASON_MISSING]
                        }
                        progress.update(1)
                        continue
                    if not _reference_change_reason_is_valid(state_qmeta):
                        filtered_keys[state_key] = {
                            "reason_codes": [_REASON_REFERENCE_CHANGE_REASON_INVALID]
                        }
                        progress.update(1)
                        continue
                    before_sig = value_signature(before_value)
                    after_sig = value_signature(after_value)
                    reason_sig = value_signature(reference_change_reason)
                    cache_key = (
                        f"{state_key}::{before_sig}::{after_sig}::{reason_sig}::{CHANGE_TRACKING_PACK_VERSION}"
                    )
                    cached_item = reuse_cache.get(cache_key)
                    if isinstance(cached_item, dict):
                        item = copy.deepcopy(cached_item)
                        item["pack_source"] = "reused"
                        cp_reused += 1
                    else:
                        before_template = fill_blank_template(before_value)
                        after_template = fill_blank_template(after_value)
                        identity = {
                            "state_key": state_key,
                            "before_signature": before_sig,
                            "after_signature": after_sig,
                            "reference_change_reason_signature": reason_sig,
                            "pack_version": CHANGE_TRACKING_PACK_VERSION,
                        }
                        item = {
                            "item_id": _stable_item_id("ctp", identity),
                            "state_key": state_key,
                            "question_text": _change_tracking_question_text(
                                state_key=state_key,
                                before_template=before_template,
                                after_template=after_template,
                            ),
                            "before_template": before_template,
                            "after_template": after_template,
                            "retrieval_query": _change_tracking_question_text(
                                state_key=state_key,
                                before_template=before_template,
                                after_template=after_template,
                            ),
                            "reference_change_reason": reference_change_reason,
                            "before_scoring_points": build_value_scoring_points(
                                state_key=state_key,
                                value=before_value,
                                generator_client=generator_client,
                                validator_client=validator_client,
                                prefix=f"ctp_before_{state_key.replace(':', '_')}",
                            ),
                            "after_scoring_points": build_value_scoring_points(
                                state_key=state_key,
                                value=after_value,
                                generator_client=generator_client,
                                validator_client=validator_client,
                                prefix=f"ctp_after_{state_key.replace(':', '_')}",
                            ),
                            "change_reason_scoring_points": build_change_reason_scoring_points(
                                state_key=state_key,
                                before_value=before_value,
                                after_value=after_value,
                                reference_change_reason=reference_change_reason,
                                generator_client=generator_client,
                                validator_client=validator_client,
                                prefix=f"ctp_reason_{state_key.replace(':', '_')}",
                            ),
                            "pack_source": "computed",
                            "pack_identity": identity,
                        }
                        reuse_cache[cache_key] = copy.deepcopy(item)
                        cp_computed += 1
                    pack_keys[state_key] = item
                    progress.update(1)

            checkpoint["change_tracking_pack"] = {
                "version": CHANGE_TRACKING_PACK_VERSION,
                "pack_authoring": "point_based_vnext",
                "scoring_points_version": SCORING_POINTS_VERSION,
                "previous_checkpoint_id": prev_checkpoint_id,
                "previous_cutoff_ts": prev_checkpoint_ts,
                "keys": pack_keys,
            }
            if filtered_keys:
                checkpoint["change_tracking_pack"]["filtered_keys"] = filtered_keys
            if show_progress:
                print(
                    "[Stage2][change_tracking][checkpoint]"
                    f" checkpoint_id={checkpoint_id},"
                    f" keys={len(pack_keys)},"
                    f" filtered={len(filtered_keys)},"
                    f" reused={cp_reused},"
                    f" computed={cp_computed}"
                )
            prev_checkpoint_id = checkpoint_id
            prev_checkpoint_ts = checkpoint_ts
            prev_validated_flat = validated_flat
            prev_resolved_flat = resolved_flat
    finally:
        progress.close()

    return benchmark

def _base_apply_item_failures(
    item: Dict[str, Any],
    *,
    state_key: str,
    state_value: Any,
    task_contract_version: str,
) -> List[str]:
    failed: List[str] = []
    if task_contract_is_v2(task_contract_version):
        expected_family = _infer_task_c_v2_service_family(state_key)
        structured_v2 = _task_c_v2_uses_structured_output(expected_family)
        if not str(item.get("scenario") or "").strip():
            failed.append(_REASON_TASK_C_V2_MISSING_SCENARIO)
        if not str(item.get("task_instruction") or "").strip():
            failed.append(_REASON_TASK_C_V2_MISSING_TASK_INSTRUCTION)
        if structured_v2:
            if item.get("output_template") is None:
                failed.append(_REASON_TASK_C_V2_MISSING_OUTPUT_TEMPLATE)
            if item.get("reference_output") is None:
                failed.append(_REASON_TASK_C_V2_MISSING_REFERENCE_OUTPUT)
        else:
            if not str(item.get("reference_answer") or "").strip():
                failed.append(_REASON_TASK_C_V2_MISSING_REFERENCE_ANSWER)
        if structured_v2 and item.get("output_template") is not None and item.get("reference_output") is not None:
            output_template = item.get("output_template")
            reference_output = item.get("reference_output")
            if not isinstance(output_template, dict) or not isinstance(reference_output, dict):
                failed.append(_REASON_TASK_C_V2_NOT_STRUCTURED_SERVICE_OBJECT)
            else:
                if output_template != _task_c_v2_fill_template(reference_output):
                    failed.append(_REASON_TASK_C_V2_OUTPUT_TEMPLATE_MISMATCH)
                if reference_output == state_value:
                    failed.append(_REASON_TASK_C_V2_RAW_STATE_MIRROR)
        return failed
    if any(key in item for key in ("service_category", "question", "reference_answer")):
        if not str(item.get("service_category") or "").strip():
            failed.append("missing_service_category")
        if not str(item.get("question") or item.get("apply_question") or "").strip():
            failed.append("missing_question")
        if not str(item.get("reference_answer") or item.get("apply_reference_answer") or "").strip():
            failed.append("missing_reference_answer")
    return failed


def _normalize_generated_items(
    raw_out: Any,
    item_count: int,
    *,
    state_key: str,
    state_value: Any,
    task_contract_version: str,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_out, dict):
        return []
    raw_items = raw_out.get("items")
    if task_contract_is_v2(task_contract_version) and not isinstance(raw_items, list):
        raw_item = raw_out.get("item")
        if isinstance(raw_item, dict):
            raw_items = [raw_item]
    if not isinstance(raw_items, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        if task_contract_is_v2(task_contract_version):
            expected_family = _infer_task_c_v2_service_family(state_key)
            normalized = {
                "qa_id": f"q{idx+1}",
                "service_family": expected_family,
                "scenario": str(item.get("scenario") or "").strip(),
                "task_instruction": _task_c_v2_task_instruction(
                    expected_family,
                    item.get("task_instruction"),
                ),
                "reference_anchors": copy.deepcopy(item.get("reference_anchors"))
                if isinstance(item.get("reference_anchors"), list)
                else [],
                "retrieval_query": "",
            }
            if _task_c_v2_uses_structured_output(expected_family):
                normalized["output_template"] = item.get("output_template")
                normalized["reference_output"] = item.get("reference_output")
            else:
                normalized["reference_answer"] = str(item.get("reference_answer") or "").strip()
            out.append(normalized)
            continue
        service_category = str(item.get("service_category") or "").strip()
        question = str(item.get("question") or item.get("apply_question") or "").strip()
        reference_answer = str(item.get("reference_answer") or item.get("apply_reference_answer") or "").strip()
        out.append(
            {
                "qa_id": f"q{idx+1}",
                "service_category": service_category,
                "question": question,
                "reference_answer": reference_answer,
                "apply_scenario": str(item.get("apply_scenario") or "").strip(),
                "apply_question": question,
                "apply_reference_answer": reference_answer,
                "retrieval_query": "",
            }
        )
    return out[: max(1, int(item_count))]


def _derive_apply_rubric_alias(points: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        text = str(point.get("point_text") or "").strip()
        if text:
            out.append(text)
    return out


def build_rq3_apply_retrieval_query(
    *,
    service_category: str = "",
    service_family: str = "",
    scenario: str = "",
    task_instruction: str = "",
    output_template: Any = None,
    question: str = "",
    apply_scenario: str = "",
    apply_question: str = "",
) -> str:
    service_family = str(service_family or "").strip()
    scenario = str(scenario or "").strip()
    task_instruction = str(task_instruction or "").strip()
    service_category = str(service_category or "").strip()
    question = str(question or apply_question or "").strip()
    if service_family:
        return build_task_c_task_body(
            scenario=scenario,
            task_instruction=task_instruction,
            output_template=output_template,
        )

    parts = []
    if service_category:
        parts.append("Service category:\n{}".format(service_category))
    if str(apply_scenario or "").strip():
        parts.append("Service scenario:\n{}".format(str(apply_scenario or "").strip()))
    if question:
        parts.append("Question:\n{}".format(question))
    return "\n\n".join(parts).strip()


def _refresh_reused_apply_key_node(
    key_node: Dict[str, Any],
    *,
    state_sig: str,
    evidence_sig: str,
    evidence_ids: Sequence[str],
    apply_pack_prompt_version: str,
    apply_pack_version: str,
) -> Dict[str, Any]:
    refreshed = copy.deepcopy(key_node)
    identity = dict(refreshed.get("pack_identity") or {})
    identity["validated_state_value_signature"] = state_sig
    identity["evidence_signature"] = evidence_sig
    identity["prompt_version"] = apply_pack_prompt_version
    identity["pack_version"] = apply_pack_version
    refreshed["pack_identity"] = identity
    for field_name in ("items", "discarded_items"):
        entries = refreshed.get(field_name)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            item["gold_memory_evidence_app_log_ids"] = list(evidence_ids)
    return refreshed


def _call_json(client: Any, prompt: str) -> Any:
    try:
        return client.ask(prompt, response_type="json")
    except Exception as exc:
        return {"_error": str(exc)}


def _normalize_apply_semantic_criteria(
    raw: Any,
    *,
    task_contract_version: str,
    service_family: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized: List[Dict[str, Any]] = []
    failed_rules: List[str] = []
    expected_criteria = _apply_validation_semantic_criteria(
        task_contract_version,
        service_family=service_family,
    )
    criteria = raw.get("criteria") if isinstance(raw, dict) else None
    if not isinstance(criteria, list):
        return [], ["llm_invalid"]
    if len(criteria) != len(expected_criteria):
        failed_rules.append("llm_invalid")
    seen: Set[str] = set()
    for idx, expected in enumerate(expected_criteria):
        item = criteria[idx] if idx < len(criteria) else None
        if not isinstance(item, dict):
            failed_rules.append("llm_invalid")
            normalized.append(
                {"criterion": expected, "pass": False, "analysis": ""}
            )
            continue
        criterion = str(item.get("criterion") or "").strip()
        analysis = str(item.get("analysis") or "").strip()
        passed = item.get("pass")
        if criterion != expected or criterion in seen:
            failed_rules.append("llm_invalid")
        if criterion not in expected_criteria:
            failed_rules.append("llm_invalid")
        if not isinstance(passed, bool):
            failed_rules.append("llm_invalid")
            passed = False
        if not analysis:
            failed_rules.append("llm_invalid")
        normalized.append(
            {
                "criterion": expected,
                "pass": bool(passed) if isinstance(passed, bool) else False,
                "analysis": analysis,
            }
        )
        seen.add(criterion)
    if seen != set(expected_criteria):
        failed_rules.append("llm_invalid")
    deduped: List[str] = []
    for rule in failed_rules:
        if rule and rule not in deduped:
            deduped.append(rule)
    return normalized, deduped


def _validate_apply_item(
    *,
    validator_client: Any,
    state_key: str,
    state_value: Any,
    item: Dict[str, Any],
    task_contract_version: str = LEGACY_TASK_CONTRACT_VERSION,
) -> Tuple[bool, Dict[str, Any]]:
    # Task C v2 item validation is semantic-only; scoring points are materialized later.
    if task_contract_is_v2(task_contract_version):
        prompt = build_task_c_v2_validation_prompt(
            state_key=state_key,
            state_value=state_value,
            service_family=str(item.get("service_family") or ""),
            scenario=str(item.get("scenario") or ""),
            task_instruction=str(item.get("task_instruction") or ""),
            output_template=item.get("output_template"),
            reference_output=item.get("reference_output"),
            reference_anchors=item.get("reference_anchors"),
            reference_answer=str(item.get("reference_answer") or ""),
        )
    else:
        prompt = build_rq3_apply_validation_prompt(
            state_key=state_key,
            state_value=state_value,
            service_category=str(item.get("service_category") or ""),
            question=str(item.get("question") or item.get("apply_question") or ""),
            reference_answer=str(item.get("reference_answer") or item.get("apply_reference_answer") or ""),
            apply_scenario=item.get("apply_scenario", ""),
            apply_question=str(item.get("question") or item.get("apply_question") or ""),
            apply_reference_answer=str(item.get("reference_answer") or item.get("apply_reference_answer") or ""),
        )
    raw = _call_json(validator_client, prompt)
    failed_rules = list(
        _base_apply_item_failures(
            item,
            state_key=state_key,
            state_value=state_value,
            task_contract_version=task_contract_version,
        )
    )
    semantic_criteria, schema_failures = _normalize_apply_semantic_criteria(
        raw,
        task_contract_version=task_contract_version,
        service_family=str(item.get("service_family") or ""),
    )
    for rule in schema_failures:
        if rule not in failed_rules:
            failed_rules.append(rule)
    for item_criterion in semantic_criteria:
        if not bool(item_criterion.get("pass")):
            name = str(item_criterion.get("criterion") or "").strip()
            if name and name not in failed_rules:
                failed_rules.append(name)
    is_valid = (
        (not failed_rules)
        and len(semantic_criteria)
        == len(
            _apply_validation_semantic_criteria(
                task_contract_version,
                service_family=str(item.get("service_family") or ""),
            )
        )
        and all(bool(item_criterion.get("pass")) for item_criterion in semantic_criteria)
    )
    payload = {
        "is_valid": is_valid,
        "semantic_criteria": semantic_criteria,
        "failed_rules": failed_rules,
    }
    return is_valid, payload


def _rewrite_apply_item(
    *,
    generator_client: Any,
    state_key: str,
    state_value: Any,
    item: Dict[str, Any],
    validation_payload: Dict[str, Any],
    task_contract_version: str = LEGACY_TASK_CONTRACT_VERSION,
) -> Dict[str, Any]:
    if task_contract_is_v2(task_contract_version):
        prompt = build_task_c_v2_rewrite_prompt(
            state_key=state_key,
            state_value=state_value,
            service_family=str(item.get("service_family") or _infer_task_c_v2_service_family(state_key)),
            scenario=str(item.get("scenario") or ""),
            task_instruction=str(item.get("task_instruction") or ""),
            output_template=item.get("output_template"),
            reference_output=item.get("reference_output"),
            reference_anchors=item.get("reference_anchors"),
            failed_rules=list(validation_payload.get("failed_rules") or []),
            semantic_criteria=list(validation_payload.get("semantic_criteria") or []),
            reference_answer=str(item.get("reference_answer") or ""),
        )
    else:
        prompt = build_rq3_apply_rewrite_prompt(
            state_key=state_key,
            state_value=state_value,
            service_category=str(item.get("service_category") or ""),
            question=str(item.get("question") or item.get("apply_question") or ""),
            reference_answer=str(item.get("reference_answer") or item.get("apply_reference_answer") or ""),
            apply_scenario=item.get("apply_scenario", ""),
            apply_question=str(item.get("question") or item.get("apply_question") or ""),
            apply_reference_answer=str(item.get("reference_answer") or item.get("apply_reference_answer") or ""),
            failed_rules=list(validation_payload.get("failed_rules") or []),
            semantic_criteria=list(validation_payload.get("semantic_criteria") or []),
        )
    raw = _call_json(generator_client, prompt)
    if not isinstance(raw, dict):
        return item
    if task_contract_is_v2(task_contract_version):
        def _rewrite_string_delta(field_name: str) -> str:
            if field_name in raw and raw.get(field_name) is not None:
                return str(raw.get(field_name) or "").strip()
            return str(item.get(field_name) or "").strip()

        def _rewrite_object_delta(field_name: str) -> Any:
            if field_name in raw:
                return raw.get(field_name)
            return item.get(field_name)

        rewritten = {
            "qa_id": str(item.get("qa_id") or ""),
            "service_family": str(item.get("service_family") or _infer_task_c_v2_service_family(state_key)).strip(),
            "scenario": _rewrite_string_delta("scenario"),
            "task_instruction": _task_c_v2_task_instruction(
                str(item.get("service_family") or _infer_task_c_v2_service_family(state_key)),
                _rewrite_string_delta("task_instruction"),
            ),
            "reference_anchors": copy.deepcopy(raw.get("reference_anchors"))
            if isinstance(raw.get("reference_anchors"), list)
            else (
                copy.deepcopy(item.get("reference_anchors"))
                if isinstance(item.get("reference_anchors"), list)
                else []
            ),
            "retrieval_query": "",
        }
        if _task_c_v2_uses_structured_output(rewritten["service_family"]):
            rewritten["output_template"] = _rewrite_object_delta("output_template")
            rewritten["reference_output"] = _rewrite_object_delta("reference_output")
        else:
            rewritten["reference_answer"] = _rewrite_string_delta("reference_answer")
        return rewritten
    question = str(raw.get("question") or item.get("question") or item.get("apply_question") or "").strip()
    reference_answer = str(
        raw.get("reference_answer") or item.get("reference_answer") or item.get("apply_reference_answer") or ""
    ).strip()
    return {
        "qa_id": str(item.get("qa_id") or ""),
        "service_category": str(raw.get("service_category") or item.get("service_category") or "").strip(),
        "question": question,
        "reference_answer": reference_answer,
        "apply_scenario": str(raw.get("apply_scenario") or item.get("apply_scenario") or "").strip(),
        "apply_question": question,
        "apply_reference_answer": reference_answer,
        "retrieval_query": "",
    }


def _build_apply_key_node(
    *,
    generator_client: Any,
    validator_client: Any,
    checkpoint_ts: str,
    state_key: str,
    state_value: Any,
    state_observability: Dict[str, Any],
    item_count_per_key: int,
    max_rewrites: int,
    save_raw: bool,
    task_contract_version: str,
) -> Dict[str, Any]:
    if task_contract_is_v2(task_contract_version):
        prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp=checkpoint_ts,
            state_key=state_key,
            state_value=state_value,
            service_family=_infer_task_c_v2_service_family(state_key),
        )
    else:
        prompt = build_rq3_apply_question_pack_prompt(
            checkpoint_timestamp=checkpoint_ts,
            state_key=state_key,
            state_value=state_value,
            item_count=item_count_per_key,
        )
    generated = _call_json(generator_client, prompt)
    items = _normalize_generated_items(
        generated,
        item_count_per_key,
        state_key=state_key,
        state_value=state_value,
        task_contract_version=task_contract_version,
    )
    accepted: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    gold_memory_evidence_app_log_ids = collect_state_evidence_ids(state_observability)

    for idx, item in enumerate(items):
        candidate = dict(item)
        if not candidate.get("qa_id"):
            candidate["qa_id"] = f"q{idx + 1}"
        answer_scoring_points: List[Dict[str, Any]] = []
        final_qa_validation = {
            "is_valid": False,
            "semantic_criteria": [],
            "failed_rules": ["unknown"],
            "rewrite_attempts": 0,
        }
        qa_passed = False
        for attempt in range(max(0, int(max_rewrites)) + 1):
            is_semantically_valid, qa_validation_payload = _validate_apply_item(
                validator_client=validator_client,
                state_key=state_key,
                state_value=state_value,
                item=candidate,
                task_contract_version=task_contract_version,
            )
            final_qa_validation = {
                "is_valid": bool(is_semantically_valid),
                "semantic_criteria": list(qa_validation_payload.get("semantic_criteria") or []),
                "failed_rules": list(qa_validation_payload.get("failed_rules") or []),
                "rewrite_attempts": attempt,
            }
            if is_semantically_valid:
                qa_passed = True
                break
            if attempt < int(max_rewrites):
                candidate = _rewrite_apply_item(
                    generator_client=generator_client,
                    state_key=state_key,
                    state_value=state_value,
                    item=candidate,
                    validation_payload=qa_validation_payload,
                    task_contract_version=task_contract_version,
                )
        if not qa_passed:
            if task_contract_is_v2(task_contract_version):
                structured_v2 = _task_c_v2_uses_structured_output(str(candidate.get("service_family") or ""))
                discarded.append(
                    {
                        "qa_id": str(candidate.get("qa_id") or ""),
                        "service_family": str(candidate.get("service_family") or ""),
                        "scenario": str(candidate.get("scenario") or ""),
                        "task_instruction": str(candidate.get("task_instruction") or ""),
                        "reference_anchors": copy.deepcopy(candidate.get("reference_anchors"))
                        if isinstance(candidate.get("reference_anchors"), list)
                        else [],
                        "reference_answer": str(candidate.get("reference_answer") or ""),
                        "output_template": candidate.get("output_template") if structured_v2 else None,
                        "reference_output": candidate.get("reference_output") if structured_v2 else None,
                        "retrieval_query": build_rq3_apply_retrieval_query(
                            service_family=str(candidate.get("service_family") or ""),
                            scenario=str(candidate.get("scenario") or ""),
                            task_instruction=str(candidate.get("task_instruction") or ""),
                            output_template=candidate.get("output_template") if structured_v2 else None,
                        ),
                        "answer_scoring_points": [],
                        "gold_memory_evidence_app_log_ids": list(gold_memory_evidence_app_log_ids),
                        "item_validation": {**final_qa_validation, "manual_review_required": True},
                        "scoring_validation": None,
                    }
                )
            else:
                discarded.append(
                    {
                        "qa_id": str(candidate.get("qa_id") or ""),
                        "service_category": str(candidate.get("service_category") or ""),
                        "question": str(candidate.get("question") or candidate.get("apply_question") or ""),
                        "reference_answer": str(candidate.get("reference_answer") or candidate.get("apply_reference_answer") or ""),
                        "rubric": [],
                        "apply_scenario": str(candidate.get("apply_scenario") or ""),
                        "apply_question": str(candidate.get("question") or candidate.get("apply_question") or ""),
                        "apply_reference_answer": str(candidate.get("reference_answer") or candidate.get("apply_reference_answer") or ""),
                        "retrieval_query": build_rq3_apply_retrieval_query(
                            service_category=str(candidate.get("service_category") or ""),
                            question=str(candidate.get("question") or candidate.get("apply_question") or ""),
                            apply_scenario=str(candidate.get("apply_scenario") or ""),
                        ),
                        "answer_scoring_points": [],
                        "gold_memory_evidence_app_log_ids": list(gold_memory_evidence_app_log_ids),
                        "qa_validation": {**final_qa_validation, "manual_review_required": True},
                        "atomic_fact_validation": None,
                    }
                )
            continue

        if task_contract_is_v2(task_contract_version):
            structured_v2 = _task_c_v2_uses_structured_output(str(candidate.get("service_family") or ""))
            # Only after the item itself passes semantic validation do we materialize scoring points.
            if structured_v2:
                answer_scoring_points, scoring_validation = build_task_c_v2_structured_answer_scoring_points(
                    state_key=state_key,
                    state_value=state_value,
                    service_family=str(candidate.get("service_family") or ""),
                    output_template=candidate.get("output_template"),
                    reference_output=candidate.get("reference_output"),
                    prefix=f"aqp_{state_key.replace(':', '_')}_{str(candidate.get('qa_id') or f'q{idx+1}')}",
                )
            else:
                answer_scoring_points, scoring_validation = build_task_c_v2_user_communication_answer_scoring_points(
                    state_key=state_key,
                    state_value=state_value,
                    reference_answer=str(candidate.get("reference_answer") or ""),
                    prefix=f"aqp_{state_key.replace(':', '_')}_{str(candidate.get('qa_id') or f'q{idx+1}')}",
                )
            candidate["retrieval_query"] = build_rq3_apply_retrieval_query(
                service_family=str(candidate.get("service_family") or ""),
                scenario=str(candidate.get("scenario") or ""),
                task_instruction=str(candidate.get("task_instruction") or ""),
                output_template=candidate.get("output_template") if structured_v2 else None,
            )
            scoring_needs_manual_review = not bool(scoring_validation.get("is_valid"))
            accepted_item = {
                "qa_id": str(candidate.get("qa_id") or ""),
                "service_family": str(candidate.get("service_family") or ""),
                "scenario": str(candidate.get("scenario") or ""),
                "task_instruction": str(candidate.get("task_instruction") or ""),
                "retrieval_query": str(candidate.get("retrieval_query") or ""),
                "answer_scoring_points": answer_scoring_points,
                "reference_anchors": copy.deepcopy(candidate.get("reference_anchors"))
                if isinstance(candidate.get("reference_anchors"), list)
                else [],
                "gold_memory_evidence_app_log_ids": list(gold_memory_evidence_app_log_ids),
                "item_validation": {**final_qa_validation, "manual_review_required": False},
                "scoring_validation": {
                    **scoring_validation,
                    "manual_review_required": scoring_needs_manual_review,
                },
            }
            if structured_v2:
                accepted_item["output_template"] = candidate.get("output_template")
                accepted_item["reference_output"] = candidate.get("reference_output")
            else:
                accepted_item["reference_answer"] = str(candidate.get("reference_answer") or "")
            accepted.append(accepted_item)
        else:
            answer_scoring_points, atomic_fact_validation = build_validated_apply_answer_scoring_points(
                state_key=state_key,
                state_value=state_value,
                service_category=str(candidate.get("service_category") or ""),
                apply_scenario=str(candidate.get("apply_scenario") or ""),
                apply_question=str(candidate.get("question") or candidate.get("apply_question") or ""),
                apply_reference_answer=str(candidate.get("reference_answer") or candidate.get("apply_reference_answer") or ""),
                generator_client=generator_client,
                validator_client=validator_client,
                prefix=f"aqp_{state_key.replace(':', '_')}_{str(candidate.get('qa_id') or f'q{idx+1}')}",
                max_rewrites=max_rewrites,
            )
            candidate["retrieval_query"] = build_rq3_apply_retrieval_query(
                service_category=str(candidate.get("service_category") or ""),
                question=str(candidate.get("question") or candidate.get("apply_question") or ""),
                apply_scenario=str(candidate.get("apply_scenario") or ""),
            )
            accepted.append(
                {
                    "qa_id": str(candidate.get("qa_id") or ""),
                    "service_category": str(candidate.get("service_category") or ""),
                    "question": str(candidate.get("question") or candidate.get("apply_question") or ""),
                    "reference_answer": str(candidate.get("reference_answer") or candidate.get("apply_reference_answer") or ""),
                    "rubric": _derive_apply_rubric_alias(answer_scoring_points),
                    "apply_scenario": str(candidate.get("apply_scenario") or ""),
                    "apply_question": str(candidate.get("question") or candidate.get("apply_question") or ""),
                    "apply_reference_answer": str(candidate.get("reference_answer") or candidate.get("apply_reference_answer") or ""),
                    "retrieval_query": str(candidate.get("retrieval_query") or ""),
                    "answer_scoring_points": answer_scoring_points,
                    "gold_memory_evidence_app_log_ids": list(gold_memory_evidence_app_log_ids),
                    "qa_validation": {**final_qa_validation, "manual_review_required": False},
                    "atomic_fact_validation": {**atomic_fact_validation, "manual_review_required": False},
                }
            )

    raw_record: Optional[Dict[str, Any]] = None
    if save_raw:
        raw_record = {
            "key": state_key,
            "prompt": prompt,
            "raw_model_output": generated,
            "accepted_count": len(accepted),
            "discarded_count": len(discarded),
            "discarded_items": discarded,
        }
    return {
        "accepted": accepted,
        "discarded": discarded,
        "raw_record": raw_record,
    }


def build_apply_service_pack_inplace(
    *,
    benchmark: Dict[str, Any],
    generator_client: Any,
    validator_client: Any,
    provider: str,
    model: str,
    validator_provider: str,
    validator_model: str,
    item_count_per_key: int,
    max_rewrites: int,
    save_raw: bool,
    reuse_scope: str = "key_value_signature",
    apply_workers: int = 1,
    save_every_apply_keys: int = 0,
    save_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    _require_validated_benchmark(benchmark)
    task_contract_version = infer_task_contract_version(benchmark)
    apply_pack_version = _apply_pack_version(task_contract_version)
    apply_pack_prompt_version = _apply_pack_prompt_version(task_contract_version)
    effective_item_count_per_key = 1 if task_contract_is_v2(task_contract_version) else int(item_count_per_key)
    reuse_scope = str(reuse_scope or "key_value_signature").strip().lower()
    if reuse_scope not in {"key", "key_value_signature"}:
        raise ValueError("reuse_scope must be one of: key|key_value_signature")
    apply_workers = max(1, int(apply_workers))

    reuse_cache: Dict[str, Dict[str, Any]] = {}
    checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return benchmark

    total_keys, _ = _apply_candidate_counts(benchmark)
    progress = tqdm(
        total=total_keys,
        desc="TCE Stage2 apply",
        unit="key",
        disable=(not show_progress),
    )
    save_every_apply_keys = max(0, int(save_every_apply_keys))
    processed_apply_keys = 0

    try:
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
            checkpoint_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
            checkpoint.pop("rq3_know_apply", None)
            validated_flat = _validated_snapshot_flat(checkpoint)
            key_payload: Dict[str, Any] = {}
            raw_records: List[Dict[str, Any]] = []
            cp_reused = 0
            cp_computed = 0
            cp_accepted_items = 0
            cp_discarded_items = 0
            checkpoint["rq3_apply_service_qa"] = {
                "version": apply_pack_version,
                "scoring_points_version": SCORING_POINTS_VERSION,
                "generator": {
                    "provider": provider,
                    "model": model,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                "validator": {
                    "provider": validator_provider,
                    "model": validator_model,
                    "policy": {
                        "max_rewrites": int(max_rewrites),
                        "rule_and_llm_validation": True,
                    },
                },
                "pair_count_per_key": int(effective_item_count_per_key),
                "reuse_scope": reuse_scope,
                "state_validate_only": False,
                "keys": key_payload,
            }
            if save_raw:
                checkpoint["rq3_apply_service_qa"]["records"] = raw_records

            obs_flat = flatten_snapshot(checkpoint.get("state_observability") or {})
            pending_keys: Dict[str, Tuple[Any, str, str, Dict[str, Any], str]] = {}
            resolved_flat, filtered_keys = _resolve_task_a_candidates_from_flat(
                validated_flat,
                task_contract_version=task_contract_version,
            )
            checkpoint["rq3_apply_service_qa"]["filtered_keys"] = filtered_keys
            for state_key in sorted(validated_flat.keys()):
                if show_progress:
                    progress.set_postfix(checkpoint_id=checkpoint_id, state_key=state_key)
                state_value = resolved_flat.get(state_key)
                if state_key not in resolved_flat:
                    processed_apply_keys += 1
                    if (
                        save_callback is not None
                        and save_every_apply_keys > 0
                        and processed_apply_keys % save_every_apply_keys == 0
                    ):
                        save_callback(benchmark)
                    progress.update(1)
                    continue
                state_sig = value_signature(state_value)
                state_obs = obs_flat.get(state_key) if isinstance(obs_flat.get(state_key), dict) else {}
                evidence_ids = sorted(collect_state_evidence_ids(state_obs))
                evidence_sig = value_signature(evidence_ids)
                cache_key: Optional[str] = None
                if reuse_scope == "key":
                    cache_key = state_key
                elif reuse_scope == "key_value_signature":
                    cache_key = f"{state_key}::{state_sig}::{apply_pack_prompt_version}"

                cached = reuse_cache.get(cache_key or "")
                if cache_key is not None and isinstance(cached, dict):
                    reused_node = _refresh_reused_apply_key_node(
                        cached,
                        state_sig=state_sig,
                        evidence_sig=evidence_sig,
                        evidence_ids=evidence_ids,
                        apply_pack_prompt_version=apply_pack_prompt_version,
                        apply_pack_version=apply_pack_version,
                    )
                    reused_node["pack_source"] = "reused"
                    if task_contract_is_v2(task_contract_version) and not reused_node.get("items"):
                        filtered_keys[state_key] = {
                            "reason_codes": [_REASON_TASK_C_V2_NO_ACCEPTED_ITEMS],
                            "manual_review_required": True,
                            "discarded_count": len(reused_node.get("discarded_items") or []),
                        }
                    else:
                        key_payload[state_key] = reused_node
                    cp_reused += 1
                    cp_accepted_items += len(reused_node.get("items") or [])
                    cp_discarded_items += len(reused_node.get("discarded_items") or [])
                    processed_apply_keys += 1
                    if (
                        save_callback is not None
                        and save_every_apply_keys > 0
                        and processed_apply_keys % save_every_apply_keys == 0
                    ):
                        save_callback(benchmark)
                    progress.update(1)
                    continue
                pending_keys[state_key] = (state_value, state_sig, cache_key or "", state_obs, evidence_sig)

            with ThreadPoolExecutor(max_workers=apply_workers) as executor:
                submitted: Dict[Future, Tuple[str, Any, str, str]] = {}
                for state_key, (state_value, state_sig, cache_key_value, state_obs, evidence_sig) in pending_keys.items():
                    future = executor.submit(
                        _build_apply_key_node,
                        generator_client=generator_client,
                        validator_client=validator_client,
                        checkpoint_ts=checkpoint_ts,
                        state_key=state_key,
                        state_value=state_value,
                        state_observability=state_obs,
                        item_count_per_key=effective_item_count_per_key,
                        max_rewrites=max_rewrites,
                        save_raw=save_raw,
                        task_contract_version=task_contract_version,
                    )
                    submitted[future] = (state_key, state_sig, cache_key_value, evidence_sig)

                for future in as_completed(submitted):
                    state_key, state_sig, cache_key_value, evidence_sig = submitted[future]
                    result = future.result()
                    accepted = list(result.get("accepted") or [])
                    discarded = list(result.get("discarded") or [])
                    identity = {
                        "state_key": state_key,
                        "validated_state_value_signature": state_sig,
                        "evidence_signature": evidence_sig,
                        "prompt_version": apply_pack_prompt_version,
                        "pack_version": apply_pack_version,
                    }
                    key_node = {
                        "pack_source": "computed",
                        "pack_identity": identity,
                        "items": accepted,
                    }
                    if discarded:
                        key_node["discarded_items"] = discarded
                    if task_contract_is_v2(task_contract_version) and not accepted:
                        filtered_keys[state_key] = {
                            "reason_codes": [_REASON_TASK_C_V2_NO_ACCEPTED_ITEMS],
                            "manual_review_required": True,
                            "discarded_count": len(discarded),
                        }
                    else:
                        key_payload[state_key] = key_node
                    if cache_key_value and accepted:
                        reuse_cache[cache_key_value] = copy.deepcopy(key_node)
                    cp_computed += 1
                    cp_accepted_items += len(accepted)
                    cp_discarded_items += len(discarded)
                    raw_record = result.get("raw_record")
                    if save_raw and isinstance(raw_record, dict):
                        raw_records.append(raw_record)
                    processed_apply_keys += 1
                    if (
                        save_callback is not None
                        and save_every_apply_keys > 0
                        and processed_apply_keys % save_every_apply_keys == 0
                    ):
                        save_callback(benchmark)
                    progress.update(1)

            if save_callback is not None:
                save_callback(benchmark)
            if show_progress:
                print(
                    "[Stage2][apply][checkpoint]"
                    f" checkpoint_id={checkpoint_id},"
                    f" candidate_keys={len(validated_flat)},"
                    f" filtered_keys={len(filtered_keys)},"
                    f" reused_keys={cp_reused},"
                    f" computed_keys={cp_computed},"
                    f" accepted_items={cp_accepted_items},"
                    f" discarded_items={cp_discarded_items}"
                )
    finally:
        progress.close()

    return benchmark


def build_task_packs(
    *,
    benchmark: Dict[str, Any],
    tasks: Sequence[str],
    generator_client: Optional[Any] = None,
    validator_client: Optional[Any] = None,
    provider: str = "",
    model: str = "",
    validator_provider: str = "",
    validator_model: str = "",
    item_count_per_key: int = 1,
    max_rewrites: int = 2,
    save_raw: bool = False,
    reuse_scope: str = "key_value_signature",
    max_checkpoints: Optional[int] = None,
    apply_workers: int = 1,
    save_every_apply_keys: int = 0,
    save_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    show_progress: bool = False,
    enable_state_completion_scoring_points: bool = True,
    task_contract_version: str = CURRENT_TASK_CONTRACT_VERSION,
    research_frame_version: str = RESEARCH_FRAME_VERSION_V2,
    canonical_research_doc: str = CANONICAL_RESEARCH_DOC_V2,
) -> Dict[str, Any]:
    _require_validated_benchmark(benchmark)
    apply_contract_metadata(
        benchmark,
        task_contract_version=task_contract_version,
        research_frame_version=research_frame_version,
        canonical_research_doc=canonical_research_doc,
        overwrite=False,
    )
    effective_contract_version = infer_task_contract_version(benchmark)
    normalized = normalize_stage2_tasks(
        tasks,
        task_contract_version=effective_contract_version,
    )
    if max_checkpoints is not None:
        checkpoints = benchmark.get("checkpoints", [])
        if isinstance(checkpoints, list):
            benchmark["checkpoints"] = checkpoints[: max(0, int(max_checkpoints))]
            benchmark["total_checkpoints"] = len(benchmark["checkpoints"])
    if task_contract_is_v2(effective_contract_version):
        for checkpoint in _iter_checkpoint_dicts(benchmark):
            checkpoint.pop("change_tracking_pack", None)

    if show_progress:
        checkpoint_count = len(_iter_checkpoint_dicts(benchmark))
        print(
            "[Stage2][pre]"
            f" checkpoints={checkpoint_count},"
            f" tasks={','.join(sorted(normalized))}"
        )
        if "state_completion" in normalized:
            total_count, checkpoint_counts = _state_completion_counts(benchmark)
            _print_stage2_task_summary(
                task_name="state_completion",
                total_count=total_count,
                checkpoint_counts=checkpoint_counts,
            )
        if "change_tracking" in normalized:
            total_count, checkpoint_counts = _change_tracking_counts(benchmark)
            _print_stage2_task_summary(
                task_name="change_tracking",
                total_count=total_count,
                checkpoint_counts=checkpoint_counts,
            )
        if "apply" in normalized:
            total_count, checkpoint_counts = _apply_candidate_counts(benchmark)
            print(
                f"[Stage2][pre][apply] total_keys={total_count},"
                f" expected_items={total_count * max(1, int(item_count_per_key))}"
            )
            for checkpoint_id, count in checkpoint_counts:
                print(
                    "[Stage2][pre][apply][checkpoint]"
                    f" checkpoint_id={checkpoint_id},"
                    f" candidate_keys={count},"
                    f" expected_items={count * max(1, int(item_count_per_key))}"
                )

    if "state_completion" in normalized:
        build_state_completion_pack_inplace(
            benchmark=benchmark,
            generator_client=generator_client if enable_state_completion_scoring_points else None,
            validator_client=validator_client if enable_state_completion_scoring_points else None,
            show_progress=show_progress,
            enable_scoring_points=enable_state_completion_scoring_points,
        )
    if "change_tracking" in normalized:
        build_change_tracking_pack_inplace(
            benchmark=benchmark,
            generator_client=generator_client,
            validator_client=validator_client,
            show_progress=show_progress,
        )
    if "apply" in normalized:
        if generator_client is None or validator_client is None:
            raise ValueError("apply pack build requires generator_client and validator_client")
        build_apply_service_pack_inplace(
            benchmark=benchmark,
            generator_client=generator_client,
            validator_client=validator_client,
            provider=provider,
            model=model,
            validator_provider=validator_provider,
            validator_model=validator_model,
            item_count_per_key=item_count_per_key,
            max_rewrites=max_rewrites,
            save_raw=save_raw,
            reuse_scope=reuse_scope,
            apply_workers=apply_workers,
            save_every_apply_keys=save_every_apply_keys,
            save_callback=save_callback,
            show_progress=show_progress,
        )
    return benchmark


def build_state_completion_targets_from_pack(
    checkpoint: Dict[str, Any],
) -> Optional[Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    payload = checkpoint.get("state_completion_pack")
    if not isinstance(payload, dict):
        return None
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, dict):
        return None

    target_keys: List[str] = []
    target_value_templates: Dict[str, Any] = {}
    target_key_status: Dict[str, Any] = {}
    prebuilt_task_records: Dict[str, Any] = {}
    for state_key in sorted(keys_obj.keys()):
        item = keys_obj.get(state_key)
        if not isinstance(item, dict):
            continue
        answer_template = item.get("answer_template")
        if answer_template is None:
            continue
        target_keys.append(state_key)
        target_value_templates[state_key] = copy.deepcopy(answer_template)
        target_key_status[state_key] = {
            "observable": True,
            "valid": True,
            "selected": True,
            "template_source": "state_completion_pack",
            "pack_source": str(item.get("pack_source") or ""),
        }
        prebuilt_task_records[state_key] = {
            "item_id": str(item.get("item_id") or ""),
            "question_text": str(item.get("question_text") or ""),
            "retrieval_query": str(item.get("retrieval_query") or ""),
            "pack_identity": copy.deepcopy(item.get("pack_identity") or {}),
        }
    return target_keys, target_value_templates, target_key_status, prebuilt_task_records


def build_state_completion_combined_task_text(
    *,
    checkpoint: Dict[str, Any],
    prebuilt_task_records: Dict[str, Any],
) -> str:
    checkpoint_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
    lines = [
        f"- {state_key}: {str((prebuilt_task_records.get(state_key) or {}).get('question_text') or '').strip()}"
        for state_key in sorted(prebuilt_task_records.keys())
    ]
    joined = "\n".join(line for line in lines if line.strip())
    return f"As of {checkpoint_ts}, answer all validated state-completion items below.\n{joined}".strip()


def build_state_completion_combined_retrieval_query(
    *,
    checkpoint: Dict[str, Any],
    prebuilt_task_records: Dict[str, Any],
) -> str:
    checkpoint_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
    queries = [
        str((prebuilt_task_records.get(state_key) or {}).get("retrieval_query") or "").strip()
        for state_key in sorted(prebuilt_task_records.keys())
    ]
    queries = [query for query in queries if query]
    if not queries:
        return f"As of {checkpoint_ts}, retrieve memory needed for validated state completion."
    return "\n".join(queries)


def build_change_targets_from_pack(
    checkpoint: Dict[str, Any],
) -> Optional[Tuple[List[str], Dict[str, Any], str, Dict[str, Any]]]:
    payload = checkpoint.get("change_tracking_pack")
    if not isinstance(payload, dict):
        return None
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, dict):
        return None
    previous_cutoff_ts = str(payload.get("previous_cutoff_ts") or "")
    changed_keys: List[str] = []
    changed_templates: Dict[str, Any] = {}
    prebuilt_task_records: Dict[str, Any] = {}
    for state_key in sorted(keys_obj.keys()):
        item = keys_obj.get(state_key)
        if not isinstance(item, dict):
            continue
        before_template = copy.deepcopy(item.get("before_template"))
        after_template = copy.deepcopy(item.get("after_template"))
        if before_template is None or after_template is None:
            continue
        changed_keys.append(state_key)
        changed_templates[state_key] = {
            "before": before_template,
            "after": after_template,
            "change_reason": CHANGE_REASON_TEMPLATE,
            "evidence": copy.deepcopy(EVIDENCE_TEMPLATE),
        }
        prebuilt_task_records[state_key] = {
            "item_id": str(item.get("item_id") or ""),
            "question_text": str(item.get("question_text") or ""),
            "retrieval_query": str(item.get("retrieval_query") or ""),
            "pack_identity": copy.deepcopy(item.get("pack_identity") or {}),
        }
    return changed_keys, changed_templates, previous_cutoff_ts, prebuilt_task_records


def extract_pack_keys(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, dict):
        return []
    return sorted(str(key) for key in keys_obj.keys())
