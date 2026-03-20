"""Build a state-centric timeline viewer payload for TCE artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .evaluation import flatten_observability, flatten_snapshot, value_f1

VIEWER_SCHEMA_VERSION = "tce_state_timeline_viewer_v1"
LEGACY_TASK_A_JUDGE_FIELDS = (
    "correctness",
    "completeness",
    "specificity",
    "groundedness",
)
RUBRIC_FIELDS = (
    "core_value_match",
    "constraint_coverage",
    "value_precision",
    "unsupported_inference_control",
)
OPTION_LABEL_PATTERN = re.compile(r"(?i)(?:^|\b)(?:option\s*)?([A-Z])(?:[\)\].:]\s*|\b)")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_viewer_payload(
    benchmark_payload: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    app_logs_payload: Sequence[Mapping[str, Any]],
    *,
    source_files: Optional[Mapping[str, str]] = None,
    compact: bool = False,
) -> Dict[str, Any]:
    benchmark_checkpoints = _checkpoint_map(benchmark_payload.get("checkpoints"))
    prediction_checkpoints = _prediction_checkpoint_map(prediction_payload)
    eval_checkpoints = _checkpoint_map(eval_payload.get("checkpoints"))
    checkpoint_ids = _ordered_checkpoint_ids(benchmark_checkpoints, prediction_checkpoints, eval_checkpoints)
    app_logs_by_id = _app_log_map(app_logs_payload)

    states_index: Dict[str, Dict[str, Any]] = {}
    timeline: Dict[str, Dict[str, Any]] = {}
    referenced_log_ids: Set[str] = set()

    for checkpoint_id in checkpoint_ids:
        benchmark_cp = benchmark_checkpoints.get(checkpoint_id, {})
        prediction_cp = prediction_checkpoints.get(checkpoint_id, {})
        eval_cp = eval_checkpoints.get(checkpoint_id, {})

        expected_flat = flatten_snapshot(benchmark_cp.get("expected_snapshot_state") or {})
        validated_flat = flatten_snapshot(benchmark_cp.get("validated_snapshot_state") or {})
        observability_flat = flatten_observability(benchmark_cp.get("state_observability") or {})

        state_completion_keys = _extract_pack_key_dict(benchmark_cp.get("state_completion_pack"))
        change_tracking_keys = _extract_pack_key_dict(benchmark_cp.get("change_tracking_pack"))
        apply_key_dict = _extract_pack_key_dict(benchmark_cp.get("rq3_apply_service_qa"))

        prediction_snapshot = flatten_snapshot(prediction_cp.get("snapshot_state") or {})
        prediction_evidence = prediction_cp.get("evidence") or {}
        prediction_change = prediction_cp.get("change_analysis") or {}
        prediction_rq3 = prediction_cp.get("rq3_apply_answers") or {}

        task_a_inputs = _build_task_a_input_map(prediction_cp)
        task_b_inputs = _build_task_b_input_map(prediction_cp)
        task_c_inputs = _build_task_c_input_map(prediction_cp)

        task_a_slot_eval = eval_cp.get("snapshot_slot_eval_by_key") or eval_cp.get("snapshot_slot_judgments_by_key") or {}
        task_b_slot_eval = eval_cp.get("change_slot_eval_by_key") or eval_cp.get("change_slot_judgments_by_key") or {}
        task_c_slot_eval = eval_cp.get("rq3_apply_slot_eval_by_item") or eval_cp.get("rq3_apply_answer_slot_judgments_by_item") or {}
        task_a_legacy_judgments = eval_cp.get("llm_judge_judgments") or {}
        task_a_raw_outputs = (
            eval_cp.get("snapshot_slot_judge_raw_output_by_key")
            or eval_cp.get("llm_judge_raw_output")
            or {}
        )
        task_b_legacy_judgments = eval_cp.get("change_llm_judge_judgments") or {}
        task_b_raw_outputs = (
            eval_cp.get("change_slot_judge_raw_output_by_key")
            or eval_cp.get("change_llm_judge_raw_output")
            or {}
        )
        task_c_legacy_judgments = eval_cp.get("rq3_llm_judge_judgments") or {}
        task_a_point_scores = eval_cp.get("snapshot_point_scores_by_key") or {}
        task_a_point_judgments = eval_cp.get("snapshot_point_judgments_by_key") or {}
        task_b_point_scores = eval_cp.get("change_point_scores_by_key") or {}
        task_b_point_judgments = eval_cp.get("change_point_judgments_by_key") or {}
        task_c_point_scores = eval_cp.get("rq3_apply_answer_point_scores_by_item") or {}
        task_c_point_judgments = eval_cp.get("rq3_apply_answer_point_judgments_by_item") or {}
        task_b_groundtruth = eval_cp.get("groundtruth_change") or {}

        state_keys = sorted(
            set(expected_flat)
            | set(validated_flat)
            | set(state_completion_keys)
            | set(change_tracking_keys)
            | set(apply_key_dict)
            | set(prediction_snapshot)
            | set(prediction_change)
            | set(prediction_rq3)
        )

        changed_key_set = set(change_tracking_keys)
        for state_key in state_keys:
            state_meta = states_index.setdefault(
                state_key,
                {
                    "state_key": state_key,
                    "state_category": _state_category(state_key),
                    "display_name": _display_name(state_key),
                    "appears_in_checkpoints": [],
                    "changed_checkpoints": [],
                    "tasks_present": set(),
                },
            )

            exists_in_expected = state_key in expected_flat
            exists_in_validated = state_key in validated_flat
            is_changed = state_key in changed_key_set
            if exists_in_expected or exists_in_validated or state_key in prediction_snapshot:
                state_meta["appears_in_checkpoints"].append(checkpoint_id)
            if is_changed:
                state_meta["changed_checkpoints"].append(checkpoint_id)

            task_a_eval = None
            if exists_in_validated or state_key in prediction_snapshot:
                task_a_eval = _build_task_a_eval(
                    state_key=state_key,
                    expected_value=validated_flat.get(state_key),
                    predicted_value=prediction_snapshot.get(state_key),
                    expected_evidence_ids=_expected_evidence_ids(observability_flat.get(state_key)),
                    predicted_evidence=prediction_evidence.get(state_key),
                    slot_payload=task_a_slot_eval.get(state_key),
                    judge_payload=task_a_legacy_judgments.get(state_key),
                    judge_raw_output=task_a_raw_outputs.get(state_key),
                    point_score=task_a_point_scores.get(state_key),
                    point_judgments=task_a_point_judgments.get(state_key),
                )
            task_b_payload = _build_task_b_payload(
                state_key=state_key,
                predicted_change=prediction_change.get(state_key),
                expected_change=task_b_groundtruth.get(state_key),
                expected_change_evidence=eval_cp.get("groundtruth_change_evidence", {}).get(state_key),
                slot_payload=task_b_slot_eval.get(state_key),
                judge_payload=task_b_legacy_judgments.get(state_key),
                judge_raw_output=task_b_raw_outputs.get(state_key),
                point_scores=task_b_point_scores.get(state_key),
                point_judgments=task_b_point_judgments.get(state_key),
                model_input=task_b_inputs.get(state_key),
                app_logs_by_id=app_logs_by_id,
                compact=compact,
            )

            apply_spec = apply_key_dict.get(state_key) or {}
            task_c_items = _build_task_c_items(
                state_key=state_key,
                apply_spec=apply_spec,
                predicted_payload=prediction_rq3.get(state_key),
                input_records=task_c_inputs,
                legacy_judgments=task_c_legacy_judgments,
                slot_judgments=task_c_slot_eval,
                answer_point_scores_by_item=task_c_point_scores,
                answer_point_judgments_by_item=task_c_point_judgments,
                app_logs_by_id=app_logs_by_id,
                compact=compact,
            )

            model_inputs = {
                "task_a": _build_task_a_model_inputs(
                    state_key=state_key,
                    input_record=task_a_inputs.get(state_key),
                    app_logs_by_id=app_logs_by_id,
                    compact=compact,
                ),
                "task_b": task_b_payload.get("model_inputs"),
                "task_c": [item.get("model_inputs") for item in task_c_items if item.get("model_inputs")],
            }

            for log_id in _collect_log_ids_from_cell(task_a_eval, task_b_payload, task_c_items, model_inputs):
                if log_id in app_logs_by_id:
                    referenced_log_ids.add(log_id)

            if state_key in state_completion_keys or exists_in_validated:
                state_meta["tasks_present"].add("A")
            if is_changed:
                state_meta["tasks_present"].add("B")
            if task_c_items:
                state_meta["tasks_present"].add("C")

            timeline.setdefault(state_key, {})[checkpoint_id] = {
                "exists_in_expected": exists_in_expected,
                "exists_in_validated": exists_in_validated,
                "is_changed": is_changed,
                "presence_status": _presence_status(exists_in_expected, exists_in_validated),
                "expected_state_value": expected_flat.get(state_key),
                "validated_state_value": validated_flat.get(state_key),
                "state_questionability": benchmark_cp.get("state_questionability", {}).get(state_key),
                "predicted_task_a_value": prediction_snapshot.get(state_key),
                "task_a_evidence": _normalize_evidence_records(prediction_evidence.get(state_key)),
                "task_a_eval": task_a_eval,
                "task_b_payload": task_b_payload,
                "task_c_items": task_c_items,
                "model_inputs": model_inputs,
            }

    _fill_absent_cells(timeline, checkpoint_ids)
    _annotate_transitions(timeline, checkpoint_ids)

    states = []
    for state_key in sorted(states_index):
        state_meta = states_index[state_key]
        state_meta["appears_in_checkpoints"] = _dedupe_preserve_order(state_meta["appears_in_checkpoints"])
        state_meta["changed_checkpoints"] = _dedupe_preserve_order(state_meta["changed_checkpoints"])
        state_meta["tasks_present"] = sorted(state_meta["tasks_present"])
        states.append(state_meta)

    app_logs_subset = {
        log_id: _normalize_app_log(app_logs_by_id[log_id], compact=compact)
        for log_id in sorted(referenced_log_ids)
        if log_id in app_logs_by_id
    }

    return {
        "meta": {
            "user_id": _infer_user_id(source_files or {}),
            "checkpoint_ids": checkpoint_ids,
            "source_files": dict(source_files or {}),
            "schema_version": VIEWER_SCHEMA_VERSION,
            "eval_protocol_version": _infer_eval_protocol_version(eval_payload),
            "compact_mode": compact,
        },
        "states": states,
        "timeline": timeline,
        "app_logs_by_id": app_logs_subset,
    }


def write_viewer_payload(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_checkpoint_map(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return {}
    return {
        str(item.get("checkpoint_id")): item
        for item in predictions
        if isinstance(item, dict) and item.get("checkpoint_id")
    }


def _checkpoint_map(items: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("checkpoint_id")): item
        for item in items
        if isinstance(item, dict) and item.get("checkpoint_id")
    }


def _ordered_checkpoint_ids(*maps: Mapping[str, Any]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for mapping in maps:
        for checkpoint_id in mapping.keys():
            if checkpoint_id not in seen:
                seen.add(checkpoint_id)
                ordered.append(checkpoint_id)
    return ordered


def _state_category(state_key: str) -> str:
    return state_key.split(":", 1)[0] if ":" in state_key else "unknown"


def _display_name(state_key: str) -> str:
    tail = state_key.split(":", 1)[1] if ":" in state_key else state_key
    return tail.replace("_", " ")


def _extract_pack_key_dict(pack: Any) -> Dict[str, Any]:
    if not isinstance(pack, dict):
        return {}
    keys = pack.get("keys")
    return keys if isinstance(keys, dict) else {}


def _normalize_evidence_records(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in value:
        app_log_id = ""
        evidence_content = ""
        if isinstance(item, dict):
            app_log_id = str(item.get("app_log_id") or "").strip()
            evidence_content = str(item.get("evidence_content") or "").strip()
        elif item is not None:
            app_log_id = str(item).strip()
        key = (app_log_id, evidence_content)
        if key in seen:
            continue
        if app_log_id or evidence_content:
            seen.add(key)
            out.append({"app_log_id": app_log_id, "evidence_content": evidence_content})
    return out


def _expected_evidence_ids(observability_item: Any) -> List[str]:
    if not isinstance(observability_item, dict):
        return []
    raw = observability_item.get("evidence_app_log_ids")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    last_id = observability_item.get("last_app_log_id")
    if last_id is None:
        return []
    text = str(last_id).strip()
    return [text] if text else []


def _score_evidence_ids(expected_ids: Sequence[str], predicted_records: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    exp = {str(item) for item in expected_ids if str(item).strip()}
    pred = {
        str(item.get("app_log_id") or "").strip()
        for item in predicted_records
        if isinstance(item, Mapping) and str(item.get("app_log_id") or "").strip()
    }
    tp = len(exp & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(exp) if exp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "expected_ids": sorted(exp),
        "predicted_ids": sorted(pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exp == pred,
    }


def _score_evidence_structure(predicted_records: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    records = [item for item in predicted_records if isinstance(item, Mapping)]
    if not records:
        return {
            "app_log_id_nonempty_rate": 0.0,
            "evidence_content_nonempty_rate": 0.0,
            "evidence_content_with_id_rate": 0.0,
        }
    total = len(records)
    with_id = 0
    with_content = 0
    with_both = 0
    for item in records:
        has_id = bool(str(item.get("app_log_id") or "").strip())
        has_content = bool(str(item.get("evidence_content") or "").strip())
        if has_id:
            with_id += 1
        if has_content:
            with_content += 1
        if has_id and has_content:
            with_both += 1
    return {
        "app_log_id_nonempty_rate": with_id / total,
        "evidence_content_nonempty_rate": with_content / total,
        "evidence_content_with_id_rate": with_both / total,
    }


def _normalize_judge_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    rubrics: Dict[str, float] = {}
    for field in RUBRIC_FIELDS:
        value = payload.get(field)
        if isinstance(value, (int, float)):
            rubrics[field] = float(value)
    legacy: Dict[str, float] = {}
    for field in LEGACY_TASK_A_JUDGE_FIELDS:
        value = payload.get(field)
        if isinstance(value, (int, float)):
            legacy[field] = float(value)
    score_1_5 = 0.0
    source = ""
    if rubrics:
        score_1_5 = sum(rubrics.values()) / len(rubrics)
        source = "rubric4"
    elif legacy:
        score_1_5 = sum(legacy.values()) / len(legacy)
        source = "legacy"
    else:
        return None
    return {
        "source": source,
        "rubrics": rubrics,
        "legacy_fields": legacy,
        "reason": "",
        "score_1_5": score_1_5,
        "score_0_1": max(0.0, min(1.0, (score_1_5 - 1.0) / 4.0)),
    }


def _split_merged_slot_judgments(judgments: Sequence[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    slot_context: List[Dict[str, Any]] = []
    normalized_judgments: List[Dict[str, Any]] = []
    for item in judgments:
        if not isinstance(item, Mapping):
            continue
        slot_context.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"analysis", "correct"}
            }
        )
        normalized_judgments.append(
            {
                "point_id": str(item.get("point_id") or ""),
                "analysis": str(item.get("analysis") or ""),
                "correct": bool(item.get("correct")),
            }
        )
    return slot_context, normalized_judgments


def _normalize_slot_eval_payload(slot_payload: Any, *, point_score: Any = None, point_judgments: Any = None) -> Optional[Dict[str, Any]]:
    if isinstance(slot_payload, Mapping):
        if isinstance(slot_payload.get("slot_context"), list):
            return {
                "score_0_1": slot_payload.get("score_0_1"),
                "slot_count": slot_payload.get("slot_count"),
                "slot_context": list(slot_payload.get("slot_context") or []),
                "judgments": list(slot_payload.get("judgments") or []),
            }
        if isinstance(slot_payload.get("judgments"), list):
            slot_context, normalized_judgments = _split_merged_slot_judgments(list(slot_payload.get("judgments") or []))
            return {
                "score_0_1": slot_payload.get("score_0_1"),
                "slot_count": slot_payload.get("slot_count", len(slot_context)),
                "slot_context": slot_context,
                "judgments": normalized_judgments,
            }
    if isinstance(point_judgments, list):
        try:
            point_score_value = float(point_score)
        except Exception:
            point_score_value = None
        slot_context, normalized_judgments = _split_merged_slot_judgments(point_judgments)
        return {
            "score_0_1": point_score_value,
            "slot_count": len(slot_context),
            "slot_context": slot_context,
            "judgments": normalized_judgments,
        }
    return None


def _build_task_a_eval(
    *,
    state_key: str,
    expected_value: Any,
    predicted_value: Any,
    expected_evidence_ids: Sequence[str],
    predicted_evidence: Any,
    slot_payload: Any,
    judge_payload: Any,
    judge_raw_output: Any,
    point_score: Any,
    point_judgments: Any,
) -> Dict[str, Any]:
    predicted_records = _normalize_evidence_records(predicted_evidence)
    judge = _normalize_judge_payload(judge_payload)
    if judge:
        judge["reason"] = _extract_judge_reason(judge_raw_output, state_key)
    slot_eval = _normalize_slot_eval_payload(
        slot_payload,
        point_score=point_score,
        point_judgments=point_judgments,
    )
    return {
        "state_key": state_key,
        "value_f1": value_f1(expected_value, predicted_value),
        "exact_match": expected_value == predicted_value,
        "evidence_id_metrics": _score_evidence_ids(expected_evidence_ids, predicted_records),
        "evidence_structure": _score_evidence_structure(predicted_records),
        "slot_eval": slot_eval,
        "judge": judge,
    }


def _build_task_b_payload(
    *,
    state_key: str,
    predicted_change: Any,
    expected_change: Any,
    expected_change_evidence: Any,
    slot_payload: Any,
    judge_payload: Any,
    judge_raw_output: Any,
    point_scores: Any,
    point_judgments: Any,
    model_input: Any,
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    compact: bool = False,
) -> Dict[str, Any]:
    applicable = isinstance(expected_change, dict) or isinstance(predicted_change, dict)
    predicted_change = predicted_change if isinstance(predicted_change, dict) else {}
    expected_change = expected_change if isinstance(expected_change, dict) else {}
    predicted_evidence = _normalize_evidence_records(predicted_change.get("evidence"))
    expected_evidence_ids = _expected_change_evidence_ids(expected_change_evidence)
    model_inputs = _build_change_model_inputs(model_input, app_logs_by_id, compact=compact)
    judge = _normalize_judge_payload(judge_payload)
    if judge:
        judge["reason"] = _extract_judge_reason(judge_raw_output, state_key)
    slot_eval = None
    if isinstance(slot_payload, Mapping):
        slot_eval = {
            "before": _normalize_slot_eval_payload(slot_payload.get("before")),
            "after": _normalize_slot_eval_payload(slot_payload.get("after")),
            "state_predict": _normalize_slot_eval_payload(slot_payload.get("state_predict")),
            "change_reason": _normalize_slot_eval_payload(slot_payload.get("change_reason")),
        }
    elif isinstance(point_scores, Mapping) or isinstance(point_judgments, Mapping):
        point_eval_scores = dict(point_scores) if isinstance(point_scores, Mapping) else {}
        point_eval_judgments = dict(point_judgments) if isinstance(point_judgments, Mapping) else {}
        slot_eval = {
            "before": _normalize_slot_eval_payload(
                None,
                point_score=point_eval_scores.get("before_point_score"),
                point_judgments=point_eval_judgments.get("before"),
            ),
            "after": _normalize_slot_eval_payload(
                None,
                point_score=point_eval_scores.get("after_point_score"),
                point_judgments=point_eval_judgments.get("after"),
            ),
            "state_predict": _normalize_slot_eval_payload(
                None,
                point_score=point_eval_scores.get("state_predict_score"),
                point_judgments=point_eval_judgments.get("state_predict"),
            ),
            "change_reason": _normalize_slot_eval_payload(
                None,
                point_score=point_eval_scores.get("reason_score"),
                point_judgments=point_eval_judgments.get("change_reason"),
            ),
        }
    return {
        "applicable": applicable,
        "before": predicted_change.get("before"),
        "after": predicted_change.get("after"),
        "change_reason": predicted_change.get("change_reason"),
        "evidence": predicted_evidence,
        "expected_before": expected_change.get("before"),
        "expected_after": expected_change.get("after"),
        "expected_change_reason": expected_change.get("change_reason"),
        "eval": {
            "before_after_f1": _change_pair_f1(expected_change, predicted_change),
            "before_after_exact": (
                expected_change.get("before") == predicted_change.get("before")
                and expected_change.get("after") == predicted_change.get("after")
            ),
            "evidence_id_metrics": _score_evidence_ids(expected_evidence_ids, predicted_evidence),
            "evidence_structure": _score_evidence_structure(predicted_evidence),
            "slot_eval": slot_eval,
            "judge": judge,
        },
        "model_inputs": model_inputs,
    }


def _expected_change_evidence_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _change_pair_f1(expected_change: Mapping[str, Any], predicted_change: Mapping[str, Any]) -> float:
    before = value_f1(expected_change.get("before"), predicted_change.get("before"))
    after = value_f1(expected_change.get("after"), predicted_change.get("after"))
    return (before + after) / 2.0


def _build_task_c_items(
    *,
    state_key: str,
    apply_spec: Any,
    predicted_payload: Any,
    input_records: Mapping[Tuple[str, str], Mapping[str, Any]],
    legacy_judgments: Mapping[str, Any],
    slot_judgments: Mapping[str, Any],
    answer_point_scores_by_item: Mapping[str, Any],
    answer_point_judgments_by_item: Mapping[str, Any],
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    compact: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(apply_spec, dict):
        return []
    expected_items = apply_spec.get("items")
    if not isinstance(expected_items, list):
        return []
    predicted_items = {}
    if isinstance(predicted_payload, dict) and isinstance(predicted_payload.get("items"), list):
        for item in predicted_payload.get("items") or []:
            if isinstance(item, dict) and item.get("qa_id"):
                predicted_items[str(item["qa_id"])] = item

    out: List[Dict[str, Any]] = []
    for expected_item in expected_items:
        if not isinstance(expected_item, dict):
            continue
        qa_id = str(expected_item.get("qa_id") or "")
        predicted_item = predicted_items.get(qa_id, {})
        predicted_evidence = _normalize_evidence_records(predicted_item.get("evidence"))
        reference_answer = expected_item.get("reference_answer", expected_item.get("apply_reference_answer"))
        predicted_answer = predicted_item.get("answer")
        item_id = f"{state_key}::{qa_id}"
        slot_payload = slot_judgments.get(item_id) if isinstance(slot_judgments, Mapping) else None
        point_payload = answer_point_judgments_by_item.get(item_id) if isinstance(answer_point_judgments_by_item, Mapping) else None
        out.append(
            {
                "qa_id": qa_id,
                "service_category": expected_item.get("service_category"),
                "scenario": expected_item.get("apply_scenario"),
                "question": expected_item.get("question", expected_item.get("apply_question")),
                "reference_answer": reference_answer,
                "rubric": expected_item.get("rubric"),
                "answer_scoring_points": expected_item.get("answer_scoring_points"),
                "gold_memory_evidence_app_log_ids": expected_item.get("gold_memory_evidence_app_log_ids"),
                "predicted_answer": predicted_answer,
                "evidence": predicted_evidence,
                "eval": {
                    "slot_eval": _normalize_slot_eval_payload(
                        slot_payload,
                        point_score=(
                            answer_point_scores_by_item.get(item_id)
                            if isinstance(answer_point_scores_by_item, Mapping)
                            else None
                        ),
                        point_judgments=(
                            list(point_payload.get("judgments") or [])
                            if isinstance(point_payload, Mapping)
                            else None
                        ),
                    ),
                    "evidence_id_metrics": _score_evidence_ids(
                        expected_item.get("gold_memory_evidence_app_log_ids") or [],
                        predicted_evidence,
                    ),
                    "evidence_structure": _score_evidence_structure(predicted_evidence),
                },
                "model_inputs": _build_apply_model_inputs(
                    input_records.get((state_key, qa_id)),
                    app_logs_by_id,
                    compact=compact,
                ),
            }
        )
    return out


def _extract_option_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = OPTION_LABEL_PATTERN.search(raw)
    return str(match.group(1)).upper().strip() if match else ""


def _extract_judge_reason(raw_output: Any, state_key: str) -> str:
    if not isinstance(raw_output, Mapping):
        return ""
    judgments = raw_output.get("judgments")
    if not isinstance(judgments, list):
        return ""
    for item in judgments:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("key") or "") == state_key:
            return str(item.get("reason") or "").strip()
    return ""


def _build_task_a_input_map(prediction_checkpoint: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata = prediction_checkpoint.get("metadata") or {}
    target_keys = metadata.get("target_keys") if isinstance(metadata.get("target_keys"), list) else []
    prompts = metadata.get("prompt") if isinstance(metadata.get("prompt"), list) else []
    raw_payload = metadata.get("raw_model_output")
    if isinstance(raw_payload, list):
        raws = raw_payload
    elif isinstance(raw_payload, Mapping):
        raw_records = raw_payload.get("records")
        raws = raw_records if isinstance(raw_records, list) else []
    else:
        raws = []
    retrievals = metadata.get("per_key_retrieval") if isinstance(metadata.get("per_key_retrieval"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for idx, key in enumerate(target_keys):
        state_key = str(key)
        out[state_key] = {
            "prompt": prompts[idx] if idx < len(prompts) else None,
            "raw_model_output": raws[idx] if idx < len(raws) else None,
        }
    for record in raws:
        if not isinstance(record, Mapping):
            continue
        state_key = str(record.get("key") or "").strip()
        if not state_key:
            continue
        node = out.setdefault(state_key, {})
        if node.get("prompt") is None:
            node["prompt"] = record.get("prompt")
        node["raw_model_output"] = record.get("raw_model_output")
    for record in retrievals:
        if not isinstance(record, dict):
            continue
        state_key = str(record.get("key") or "")
        if not state_key:
            continue
        out.setdefault(state_key, {}).update(record)
    return out


def _build_task_b_input_map(prediction_checkpoint: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata = prediction_checkpoint.get("metadata") or {}
    change_reasoning = metadata.get("change_reasoning") or {}
    records = change_reasoning.get("per_key_records") if isinstance(change_reasoning.get("per_key_records"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("key") or "")
        if key:
            out[key] = record
    return out


def _build_task_c_input_map(prediction_checkpoint: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    metadata = prediction_checkpoint.get("metadata") or {}
    rq3_apply = metadata.get("rq3_apply") or {}
    records = rq3_apply.get("records") if isinstance(rq3_apply.get("records"), list) else []
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("key") or "")
        qa_id = str(record.get("qa_id") or "")
        if key and qa_id:
            out[(key, qa_id)] = record
    return out


def _lookup_logs(
    log_ids: Iterable[str],
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    *,
    compact: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in log_ids:
        log_id = str(item or "").strip()
        if not log_id or log_id in seen or log_id not in app_logs_by_id:
            continue
        seen.add(log_id)
        out.append(_normalize_app_log(app_logs_by_id[log_id], compact=compact))
    return out


def _build_task_a_model_inputs(
    *,
    state_key: str,
    input_record: Optional[Mapping[str, Any]],
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    compact: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(input_record, Mapping):
        return None
    context_ids = input_record.get("context_log_ids")
    if not isinstance(context_ids, list):
        context_ids = (
            (input_record.get("retrieval_metadata") or {}).get("retrieved_app_log_ids")
            if isinstance(input_record.get("retrieval_metadata"), dict)
            else []
        )
    return {
        "state_key": state_key,
        "retrieval_query": input_record.get("retrieval_query"),
        "user_memory_log_ids": [str(item) for item in context_ids or []],
        "user_memory_logs": _lookup_logs(context_ids or [], app_logs_by_id, compact=compact),
        "prompt": None if compact else input_record.get("prompt"),
        "raw_model_output": input_record.get("raw_model_output"),
    }


def _build_change_model_inputs(
    input_record: Any,
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    compact: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(input_record, Mapping):
        return None
    retrieval_metadata = input_record.get("retrieval_metadata")
    retrieved_ids = []
    if isinstance(retrieval_metadata, dict):
        raw = retrieval_metadata.get("retrieved_app_log_ids")
        if isinstance(raw, list):
            retrieved_ids = [str(item) for item in raw]
    return {
        "retrieval_query": input_record.get("retrieval_query"),
        "user_memory_log_ids": retrieved_ids,
        "user_memory_logs": _lookup_logs(retrieved_ids, app_logs_by_id, compact=compact),
        "prompt": None if compact else input_record.get("prompt"),
        "raw_model_output": input_record.get("raw_model_output"),
    }


def _build_apply_model_inputs(
    input_record: Any,
    app_logs_by_id: Mapping[str, Mapping[str, Any]],
    compact: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(input_record, Mapping):
        return None
    retrieval_metadata = input_record.get("retrieval_metadata")
    retrieved_ids = []
    if isinstance(retrieval_metadata, dict):
        raw = retrieval_metadata.get("retrieved_app_log_ids")
        if isinstance(raw, list):
            retrieved_ids = [str(item) for item in raw]
    return {
        "retrieval_query": input_record.get("retrieval_query"),
        "user_memory_log_ids": retrieved_ids,
        "user_memory_logs": _lookup_logs(retrieved_ids, app_logs_by_id, compact=compact),
        "prompt": None if compact else input_record.get("prompt"),
        "raw_model_output": input_record.get("raw_model_output"),
    }


def _app_log_map(app_logs_payload: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in app_logs_payload:
        if not isinstance(item, Mapping):
            continue
        log_id = str(item.get("app_log_id") or "").strip()
        if log_id:
            out[log_id] = dict(item)
    return out


def _normalize_app_log(log: Mapping[str, Any], *, compact: bool = False) -> Dict[str, Any]:
    normalized = {
        "app_log_id": log.get("app_log_id"),
        "timestamp": log.get("timestamp"),
        "app_name": log.get("app_name"),
        "api_name": log.get("api_name"),
        "summary_text": _summarize_log(log),
    }
    if not compact:
        normalized["request"] = log.get("request")
        normalized["response"] = log.get("response")
    return normalized


def _summarize_log(log: Mapping[str, Any]) -> str:
    request = log.get("request")
    response = log.get("response")
    candidates = [
        _dig_text(request, "message"),
        _dig_text(request, "content"),
        _dig_text(request, "title"),
        _dig_text(response, "assistant_response", "content"),
        _dig_text(response, "page", "title"),
        _dig_text(response, "page", "content"),
        _dig_text(response, "content"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return _truncate(text.replace("\\n", " "), 240)
    return ""


def _dig_text(value: Any, *path: str) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _collect_log_ids_from_cell(
    task_a_eval: Mapping[str, Any],
    task_b_payload: Mapping[str, Any],
    task_c_items: Sequence[Mapping[str, Any]],
    model_inputs: Mapping[str, Any],
) -> Set[str]:
    ids: Set[str] = set()
    if isinstance(task_a_eval, Mapping):
        for record in task_a_eval.get("evidence_id_metrics", {}).get("expected_ids", []):
            ids.add(str(record))
        for record in task_a_eval.get("evidence_id_metrics", {}).get("predicted_ids", []):
            ids.add(str(record))
    for record in task_b_payload.get("evidence", []):
        log_id = str(record.get("app_log_id") or "").strip()
        if log_id:
            ids.add(log_id)
    for record in task_b_payload.get("eval", {}).get("evidence_id_metrics", {}).get("expected_ids", []):
        ids.add(str(record))
    for item in task_c_items:
        for record in item.get("evidence", []):
            log_id = str(record.get("app_log_id") or "").strip()
            if log_id:
                ids.add(log_id)
        model_input = item.get("model_inputs") or {}
        for log_id in model_input.get("user_memory_log_ids", []):
            ids.add(str(log_id))
    task_a_input = model_inputs.get("task_a") or {}
    for log_id in task_a_input.get("user_memory_log_ids", []):
        ids.add(str(log_id))
    task_b_input = model_inputs.get("task_b") or {}
    for log_id in task_b_input.get("user_memory_log_ids", []):
        ids.add(str(log_id))
    return ids


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _infer_user_id(source_files: Mapping[str, str]) -> str:
    for value in source_files.values():
        text = str(value)
        match = re.search(r"(\d{3}_user_\d{3})", text)
        if match:
            return match.group(1)
    return ""


def _infer_eval_protocol_version(eval_payload: Mapping[str, Any]) -> str:
    for checkpoint in eval_payload.get("checkpoints", []) or []:
        if isinstance(checkpoint, Mapping) and (
            checkpoint.get("snapshot_slot_eval_by_key")
            or checkpoint.get("change_slot_eval_by_key")
            or checkpoint.get("rq3_apply_slot_eval_by_item")
        ):
            return "slot_llm_judge_v2"
        if isinstance(checkpoint, Mapping) and (
            checkpoint.get("snapshot_slot_judgments_by_key")
            or checkpoint.get("change_slot_judgments_by_key")
            or checkpoint.get("rq3_apply_answer_slot_judgments_by_item")
        ):
            return "slot_llm_judge_v1"
        if isinstance(checkpoint, Mapping) and checkpoint.get("rq3_apply_answer_point_judgments_by_item"):
            return "point_based_vnext"
        if isinstance(checkpoint, Mapping) and checkpoint.get("rq3_llm_judge_judgments"):
            return "legacy_task_c_llm_judge"
    return "task_c_deterministic_only"


def _presence_status(exists_in_expected: bool, exists_in_validated: bool) -> str:
    if exists_in_validated:
        return "validated_present"
    if exists_in_expected:
        return "expected_but_filtered"
    return "absent_from_expected"


def _annotate_transitions(timeline: Dict[str, Dict[str, Any]], checkpoint_ids: Sequence[str]) -> None:
    for state_key, per_cp in timeline.items():
        previous: Optional[Dict[str, Any]] = None
        for checkpoint_id in checkpoint_ids:
            cell = per_cp.get(checkpoint_id)
            if cell is None:
                continue
            cell["transition_from_previous"] = _transition_from_previous(previous, cell)
            previous = cell


def _fill_absent_cells(timeline: Dict[str, Dict[str, Any]], checkpoint_ids: Sequence[str]) -> None:
    for state_key, per_cp in timeline.items():
        for checkpoint_id in checkpoint_ids:
            if checkpoint_id in per_cp:
                continue
            per_cp[checkpoint_id] = {
                "exists_in_expected": False,
                "exists_in_validated": False,
                "is_changed": False,
                "presence_status": "absent_from_expected",
                "expected_state_value": None,
                "validated_state_value": None,
                "state_questionability": None,
                "predicted_task_a_value": None,
                "task_a_evidence": [],
                "task_a_eval": {
                    "state_key": state_key,
                    "value_f1": 0.0,
                    "exact_match": False,
                    "evidence_id_metrics": {
                        "expected_ids": [],
                        "predicted_ids": [],
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "exact_match": True,
                    },
                    "evidence_structure": {
                        "app_log_id_nonempty_rate": 0.0,
                        "evidence_content_nonempty_rate": 0.0,
                        "evidence_content_with_id_rate": 0.0,
                    },
                    "judge": None,
                },
                "task_b_payload": {
                    "applicable": False,
                    "before": None,
                    "after": None,
                    "change_reason": None,
                    "evidence": [],
                    "expected_before": None,
                    "expected_after": None,
                    "expected_change_reason": None,
                    "eval": {
                        "before_after_f1": 0.0,
                        "before_after_exact": False,
                        "evidence_id_metrics": {
                            "expected_ids": [],
                            "predicted_ids": [],
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "exact_match": True,
                        },
                        "evidence_structure": {
                            "app_log_id_nonempty_rate": 0.0,
                            "evidence_content_nonempty_rate": 0.0,
                            "evidence_content_with_id_rate": 0.0,
                        },
                        "judge": None,
                    },
                    "model_inputs": None,
                },
                "task_c_items": [],
                "model_inputs": {"task_a": None, "task_b": None, "task_c": []},
            }


def _transition_from_previous(previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]) -> Dict[str, Any]:
    current_expected = bool(current.get("exists_in_expected"))
    current_validated = bool(current.get("exists_in_validated"))
    current_q = current.get("state_questionability") if isinstance(current.get("state_questionability"), Mapping) else {}
    current_reasons = list(current_q.get("reason_codes") or [])

    if previous is None:
        if current_validated:
            return {
                "status": "initial_validated",
                "summary": "Present and validated at the first visible checkpoint for this state.",
                "reason_codes": current_reasons,
            }
        if current_expected:
            return {
                "status": "initial_filtered",
                "summary": "Present in expected snapshot but filtered by validation at the first visible checkpoint.",
                "reason_codes": current_reasons,
            }
        return {
            "status": "initial_absent",
            "summary": "Not present in the expected snapshot at this checkpoint.",
            "reason_codes": [],
        }

    prev_expected = bool(previous.get("exists_in_expected"))
    prev_validated = bool(previous.get("exists_in_validated"))

    if not prev_expected and not current_expected:
        return {
            "status": "still_absent",
            "summary": "Still not present in the expected snapshot.",
            "reason_codes": [],
        }
    if not prev_expected and current_validated:
        return {
            "status": "appeared_and_validated",
            "summary": "Newly appeared in the expected snapshot and passed validation.",
            "reason_codes": current_reasons,
        }
    if not prev_expected and current_expected and not current_validated:
        return {
            "status": "appeared_but_filtered",
            "summary": "Newly appeared in the expected snapshot but was filtered by validation.",
            "reason_codes": current_reasons,
        }
    if prev_expected and not current_expected:
        return {
            "status": "dropped_from_expected",
            "summary": "Was present previously but is no longer in the expected snapshot. This is an upstream timeline/state construction change, not a Stage 1 validation drop.",
            "reason_codes": [],
        }
    if prev_expected and current_expected and prev_validated and current_validated:
        return {
            "status": "still_validated",
            "summary": "Still present and still validated.",
            "reason_codes": current_reasons,
        }
    if prev_expected and current_expected and (not prev_validated) and (not current_validated):
        return {
            "status": "still_filtered",
            "summary": "Still present in the expected snapshot but still filtered by validation.",
            "reason_codes": current_reasons,
        }
    if prev_expected and current_expected and prev_validated and (not current_validated):
        return {
            "status": "lost_validation",
            "summary": "Still present in the expected snapshot, but no longer passes validation at this checkpoint.",
            "reason_codes": current_reasons,
        }
    if prev_expected and current_expected and (not prev_validated) and current_validated:
        return {
            "status": "regained_validation",
            "summary": "Still present in the expected snapshot and now passes validation.",
            "reason_codes": current_reasons,
        }
    return {
        "status": "unknown_transition",
        "summary": "Transition could not be categorized cleanly.",
        "reason_codes": current_reasons,
    }
