"""Core evaluation logic shared by TCE entrypoints."""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .scoring_points import extract_value_at_path
from .task_packs import extract_pack_keys
from tce_contracts import infer_task_contract_version, task_contract_is_v2

EXCLUDED_VALUE_FIELDS = {"priority", "schedule_date", "schedule_dates"}
POINT_TYPE_FIELD = "field"
POINT_TYPE_LIST_ITEM = "list_item"
POINT_TYPE_MICRO = "micro"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return _safe_div(2 * p * r, p + r) if (p + r) else 0.0


def _drop_excluded_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in EXCLUDED_VALUE_FIELDS:
                continue
            out[k] = _drop_excluded_fields(v)
        return out
    if isinstance(value, list):
        return [_drop_excluded_fields(v) for v in value]
    return value


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _collect_leaf_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(_collect_leaf_texts(v))
        return out
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_collect_leaf_texts(item))
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    try:
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    except Exception:
        return [str(value)]


def _value_f1(expected_value: Any, predicted_value: Any) -> float:
    exp_tokens: List[str] = []
    for leaf in _collect_leaf_texts(expected_value):
        exp_tokens.extend(_tokenize_text(leaf))

    pred_tokens: List[str] = []
    for leaf in _collect_leaf_texts(predicted_value):
        pred_tokens.extend(_tokenize_text(leaf))

    if not exp_tokens and not pred_tokens:
        return 1.0
    if not exp_tokens or not pred_tokens:
        return 0.0

    exp_counter = Counter(exp_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((exp_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def value_f1(expected_value: Any, predicted_value: Any) -> float:
    """Public wrapper for value-level token F1 used in TCE evaluation."""
    return _value_f1(expected_value, predicted_value)


def normalize_predictions(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize prediction payload to checkpoint_id -> prediction mapping."""
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if not isinstance(raw, list):
        raise ValueError("Prediction file must be a list or {'predictions': [...]}.")

    by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = item.get("checkpoint_id")
        if cid:
            by_id[str(cid)] = item
    return by_id


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


def flatten_observability(observability: Any) -> Dict[str, Any]:
    if not isinstance(observability, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in observability.keys()):
        return {str(k): v for k, v in observability.items()}

    flat: Dict[str, Any] = {}
    for category, content in observability.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def _extract_evidence_ids(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in _extract_evidence_records(value):
        log_id = str(item.get("app_log_id") or "").strip()
        if log_id and log_id not in seen:
            seen.add(log_id)
            out.append(log_id)
    return out


def _extract_evidence_records(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for item in value:
        log_id = ""
        evidence_content = ""
        if isinstance(item, dict):
            candidate = item.get("app_log_id")
            if candidate is not None:
                log_id = str(candidate).strip()
            evidence_content = str(item.get("evidence_content") or "").strip()
        elif isinstance(item, (str, int, float)):
            log_id = str(item).strip()
        dedupe_key = (log_id, evidence_content)
        if dedupe_key in seen:
            continue
        if log_id or evidence_content:
            seen.add(dedupe_key)
            out.append(
                {
                    "app_log_id": log_id,
                    "evidence_content": evidence_content,
                }
            )
    return out


def _score_evidence_content_structural(records_by_key: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    id_nonempty_rates: List[float] = []
    nonempty_rates: List[float] = []
    paired_rates: List[float] = []
    for key in sorted(records_by_key.keys()):
        records = records_by_key.get(key, [])
        if not records:
            id_nonempty_rates.append(0.0)
            nonempty_rates.append(0.0)
            paired_rates.append(0.0)
            continue
        id_nonempty = 0
        nonempty = 0
        paired = 0
        for record in records:
            content = str(record.get("evidence_content") or "").strip()
            log_id = str(record.get("app_log_id") or "").strip()
            if log_id:
                id_nonempty += 1
            if content:
                nonempty += 1
            if content and log_id:
                paired += 1
        denom = len(records)
        id_nonempty_rates.append(id_nonempty / denom)
        nonempty_rates.append(nonempty / denom)
        paired_rates.append(paired / denom)
    return {
        "evidence_app_log_id_nonempty_rate": sum(id_nonempty_rates) / len(id_nonempty_rates) if id_nonempty_rates else 0.0,
        "evidence_content_nonempty_rate": sum(nonempty_rates) / len(nonempty_rates) if nonempty_rates else 0.0,
        "evidence_content_with_id_rate": sum(paired_rates) / len(paired_rates) if paired_rates else 0.0,
    }
def _expected_evidence_by_key(checkpoint: Dict[str, Any], target_keys: Sequence[str]) -> Dict[str, List[str]]:
    obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
    out: Dict[str, List[str]] = {}
    for key in target_keys:
        obs = obs_flat.get(key)
        if not isinstance(obs, dict):
            out[key] = []
            continue
        ids = _extract_evidence_ids(obs.get("evidence_app_log_ids"))
        if not ids:
            # Backward compatibility for old benchmarks.
            last_id = obs.get("last_app_log_id")
            ids = [str(last_id)] if last_id is not None and str(last_id).strip() else []
        out[key] = ids
    return out


def _predicted_evidence_by_key(prediction: Dict[str, Any], target_keys: Sequence[str]) -> Dict[str, List[str]]:
    raw = prediction.get("evidence")
    out: Dict[str, List[str]] = {}
    if isinstance(raw, dict):
        for key in target_keys:
            out[key] = _extract_evidence_ids(raw.get(key))
    else:
        for key in target_keys:
            out[key] = []
    return out


def _predicted_evidence_records_by_key(
    prediction: Dict[str, Any],
    target_keys: Sequence[str],
) -> Dict[str, List[Dict[str, str]]]:
    raw = prediction.get("evidence")
    out: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(raw, dict):
        for key in target_keys:
            out[key] = _extract_evidence_records(raw.get(key))
    else:
        for key in target_keys:
            out[key] = []
    return out


def _extract_change_payload_by_key(
    payload: Any,
    target_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, dict):
        for key in target_keys:
            out[str(key)] = {
                "before": None,
                "after": None,
                "change_reason": "",
                "evidence": [],
            }
        return out
    for key in target_keys:
        item = payload.get(str(key))
        if not isinstance(item, dict):
            out[str(key)] = {
                "before": None,
                "after": None,
                "change_reason": "",
                "evidence": [],
            }
            continue
        reason_or_attr = item.get("change_reason")
        if reason_or_attr is None:
            reason_or_attr = item.get("reason", "")
        out[str(key)] = {
            "before": item.get("before"),
            "after": item.get("after"),
            "change_reason": str(reason_or_attr or ""),
            "evidence": _extract_evidence_records(item.get("evidence")),
        }
    return out


def _extract_rq3_pack_by_key(
    checkpoint: Dict[str, Any],
    *,
    task_contract_version: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if "rq3_know_apply" in checkpoint:
        raise ValueError("Legacy field rq3_know_apply is no longer supported. Use rq3_apply_service_qa.")
    payload = checkpoint.get("rq3_apply_service_qa")
    if not isinstance(payload, dict):
        return {}
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, dict):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, item in keys_obj.items():
        if not isinstance(item, dict):
            continue
        items = item.get("items")
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for qa_item in items:
            if not isinstance(qa_item, dict):
                continue
            if task_contract_is_v2(task_contract_version):
                normalized.append(
                    {
                        "qa_id": str(qa_item.get("qa_id") or ""),
                        "service_family": str(qa_item.get("service_family") or ""),
                        "scenario": str(qa_item.get("scenario") or ""),
                        "task_instruction": str(qa_item.get("task_instruction") or ""),
                        "reference_answer": str(qa_item.get("reference_answer") or ""),
                        "output_template": qa_item.get("output_template"),
                        "reference_output": qa_item.get("reference_output"),
                        "answer_scoring_points": list(qa_item.get("answer_scoring_points") or []),
                        "gold_memory_evidence_app_log_ids": _extract_evidence_ids(
                            qa_item.get("gold_memory_evidence_app_log_ids")
                        ),
                    }
                )
            else:
                normalized.append(
                    {
                        "qa_id": str(qa_item.get("qa_id") or ""),
                        "service_category": str(qa_item.get("service_category") or ""),
                        "question": str(qa_item.get("question") or qa_item.get("apply_question") or ""),
                        "reference_answer": str(qa_item.get("reference_answer") or qa_item.get("apply_reference_answer") or ""),
                        "rubric": list(qa_item.get("rubric") or []),
                        "apply_scenario": str(qa_item.get("apply_scenario") or ""),
                        "apply_question": str(qa_item.get("question") or qa_item.get("apply_question") or ""),
                        "apply_reference_answer": str(qa_item.get("reference_answer") or qa_item.get("apply_reference_answer") or ""),
                        "answer_scoring_points": list(qa_item.get("answer_scoring_points") or []),
                        "gold_memory_evidence_app_log_ids": _extract_evidence_ids(
                            qa_item.get("gold_memory_evidence_app_log_ids")
                        ),
                    }
                )
        if normalized:
            out[str(key)] = normalized
    return out


def _extract_rq3_apply_answers_by_key(prediction: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if "rq3_answers" in prediction:
        raise ValueError("Legacy field rq3_answers is no longer supported. Use rq3_apply_answers.")
    payload = prediction.get("rq3_apply_answers")
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        items = item.get("items")
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for qa_item in items:
            if not isinstance(qa_item, dict):
                continue
            payload = {
                "qa_id": str(qa_item.get("qa_id") or ""),
                "answer": str(qa_item.get("answer") or ""),
                "output": qa_item.get("output"),
                "evidence": _extract_evidence_records(qa_item.get("evidence")),
                "service_family": str(qa_item.get("service_family") or ""),
            }
            normalized.append(payload)
        if normalized:
            out[str(key)] = normalized
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_list_item_value(predicted_blob: Any, target_path: str, index: int) -> Any:
    container = extract_value_at_path(predicted_blob, target_path)
    if isinstance(container, list) and 0 <= index < len(container):
        return container[index]
    return None


def _materialize_slots(
    points: Sequence[Dict[str, Any]],
    predicted_blob: Any,
    *,
    slot_group: Optional[str] = None,
) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    list_item_occurrence_by_path: Dict[str, int] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        point_id = str(point.get("point_id") or "").strip()
        if not point_id:
            continue
        point_type = str(point.get("point_type") or "").strip().lower()
        target_path = str(point.get("target_path") or "")
        if point_type == POINT_TYPE_FIELD:
            predicted_value = extract_value_at_path(predicted_blob, target_path)
            list_index = None
        elif point_type == POINT_TYPE_LIST_ITEM:
            list_index = list_item_occurrence_by_path.get(target_path, 0)
            list_item_occurrence_by_path[target_path] = list_index + 1
            predicted_value = _extract_list_item_value(predicted_blob, target_path, list_index)
        else:
            predicted_value = extract_value_at_path(predicted_blob, target_path) if target_path else predicted_blob
            list_index = None
        slot: Dict[str, Any] = {
            "point_id": point_id,
            "point_type": point_type,
            "polarity": str(point.get("polarity") or "positive"),
            "predicted_value": predicted_value,
        }
        if point_type in {POINT_TYPE_FIELD, POINT_TYPE_LIST_ITEM}:
            slot["reference_value"] = point.get("reference_value", point.get("point_text"))
        elif point_type == POINT_TYPE_MICRO:
            slot["point_text"] = str(point.get("point_text") or "")
        else:
            if point.get("reference_value") is not None:
                slot["reference_value"] = point.get("reference_value")
            if point.get("point_text"):
                slot["point_text"] = str(point.get("point_text") or "")
        if target_path:
            slot["target_path"] = target_path
        if list_index is not None:
            slot["list_index"] = int(list_index)
        if slot_group:
            slot["slot_group"] = slot_group
        slots.append(slot)
    return slots


def _score_id_metrics(
    expected_by_unit: Dict[str, List[str]],
    predicted_by_unit: Dict[str, List[str]],
    *,
    prefix: str,
    suffix: str = "",
    include_exact: bool = False,
) -> Dict[str, float]:
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    exacts: List[float] = []
    for unit_id in sorted(expected_by_unit.keys()):
        exp = set(expected_by_unit.get(unit_id, []))
        pred = set(predicted_by_unit.get(unit_id, []))
        if not exp:
            recall = 1.0 if not pred else 0.0
        else:
            recall = len(exp & pred) / len(exp)
        if not pred:
            precision = 1.0 if not exp else 0.0
        else:
            precision = len(exp & pred) / len(pred)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        exacts.append(1.0 if exp == pred else 0.0)
    out = {
        f"{prefix}_precision{suffix}": _mean(precisions),
        f"{prefix}_recall{suffix}": _mean(recalls),
        f"{prefix}_f1{suffix}": _mean(f1s),
    }
    if include_exact:
        out[f"{prefix}_exact_match{suffix}"] = _mean(exacts)
    return out


def _retrieved_ids_from_metadata(metadata: Any) -> Optional[List[str]]:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("retrieved_app_log_ids")
    if not isinstance(raw, list):
        return None
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _predicted_snapshot_retrieval_by_key(
    prediction: Dict[str, Any],
    target_keys: Sequence[str],
) -> Tuple[Dict[str, List[str]], bool]:
    out: Dict[str, List[str]] = {str(key): [] for key in target_keys}
    metadata = prediction.get("metadata")
    if not isinstance(metadata, dict):
        return out, False
    records = metadata.get("per_key_retrieval")
    if not isinstance(records, list):
        return out, False
    available = False
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("key") or "").strip()
        if not key or key not in out:
            continue
        ids = _retrieved_ids_from_metadata(record.get("retrieval_metadata"))
        if ids is None:
            continue
        available = True
        out[key] = ids
    return out, available


def _predicted_change_retrieval_by_key(
    prediction: Dict[str, Any],
    target_keys: Sequence[str],
) -> Tuple[Dict[str, List[str]], bool]:
    out: Dict[str, List[str]] = {str(key): [] for key in target_keys}
    metadata = prediction.get("metadata")
    if not isinstance(metadata, dict):
        return out, False
    change_reasoning = metadata.get("change_reasoning")
    if not isinstance(change_reasoning, dict):
        return out, False
    records = change_reasoning.get("per_key_records")
    if not isinstance(records, list):
        return out, False
    available = False
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("key") or "").strip()
        if not key or key not in out:
            continue
        ids = _retrieved_ids_from_metadata(record.get("retrieval_metadata"))
        if ids is None:
            continue
        available = True
        out[key] = ids
    return out, available


def _predicted_apply_retrieval_by_item(
    prediction: Dict[str, Any],
    expected_item_ids: Sequence[str],
) -> Tuple[Dict[str, List[str]], bool]:
    out: Dict[str, List[str]] = {str(item_id): [] for item_id in expected_item_ids}
    metadata = prediction.get("metadata")
    if not isinstance(metadata, dict):
        return out, False
    rq3_apply = metadata.get("rq3_apply")
    if not isinstance(rq3_apply, dict):
        return out, False
    records = rq3_apply.get("records")
    if not isinstance(records, list):
        return out, False
    available = False
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("key") or "").strip()
        qa_id = str(record.get("qa_id") or "").strip()
        if not key or not qa_id:
            continue
        item_id = f"{key}::{qa_id}"
        if item_id not in out:
            continue
        ids = _retrieved_ids_from_metadata(record.get("retrieval_metadata"))
        if ids is None:
            continue
        available = True
        out[item_id] = ids
    return out, available


def score_evidence(expected_by_key: Dict[str, List[str]], predicted_by_key: Dict[str, List[str]]) -> Dict[str, float]:
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    exacts: List[float] = []

    for key in sorted(expected_by_key.keys()):
        exp = set(expected_by_key.get(key, []))
        pred = set(predicted_by_key.get(key, []))

        if len(exp) == 0:
            recall = 1.0 if len(pred) == 0 else 0.0
        else:
            recall = len(exp & pred) / len(exp)

        if len(pred) == 0:
            precision = 1.0 if len(exp) == 0 else 0.0
        else:
            precision = len(exp & pred) / len(pred)

        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        exact = 1.0 if exp == pred else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        exacts.append(exact)

    return {
        "snapshot_evidence_recall_mean_on_expected": sum(recalls) / len(recalls) if recalls else 0.0,
        "snapshot_evidence_precision_mean_on_expected": sum(precisions) / len(precisions) if precisions else 0.0,
        "snapshot_evidence_f1_mean_on_expected": sum(f1s) / len(f1s) if f1s else 0.0,
        "snapshot_evidence_exact_match_mean_on_expected": sum(exacts) / len(exacts) if exacts else 0.0,
    }


def score_change_evidence(expected_by_key: Dict[str, List[str]], predicted_by_key: Dict[str, List[str]]) -> Dict[str, float]:
    base = score_evidence(expected_by_key, predicted_by_key)
    return {
        "change_evidence_recall_mean_on_changed": base.get("snapshot_evidence_recall_mean_on_expected", 0.0),
        "change_evidence_precision_mean_on_changed": base.get("snapshot_evidence_precision_mean_on_expected", 0.0),
        "change_evidence_f1_mean_on_changed": base.get("snapshot_evidence_f1_mean_on_expected", 0.0),
    }


def score_evidence_content(records_by_key: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    base = _score_evidence_content_structural(records_by_key)
    return {
        "snapshot_evidence_app_log_id_nonempty_rate_mean_on_expected": base.get("evidence_app_log_id_nonempty_rate", 0.0),
        "snapshot_evidence_content_nonempty_rate_mean_on_expected": base.get("evidence_content_nonempty_rate", 0.0),
        "snapshot_evidence_content_with_id_rate_mean_on_expected": base.get("evidence_content_with_id_rate", 0.0),
    }


def score_change_evidence_content(records_by_key: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    base = _score_evidence_content_structural(records_by_key)
    return {
        "change_evidence_app_log_id_nonempty_rate_mean_on_changed": base.get("evidence_app_log_id_nonempty_rate", 0.0),
        "change_evidence_content_nonempty_rate_mean_on_changed": base.get("evidence_content_nonempty_rate", 0.0),
        "change_evidence_content_with_id_rate_mean_on_changed": base.get("evidence_content_with_id_rate", 0.0),
    }


def score_snapshot(expected: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, float]:
    exp_keys = set(expected.keys())
    pred_keys = set(predicted.keys())
    tp = len(exp_keys & pred_keys)
    fp = len(pred_keys - exp_keys)
    fn = len(exp_keys - pred_keys)

    value_correct = sum(1 for k in exp_keys & pred_keys if expected[k] == predicted[k])
    per_key_f1: List[float] = []
    for key in sorted(exp_keys):
        per_key_f1.append(_value_f1(expected.get(key), predicted.get(key)))

    value_f1_mean = sum(per_key_f1) / len(per_key_f1) if per_key_f1 else 0.0
    return {
        "snapshot_key_precision": _safe_div(tp, tp + fp),
        "snapshot_key_recall": _safe_div(tp, tp + fn),
        "snapshot_key_f1": _f1(tp, fp, fn),
        "snapshot_value_accuracy_on_expected": _safe_div(value_correct, len(exp_keys)),
        "snapshot_value_f1_mean_on_expected": value_f1_mean,
        "snapshot_exact_match": 1.0 if expected == predicted else 0.0,
    }


def mean_numeric_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals: List[float] = []
        for row in rows:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                vals.append(float(value))
                continue
            try:
                vals.append(float(value))
            except Exception:
                continue
        if vals:
            out[f"{key}_mean"] = sum(vals) / len(vals)
    return out


def evaluate_checkpoints(
    benchmark: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    *,
    include_internal_payload: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    benchmark_contract_version = infer_task_contract_version(benchmark)
    rows: List[Dict[str, Any]] = []
    evaluated = 0
    prev_exp_snapshot: Optional[Dict[str, Any]] = None
    prev_validated_snapshot: Optional[Dict[str, Any]] = None

    for checkpoint in benchmark.get("checkpoints", []):
        checkpoint_id = checkpoint.get("checkpoint_id")
        prediction = predictions.get(str(checkpoint_id))
        current_raw_snapshot = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
        current_raw_snapshot = {k: _drop_excluded_fields(v) for k, v in current_raw_snapshot.items()}
        current_validated_snapshot = flatten_snapshot(checkpoint.get("validated_snapshot_state") or {})
        if prediction is None:
            prev_exp_snapshot = current_raw_snapshot
            prev_validated_snapshot = current_validated_snapshot
            continue

        state_completion_pack = checkpoint.get("state_completion_pack")
        state_completion_pack_keys = (
            state_completion_pack.get("keys") if isinstance(state_completion_pack, dict) else {}
        )
        has_state_completion_pack = isinstance(state_completion_pack, dict) and isinstance(
            state_completion_pack_keys,
            dict,
        )
        state_completion_keys = extract_pack_keys(state_completion_pack)
        if has_state_completion_pack:
            exp_snapshot = {k: current_validated_snapshot.get(k) for k in state_completion_keys}
        else:
            exp_snapshot = current_raw_snapshot

        raw_pred_snapshot = flatten_snapshot(prediction.get("snapshot_state") or {})
        if has_state_completion_pack:
            pred_snapshot = {k: raw_pred_snapshot.get(k, None) for k in exp_snapshot.keys()}
        else:
            raw_pred_snapshot = {k: _drop_excluded_fields(v) for k, v in raw_pred_snapshot.items()}
            pred_snapshot = {k: raw_pred_snapshot.get(k, None) for k in exp_snapshot.keys()}

        exp_evidence = _expected_evidence_by_key(checkpoint, sorted(exp_snapshot.keys()))
        pred_evidence = _predicted_evidence_by_key(prediction, sorted(exp_snapshot.keys()))
        pred_evidence_records = _predicted_evidence_records_by_key(prediction, sorted(exp_snapshot.keys()))

        snapshot_slots_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for key in sorted(exp_snapshot.keys()):
            pack_item = state_completion_pack_keys.get(key) if isinstance(state_completion_pack_keys, dict) else None
            points = list((pack_item or {}).get("scoring_points") or []) if isinstance(pack_item, dict) else []
            if not points:
                continue
            slots = _materialize_slots(points, pred_snapshot.get(key))
            if slots:
                snapshot_slots_by_key[key] = slots

        snapshot_retrieval_by_key, snapshot_retrieval_available = _predicted_snapshot_retrieval_by_key(
            prediction,
            sorted(exp_snapshot.keys()),
        )

        changed_keys: List[str] = []
        expected_change: Dict[str, Dict[str, Any]] = {}
        expected_change_evidence: Dict[str, List[str]] = {}
        change_tracking_pack = checkpoint.get("change_tracking_pack")
        change_pack_keys_obj = (
            change_tracking_pack.get("keys") if isinstance(change_tracking_pack, dict) else {}
        )
        has_change_tracking_pack = isinstance(change_tracking_pack, dict) and isinstance(
            change_pack_keys_obj,
            dict,
        )
        change_pack_keys = extract_pack_keys(change_tracking_pack)
        if has_change_tracking_pack and prev_validated_snapshot is not None:
            changed_keys = list(change_pack_keys)
            for key in changed_keys:
                pack_item = change_pack_keys_obj.get(key) if isinstance(change_pack_keys_obj, dict) else {}
                expected_change[key] = {
                    "before": prev_validated_snapshot.get(key),
                    "after": current_validated_snapshot.get(key),
                    "change_reason": str((pack_item or {}).get("reference_change_reason") or ""),
                }
                expected_change_evidence[key] = exp_evidence.get(key, [])
        else:
            obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
            if prev_exp_snapshot is not None:
                for key in sorted(exp_snapshot.keys()):
                    obs = obs_flat.get(key)
                    is_observable_and_valid = bool(isinstance(obs, dict) and obs.get("is_valid"))
                    if not is_observable_and_valid:
                        continue
                    before_v = prev_exp_snapshot.get(key)
                    after_v = exp_snapshot.get(key)
                    if before_v != after_v:
                        changed_keys.append(key)
                        expected_change[key] = {
                            "before": before_v,
                            "after": after_v,
                            "change_reason": "",
                        }
                        expected_change_evidence[key] = exp_evidence.get(key, [])

        pred_change_raw = prediction.get("change_analysis") or {}
        pred_change = _extract_change_payload_by_key(pred_change_raw, changed_keys)
        pred_change_evidence = {
            k: _extract_evidence_ids((pred_change.get(k, {}) or {}).get("evidence"))
            for k in changed_keys
        }
        pred_change_evidence_records = {
            k: list((pred_change.get(k, {}) or {}).get("evidence", []))
            for k in changed_keys
        }
        change_retrieval_by_key, change_retrieval_available = _predicted_change_retrieval_by_key(
            prediction,
            changed_keys,
        )

        before_exacts: List[float] = []
        after_exacts: List[float] = []
        before_f1s: List[float] = []
        after_f1s: List[float] = []
        legacy_reason_nonempty: List[float] = []
        change_slots_by_key: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for key in changed_keys:
            exp_item = expected_change.get(key, {})
            pred_item = pred_change.get(key, {})
            exp_before = exp_item.get("before")
            exp_after = exp_item.get("after")
            pred_before = pred_item.get("before")
            pred_after = pred_item.get("after")
            before_exacts.append(1.0 if exp_before == pred_before else 0.0)
            after_exacts.append(1.0 if exp_after == pred_after else 0.0)
            before_f1s.append(_value_f1(exp_before, pred_before))
            after_f1s.append(_value_f1(exp_after, pred_after))
            reason_text = str(pred_item.get("change_reason", "") or "").strip()
            legacy_reason_nonempty.append(1.0 if reason_text else 0.0)

            pack_item = change_pack_keys_obj.get(key) if isinstance(change_pack_keys_obj, dict) else None
            before_points = list((pack_item or {}).get("before_scoring_points") or []) if isinstance(pack_item, dict) else []
            after_points = list((pack_item or {}).get("after_scoring_points") or []) if isinstance(pack_item, dict) else []
            reason_points = list((pack_item or {}).get("change_reason_scoring_points") or []) if isinstance(pack_item, dict) else []

            if before_points:
                change_slots_by_key.setdefault(key, {})["before"] = _materialize_slots(
                    before_points,
                    pred_before,
                    slot_group="before",
                )
            if after_points:
                change_slots_by_key.setdefault(key, {})["after"] = _materialize_slots(
                    after_points,
                    pred_after,
                    slot_group="after",
                )
            if reason_points:
                change_slots_by_key.setdefault(key, {})["change_reason"] = _materialize_slots(
                    reason_points,
                    reason_text,
                    slot_group="change_reason",
                )

        change_before_after_correctness = (
            (sum(before_exacts) + sum(after_exacts)) / (2 * len(changed_keys))
            if changed_keys
            else 0.0
        )
        change_before_after_f1 = (
            (sum(before_f1s) + sum(after_f1s)) / (2 * len(changed_keys))
            if changed_keys
            else 0.0
        )
        change_evidence_scores = score_change_evidence(expected_change_evidence, pred_change_evidence)
        change_evidence_content_scores = score_change_evidence_content(pred_change_evidence_records)
        snapshot_evidence_content_scores = score_evidence_content(pred_evidence_records)

        expected_rq3 = _extract_rq3_pack_by_key(
            checkpoint,
            task_contract_version=benchmark_contract_version,
        )
        pred_rq3 = _extract_rq3_apply_answers_by_key(prediction)
        rq3_item_count = sum(len(v) for v in expected_rq3.values())
        rq3_expected_keys = set(expected_rq3.keys())
        rq3_answered_keys = {k for k in rq3_expected_keys if k in pred_rq3 and isinstance(pred_rq3.get(k), list)}
        rq3_key_coverage = (len(rq3_answered_keys) / len(rq3_expected_keys)) if rq3_expected_keys else 0.0
        rq3_slots_by_item: Dict[str, Dict[str, Any]] = {}
        rq3_expected_evidence_by_item: Dict[str, List[str]] = {}
        rq3_predicted_evidence_by_item: Dict[str, List[str]] = {}
        rq3_evidence_records_by_item: Dict[str, List[Dict[str, str]]] = {}
        rq3_expected_item_ids: List[str] = []
        for key in sorted(rq3_expected_keys):
            pred_by_id = {
                str(item.get("qa_id") or ""): item
                for item in pred_rq3.get(key, [])
                if isinstance(item, dict)
            }
            for exp_item in expected_rq3.get(key, []):
                if not isinstance(exp_item, dict):
                    continue
                qa_id = str(exp_item.get("qa_id") or "")
                item_id = f"{key}::{qa_id}"
                rq3_expected_item_ids.append(item_id)
                pred_item = pred_by_id.get(qa_id, {})
                predicted_evidence_records = list((pred_item or {}).get("evidence") or [])
                rq3_expected_evidence_by_item[item_id] = _extract_evidence_ids(
                    exp_item.get("gold_memory_evidence_app_log_ids")
                )
                rq3_predicted_evidence_by_item[item_id] = _extract_evidence_ids(predicted_evidence_records)
                rq3_evidence_records_by_item[item_id] = predicted_evidence_records

                answer_scoring_points = exp_item.get("answer_scoring_points")
                if not isinstance(answer_scoring_points, list) or not answer_scoring_points:
                    raise ValueError(
                        "Task C pack item is missing non-empty answer_scoring_points[]. "
                        "Current protocol does not allow option-style fallback during evaluation "
                        f"(checkpoint_id={checkpoint_id}, state_key={key}, qa_id={qa_id})."
                    )
                predicted_blob = (
                    str((pred_item or {}).get("answer") or "")
                    if (
                        task_contract_is_v2(benchmark_contract_version)
                        and str(exp_item.get("service_family") or "") == "user_communication"
                    )
                    else (
                        (pred_item or {}).get("output")
                        if task_contract_is_v2(benchmark_contract_version)
                        else str((pred_item or {}).get("answer") or "")
                    )
                )
                rq3_slots_by_item[item_id] = {
                    "state_key": key,
                    "qa_id": qa_id,
                    "service_family": str(exp_item.get("service_family") or ""),
                    "scenario": str(exp_item.get("scenario") or exp_item.get("apply_scenario") or ""),
                    "task_instruction": str(exp_item.get("task_instruction") or exp_item.get("question") or exp_item.get("apply_question") or ""),
                    "reference_answer": str(
                        exp_item.get("reference_answer") or exp_item.get("apply_reference_answer") or ""
                    ),
                    "reference_output": exp_item.get("reference_output"),
                    "predicted_answer": str((pred_item or {}).get("answer") or ""),
                    "predicted_output": (pred_item or {}).get("output"),
                    "slots": _materialize_slots(
                        answer_scoring_points,
                        predicted_blob,
                    ),
                }

        rq3_content_scores = _score_evidence_content_structural(rq3_evidence_records_by_item)
        rq3_apply_evidence_scores = _score_id_metrics(
            rq3_expected_evidence_by_item,
            rq3_predicted_evidence_by_item,
            prefix="rq3_apply_evidence",
        )
        rq3_apply_retrieval_by_item, rq3_apply_retrieval_available = _predicted_apply_retrieval_by_item(
            prediction,
            rq3_expected_item_ids,
        )

        row: Dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            **score_snapshot(exp_snapshot, pred_snapshot),
            **score_evidence(exp_evidence, pred_evidence),
            **snapshot_evidence_content_scores,
            "snapshot_retrieval_metrics_available": snapshot_retrieval_available,
            "change_item_count": float(len(changed_keys)),
            "change_before_after_correctness_mean_on_changed": change_before_after_correctness,
            "change_before_after_f1_mean_on_changed": change_before_after_f1,
            **change_evidence_scores,
            **change_evidence_content_scores,
            "change_retrieval_metrics_available": change_retrieval_available,
            "rq3_apply_item_count": float(rq3_item_count),
            "rq3_apply_key_coverage": float(rq3_key_coverage),
            **rq3_apply_evidence_scores,
            "rq3_apply_evidence_app_log_id_nonempty_rate": rq3_content_scores.get("evidence_app_log_id_nonempty_rate", 0.0),
            "rq3_apply_evidence_content_nonempty_rate": rq3_content_scores.get("evidence_content_nonempty_rate", 0.0),
            "rq3_apply_evidence_content_with_id_rate": rq3_content_scores.get("evidence_content_with_id_rate", 0.0),
            "rq3_apply_retrieval_metrics_available": rq3_apply_retrieval_available,
        }
        if snapshot_retrieval_available:
            row.update(
                _score_id_metrics(
                    exp_evidence,
                    snapshot_retrieval_by_key,
                    prefix="snapshot_retrieval",
                    suffix="_mean_on_expected",
                )
            )
        if not any(
            isinstance(groups, dict) and groups.get("change_reason")
            for groups in change_slots_by_key.values()
        ):
            row["change_reason_mean_on_changed"] = _mean(legacy_reason_nonempty)
        if change_retrieval_available:
            row.update(
                _score_id_metrics(
                    expected_change_evidence,
                    change_retrieval_by_key,
                    prefix="change_retrieval",
                    suffix="_mean_on_changed",
                )
            )
        if rq3_apply_retrieval_available:
            row.update(
                _score_id_metrics(
                    rq3_expected_evidence_by_item,
                    rq3_apply_retrieval_by_item,
                    prefix="rq3_apply_retrieval",
                )
            )

        if include_internal_payload:
            row.update(
                {
                    "_expected_snapshot": exp_snapshot,
                    "_pred_snapshot": pred_snapshot,
                    "_expected_evidence": exp_evidence,
                    "_pred_evidence": pred_evidence,
                    "_pred_evidence_records": pred_evidence_records,
                    "_expected_change": expected_change,
                    "_pred_change": pred_change,
                    "_expected_change_evidence": expected_change_evidence,
                    "_pred_change_evidence": pred_change_evidence,
                    "_pred_change_evidence_records": pred_change_evidence_records,
                    "_changed_keys": changed_keys,
                    "_expected_rq3": expected_rq3,
                    "_pred_rq3": pred_rq3,
                    "_snapshot_slots_by_key": snapshot_slots_by_key,
                    "_change_slots_by_key": change_slots_by_key,
                    "_rq3_apply_slots_by_item": rq3_slots_by_item,
                }
            )

        rows.append(row)
        evaluated += 1
        prev_exp_snapshot = current_raw_snapshot
        prev_validated_snapshot = current_validated_snapshot

    return rows, evaluated
