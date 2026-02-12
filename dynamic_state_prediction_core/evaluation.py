"""Core evaluation logic shared by dynamic state prediction entrypoints."""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

EXCLUDED_VALUE_FIELDS = {"priority", "schedule_date", "schedule_dates"}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return _safe_div(2 * p * r, p + r) if (p + r) else 0.0


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


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


def _value_f1(expected_value: Any, predicted_value: Any) -> float:
    exp_tokens = _tokenize_text(_value_to_text(expected_value))
    pred_tokens = _tokenize_text(_value_to_text(predicted_value))

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
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        log_id = None
        if isinstance(item, dict):
            candidate = item.get("app_log_id")
            if candidate is not None:
                log_id = str(candidate).strip()
        elif isinstance(item, (str, int, float)):
            log_id = str(item).strip()
        if log_id and log_id not in seen:
            seen.add(log_id)
            out.append(log_id)
    return out


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
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
    return out


def evaluate_checkpoints(
    benchmark: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    *,
    include_internal_payload: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    evaluated = 0

    for checkpoint in benchmark.get("checkpoints", []):
        checkpoint_id = checkpoint.get("checkpoint_id")
        prediction = predictions.get(str(checkpoint_id))
        if prediction is None:
            continue

        exp_snapshot = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
        exp_snapshot = {k: _drop_excluded_fields(v) for k, v in exp_snapshot.items()}
        raw_pred_snapshot = flatten_snapshot(prediction.get("snapshot_state") or {})
        raw_pred_snapshot = {k: _drop_excluded_fields(v) for k, v in raw_pred_snapshot.items()}
        # In dynamic state prediction mode, keys are fixed by benchmark.
        # Align prediction to expected keys and evaluate value quality only.
        pred_snapshot = {k: raw_pred_snapshot.get(k, None) for k in exp_snapshot.keys()}
        exp_evidence = _expected_evidence_by_key(checkpoint, sorted(exp_snapshot.keys()))
        pred_evidence = _predicted_evidence_by_key(prediction, sorted(exp_snapshot.keys()))

        row: Dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            **score_snapshot(exp_snapshot, pred_snapshot),
            **score_evidence(exp_evidence, pred_evidence),
        }
        if include_internal_payload:
            row.update(
                {
                    "_expected_snapshot": exp_snapshot,
                    "_pred_snapshot": pred_snapshot,
                    "_expected_evidence": exp_evidence,
                    "_pred_evidence": pred_evidence,
                }
            )

        rows.append(row)
        evaluated += 1

    return rows, evaluated
