"""Core evaluation logic shared by state abstraction entrypoints."""

from typing import Any, Dict, List, Sequence, Set, Tuple


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return _safe_div(2 * p * r, p + r) if (p + r) else 0.0


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


def flatten_delta(delta: Any) -> Dict[str, Any]:
    if not isinstance(delta, dict):
        return {"added": {}, "updated": {}, "removed": []}

    added_raw = delta.get("added") or {}
    updated_raw = delta.get("updated") or {}
    removed_raw = delta.get("removed") or []

    added: Dict[str, Any] = {}
    if isinstance(added_raw, dict):
        for key, value in added_raw.items():
            if isinstance(key, str) and ":" in key:
                added[key] = value
            elif isinstance(value, dict):
                for name, nested in value.items():
                    added[f"{key}:{name}"] = nested

    updated: Dict[str, Any] = {}
    if isinstance(updated_raw, dict):
        for key, value in updated_raw.items():
            if isinstance(key, str) and ":" in key:
                updated[key] = value
            elif isinstance(value, dict):
                for name, nested in value.items():
                    updated[f"{key}:{name}"] = nested

    removed: List[str] = []
    if isinstance(removed_raw, list):
        removed = [str(x) for x in removed_raw]
    elif isinstance(removed_raw, dict):
        for category, names in removed_raw.items():
            if isinstance(names, list):
                for name in names:
                    removed.append(f"{category}:{name}")

    return {"added": added, "updated": updated, "removed": removed}


def score_snapshot(expected: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, float]:
    exp_keys = set(expected.keys())
    pred_keys = set(predicted.keys())
    tp = len(exp_keys & pred_keys)
    fp = len(pred_keys - exp_keys)
    fn = len(exp_keys - pred_keys)

    value_correct = sum(1 for k in exp_keys & pred_keys if expected[k] == predicted[k])
    return {
        "snapshot_key_precision": _safe_div(tp, tp + fp),
        "snapshot_key_recall": _safe_div(tp, tp + fn),
        "snapshot_key_f1": _f1(tp, fp, fn),
        "snapshot_value_accuracy_on_expected": _safe_div(value_correct, len(exp_keys)),
        "snapshot_exact_match": 1.0 if expected == predicted else 0.0,
    }


def _delta_to_ops(delta: Dict[str, Any]) -> Set[Tuple[str, str]]:
    ops: Set[Tuple[str, str]] = set()
    for key in (delta.get("added") or {}).keys():
        ops.add(("add", str(key)))
    for key in (delta.get("updated") or {}).keys():
        ops.add(("update", str(key)))
    for key in (delta.get("removed") or []):
        ops.add(("remove", str(key)))
    return ops


def score_delta(expected: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, float]:
    exp_ops = _delta_to_ops(expected)
    pred_ops = _delta_to_ops(predicted)
    tp = len(exp_ops & pred_ops)
    fp = len(pred_ops - exp_ops)
    fn = len(exp_ops - pred_ops)

    return {
        "delta_op_precision": _safe_div(tp, tp + fp),
        "delta_op_recall": _safe_div(tp, tp + fn),
        "delta_op_f1": _f1(tp, fp, fn),
        "delta_exact_match": 1.0 if expected == predicted else 0.0,
    }


def score_uncertainty(
    expected_snapshot: Dict[str, Any],
    predicted_snapshot: Dict[str, Any],
    uncertainty: Any,
) -> Dict[str, float]:
    pairs: List[Tuple[float, float]] = []
    if isinstance(uncertainty, dict):
        for key, value in uncertainty.items():
            if key not in predicted_snapshot:
                continue
            try:
                prob_error = float(value)
            except Exception:
                continue
            prob_error = min(1.0, max(0.0, prob_error))
            is_error = (
                0.0
                if (key in expected_snapshot and expected_snapshot[key] == predicted_snapshot[key])
                else 1.0
            )
            pairs.append((prob_error, is_error))

    if not pairs:
        return {"uncertainty_coverage": 0.0, "uncertainty_brier_error": 0.0}

    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    coverage = len(pairs) / max(len(predicted_snapshot), 1)
    return {"uncertainty_coverage": coverage, "uncertainty_brier_error": brier}


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
        exp_delta = flatten_delta(checkpoint.get("expected_delta_from_previous") or {})
        pred_snapshot = flatten_snapshot(prediction.get("snapshot_state") or {})
        pred_delta = flatten_delta(prediction.get("delta_prediction") or {})
        pred_uncertainty = prediction.get("uncertainty") or {}

        row: Dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            **score_snapshot(exp_snapshot, pred_snapshot),
            **score_delta(exp_delta, pred_delta),
            **score_uncertainty(exp_snapshot, pred_snapshot, pred_uncertainty),
        }
        if include_internal_payload:
            row.update(
                {
                    "_expected_snapshot": exp_snapshot,
                    "_expected_delta": exp_delta,
                    "_pred_snapshot": pred_snapshot,
                    "_pred_delta": pred_delta,
                }
            )

        rows.append(row)
        evaluated += 1

    return rows, evaluated
