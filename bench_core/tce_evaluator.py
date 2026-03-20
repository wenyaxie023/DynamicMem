
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tce_core.evaluation import (
    evaluate_checkpoints,
    mean_numeric_fields,
    normalize_predictions,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:
            dumped = value.model_dump()
        return _json_safe(dumped)
    return value


def align_predictions_to_benchmark(
    benchmark: Dict[str, Any],
    raw_pred: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Align prediction checkpoint ids to benchmark by metadata.checkpoint_timestamp."""
    report = {
        "total_predictions": 0,
        "aligned_by_timestamp": 0,
        "unmatched_timestamp": 0,
        "kept_original_id_no_timestamp": 0,
    }
    if not isinstance(raw_pred, dict) or "predictions" not in raw_pred:
        return raw_pred, report
    preds = raw_pred.get("predictions")
    if not isinstance(preds, list):
        return raw_pred, report

    cp_by_ts: Dict[str, str] = {}
    for cp in benchmark.get("checkpoints", []):
        if not isinstance(cp, dict):
            continue
        cid = cp.get("checkpoint_id")
        ts = (cp.get("as_of") or {}).get("timestamp")
        if cid and ts and str(ts) not in cp_by_ts:
            cp_by_ts[str(ts)] = str(cid)

    aligned: List[Dict[str, Any]] = []
    report["total_predictions"] = len(preds)
    for item in preds:
        if not isinstance(item, dict):
            aligned.append(item)
            continue
        out = dict(item)
        md = out.get("metadata") or {}
        ts = md.get("checkpoint_timestamp") if isinstance(md, dict) else None
        if ts is not None:
            old = out.get("checkpoint_id")
            md2 = dict(md) if isinstance(md, dict) else {}
            if str(ts) in cp_by_ts:
                new = cp_by_ts[str(ts)]
                out["checkpoint_id"] = new
                md2["original_checkpoint_id"] = old
                md2["alignment_status"] = "aligned_by_timestamp"
                report["aligned_by_timestamp"] += 1
            else:
                out["checkpoint_id"] = f"unmatched_timestamp::{old}"
                md2["original_checkpoint_id"] = old
                md2["alignment_status"] = "unmatched_timestamp"
                report["unmatched_timestamp"] += 1
            out["metadata"] = md2
        else:
            report["kept_original_id_no_timestamp"] += 1
        aligned.append(out)

    out_payload = dict(raw_pred)
    out_payload["predictions"] = aligned
    return out_payload, report


def evaluate_tce_rows(
    benchmark: Dict[str, Any],
    raw_prediction: Any,
    *,
    align_by_timestamp: bool = True,
    include_internal_payload: bool = False,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    aligned = raw_prediction
    align_report = {
        "total_predictions": 0,
        "aligned_by_timestamp": 0,
        "unmatched_timestamp": 0,
        "kept_original_id_no_timestamp": 0,
    }
    if align_by_timestamp:
        aligned, align_report = align_predictions_to_benchmark(benchmark, raw_prediction)

    pred_by_id = normalize_predictions(aligned)
    rows, evaluated = evaluate_checkpoints(
        benchmark,
        pred_by_id,
        include_internal_payload=include_internal_payload,
    )
    return rows, evaluated, align_report


def build_tce_result_payload(
    benchmark: Dict[str, Any],
    checkpoint_rows: List[Dict[str, Any]],
    evaluated: int,
    *,
    save_eyeball: bool = False,
    strip_internal_payload: bool = True,
) -> Dict[str, Any]:
    rows = []
    for row in checkpoint_rows:
        out = dict(row)
        if save_eyeball:
            out["groundtruth_snapshot"] = _json_safe(out.get("_expected_snapshot", {}))
            out["prediction_snapshot"] = _json_safe(out.get("_pred_snapshot", {}))
            out["groundtruth_evidence"] = _json_safe(out.get("_expected_evidence", {}))
            out["prediction_evidence"] = _json_safe(out.get("_pred_evidence", {}))
            out["groundtruth_change"] = _json_safe(out.get("_expected_change", {}))
            out["prediction_change"] = _json_safe(out.get("_pred_change", {}))
            out["groundtruth_change_evidence"] = _json_safe(out.get("_expected_change_evidence", {}))
            out["prediction_change_evidence"] = _json_safe(out.get("_pred_change_evidence", {}))
            out["groundtruth_rq3_apply"] = _json_safe(out.get("_expected_rq3", {}))
            out["prediction_rq3_apply"] = _json_safe(out.get("_pred_rq3", {}))
        if strip_internal_payload:
            out.pop("_expected_snapshot", None)
            out.pop("_pred_snapshot", None)
            out.pop("_expected_evidence", None)
            out.pop("_pred_evidence", None)
            out.pop("_pred_evidence_records", None)
            out.pop("_expected_change", None)
            out.pop("_pred_change", None)
            out.pop("_expected_change_evidence", None)
            out.pop("_pred_change_evidence", None)
            out.pop("_pred_change_evidence_records", None)
            out.pop("_changed_keys", None)
            out.pop("_expected_rq3", None)
            out.pop("_pred_rq3", None)
            out.pop("_snapshot_slots_by_key", None)
            out.pop("_change_slots_by_key", None)
            out.pop("_rq3_apply_slots_by_item", None)
        rows.append(out)

    summary = mean_numeric_fields(
        [{k: v for k, v in row.items() if k != "checkpoint_id"} for row in rows]
    )

    return {
        "user_id": benchmark.get("user_id"),
        "total_checkpoints": benchmark.get("total_checkpoints", len(rows)),
        "evaluated_checkpoints": evaluated,
        "skipped_checkpoints": max(benchmark.get("total_checkpoints", 0) - evaluated, 0),
        "summary": summary,
        "checkpoints": rows,
    }
