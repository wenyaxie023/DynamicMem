#!/usr/bin/env python3
"""Build a reusable analysis pack for milestone-style TCE reporting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    mcolors = None  # type: ignore[assignment]

from bench_core.tce_evaluator import align_predictions_to_benchmark
from tce_core.evaluation import flatten_observability, normalize_predictions


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_evidence_records(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for item in value:
        log_id = ""
        evidence_content = ""
        if isinstance(item, Mapping):
            log_id = str(item.get("app_log_id") or "").strip()
            evidence_content = str(item.get("evidence_content") or "").strip()
        elif isinstance(item, (str, int, float)):
            log_id = str(item).strip()
        dedupe_key = (log_id, evidence_content)
        if dedupe_key in seen:
            continue
        if log_id or evidence_content:
            seen.add(dedupe_key)
            out.append({"app_log_id": log_id, "evidence_content": evidence_content})
    return out


def _extract_evidence_ids(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for record in _extract_evidence_records(value):
        log_id = str(record.get("app_log_id") or "").strip()
        if log_id and log_id not in seen:
            seen.add(log_id)
            out.append(log_id)
    return out


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _score_id_metrics(expected_ids: Sequence[str], predicted_ids: Sequence[str]) -> Dict[str, float]:
    exp = {str(v) for v in expected_ids if str(v).strip()}
    pred = {str(v) for v in predicted_ids if str(v).strip()}
    if not exp:
        recall = 1.0 if not pred else 0.0
    else:
        recall = len(exp & pred) / len(exp)
    if not pred:
        precision = 1.0 if not exp else 0.0
    else:
        precision = len(exp & pred) / len(pred)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def _round_or_none(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _expected_state_evidence_by_key(checkpoint: Mapping[str, Any], target_keys: Sequence[str]) -> Dict[str, List[str]]:
    obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
    out: Dict[str, List[str]] = {}
    for key in target_keys:
        obs = obs_flat.get(key)
        if not isinstance(obs, Mapping):
            out[str(key)] = []
            continue
        ids = _extract_evidence_ids(obs.get("evidence_app_log_ids"))
        if not ids:
            last_id = obs.get("last_app_log_id")
            if last_id is not None and str(last_id).strip():
                ids = [str(last_id).strip()]
        out[str(key)] = ids
    return out


def _extract_rq3_pack_by_key(checkpoint: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    payload = checkpoint.get("rq3_apply_service_qa")
    if not isinstance(payload, Mapping):
        return {}
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, Mapping):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, node in keys_obj.items():
        if not isinstance(node, Mapping):
            continue
        items = node.get("items")
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            normalized.append(
                {
                    "qa_id": str(item.get("qa_id") or ""),
                    "question": str(item.get("question") or ""),
                    "reference_answer": str(item.get("reference_answer") or ""),
                    "gold_memory_evidence_app_log_ids": _extract_evidence_ids(
                        item.get("gold_memory_evidence_app_log_ids")
                    ),
                }
            )
        if normalized:
            out[str(key)] = normalized
    return out


def _extract_rq3_answers_by_key(prediction: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    payload = prediction.get("rq3_apply_answers")
    if not isinstance(payload, Mapping):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, node in payload.items():
        if not isinstance(node, Mapping):
            continue
        items = node.get("items")
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            normalized.append(
                {
                    "qa_id": str(item.get("qa_id") or ""),
                    "answer": str(item.get("answer") or ""),
                    "evidence": _extract_evidence_records(item.get("evidence")),
                }
            )
        if normalized:
            out[str(key)] = normalized
    return out


def _build_checkpoint_meta(checkpoint: Mapping[str, Any], order_index: int) -> Dict[str, Any]:
    sampling = checkpoint.get("sampling") or {}
    params = sampling.get("params") or {}
    actual_tokens = params.get("actual_tokens_at_cutoff")
    total_tokens = params.get("total_tokens")
    ratio = _safe_div(float(actual_tokens), float(total_tokens)) if actual_tokens is not None and total_tokens else None
    return {
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "checkpoint_order": int(order_index),
        "anchor_timestamp": str(params.get("anchor_timestamp") or (checkpoint.get("as_of") or {}).get("timestamp") or ""),
        "actual_tokens_at_cutoff": int(actual_tokens) if actual_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
        "token_exposure_ratio": ratio,
    }


def _classify_trend(values: Sequence[Optional[float]]) -> str:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return "insufficient data"
    diffs = [b - a for a, b in zip(clean, clean[1:])]
    eps = 1e-9
    if all(d <= eps for d in diffs) and any(d < -eps for d in diffs):
        return "monotonic drop"
    if all(d >= -eps for d in diffs) and any(d > eps for d in diffs):
        return "monotonic increase"
    return "non-monotonic"


def _format_number(value: Optional[float], ndigits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def _stddev(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    mean_v = sum(clean) / len(clean)
    return math.sqrt(sum((v - mean_v) ** 2 for v in clean) / len(clean))


def _checkpoint_tick_labels(checkpoint_rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for row in checkpoint_rows:
        checkpoint_id = str(row["checkpoint_id"]).replace("cal_quarterly_", "cp")
        tokens = row.get("actual_tokens_at_cutoff")
        if tokens is None:
            labels.append(checkpoint_id)
            continue
        token_int = int(tokens)
        token_label = f"{token_int / 1000:.0f}K" if token_int >= 1000 else str(token_int)
        labels.append(f"{checkpoint_id}\n{token_label}")
    return labels


def _build_task_a_units(
    benchmark_cp: Mapping[str, Any],
    prediction_cp: Mapping[str, Any],
    eval_cp: Mapping[str, Any],
    cp_meta: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    slot_eval = eval_cp.get("snapshot_slot_eval_by_key") or {}
    target_keys = sorted(slot_eval.keys())
    gold_by_key = _expected_state_evidence_by_key(benchmark_cp, target_keys)
    pred_evidence = prediction_cp.get("evidence") if isinstance(prediction_cp.get("evidence"), Mapping) else {}
    rows: List[Dict[str, Any]] = []
    for state_key in target_keys:
        payload = slot_eval.get(state_key) or {}
        metrics = _score_id_metrics(gold_by_key.get(state_key, []), _extract_evidence_ids(pred_evidence.get(state_key)))
        rows.append(
            {
                "checkpoint_id": cp_meta["checkpoint_id"],
                "checkpoint_order": cp_meta["checkpoint_order"],
                "anchor_timestamp": cp_meta["anchor_timestamp"],
                "actual_tokens_at_cutoff": cp_meta["actual_tokens_at_cutoff"],
                "total_tokens": cp_meta["total_tokens"],
                "token_exposure_ratio": cp_meta["token_exposure_ratio"],
                "state_key": state_key,
                "score": payload.get("score_0_1"),
                "slot_count": payload.get("slot_count"),
                **metrics,
            }
        )
    return rows


def _build_task_b_units(
    benchmark_cp: Mapping[str, Any],
    prediction_cp: Mapping[str, Any],
    eval_cp: Mapping[str, Any],
    cp_meta: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    slot_eval = eval_cp.get("change_slot_eval_by_key") or {}
    target_keys = sorted(slot_eval.keys())
    gold_by_key = _expected_state_evidence_by_key(benchmark_cp, target_keys)
    pred_change = prediction_cp.get("change_analysis") if isinstance(prediction_cp.get("change_analysis"), Mapping) else {}
    rows: List[Dict[str, Any]] = []
    for state_key in target_keys:
        node = slot_eval.get(state_key) or {}
        pred_node = pred_change.get(state_key) if isinstance(pred_change.get(state_key), Mapping) else {}
        pred_ids = _extract_evidence_ids(pred_node.get("evidence"))
        metrics = _score_id_metrics(gold_by_key.get(state_key, []), pred_ids)
        rows.append(
            {
                "checkpoint_id": cp_meta["checkpoint_id"],
                "checkpoint_order": cp_meta["checkpoint_order"],
                "anchor_timestamp": cp_meta["anchor_timestamp"],
                "actual_tokens_at_cutoff": cp_meta["actual_tokens_at_cutoff"],
                "total_tokens": cp_meta["total_tokens"],
                "token_exposure_ratio": cp_meta["token_exposure_ratio"],
                "state_key": state_key,
                "state_predict_score": ((node.get("state_predict") or {}).get("score_0_1")),
                "change_reason_score": ((node.get("change_reason") or {}).get("score_0_1")),
                "state_predict_slot_count": ((node.get("state_predict") or {}).get("slot_count")),
                "change_reason_slot_count": ((node.get("change_reason") or {}).get("slot_count")),
                "before_slot_count": ((node.get("before") or {}).get("slot_count")),
                "after_slot_count": ((node.get("after") or {}).get("slot_count")),
                **metrics,
            }
        )
    return rows


def _build_task_c_units(
    benchmark_cp: Mapping[str, Any],
    prediction_cp: Mapping[str, Any],
    eval_cp: Mapping[str, Any],
    cp_meta: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    slot_eval = eval_cp.get("rq3_apply_slot_eval_by_item") or {}
    pack_by_key = _extract_rq3_pack_by_key(benchmark_cp)
    pred_by_key = _extract_rq3_answers_by_key(prediction_cp)
    rows: List[Dict[str, Any]] = []
    for item_id in sorted(slot_eval.keys()):
        payload = slot_eval.get(item_id) or {}
        state_key = str(payload.get("state_key") or item_id.split("::", 1)[0])
        qa_id = str(payload.get("qa_id") or item_id.rsplit("::", 1)[-1])
        gold_item = next((item for item in pack_by_key.get(state_key, []) if str(item.get("qa_id")) == qa_id), {})
        pred_item = next((item for item in pred_by_key.get(state_key, []) if str(item.get("qa_id")) == qa_id), {})
        metrics = _score_id_metrics(
            gold_item.get("gold_memory_evidence_app_log_ids", []),
            _extract_evidence_ids(pred_item.get("evidence")),
        )
        rows.append(
            {
                "checkpoint_id": cp_meta["checkpoint_id"],
                "checkpoint_order": cp_meta["checkpoint_order"],
                "anchor_timestamp": cp_meta["anchor_timestamp"],
                "actual_tokens_at_cutoff": cp_meta["actual_tokens_at_cutoff"],
                "total_tokens": cp_meta["total_tokens"],
                "token_exposure_ratio": cp_meta["token_exposure_ratio"],
                "state_key": state_key,
                "qa_id": qa_id,
                "item_id": item_id,
                "score": payload.get("score_0_1"),
                "slot_count": payload.get("slot_count"),
                **metrics,
            }
        )
    return rows


def _build_rq3_overlap_rows(task_a_units: Sequence[Dict[str, Any]], task_c_units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    a_by_cp: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in task_a_units:
        score = row.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        a_by_cp[str(row["checkpoint_id"])][str(row["state_key"])] = float(score)

    c_scores_by_cp_key: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in task_c_units:
        score = row.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        c_scores_by_cp_key[str(row["checkpoint_id"])][str(row["state_key"])].append(float(score))

    rows: List[Dict[str, Any]] = []
    for checkpoint_id in sorted(set(a_by_cp.keys()) | set(c_scores_by_cp_key.keys())):
        a_map = a_by_cp.get(checkpoint_id, {})
        c_map = {k: sum(v) / len(v) for k, v in c_scores_by_cp_key.get(checkpoint_id, {}).items() if v}
        overlap = sorted(set(a_map.keys()) & set(c_map.keys()))
        if not overlap:
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "overlap_state_count": 0,
                    "task_a_overlap_score_mean": None,
                    "task_c_overlap_score_mean": None,
                    "apply_minus_know_mean": None,
                }
            )
            continue
        task_a_vals = [a_map[k] for k in overlap]
        task_c_vals = [c_map[k] for k in overlap]
        gaps = [c_map[k] - a_map[k] for k in overlap]
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "overlap_state_count": len(overlap),
                "task_a_overlap_score_mean": _mean(task_a_vals),
                "task_c_overlap_score_mean": _mean(task_c_vals),
                "apply_minus_know_mean": _mean(gaps),
            }
        )
    return rows


def _build_state_summary_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    score_field: str,
    rank_field_name: str,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    by_state: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        state_key = str(row.get("state_key") or "")
        score = row.get(score_field)
        if not state_key or score is None:
            continue
        score_f = float(score)
        if math.isfinite(score_f):
            by_state[state_key].append(score_f)

    summary_rows: List[Dict[str, Any]] = []
    for state_key, values in by_state.items():
        summary_rows.append(
            {
                "state_key": state_key,
                rank_field_name: _stddev(values),
                "mean_score": _mean(values),
                "std_score": _stddev(values),
                "min_score": min(values) if values else None,
                "max_score": max(values) if values else None,
                "non_null_checkpoint_count": len(values),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -1 if row.get(rank_field_name) is None else -float(row.get(rank_field_name)),
            str(row["state_key"]),
        )
    )
    if top_n is not None:
        return summary_rows[:top_n]
    return summary_rows


def _build_rq3_gap_state_rows(
    task_a_units: Sequence[Dict[str, Any]],
    task_c_units: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    a_by_cp_state: Dict[Tuple[str, str], float] = {}
    for row in task_a_units:
        score = row.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        a_by_cp_state[(str(row["checkpoint_id"]), str(row["state_key"]))] = float(score)

    c_scores_by_cp_state: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in task_c_units:
        score = row.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        c_scores_by_cp_state[(str(row["checkpoint_id"]), str(row["state_key"]))].append(float(score))

    out: List[Dict[str, Any]] = []
    keys = sorted(set(a_by_cp_state.keys()) & set(c_scores_by_cp_state.keys()))
    for checkpoint_id, state_key in keys:
        c_state_score = _mean(c_scores_by_cp_state[(checkpoint_id, state_key)])
        a_state_score = a_by_cp_state[(checkpoint_id, state_key)]
        if c_state_score is None:
            continue
        out.append(
            {
                "checkpoint_id": checkpoint_id,
                "state_key": state_key,
                "task_a_score": a_state_score,
                "task_c_score": c_state_score,
                "gap": c_state_score - a_state_score,
            }
        )
    return out


def _build_gap_summary_rows(gap_rows: Sequence[Dict[str, Any]], *, descending: bool, top_n: int) -> List[Dict[str, Any]]:
    by_state: Dict[str, List[float]] = defaultdict(list)
    for row in gap_rows:
        gap = row.get("gap")
        state_key = str(row.get("state_key") or "")
        if not state_key or gap is None:
            continue
        gap_f = float(gap)
        if math.isfinite(gap_f):
            by_state[state_key].append(gap_f)

    summary_rows: List[Dict[str, Any]] = []
    for state_key, values in by_state.items():
        summary_rows.append(
            {
                "state_key": state_key,
                "mean_gap": _mean(values),
                "std_gap": _stddev(values),
                "min_gap": min(values) if values else None,
                "max_gap": max(values) if values else None,
                "non_null_checkpoint_count": len(values),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -(float(row["mean_gap"])) if descending and row.get("mean_gap") is not None else (
                float(row["mean_gap"]) if row.get("mean_gap") is not None else float("inf")
            ),
            str(row["state_key"]),
        )
    )
    return summary_rows[:top_n]


def _build_correlation_rows(
    task_a_units: Sequence[Dict[str, Any]],
    task_b_units: Sequence[Dict[str, Any]],
    task_c_units: Sequence[Dict[str, Any]],
    *,
    high_evidence: float = 0.5,
    low_evidence: float = 0.1,
    high_score: float = 0.8,
    low_score: float = 0.2,
) -> List[Dict[str, Any]]:
    def build_rows(task: str, score_variant: str, units: Sequence[Dict[str, Any]], score_field: str) -> Dict[str, Any]:
        valid = []
        high_ev_low_score = 0
        low_ev_high_score = 0
        for row in units:
            score = row.get(score_field)
            ev = row.get("evidence_f1")
            if score is None or ev is None:
                continue
            score_f = float(score)
            ev_f = float(ev)
            if not (math.isfinite(score_f) and math.isfinite(ev_f)):
                continue
            valid.append((ev_f, score_f))
            if ev_f >= high_evidence and score_f <= low_score:
                high_ev_low_score += 1
            if ev_f <= low_evidence and score_f >= high_score:
                low_ev_high_score += 1
        xs = [x for x, _ in valid]
        ys = [y for _, y in valid]
        return {
            "task": task,
            "score_variant": score_variant,
            "unit_count": len(valid),
            "pearson_score_vs_evidence_f1": _pearson(xs, ys),
            "spearman_score_vs_evidence_f1": _spearman(xs, ys),
            "high_evidence_low_score_count": high_ev_low_score,
            "low_evidence_high_score_count": low_ev_high_score,
        }

    return [
        build_rows("task_a", "score", task_a_units, "score"),
        build_rows("task_b", "state_predict", task_b_units, "state_predict_score"),
        build_rows("task_b", "change_reason", task_b_units, "change_reason_score"),
        build_rows("task_c", "score", task_c_units, "score"),
    ]


def _build_heatmap_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    checkpoint_rows: Sequence[Dict[str, Any]],
    score_field: str,
    order_mode: str,
) -> List[Tuple[str, List[float]]]:
    checkpoint_ids = [str(row["checkpoint_id"]) for row in checkpoint_rows]
    checkpoint_index = {checkpoint_id: i for i, checkpoint_id in enumerate(checkpoint_ids)}
    state_to_scores: Dict[str, List[float]] = defaultdict(lambda: [math.nan] * len(checkpoint_ids))
    for row in rows:
        checkpoint_id = str(row.get("checkpoint_id") or "")
        state_key = str(row.get("state_key") or "")
        score = row.get(score_field)
        if checkpoint_id not in checkpoint_index or not state_key or score is None:
            continue
        score_f = float(score)
        if math.isfinite(score_f):
            state_to_scores[state_key][checkpoint_index[checkpoint_id]] = score_f

    def sort_key(item: Tuple[str, List[float]]) -> Tuple[float, str]:
        state_key, values = item
        finite = [v for v in values if not math.isnan(v)]
        if order_mode == "mean_desc":
            metric = _mean(finite)
            return (-(metric if metric is not None else float("-inf")), state_key)
        metric = _stddev(finite)
        return (-(metric if metric is not None else float("-inf")), state_key)

    return sorted(state_to_scores.items(), key=sort_key)


def _checkpoint_color_map(checkpoint_ids: Sequence[str]) -> Dict[str, str]:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return {checkpoint_id: palette[i % len(palette)] for i, checkpoint_id in enumerate(checkpoint_ids)}


def _plot_task_lines(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    y_fields: Sequence[Tuple[str, str, str]],
    title: str,
    y_label: str,
) -> None:
    if plt is None:
        return
    xs = list(range(len(rows)))
    x_labels = _checkpoint_tick_labels(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    for field, label, color in y_fields:
        ys = [row.get(field) for row in rows]
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gap_line(path: Path, rows: Sequence[Dict[str, Any]], *, title: str) -> None:
    if plt is None:
        return
    xs = list(range(len(rows)))
    ys = [row.get("apply_minus_know_mean") for row in rows]
    x_labels = [row["checkpoint_id"].replace("cal_quarterly_", "cp") for row in rows]
    finite = [float(v) for v in ys if v is not None and math.isfinite(float(v))]
    ymin = min(finite) if finite else -0.1
    ymax = max(finite) if finite else 0.1
    padding = max(0.05, (ymax - ymin) * 0.2 if finite else 0.1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.plot(xs, ys, marker="o", color="#2ca02c")
    ax.set_title(title)
    ax.set_ylabel("apply - know")
    ax.set_xlabel("checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_state_lines(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    checkpoint_rows: Sequence[Dict[str, Any]],
    score_field: str,
    title: str,
    y_label: str,
) -> None:
    if plt is None:
        return
    checkpoint_ids = [str(row["checkpoint_id"]) for row in checkpoint_rows]
    xs = list(range(len(checkpoint_ids)))
    checkpoint_index = {checkpoint_id: i for i, checkpoint_id in enumerate(checkpoint_ids)}
    state_to_scores: Dict[str, List[float]] = defaultdict(lambda: [math.nan] * len(checkpoint_ids))

    for row in rows:
        checkpoint_id = str(row.get("checkpoint_id") or "")
        state_key = str(row.get("state_key") or "")
        score = row.get(score_field)
        if checkpoint_id not in checkpoint_index or not state_key or score is None:
            continue
        score_f = float(score)
        if not math.isfinite(score_f):
            continue
        state_to_scores[state_key][checkpoint_index[checkpoint_id]] = score_f

    ranked_states = sorted(
        state_to_scores.keys(),
        key=lambda state_key: (
            -1 if _stddev([v for v in state_to_scores[state_key] if not math.isnan(v)]) is None
            else -float(_stddev([v for v in state_to_scores[state_key] if not math.isnan(v)])),
            state_key,
        ),
    )
    overlay_states = ranked_states[:8]

    fig, ax = plt.subplots(figsize=(11, 6))
    for state_key in ranked_states:
        ax.plot(xs, state_to_scores[state_key], linewidth=0.8, alpha=0.18, color="#999999")

    overlay_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
    for idx, state_key in enumerate(overlay_states):
        ax.plot(
            xs,
            state_to_scores[state_key],
            linewidth=1.8,
            alpha=0.95,
            color=overlay_palette[idx % len(overlay_palette)],
            label=state_key.split(":", 1)[-1],
        )
    ax.set_title(f"{title}\n{len(state_to_scores)} states; top 8 variable highlighted")
    ax.set_ylabel(y_label)
    ax.set_xlabel("checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(_checkpoint_tick_labels(checkpoint_rows))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    if overlay_states:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmap(
    path: Path,
    *,
    matrix_rows: Sequence[Tuple[str, List[float]]],
    checkpoint_rows: Sequence[Dict[str, Any]],
    title: str,
    cmap_name: str,
    center_zero: bool = False,
) -> None:
    if plt is None or mcolors is None:
        return
    if not matrix_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    data = [scores for _, scores in matrix_rows]
    labels = [state_key for state_key, _ in matrix_rows]
    masked = [[math.nan if (v is None or not math.isfinite(v)) else float(v) for v in row] for row in data]

    import numpy as np

    matrix = np.array(masked, dtype=float)
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#e6e6e6")

    norm = None
    if center_zero:
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            max_abs = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig_height = max(6, min(18, 0.22 * len(labels) + 2.5))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    im = ax.imshow(masked_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("state_key")
    ax.set_xticks(range(len(checkpoint_rows)))
    ax.set_xticklabels(_checkpoint_tick_labels(checkpoint_rows))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("score" if not center_zero else "gap", rotation=270, labelpad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scatter(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    score_field: str,
    title_prefix: str,
    checkpoint_colors: Mapping[str, str],
) -> None:
    if plt is None:
        return
    valid_rows = [
        row
        for row in rows
        if row.get(score_field) is not None
        and row.get("evidence_f1") is not None
        and math.isfinite(float(row.get(score_field)))
        and math.isfinite(float(row.get("evidence_f1")))
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if valid_rows:
        for checkpoint_id in sorted({str(row["checkpoint_id"]) for row in valid_rows}):
            subset = [row for row in valid_rows if str(row["checkpoint_id"]) == checkpoint_id]
            ax.scatter(
                [float(row["evidence_f1"]) for row in subset],
                [float(row[score_field]) for row in subset],
                s=20,
                alpha=0.75,
                color=checkpoint_colors.get(checkpoint_id, "#1f77b4"),
                label=checkpoint_id.replace("cal_quarterly_", "cp"),
            )
        pearson = _pearson(
            [float(row["evidence_f1"]) for row in valid_rows],
            [float(row[score_field]) for row in valid_rows],
        )
        spearman = _spearman(
            [float(row["evidence_f1"]) for row in valid_rows],
            [float(row[score_field]) for row in valid_rows],
        )
        ax.set_title(f"{title_prefix}\nPearson={_format_number(pearson, 3)}  Spearman={_format_number(spearman, 3)}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title_prefix)
    ax.set_xlabel("evidence F1")
    ax.set_ylabel("score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        values = []
        for key, _label in fields:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_format_number(value))
            else:
                values.append(str(value if value is not None else ""))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + body)


def _write_markdown_summary(
    path: Path,
    *,
    artifact_paths: Mapping[str, str],
    checkpoint_rows: Sequence[Dict[str, Any]],
    rq3_rows: Sequence[Dict[str, Any]],
    correlation_rows: Sequence[Dict[str, Any]],
) -> None:
    a_trend = _classify_trend([row.get("task_a_point_score") for row in checkpoint_rows])
    b_reason_trend = _classify_trend([row.get("task_b_reason_score") for row in checkpoint_rows])
    c_trend = _classify_trend([row.get("task_c_answer_score") for row in checkpoint_rows])
    rq3_gap_trend = _classify_trend([row.get("apply_minus_know_mean") for row in rq3_rows])

    corr_map = {(row["task"], row["score_variant"]): row for row in correlation_rows}
    task_a_corr = corr_map.get(("task_a", "score"), {})
    task_b_reason_corr = corr_map.get(("task_b", "change_reason"), {})
    task_c_corr = corr_map.get(("task_c", "score"), {})

    bullets = [
        f"- `Task A` across checkpoints is `{a_trend}`; start={_format_number(checkpoint_rows[0].get('task_a_point_score'))}, end={_format_number(checkpoint_rows[-1].get('task_a_point_score'))}.",
        f"- `Task C` across checkpoints is `{c_trend}`; start={_format_number(checkpoint_rows[0].get('task_c_answer_score'))}, end={_format_number(checkpoint_rows[-1].get('task_c_answer_score'))}.",
        f"- `Task B change_reason` is `{b_reason_trend}` over the changed checkpoints.",
        f"- `RQ3 apply-know gap` is `{rq3_gap_trend}` over overlap states.",
        f"- `Task A` evidence-vs-score Pearson={_format_number(task_a_corr.get('pearson_score_vs_evidence_f1'), 3)}.",
        f"- `Task B reason` evidence-vs-score Pearson={_format_number(task_b_reason_corr.get('pearson_score_vs_evidence_f1'), 3)}.",
        f"- `Task C` evidence-vs-score Pearson={_format_number(task_c_corr.get('pearson_score_vs_evidence_f1'), 3)}.",
    ]

    content = f"""# Milestone1 TCE Analysis Summary

## Artifacts
- benchmark: `{artifact_paths['benchmark']}`
- prediction: `{artifact_paths['prediction']}`
- eval: `{artifact_paths['eval']}`
- analysis output: `{artifact_paths['output_dir']}`

## Checkpoint Exposure Table
{_markdown_table(
    checkpoint_rows,
    [
        ('checkpoint_id', 'checkpoint'),
        ('anchor_timestamp', 'anchor timestamp'),
        ('actual_tokens_at_cutoff', 'tokens'),
        ('task_a_point_score', 'Task A'),
        ('task_b_state_score', 'Task B state'),
        ('task_b_reason_score', 'Task B reason'),
        ('task_c_answer_score', 'Task C'),
        ('rq3_mean_gap', 'RQ3 gap'),
    ],
)}

## Top-line Bullets
{chr(10).join(bullets)}

## RQ1
Using checkpoint as the primary x-axis, the current milestone1 result does not show a clean monotonic accuracy drop. `Task A` is non-monotonic across the five checkpoints, `Task C` also fluctuates rather than steadily degrading, and later checkpoints can recover relative to mid-sequence dips. This means the current single-user result is better described as exposure-conditioned fluctuation than as a simple “more exposure -> lower accuracy” story.

## RQ2
For attribution ability, the primary metric is `Task B change_reason`. The score changes substantially across checkpoints and is not monotonic, which suggests the harder part is not only reconstructing changed state but also explaining why the change happened. The paired state-predict and change-evidence F1 columns should be read as diagnostics for whether low attribution is accompanied by weaker state reconstruction or weaker evidence grounding.

## RQ3
For know-vs-apply, the gap is computed only on overlap states that have both `Task A` and `Task C` scores in the same checkpoint. Positive values mean application is easier than state reconstruction on the same underlying states. In this milestone1 run, the gap should be interpreted as a relative difficulty gap, not as proof that apply quality is intrinsically strong, because Task C evidence specificity remains future work.

## Evidence vs Score
The evidence-vs-score consistency is task-dependent. `Task A` and `Task B` can be compared against their own recomputed unit-level evidence F1, while `Task C` is judged at `(state_key, qa_id)` item level. The correlation table and scatter plots should be used together: moderate or weak correlation does not automatically imply a bug, but the outlier counts highlight cases where the answer score and evidence grounding diverge.

## Cautions
- This is a single-user (`001_user_001`) milestone1 analysis.
- `Task C` evidence specificity is known future work; current evidence F1 is still based on the present protocol.
- These plots and tables describe the frozen `v14/top20/slotjudge` artifact family only.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_analysis_pack(
    *,
    benchmark_payload: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    output_dir: Path,
    artifact_paths: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    aligned_prediction, align_report = align_predictions_to_benchmark(dict(benchmark_payload), dict(prediction_payload))
    pred_by_id = normalize_predictions(aligned_prediction)
    eval_by_id = {
        str(item.get("checkpoint_id")): item
        for item in eval_payload.get("checkpoints", [])
        if isinstance(item, Mapping) and item.get("checkpoint_id")
    }

    checkpoint_rows: List[Dict[str, Any]] = []
    task_a_units: List[Dict[str, Any]] = []
    task_b_units: List[Dict[str, Any]] = []
    task_c_units: List[Dict[str, Any]] = []

    for order_index, benchmark_cp in enumerate(benchmark_payload.get("checkpoints", []), start=1):
        if not isinstance(benchmark_cp, Mapping):
            continue
        checkpoint_id = str(benchmark_cp.get("checkpoint_id") or "")
        prediction_cp = pred_by_id.get(checkpoint_id, {})
        eval_cp = eval_by_id.get(checkpoint_id, {})
        cp_meta = _build_checkpoint_meta(benchmark_cp, order_index)

        task_a_unit_rows = _build_task_a_units(benchmark_cp, prediction_cp, eval_cp, cp_meta)
        task_b_unit_rows = _build_task_b_units(benchmark_cp, prediction_cp, eval_cp, cp_meta)
        task_c_unit_rows = _build_task_c_units(benchmark_cp, prediction_cp, eval_cp, cp_meta)

        task_a_units.extend(task_a_unit_rows)
        task_b_units.extend(task_b_unit_rows)
        task_c_units.extend(task_c_unit_rows)

        checkpoint_rows.append(
            {
                **cp_meta,
                "task_a_point_score": eval_cp.get("snapshot_point_score_mean_on_expected"),
                "task_a_evidence_f1": eval_cp.get("snapshot_evidence_f1_mean_on_expected"),
                "task_b_state_score": eval_cp.get("change_state_predict_point_score_mean_on_changed"),
                "task_b_reason_score": eval_cp.get("change_reason_point_score_mean_on_changed"),
                "task_b_evidence_f1": eval_cp.get("change_evidence_f1_mean_on_changed"),
                "task_c_answer_score": eval_cp.get("rq3_apply_answer_point_score_mean"),
                "task_c_evidence_f1": eval_cp.get("rq3_apply_evidence_f1"),
                "task_a_key_count": len(eval_cp.get("snapshot_slot_eval_by_key") or {}),
                "task_b_changed_key_count": len(eval_cp.get("change_slot_eval_by_key") or {}),
                "task_c_item_count": len(eval_cp.get("rq3_apply_slot_eval_by_item") or {}),
            }
        )

    rq3_rows = _build_rq3_overlap_rows(task_a_units, task_c_units)
    rq3_by_checkpoint = {row["checkpoint_id"]: row for row in rq3_rows}
    for row in checkpoint_rows:
        rq3 = rq3_by_checkpoint.get(row["checkpoint_id"], {})
        row["rq3_overlap_state_count"] = rq3.get("overlap_state_count", 0)
        row["rq3_task_a_overlap_score_mean"] = rq3.get("task_a_overlap_score_mean")
        row["rq3_task_c_overlap_score_mean"] = rq3.get("task_c_overlap_score_mean")
        row["rq3_mean_gap"] = rq3.get("apply_minus_know_mean")

    correlation_rows = _build_correlation_rows(task_a_units, task_b_units, task_c_units)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "checkpoint_summary.csv",
        checkpoint_rows,
        [
            "checkpoint_id",
            "checkpoint_order",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
            "token_exposure_ratio",
            "task_a_point_score",
            "task_a_evidence_f1",
            "task_b_state_score",
            "task_b_reason_score",
            "task_b_evidence_f1",
            "task_c_answer_score",
            "task_c_evidence_f1",
            "task_a_key_count",
            "task_b_changed_key_count",
            "task_c_item_count",
            "rq3_overlap_state_count",
            "rq3_mean_gap",
        ],
    )
    _write_csv(
        output_dir / "rq1_task_scores_by_checkpoint.csv",
        checkpoint_rows,
        [
            "checkpoint_id",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "token_exposure_ratio",
            "task_a_point_score",
            "task_b_state_score",
            "task_c_answer_score",
        ],
    )
    _write_csv(
        output_dir / "rq2_taskb_attribution_by_checkpoint.csv",
        checkpoint_rows,
        [
            "checkpoint_id",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "task_b_changed_key_count",
            "task_b_state_score",
            "task_b_reason_score",
            "task_b_evidence_f1",
        ],
    )
    _write_csv(
        output_dir / "rq3_know_apply_gap_by_checkpoint.csv",
        rq3_rows,
        [
            "checkpoint_id",
            "overlap_state_count",
            "task_a_overlap_score_mean",
            "task_c_overlap_score_mean",
            "apply_minus_know_mean",
        ],
    )
    _write_csv(
        output_dir / "correlation_summary.csv",
        correlation_rows,
        [
            "task",
            "score_variant",
            "unit_count",
            "pearson_score_vs_evidence_f1",
            "spearman_score_vs_evidence_f1",
            "high_evidence_low_score_count",
            "low_evidence_high_score_count",
        ],
    )
    _write_csv(
        output_dir / "task_a_units.csv",
        task_a_units,
        [
            "checkpoint_id",
            "checkpoint_order",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
            "token_exposure_ratio",
            "state_key",
            "score",
            "slot_count",
            "evidence_precision",
            "evidence_recall",
            "evidence_f1",
        ],
    )
    _write_csv(
        output_dir / "task_b_units.csv",
        task_b_units,
        [
            "checkpoint_id",
            "checkpoint_order",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
            "token_exposure_ratio",
            "state_key",
            "state_predict_score",
            "change_reason_score",
            "state_predict_slot_count",
            "change_reason_slot_count",
            "before_slot_count",
            "after_slot_count",
            "evidence_precision",
            "evidence_recall",
            "evidence_f1",
        ],
    )
    _write_csv(
        output_dir / "task_c_units.csv",
        task_c_units,
        [
            "checkpoint_id",
            "checkpoint_order",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
            "token_exposure_ratio",
            "state_key",
            "qa_id",
            "item_id",
            "score",
            "slot_count",
            "evidence_precision",
            "evidence_recall",
            "evidence_f1",
        ],
    )

    task_c_state_rows: List[Dict[str, Any]] = []
    task_c_by_checkpoint_state: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in task_c_units:
        score = row.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        task_c_by_checkpoint_state[(str(row["checkpoint_id"]), str(row["state_key"]))].append(float(score))
    cp_meta_by_id = {str(row["checkpoint_id"]): row for row in checkpoint_rows}
    for (checkpoint_id, state_key), scores in sorted(task_c_by_checkpoint_state.items()):
        cp_row = cp_meta_by_id.get(checkpoint_id, {})
        task_c_state_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_order": cp_row.get("checkpoint_order"),
                "state_key": state_key,
                "state_score": _mean(scores),
            }
        )

    task_a_rank_rows = _build_state_summary_rows(
        task_a_units,
        score_field="score",
        rank_field_name="variability_score",
        top_n=20,
    )
    task_b_state_rank_rows = _build_state_summary_rows(
        task_b_units,
        score_field="state_predict_score",
        rank_field_name="variability_score",
        top_n=20,
    )
    task_b_reason_rank_rows = _build_state_summary_rows(
        task_b_units,
        score_field="change_reason_score",
        rank_field_name="variability_score",
        top_n=20,
    )
    task_c_rank_rows = _build_state_summary_rows(
        task_c_state_rows,
        score_field="state_score",
        rank_field_name="variability_score",
        top_n=20,
    )
    rq3_gap_state_rows = _build_rq3_gap_state_rows(task_a_units, task_c_units)
    rq3_positive_gap_rows = _build_gap_summary_rows(rq3_gap_state_rows, descending=True, top_n=20)
    rq3_negative_gap_rows = _build_gap_summary_rows(rq3_gap_state_rows, descending=False, top_n=20)

    task_a_heatmap_rows = _build_heatmap_rows(
        task_a_units,
        checkpoint_rows=checkpoint_rows,
        score_field="score",
        order_mode="variance_desc",
    )
    task_b_state_heatmap_rows = _build_heatmap_rows(
        task_b_units,
        checkpoint_rows=checkpoint_rows,
        score_field="state_predict_score",
        order_mode="variance_desc",
    )
    task_b_state_order = [state_key for state_key, _ in task_b_state_heatmap_rows]
    task_b_reason_heatmap_lookup = {
        state_key: scores
        for state_key, scores in _build_heatmap_rows(
            task_b_units,
            checkpoint_rows=checkpoint_rows,
            score_field="change_reason_score",
            order_mode="variance_desc",
        )
    }
    task_b_reason_heatmap_rows = [
        (
            state_key,
            task_b_reason_heatmap_lookup.get(state_key, [math.nan] * len(checkpoint_rows)),
        )
        for state_key in task_b_state_order
    ]
    task_c_heatmap_rows = _build_heatmap_rows(
        task_c_state_rows,
        checkpoint_rows=checkpoint_rows,
        score_field="state_score",
        order_mode="variance_desc",
    )
    rq3_gap_heatmap_rows = _build_heatmap_rows(
        rq3_gap_state_rows,
        checkpoint_rows=checkpoint_rows,
        score_field="gap",
        order_mode="mean_desc",
    )

    checkpoint_colors = _checkpoint_color_map([row["checkpoint_id"] for row in checkpoint_rows])
    _plot_task_lines(
        output_dir / "rq1_task_scores_vs_checkpoint.png",
        checkpoint_rows,
        y_fields=[
            ("task_a_point_score", "Task A", "#1f77b4"),
            ("task_b_state_score", "Task B state", "#d62728"),
            ("task_c_answer_score", "Task C", "#2ca02c"),
        ],
        title="RQ1: Task Scores vs Checkpoint",
        y_label="score",
    )
    _plot_task_lines(
        output_dir / "rq2_taskb_attribution_vs_checkpoint.png",
        checkpoint_rows,
        y_fields=[
            ("task_b_reason_score", "Task B reason", "#9467bd"),
            ("task_b_state_score", "Task B state", "#ff7f0e"),
        ],
        title="RQ2: Task B Attribution vs Checkpoint",
        y_label="score",
    )
    _plot_gap_line(
        output_dir / "rq3_know_apply_gap_vs_checkpoint.png",
        rq3_rows,
        title="RQ3: Apply - Know Gap vs Checkpoint",
    )
    _plot_heatmap(
        output_dir / "task_a_state_heatmap.png",
        matrix_rows=task_a_heatmap_rows,
        checkpoint_rows=checkpoint_rows,
        title="Task A State Heatmap",
        cmap_name="viridis",
    )
    _plot_heatmap(
        output_dir / "task_b_state_predict_heatmap.png",
        matrix_rows=task_b_state_heatmap_rows,
        checkpoint_rows=checkpoint_rows,
        title="Task B State-Predict Heatmap",
        cmap_name="magma",
    )
    _plot_heatmap(
        output_dir / "task_b_change_reason_heatmap.png",
        matrix_rows=task_b_reason_heatmap_rows,
        checkpoint_rows=checkpoint_rows,
        title="Task B Change-Reason Heatmap",
        cmap_name="plasma",
    )
    _plot_heatmap(
        output_dir / "task_c_state_heatmap.png",
        matrix_rows=task_c_heatmap_rows,
        checkpoint_rows=checkpoint_rows,
        title="Task C State Heatmap",
        cmap_name="viridis",
    )
    _plot_heatmap(
        output_dir / "rq3_gap_heatmap.png",
        matrix_rows=rq3_gap_heatmap_rows,
        checkpoint_rows=checkpoint_rows,
        title="RQ3 Apply-Know Gap Heatmap",
        cmap_name="coolwarm",
        center_zero=True,
    )
    _plot_state_lines(
        output_dir / "task_a_state_lines_vs_checkpoint.png",
        task_a_units,
        checkpoint_rows=checkpoint_rows,
        score_field="score",
        title="Task A State-Level Scores vs Checkpoint",
        y_label="score",
    )
    _plot_state_lines(
        output_dir / "task_b_state_predict_lines_vs_checkpoint.png",
        task_b_units,
        checkpoint_rows=checkpoint_rows,
        score_field="state_predict_score",
        title="Task B State-Predict Scores vs Checkpoint",
        y_label="score",
    )
    _plot_state_lines(
        output_dir / "task_b_change_reason_lines_vs_checkpoint.png",
        task_b_units,
        checkpoint_rows=checkpoint_rows,
        score_field="change_reason_score",
        title="Task B Change-Reason Scores vs Checkpoint",
        y_label="score",
    )
    _plot_state_lines(
        output_dir / "task_c_state_lines_vs_checkpoint.png",
        task_c_state_rows,
        checkpoint_rows=checkpoint_rows,
        score_field="state_score",
        title="Task C State-Level Scores vs Checkpoint",
        y_label="score",
    )
    _plot_scatter(
        output_dir / "task_a_evidence_vs_score_scatter.png",
        task_a_units,
        score_field="score",
        title_prefix="Task A: Evidence F1 vs Score",
        checkpoint_colors=checkpoint_colors,
    )
    _plot_scatter(
        output_dir / "task_b_state_evidence_vs_score_scatter.png",
        task_b_units,
        score_field="state_predict_score",
        title_prefix="Task B state: Evidence F1 vs Score",
        checkpoint_colors=checkpoint_colors,
    )
    _plot_scatter(
        output_dir / "task_b_reason_evidence_vs_score_scatter.png",
        task_b_units,
        score_field="change_reason_score",
        title_prefix="Task B reason: Evidence F1 vs Score",
        checkpoint_colors=checkpoint_colors,
    )
    _plot_scatter(
        output_dir / "task_c_evidence_vs_score_scatter.png",
        task_c_units,
        score_field="score",
        title_prefix="Task C: Evidence F1 vs Score",
        checkpoint_colors=checkpoint_colors,
    )

    _write_csv(
        output_dir / "task_a_top_variable_states.csv",
        task_a_rank_rows,
        ["state_key", "variability_score", "mean_score", "std_score", "min_score", "max_score", "non_null_checkpoint_count"],
    )
    _write_csv(
        output_dir / "task_b_state_predict_top_variable_states.csv",
        task_b_state_rank_rows,
        ["state_key", "variability_score", "mean_score", "std_score", "min_score", "max_score", "non_null_checkpoint_count"],
    )
    _write_csv(
        output_dir / "task_b_change_reason_top_variable_states.csv",
        task_b_reason_rank_rows,
        ["state_key", "variability_score", "mean_score", "std_score", "min_score", "max_score", "non_null_checkpoint_count"],
    )
    _write_csv(
        output_dir / "task_c_top_variable_states.csv",
        task_c_rank_rows,
        ["state_key", "variability_score", "mean_score", "std_score", "min_score", "max_score", "non_null_checkpoint_count"],
    )
    _write_csv(
        output_dir / "rq3_top_positive_gap_states.csv",
        rq3_positive_gap_rows,
        ["state_key", "mean_gap", "std_gap", "min_gap", "max_gap", "non_null_checkpoint_count"],
    )
    _write_csv(
        output_dir / "rq3_top_negative_gap_states.csv",
        rq3_negative_gap_rows,
        ["state_key", "mean_gap", "std_gap", "min_gap", "max_gap", "non_null_checkpoint_count"],
    )

    _write_markdown_summary(
        output_dir / "milestone1_analysis_summary.md",
        artifact_paths=artifact_paths
        or {
            "benchmark": "",
            "prediction": "",
            "eval": "",
            "output_dir": str(output_dir),
        },
        checkpoint_rows=checkpoint_rows,
        rq3_rows=rq3_rows,
        correlation_rows=correlation_rows,
    )

    return {
        "checkpoint_summary_rows": checkpoint_rows,
        "task_a_units": task_a_units,
        "task_b_units": task_b_units,
        "task_c_units": task_c_units,
        "task_c_state_rows": task_c_state_rows,
        "rq3_overlap_rows": rq3_rows,
        "rq3_gap_state_rows": rq3_gap_state_rows,
        "correlation_rows": correlation_rows,
        "alignment_report": align_report,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TCE milestone analysis pack.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = _load_json(args.benchmark)
    prediction = _load_json(args.prediction)
    eval_payload = _load_json(args.eval_path)
    result = build_analysis_pack(
        benchmark_payload=benchmark,
        prediction_payload=prediction,
        eval_payload=eval_payload,
        output_dir=args.output_dir,
        artifact_paths={
            "benchmark": str(args.benchmark),
            "prediction": str(args.prediction),
            "eval": str(args.eval_path),
            "output_dir": str(args.output_dir),
        },
    )
    print(json.dumps({"output_dir": result["output_dir"], "alignment_report": result["alignment_report"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
