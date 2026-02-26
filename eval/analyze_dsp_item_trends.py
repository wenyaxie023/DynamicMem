#!/usr/bin/env python3
"""Analyze DSP predictions at item level and plot temporal trends."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_core.dsp_evaluator import align_predictions_to_benchmark
from dynamic_state_prediction_core.evaluation import flatten_snapshot, normalize_predictions, value_f1
from dynamic_state_prediction_core.pipeline import drop_excluded_fields, parse_ts


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _sort_buckets(bucket_values: Iterable[str], bucket_type: str) -> List[str]:
    items = list(set(bucket_values))
    if bucket_type == "date":
        return sorted(items)

    def key_fn(x: str) -> Tuple[int, str]:
        try:
            if x.startswith("cp_"):
                return (int(x.split("_", 1)[1]), x)
        except Exception:
            pass
        return (10**9, x)

    return sorted(items, key=key_fn)


def _rolling_for_group(
    points: List[Dict[str, Any]],
    *,
    metric_key: str,
    rolling_days: int,
    bucket_type: str,
) -> List[Optional[float]]:
    if not points:
        return []
    if rolling_days <= 1:
        return [float(p[metric_key]) for p in points]

    out: List[Optional[float]] = []
    if bucket_type == "date":
        parsed_dates = [datetime.strptime(str(p["bucket"]), "%Y-%m-%d").date() for p in points]
        for i in range(len(points)):
            right = parsed_dates[i]
            vals: List[float] = []
            for j in range(i, -1, -1):
                if (right - parsed_dates[j]).days >= rolling_days:
                    break
                vals.append(float(points[j][metric_key]))
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    for i in range(len(points)):
        left = max(0, i - rolling_days + 1)
        vals = [float(points[j][metric_key]) for j in range(left, i + 1)]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _plot_changed_vs_unchanged(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    bucket_type: str,
    rolling_days: int,
    metrics: Sequence[str],
) -> None:
    group_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_to_rows[str(row["group"])].append(row)

    if bucket_type == "date":
        x_transform = lambda x: datetime.strptime(x, "%Y-%m-%d")
    else:
        x_transform = lambda x: int(x.split("_", 1)[1]) if x.startswith("cp_") else x

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    metric_to_ax = {"exact": axes[0], "f1": axes[1]}
    metric_to_col = {"exact": "exact_rate", "f1": "value_f1_mean"}
    metric_to_label = {"exact": "Exact Rate", "f1": "Value F1 Mean"}
    colors = {"changed": "#d62728", "unchanged": "#1f77b4"}

    for metric in metrics:
        if metric not in metric_to_ax:
            continue
        ax = metric_to_ax[metric]
        col = metric_to_col[metric]
        for group in ("changed", "unchanged"):
            points = sorted(group_to_rows.get(group, []), key=lambda r: r["bucket"])
            if not points:
                continue
            xs = [x_transform(str(p["bucket"])) for p in points]
            ys = [float(p[col]) for p in points]
            ys_roll = _rolling_for_group(
                points,
                metric_key=col,
                rolling_days=rolling_days,
                bucket_type=bucket_type,
            )
            ax.plot(xs, ys, linestyle="--", alpha=0.35, color=colors[group], label=f"{group} raw")
            ax.plot(xs, ys_roll, linestyle="-", color=colors[group], label=f"{group} rolling")
        ax.set_ylabel(metric_to_label[metric])
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Date" if bucket_type == "date" else "Checkpoint Index")
    fig.suptitle("DSP Item Trend: Changed vs Unchanged")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_keys(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    bucket_type: str,
    metrics: Sequence[str],
) -> None:
    top_keys = sorted({str(r["key"]) for r in rows})
    if not top_keys:
        return

    if bucket_type == "date":
        x_transform = lambda x: datetime.strptime(x, "%Y-%m-%d")
    else:
        x_transform = lambda x: int(x.split("_", 1)[1]) if x.startswith("cp_") else x

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    metric_to_ax = {"exact": axes[0], "f1": axes[1]}
    metric_to_col = {"exact": "exact_rate", "f1": "value_f1_mean"}
    metric_to_label = {"exact": "Exact Rate", "f1": "Value F1 Mean"}

    for metric in metrics:
        if metric not in metric_to_ax:
            continue
        ax = metric_to_ax[metric]
        col = metric_to_col[metric]
        for key in top_keys:
            points = sorted((r for r in rows if str(r["key"]) == key), key=lambda r: r["bucket"])
            if not points:
                continue
            xs = [x_transform(str(p["bucket"])) for p in points]
            ys = [float(p[col]) for p in points]
            ax.plot(xs, ys, linewidth=1.5, alpha=0.9, label=key)
        ax.set_ylabel(metric_to_label[metric])
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date" if bucket_type == "date" else "Checkpoint Index")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("DSP Item Trend: Top Keys")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DSP item-level trends and plot changed/unchanged dynamics.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-change-mode", type=str, default="first_seen", choices=["first_seen"])
    parser.add_argument("--metrics", type=str, default="exact,f1", help="comma separated metrics in {exact,f1}")
    parser.add_argument("--rolling-days", type=int, default=7)
    parser.add_argument("--top-n-keys", type=int, default=12)
    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    metrics = [m for m in metrics if m in {"exact", "f1"}]
    if not metrics:
        raise ValueError("No valid metrics selected. Use --metrics exact,f1")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    raw_pred = json.loads(args.prediction.read_text(encoding="utf-8"))
    aligned_pred, align_report = align_predictions_to_benchmark(benchmark, raw_pred)
    pred_by_id = normalize_predictions(aligned_pred)

    item_rows: List[Dict[str, Any]] = []
    first_seen_by_key: Dict[str, Any] = {}
    missing_prediction_count = 0

    for cp_idx, checkpoint in enumerate(benchmark.get("checkpoints", []), start=1):
        checkpoint_id = str(checkpoint.get("checkpoint_id"))
        prediction = pred_by_id.get(checkpoint_id)
        if prediction is None:
            missing_prediction_count += 1
            continue

        cp_ts = str((checkpoint.get("as_of") or {}).get("timestamp", "")).strip()
        cp_dt = parse_ts(cp_ts) if cp_ts else datetime.max
        cp_date = cp_dt.strftime("%Y-%m-%d") if cp_dt != datetime.max else ""

        expected_snapshot = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
        expected_snapshot = {k: drop_excluded_fields(v) for k, v in expected_snapshot.items()}
        predicted_snapshot = flatten_snapshot(prediction.get("snapshot_state") or {})
        predicted_snapshot = {k: drop_excluded_fields(v) for k, v in predicted_snapshot.items()}

        predict_per_key = bool((prediction.get("metadata") or {}).get("predict_per_key", False))
        for key in sorted(expected_snapshot.keys()):
            expected_value = expected_snapshot.get(key)
            predicted_value = predicted_snapshot.get(key, None)

            if key not in first_seen_by_key:
                first_seen_by_key[key] = expected_value
                changed_from_first = False
            else:
                changed_from_first = expected_value != first_seen_by_key[key]

            item_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_index": cp_idx,
                    "checkpoint_timestamp": cp_ts,
                    "date": cp_date,
                    "key": key,
                    "changed_from_first": int(changed_from_first),
                    "change_group": "changed" if changed_from_first else "unchanged",
                    "value_exact": 1.0 if expected_value == predicted_value else 0.0,
                    "value_f1": float(value_f1(expected_value, predicted_value)),
                    "expected_value_json": _json_cell(expected_value),
                    "predicted_value_json": _json_cell(predicted_value),
                    "predict_per_key": int(predict_per_key),
                }
            )

    if not item_rows:
        raise RuntimeError("No item rows were generated. Check benchmark/prediction alignment and file contents.")

    use_date_bucket = all(str(r.get("date", "")).strip() for r in item_rows)
    bucket_type = "date" if use_date_bucket else "checkpoint_index"
    if not use_date_bucket:
        print("[WARN] Missing/invalid checkpoint timestamp detected. Fallback to checkpoint-index bucket.")

    for row in item_rows:
        row["bucket_type"] = bucket_type
        row["bucket"] = str(row["date"]) if bucket_type == "date" else f"cp_{int(row['checkpoint_index']):04d}"

    changed_daily_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in item_rows:
        key = (str(row["bucket"]), str(row["change_group"]))
        acc = changed_daily_map.setdefault(
            key,
            {"item_count": 0.0, "exact_sum": 0.0, "f1_sum": 0.0},
        )
        acc["item_count"] += 1.0
        acc["exact_sum"] += float(row["value_exact"])
        acc["f1_sum"] += float(row["value_f1"])

    sorted_buckets = _sort_buckets((k[0] for k in changed_daily_map.keys()), bucket_type)
    changed_vs_unchanged_daily: List[Dict[str, Any]] = []
    for group in ("changed", "unchanged"):
        group_rows: List[Dict[str, Any]] = []
        for bucket in sorted_buckets:
            acc = changed_daily_map.get((bucket, group))
            if acc is None:
                continue
            count = acc["item_count"]
            group_rows.append(
                {
                    "bucket": bucket,
                    "bucket_type": bucket_type,
                    "group": group,
                    "item_count": int(count),
                    "exact_rate": acc["exact_sum"] / count if count else 0.0,
                    "value_f1_mean": acc["f1_sum"] / count if count else 0.0,
                }
            )
        roll_exact = _rolling_for_group(
            group_rows,
            metric_key="exact_rate",
            rolling_days=args.rolling_days,
            bucket_type=bucket_type,
        )
        roll_f1 = _rolling_for_group(
            group_rows,
            metric_key="value_f1_mean",
            rolling_days=args.rolling_days,
            bucket_type=bucket_type,
        )
        for i, row in enumerate(group_rows):
            row["exact_rate_rolling"] = roll_exact[i]
            row["value_f1_mean_rolling"] = roll_f1[i]
        changed_vs_unchanged_daily.extend(group_rows)

    key_daily_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    key_counter = Counter()
    for row in item_rows:
        key_counter[str(row["key"])] += 1
        k = (str(row["bucket"]), str(row["key"]))
        acc = key_daily_map.setdefault(
            k,
            {"item_count": 0.0, "exact_sum": 0.0, "f1_sum": 0.0},
        )
        acc["item_count"] += 1.0
        acc["exact_sum"] += float(row["value_exact"])
        acc["f1_sum"] += float(row["value_f1"])

    key_daily_all: List[Dict[str, Any]] = []
    for (bucket, key), acc in key_daily_map.items():
        count = acc["item_count"]
        key_daily_all.append(
            {
                "bucket": bucket,
                "bucket_type": bucket_type,
                "key": key,
                "item_count": int(count),
                "exact_rate": acc["exact_sum"] / count if count else 0.0,
                "value_f1_mean": acc["f1_sum"] / count if count else 0.0,
            }
        )

    key_daily_all.sort(key=lambda x: (x["key"], x["bucket"]))
    grouped_key_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in key_daily_all:
        grouped_key_rows[str(row["key"])].append(row)
    for key, rows in grouped_key_rows.items():
        roll_exact = _rolling_for_group(
            rows,
            metric_key="exact_rate",
            rolling_days=args.rolling_days,
            bucket_type=bucket_type,
        )
        roll_f1 = _rolling_for_group(
            rows,
            metric_key="value_f1_mean",
            rolling_days=args.rolling_days,
            bucket_type=bucket_type,
        )
        for i, row in enumerate(rows):
            row["exact_rate_rolling"] = roll_exact[i]
            row["value_f1_mean_rolling"] = roll_f1[i]

    top_keys = [k for k, _ in key_counter.most_common(max(1, args.top_n_keys))]
    key_daily_top = [row for row in key_daily_all if str(row["key"]) in set(top_keys)]
    key_daily_top.sort(key=lambda x: (x["key"], x["bucket"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_csv = args.output_dir / "item_level_metrics.csv"
    changed_csv = args.output_dir / "changed_vs_unchanged_daily.csv"
    key_csv = args.output_dir / "key_daily_metrics.csv"
    changed_png = args.output_dir / "changed_vs_unchanged_trend.png"
    key_png = args.output_dir / "top_keys_trend.png"
    summary_json = args.output_dir / "analysis_summary.json"

    _write_csv(
        item_csv,
        item_rows,
        [
            "checkpoint_id",
            "checkpoint_index",
            "checkpoint_timestamp",
            "date",
            "bucket",
            "bucket_type",
            "key",
            "changed_from_first",
            "change_group",
            "value_exact",
            "value_f1",
            "expected_value_json",
            "predicted_value_json",
            "predict_per_key",
        ],
    )
    _write_csv(
        changed_csv,
        changed_vs_unchanged_daily,
        [
            "bucket",
            "bucket_type",
            "group",
            "item_count",
            "exact_rate",
            "value_f1_mean",
            "exact_rate_rolling",
            "value_f1_mean_rolling",
        ],
    )
    _write_csv(
        key_csv,
        key_daily_top,
        [
            "bucket",
            "bucket_type",
            "key",
            "item_count",
            "exact_rate",
            "value_f1_mean",
            "exact_rate_rolling",
            "value_f1_mean_rolling",
        ],
    )

    _plot_changed_vs_unchanged(
        changed_vs_unchanged_daily,
        changed_png,
        bucket_type=bucket_type,
        rolling_days=max(1, args.rolling_days),
        metrics=metrics,
    )
    _plot_top_keys(
        key_daily_top,
        key_png,
        bucket_type=bucket_type,
        metrics=metrics,
    )

    summary = {
        "benchmark_path": str(args.benchmark),
        "prediction_path": str(args.prediction),
        "output_dir": str(args.output_dir),
        "align_report": align_report,
        "group_change_mode": args.group_change_mode,
        "bucket_type": bucket_type,
        "metrics": metrics,
        "rolling_days": args.rolling_days,
        "top_n_keys": args.top_n_keys,
        "item_rows": len(item_rows),
        "missing_prediction_checkpoints": missing_prediction_count,
        "top_keys": top_keys,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", item_csv)
    print("Saved:", changed_csv)
    print("Saved:", key_csv)
    print("Saved:", changed_png)
    print("Saved:", key_png)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
