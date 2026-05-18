#!/usr/bin/env python3
"""Analyze TCE predictions at item level and plot temporal trends."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from bench_core.tce_evaluator import align_predictions_to_benchmark
from tce_core.evaluation import flatten_snapshot, normalize_predictions, value_f1
from tce_core.pipeline import drop_excluded_fields, parse_ts


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


def _format_token_label(value: Any) -> str:
    try:
        n = int(value)
    except Exception:
        return ""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _plot_changed_vs_unchanged(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    bucket_type: str,
    rolling_days: int,
    metrics: Sequence[str],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
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
            if metric == "exact" and group == "changed":
                for x, y, p in zip(xs, ys, points):
                    token_label = _format_token_label(p.get("actual_tokens_at_cutoff", ""))
                    if token_label:
                        ax.annotate(
                            token_label,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            fontsize=7,
                            alpha=0.8,
                        )
        ax.set_ylabel(metric_to_label[metric])
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Date" if bucket_type == "date" else "Checkpoint Index")
    fig.suptitle("TCE Item Trend: Changed vs Unchanged")
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
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
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
    fig.suptitle("TCE Item Trend: Top Keys")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TCE item-level trends and plot changed/unchanged dynamics.")
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
    sampled_id_by_timestamp: Dict[str, str] = {}
    exposure_percent_by_timestamp: Dict[str, int] = {}
    sampling_mode_by_timestamp: Dict[str, str] = {}
    anchor_timestamp_by_timestamp: Dict[str, str] = {}
    actual_tokens_by_timestamp: Dict[str, int] = {}
    total_tokens_by_timestamp: Dict[str, int] = {}
    for p in raw_pred.get("predictions", []):
        if not isinstance(p, dict):
            continue
        m = p.get("metadata") or {}
        cp_ts_raw = str(m.get("checkpoint_timestamp", "")).strip()
        if not cp_ts_raw:
            continue
        sampled_id_by_timestamp[cp_ts_raw] = str(p.get("checkpoint_id", "")).strip() or cp_ts_raw
        sampling_mode_by_timestamp[cp_ts_raw] = str(m.get("sampling_mode", "")).strip()
        sampling_params = m.get("sampling_params") or {}
        exp_pct = sampling_params.get("exposure_percent")
        if exp_pct is not None:
            exposure_percent_by_timestamp[cp_ts_raw] = int(exp_pct)
        anchor_ts = sampling_params.get("anchor_timestamp")
        if anchor_ts is not None:
            anchor_timestamp_by_timestamp[cp_ts_raw] = str(anchor_ts)
        tok = sampling_params.get("actual_tokens_at_cutoff")
        if tok is not None:
            actual_tokens_by_timestamp[cp_ts_raw] = int(tok)
        total_tok = sampling_params.get("total_tokens")
        if total_tok is not None:
            total_tokens_by_timestamp[cp_ts_raw] = int(total_tok)

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
        pred_meta = prediction.get("metadata") or {}
        sampled_checkpoint_id = str(
            pred_meta.get("sampled_checkpoint_id")
            or sampled_id_by_timestamp.get(cp_ts)
            or checkpoint_id
        )
        exposure_percent = exposure_percent_by_timestamp.get(cp_ts)
        sampling_mode = str(pred_meta.get("sampling_mode") or sampling_mode_by_timestamp.get(cp_ts) or "")
        pred_sampling_params = pred_meta.get("sampling_params") if isinstance(pred_meta.get("sampling_params"), dict) else {}
        anchor_timestamp = str(
            pred_sampling_params.get("anchor_timestamp")
            or anchor_timestamp_by_timestamp.get(cp_ts)
            or ""
        )
        actual_tokens_at_cutoff = pred_sampling_params.get("actual_tokens_at_cutoff")
        if actual_tokens_at_cutoff is None:
            actual_tokens_at_cutoff = actual_tokens_by_timestamp.get(cp_ts)
        total_tokens = pred_sampling_params.get("total_tokens")
        if total_tokens is None:
            total_tokens = total_tokens_by_timestamp.get(cp_ts)
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
                    "sampled_checkpoint_id": sampled_checkpoint_id,
                    "checkpoint_index": cp_idx,
                    "checkpoint_timestamp": cp_ts,
                    "date": cp_date,
                    "exposure_percent": exposure_percent if exposure_percent is not None else "",
                    "sampling_mode": sampling_mode,
                    "anchor_timestamp": anchor_timestamp,
                    "actual_tokens_at_cutoff": actual_tokens_at_cutoff if actual_tokens_at_cutoff is not None else "",
                    "total_tokens": total_tokens if total_tokens is not None else "",
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
    bucket_token_map: Dict[str, Dict[str, Any]] = {}
    for row in item_rows:
        key = (str(row["bucket"]), str(row["change_group"]))
        acc = changed_daily_map.setdefault(
            key,
            {"item_count": 0.0, "exact_sum": 0.0, "f1_sum": 0.0},
        )
        acc["item_count"] += 1.0
        acc["exact_sum"] += float(row["value_exact"])
        acc["f1_sum"] += float(row["value_f1"])
        bucket = str(row["bucket"])
        if bucket not in bucket_token_map:
            bucket_token_map[bucket] = {
                "actual_tokens_at_cutoff": row.get("actual_tokens_at_cutoff", ""),
                "total_tokens": row.get("total_tokens", ""),
            }

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
                    "actual_tokens_at_cutoff": bucket_token_map.get(bucket, {}).get("actual_tokens_at_cutoff", ""),
                    "total_tokens": bucket_token_map.get(bucket, {}).get("total_tokens", ""),
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

    checkpoint_change_map: Dict[str, Dict[str, Any]] = {}
    for row in item_rows:
        checkpoint_id = str(row["checkpoint_id"])
        entry = checkpoint_change_map.setdefault(
            checkpoint_id,
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_index": int(row["checkpoint_index"]),
                "sampled_checkpoint_id": str(row.get("sampled_checkpoint_id", "")),
                "checkpoint_timestamp": str(row["checkpoint_timestamp"]),
                "date": str(row["date"]),
                "exposure_percent": row.get("exposure_percent", ""),
                "sampling_mode": row.get("sampling_mode", ""),
                "anchor_timestamp": row.get("anchor_timestamp", ""),
                "actual_tokens_at_cutoff": row.get("actual_tokens_at_cutoff", ""),
                "total_tokens": row.get("total_tokens", ""),
                "total_items": 0,
                "changed_items": 0,
                "changed_keys": [],
            },
        )
        entry["total_items"] += 1
        if int(row["changed_from_first"]) == 1:
            entry["changed_items"] += 1
            entry["changed_keys"].append(str(row["key"]))

    checkpoint_change_rows: List[Dict[str, Any]] = []
    for cid, entry in checkpoint_change_map.items():
        total = int(entry["total_items"])
        changed = int(entry["changed_items"])
        changed_keys = sorted(set(entry["changed_keys"]))
        checkpoint_change_rows.append(
            {
                "checkpoint_id": cid,
                "sampled_checkpoint_id": str(entry.get("sampled_checkpoint_id", "")),
                "checkpoint_index": int(entry["checkpoint_index"]),
                "checkpoint_timestamp": str(entry["checkpoint_timestamp"]),
                "date": str(entry["date"]),
                "exposure_percent": entry.get("exposure_percent", ""),
                "sampling_mode": entry.get("sampling_mode", ""),
                "anchor_timestamp": entry.get("anchor_timestamp", ""),
                "actual_tokens_at_cutoff": entry.get("actual_tokens_at_cutoff", ""),
                "total_tokens": entry.get("total_tokens", ""),
                "total_items": total,
                "changed_items": changed,
                "changed_ratio": (changed / total) if total else 0.0,
                "has_any_change": 1 if changed > 0 else 0,
                "changed_keys_json": _json_cell(changed_keys),
            }
        )
    checkpoint_change_rows.sort(key=lambda x: int(x["checkpoint_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_csv = args.output_dir / "item_level_metrics.csv"
    changed_csv = args.output_dir / "changed_vs_unchanged_daily.csv"
    key_csv = args.output_dir / "key_daily_metrics.csv"
    checkpoint_change_csv = args.output_dir / "checkpoint_change_summary.csv"
    changed_png = args.output_dir / "changed_vs_unchanged_trend.png"
    key_png = args.output_dir / "top_keys_trend.png"
    summary_json = args.output_dir / "analysis_summary.json"

    _write_csv(
        item_csv,
        item_rows,
        [
            "checkpoint_id",
            "sampled_checkpoint_id",
            "checkpoint_index",
            "checkpoint_timestamp",
            "date",
            "exposure_percent",
            "sampling_mode",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
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
            "actual_tokens_at_cutoff",
            "total_tokens",
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
    _write_csv(
        checkpoint_change_csv,
        checkpoint_change_rows,
        [
            "checkpoint_id",
            "sampled_checkpoint_id",
            "checkpoint_index",
            "checkpoint_timestamp",
            "date",
            "exposure_percent",
            "sampling_mode",
            "anchor_timestamp",
            "actual_tokens_at_cutoff",
            "total_tokens",
            "total_items",
            "changed_items",
            "changed_ratio",
            "has_any_change",
            "changed_keys_json",
        ],
    )

    if plt is not None:
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
    else:
        print("[WARN] matplotlib is not installed; skipped trend PNG generation.")

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
    print("Saved:", checkpoint_change_csv)
    if plt is not None:
        print("Saved:", changed_png)
        print("Saved:", key_png)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
