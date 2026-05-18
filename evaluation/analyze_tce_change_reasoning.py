#!/usr/bin/env python3
"""Analyze change-reasoning outputs on exposure checkpoints."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tce_core.evaluation import flatten_snapshot, value_f1


def _drop_excluded_fields(value: Any) -> Any:
    excluded = {"priority", "schedule_date", "schedule_dates"}
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in excluded:
                continue
            out[k] = _drop_excluded_fields(v)
        return out
    if isinstance(value, list):
        return [_drop_excluded_fields(v) for v in value]
    return value


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TCE change-reasoning quality.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    pred_payload = json.loads(args.prediction.read_text(encoding="utf-8"))
    pred_by_id = {str(x.get("checkpoint_id")): x for x in pred_payload.get("predictions", []) if isinstance(x, dict)}

    checkpoints = benchmark.get("checkpoints", [])
    cp_rows: List[Dict[str, Any]] = []
    prev_snapshot: Dict[str, Any] = {}
    prev_cid = None
    prev_ts = None

    for i, cp in enumerate(checkpoints):
        cid = str(cp.get("checkpoint_id"))
        ts = str((cp.get("as_of") or {}).get("timestamp", ""))
        date = ts.split(" ")[0] if ts else ""
        cur_snapshot = flatten_snapshot(cp.get("expected_snapshot_state") or {})
        cur_snapshot = {k: _drop_excluded_fields(v) for k, v in cur_snapshot.items()}
        pred = pred_by_id.get(cid, {})
        change_analysis = pred.get("change_analysis") or {}

        if i == 0:
            prev_snapshot = cur_snapshot
            prev_cid = cid
            prev_ts = ts
            continue

        changed_keys = [k for k in sorted(set(prev_snapshot.keys()) | set(cur_snapshot.keys())) if prev_snapshot.get(k) != cur_snapshot.get(k)]
        for key in changed_keys:
            item = change_analysis.get(key) if isinstance(change_analysis, dict) else None
            if not isinstance(item, dict):
                item = {"before": None, "after": None, "change_reason": "", "evidence": []}
            pred_before = _drop_excluded_fields(item.get("before"))
            pred_after = _drop_excluded_fields(item.get("after"))
            pred_reason = str(item.get("change_reason", item.get("reason", "")) or "")
            pred_evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []

            gt_before = prev_snapshot.get(key)
            gt_after = cur_snapshot.get(key)
            before_exact = 1.0 if pred_before == gt_before else 0.0
            after_exact = 1.0 if pred_after == gt_after else 0.0
            cp_rows.append(
                {
                    "checkpoint_id": cid,
                    "date": date,
                    "prev_checkpoint_id": prev_cid or "",
                    "prev_timestamp": prev_ts or "",
                    "key": key,
                    "before_exact": before_exact,
                    "after_exact": after_exact,
                    "both_exact": 1.0 if (before_exact == 1.0 and after_exact == 1.0) else 0.0,
                    "before_f1": float(value_f1(gt_before, pred_before)),
                    "after_f1": float(value_f1(gt_after, pred_after)),
                    "reason_nonempty": 1.0 if pred_reason.strip() else 0.0,
                    "evidence_nonempty": 1.0 if len(pred_evidence) > 0 else 0.0,
                }
            )
        prev_snapshot = cur_snapshot
        prev_cid = cid
        prev_ts = ts

    if not cp_rows:
        raise RuntimeError("No changed keys found for change reasoning analysis.")

    daily = defaultdict(lambda: defaultdict(float))
    for r in cp_rows:
        k = r["date"] or r["checkpoint_id"]
        daily[k]["n"] += 1.0
        for m in ["before_exact", "after_exact", "both_exact", "before_f1", "after_f1", "reason_nonempty", "evidence_nonempty"]:
            daily[k][m] += float(r[m])

    daily_rows: List[Dict[str, Any]] = []
    for k in sorted(daily.keys()):
        n = daily[k]["n"]
        row = {"bucket": k, "item_count": int(n)}
        for m in ["before_exact", "after_exact", "both_exact", "before_f1", "after_f1", "reason_nonempty", "evidence_nonempty"]:
            row[m] = daily[k][m] / n if n else 0.0
        daily_rows.append(row)

    out_dir = args.output_dir
    item_csv = out_dir / "change_reasoning_item_metrics.csv"
    daily_csv = out_dir / "change_reasoning_daily.csv"
    fig_path = out_dir / "change_reasoning_trend.png"

    _write_csv(
        item_csv,
        cp_rows,
        [
            "checkpoint_id",
            "date",
            "prev_checkpoint_id",
            "prev_timestamp",
            "key",
            "before_exact",
            "after_exact",
            "both_exact",
            "before_f1",
            "after_f1",
            "reason_nonempty",
            "evidence_nonempty",
        ],
    )
    _write_csv(
        daily_csv,
        daily_rows,
        [
            "bucket",
            "item_count",
            "before_exact",
            "after_exact",
            "both_exact",
            "before_f1",
            "after_f1",
            "reason_nonempty",
            "evidence_nonempty",
        ],
    )

    xs = [datetime.strptime(r["bucket"], "%Y-%m-%d") if "-" in r["bucket"] else r["bucket"] for r in daily_rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xs, [r["before_exact"] for r in daily_rows], label="before_exact", marker="o")
    axes[0].plot(xs, [r["after_exact"] for r in daily_rows], label="after_exact", marker="o")
    axes[0].plot(xs, [r["both_exact"] for r in daily_rows], label="both_exact", marker="o")
    axes[0].plot(xs, [r["before_f1"] for r in daily_rows], label="before_f1", linestyle="--")
    axes[0].plot(xs, [r["after_f1"] for r in daily_rows], label="after_f1", linestyle="--")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_ylabel("Value Accuracy")

    axes[1].plot(xs, [r["reason_nonempty"] for r in daily_rows], label="reason_nonempty", marker="o")
    axes[1].plot(xs, [r["evidence_nonempty"] for r in daily_rows], label="evidence_nonempty", marker="o")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_ylabel("Coverage")
    axes[1].set_xlabel("Date")

    fig.suptitle("Change Reasoning Quality Over Time")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary = {
        "benchmark": str(args.benchmark),
        "prediction": str(args.prediction),
        "changed_item_rows": len(cp_rows),
        "daily_buckets": len(daily_rows),
        "outputs": {
            "item_csv": str(item_csv),
            "daily_csv": str(daily_csv),
            "figure": str(fig_path),
        },
    }
    (out_dir / "change_reasoning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {item_csv}")
    print(f"Saved: {daily_csv}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {out_dir / 'change_reasoning_summary.json'}")


if __name__ == "__main__":
    main()
