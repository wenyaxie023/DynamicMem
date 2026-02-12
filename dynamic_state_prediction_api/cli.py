#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

from .service import DynamicStatePredictionAPI


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic State Prediction API CLI")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List checkpoints")
    p_list.add_argument("--benchmark", type=Path, required=True)

    p_export = sub.add_parser("export", help="Export tasks for baseline runners")
    p_export.add_argument("--benchmark", type=Path, required=True)
    p_export.add_argument("--app-logs", type=Path, required=True, help="Path to app_log_large.json")
    p_export.add_argument("--mode", choices=["checkpoint", "date"], required=True)
    p_export.add_argument("--checkpoint-ids", type=str, default="", help="Comma-separated checkpoint ids")
    p_export.add_argument("--dates", type=str, default="", help="Comma-separated dates: YYYY-MM-DD")
    p_export.add_argument("--output", type=Path, required=True, help="Output tasks json")
    p_export.add_argument("--subset-benchmark-output", type=Path, default=None, help="Optional subset benchmark output")
    p_export.add_argument("--include-targets", action="store_true", help="Include GT in exported tasks (debug only)")

    p_eval = sub.add_parser("evaluate", help="Evaluate predictions")
    p_eval.add_argument("--benchmark", type=Path, required=True)
    p_eval.add_argument("--prediction", type=Path, required=True)
    p_eval.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    api = DynamicStatePredictionAPI(args.benchmark)

    if args.command == "list":
        rows = api.list_checkpoints()
        print(json.dumps({"count": len(rows), "checkpoints": rows}, ensure_ascii=False, indent=2))
        return

    if args.command == "export":
        if args.mode == "checkpoint":
            ids = _parse_csv(args.checkpoint_ids)
            if not ids:
                raise ValueError("--checkpoint-ids is required for --mode checkpoint")
            checkpoints = api.select_checkpoints_by_id(ids)
        else:
            dates = _parse_csv(args.dates)
            if not dates:
                raise ValueError("--dates is required for --mode date")
            checkpoints = api.select_checkpoints_by_date(dates)

        payload = api.export_tasks(
            checkpoints=checkpoints,
            app_logs_path=args.app_logs,
            output_path=args.output,
            include_targets=args.include_targets,
        )
        print(f"Saved tasks: {args.output}")
        print(f"Tasks: {payload.get('task_count', 0)}")

        if args.subset_benchmark_output is not None:
            api.export_subset_benchmark(checkpoints=checkpoints, output_path=args.subset_benchmark_output)
            print(f"Saved subset benchmark: {args.subset_benchmark_output}")
        return

    if args.command == "evaluate":
        result = api.evaluate_predictions(prediction_path=args.prediction, output_path=args.output)
        print(f"Saved eval: {args.output}")
        print(
            "Summary: "
            f"evaluated={result.get('evaluated_checkpoints', 0)}, "
            f"value_f1={result.get('summary', {}).get('snapshot_value_f1_mean_on_expected_mean', 0.0):.4f}, "
            f"evidence_recall={result.get('summary', {}).get('snapshot_evidence_recall_mean_on_expected_mean', 0.0):.4f}"
        )
        return


if __name__ == "__main__":
    main()
