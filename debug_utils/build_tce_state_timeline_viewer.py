#!/usr/bin/env python3.11
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.state_timeline_viewer import build_viewer_payload, load_json, write_viewer_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build viewer JSON for the TCE state timeline page.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, required=True)
    parser.add_argument("--app-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write a compact viewer payload without large request/response/prompt/raw blocks.",
    )
    args = parser.parse_args()

    payload = build_viewer_payload(
        benchmark_payload=load_json(args.benchmark),
        prediction_payload=load_json(args.prediction),
        eval_payload=load_json(args.eval_path),
        app_logs_payload=load_json(args.app_logs),
        source_files={
            "benchmark": str(args.benchmark),
            "prediction": str(args.prediction),
            "eval": str(args.eval_path),
            "app_logs": str(args.app_logs),
        },
        compact=args.compact,
    )
    write_viewer_payload(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
