#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def empty_pred(checkpoint_id: str) -> Dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "snapshot_state": {
            "user_attributes_state": {},
            "habits_state": {},
            "preferences_state": {},
        },
        "delta_prediction": {
            "added": {},
            "updated": {},
            "removed": {},
        },
        "uncertainty": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo predictor for state abstraction API.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["oracle", "empty"],
        default="oracle",
        help="oracle: copy targets in tasks (for pipeline demo only); empty: output empty baseline",
    )
    args = parser.parse_args()

    payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks: List[Dict[str, Any]] = payload.get("tasks", [])

    preds: List[Dict[str, Any]] = []
    for t in tasks:
        cid = str(t.get("checkpoint_id") or t.get("task_id"))
        if args.mode == "oracle":
            target = t.get("target") or {}
            pred = {
                "checkpoint_id": cid,
                "snapshot_state": target.get("expected_snapshot_state", {
                    "user_attributes_state": {},
                    "habits_state": {},
                    "preferences_state": {},
                }),
                "delta_prediction": target.get("expected_delta_from_previous", {
                    "added": {},
                    "updated": {},
                    "removed": {},
                }),
                "uncertainty": {},
            }
        else:
            pred = empty_pred(cid)
        preds.append(pred)

    out = {"predictions": preds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved predictions: {args.output}")
    print(f"Predictions: {len(preds)}")


if __name__ == "__main__":
    main()
