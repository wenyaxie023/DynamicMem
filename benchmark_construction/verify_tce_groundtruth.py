#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.max


def flatten_snapshot(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in snapshot.keys()):
        return {str(k): v for k, v in snapshot.items()}
    out: Dict[str, Any] = {}
    for category, content in snapshot.items():
        if isinstance(content, dict):
            for name, value in content.items():
                out[f"{category}:{name}"] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify tce_benchmark observability consistency.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-final", type=Path, required=True)
    parser.add_argument("--max-report", type=int, default=5)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    app_raw = json.loads(args.app_logs_final.read_text(encoding="utf-8"))
    logs = app_raw.get("app_logs", []) if isinstance(app_raw, dict) else app_raw
    logs = sorted(
        [x for x in logs if isinstance(x, dict)],
        key=lambda x: (parse_ts(str(x.get("timestamp", "9999-12-31 23:59:59"))), str(x.get("app_log_id", ""))),
    )

    current: Dict[str, Any] = {}
    prefix_states = []
    for log in logs:
        for ev in log.get("golden_evidence", []):
            sc = ev.get("state_category")
            sn = ev.get("state_name")
            if not sc or not sn:
                continue
            k = f"{sc}:{sn}"
            change_type = str(ev.get("change_type", "unchanged")).lower()
            if change_type in {"drop", "remove", "deleted"}:
                current.pop(k, None)
            elif "state_value" in ev:
                current[k] = ev["state_value"]
        prefix_states.append(dict(current))

    snapshot_mismatch = 0
    obs_time_violation = 0
    reported = 0

    for cp in benchmark.get("checkpoints", []):
        as_of = cp.get("as_of", {}) or {}
        idx = as_of.get("log_index")
        cp_id = cp.get("checkpoint_id")
        if not isinstance(idx, int) or idx < 0 or idx >= len(prefix_states):
            continue

        expected = flatten_snapshot(cp.get("expected_snapshot_state") or {})
        built = prefix_states[idx]
        if expected != built:
            snapshot_mismatch += 1
            if reported < args.max_report:
                print(f"[MISMATCH] {cp_id}: expected_keys={len(expected)} built_keys={len(built)}")
                reported += 1

        cp_ts = parse_ts(str(as_of.get("timestamp", "9999-12-31 23:59:59")))
        obs = cp.get("state_observability", {}) or {}
        for cat, content in obs.items():
            if not isinstance(content, dict):
                continue
            for name, meta in content.items():
                last_ts = parse_ts(str((meta or {}).get("last_timestamp", "9999-12-31 23:59:59")))
                if last_ts > cp_ts:
                    obs_time_violation += 1
                    if reported < args.max_report:
                        print(f"[OBS_VIOLATION] {cp_id} {cat}:{name} last_ts>{as_of.get('timestamp')}")
                        reported += 1

    print(
        json.dumps(
            {
                "user_id": benchmark.get("user_id"),
                "checkpoints": len(benchmark.get("checkpoints", [])),
                "snapshot_mismatch": snapshot_mismatch,
                "obs_time_violation": obs_time_violation,
                "status": "ok" if snapshot_mismatch == 0 and obs_time_violation == 0 else "issues",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
