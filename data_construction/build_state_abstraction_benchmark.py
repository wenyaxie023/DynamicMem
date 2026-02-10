#!/usr/bin/env python3
"""
Build state-abstraction checkpoints from app_logs_final.json.

This script creates evaluation checkpoints at chain-completion time points:
- snapshot_state: full reconstructed user state at this time point
- delta_from_previous: state changes relative to previous checkpoint
- observability: evidence-count metadata per state item
"""

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {ts}") from exc


def _state_key(state_category: str, state_name: str) -> str:
    return f"{state_category}:{state_name}"


def _split_state_key(key: str) -> Tuple[str, str]:
    if ":" not in key:
        return "unknown_state_category", key
    return key.split(":", 1)


def _expand_snapshot(flat_snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    expanded: Dict[str, Dict[str, Any]] = {}
    for key, value in flat_snapshot.items():
        cat, name = _split_state_key(key)
        expanded.setdefault(cat, {})[name] = value
    return expanded


def _expand_observability(flat_obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    expanded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in flat_obs.items():
        cat, name = _split_state_key(key)
        expanded.setdefault(cat, {})[name] = value
    return expanded


def _expand_delta(flat_delta: Dict[str, Any]) -> Dict[str, Any]:
    added_expanded: Dict[str, Dict[str, Any]] = {}
    for key, value in (flat_delta.get("added") or {}).items():
        cat, name = _split_state_key(key)
        added_expanded.setdefault(cat, {})[name] = value

    updated_expanded: Dict[str, Dict[str, Any]] = {}
    for key, value in (flat_delta.get("updated") or {}).items():
        cat, name = _split_state_key(key)
        updated_expanded.setdefault(cat, {})[name] = value

    removed_expanded: Dict[str, List[str]] = {}
    for key in (flat_delta.get("removed") or []):
        cat, name = _split_state_key(str(key))
        removed_expanded.setdefault(cat, []).append(name)
    for cat in removed_expanded:
        removed_expanded[cat] = sorted(removed_expanded[cat])

    return {
        "added": added_expanded,
        "updated": updated_expanded,
        "removed": removed_expanded,
    }


def _compute_delta(
    prev_snapshot: Dict[str, Any], curr_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    prev_keys = set(prev_snapshot.keys())
    curr_keys = set(curr_snapshot.keys())

    added = {k: curr_snapshot[k] for k in sorted(curr_keys - prev_keys)}
    removed = sorted(prev_keys - curr_keys)

    updated: Dict[str, Dict[str, Any]] = {}
    for key in sorted(prev_keys & curr_keys):
        if prev_snapshot[key] != curr_snapshot[key]:
            updated[key] = {"before": prev_snapshot[key], "after": curr_snapshot[key]}

    return {"added": added, "updated": updated, "removed": removed}


def _sort_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(log: Dict[str, Any]) -> Tuple[datetime, str]:
        ts = log.get("timestamp")
        if not ts:
            # push no-timestamp logs to the end deterministically
            return datetime.max, str(log.get("app_log_id", ""))
        return _parse_timestamp(ts), str(log.get("app_log_id", ""))

    return sorted(logs, key=sort_key)


def build_checkpoints(app_logs_final: Dict[str, Any]) -> Dict[str, Any]:
    app_logs = _sort_logs(app_logs_final.get("app_logs", []))

    chain_last_index: Dict[str, int] = {}
    chain_ids_by_index: Dict[int, List[str]] = defaultdict(list)
    for idx, log in enumerate(app_logs):
        chain_id = (log.get("metadata") or {}).get("chain_id", "")
        if chain_id:
            chain_last_index[chain_id] = idx
    for chain_id, idx in chain_last_index.items():
        chain_ids_by_index[idx].append(chain_id)

    completion_indices = sorted(chain_ids_by_index.keys())

    current_state: Dict[str, Any] = {}
    observability: Dict[str, Dict[str, Any]] = {}
    checkpoints: List[Dict[str, Any]] = []
    prev_snapshot: Dict[str, Any] = {}

    for idx, log in enumerate(app_logs):
        for ev in log.get("golden_evidence", []):
            state_category = ev.get("state_category")
            state_name = ev.get("state_name")
            if not state_category or not state_name:
                continue
            key = _state_key(state_category, state_name)
            change_type = str(ev.get("change_type", "unchanged")).lower()

            if change_type in {"drop", "remove", "deleted"}:
                current_state.pop(key, None)
            elif "state_value" in ev:
                current_state[key] = copy.deepcopy(ev["state_value"])

            info = observability.setdefault(key, {"evidence_count": 0})
            info["evidence_count"] += 1
            info["last_timestamp"] = log.get("timestamp")
            info["last_app_log_id"] = log.get("app_log_id")
            info["last_change_type"] = ev.get("change_type", "unchanged")

        if idx not in completion_indices:
            continue

        snapshot_flat = copy.deepcopy(current_state)
        delta_flat = _compute_delta(prev_snapshot, snapshot_flat)
        snapshot = _expand_snapshot(snapshot_flat)
        delta = _expand_delta(delta_flat)
        md = log.get("metadata") or {}

        checkpoints.append(
            {
                "checkpoint_id": f"cp_{len(checkpoints) + 1:04d}",
                "as_of": {
                    "log_index": idx,
                    "app_log_id": log.get("app_log_id"),
                    "timestamp": log.get("timestamp"),
                    "window_id": md.get("window_id"),
                    "domain": md.get("domain"),
                    "completed_chain_ids": sorted(chain_ids_by_index[idx]),
                },
                "expected_snapshot_state": snapshot,
                "expected_delta_from_previous": delta,
                "state_observability": _expand_observability(copy.deepcopy(observability)),
            }
        )
        prev_snapshot = snapshot_flat

    return {
        "user_id": app_logs_final.get("user_id"),
        "source_total_app_logs": len(app_logs),
        "total_chains": len(chain_last_index),
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build state-abstraction benchmark from app_logs_final.json."
    )
    parser.add_argument(
        "--app-logs-final",
        type=Path,
        required=True,
        help="Path to app_logs_final.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <same_dir>/state_abstraction_benchmark.json",
    )
    args = parser.parse_args()

    if not args.app_logs_final.exists():
        raise FileNotFoundError(f"File not found: {args.app_logs_final}")

    payload = json.loads(args.app_logs_final.read_text(encoding="utf-8"))
    benchmark = build_checkpoints(payload)

    output_path = args.output
    if output_path is None:
        output_path = args.app_logs_final.parent / "state_abstraction_benchmark.json"

    output_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved benchmark: {output_path}")
    print(
        "Summary: "
        f"user_id={benchmark.get('user_id')}, "
        f"checkpoints={benchmark.get('total_checkpoints')}, "
        f"chains={benchmark.get('total_chains')}"
    )


if __name__ == "__main__":
    main()
