#!/usr/bin/env python3
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from state_abstraction_core import evaluate_checkpoints, mean_numeric_fields, normalize_predictions


@dataclass
class CheckpointTask:
    task_id: str
    checkpoint_id: str
    as_of: Dict[str, Any]
    app_logs: List[Dict[str, Any]]


def _parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.max


def _normalize_app_logs(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        logs = raw
    elif isinstance(raw, dict):
        logs = raw.get("app_logs", [])
    else:
        logs = []
    logs = [x for x in logs if isinstance(x, dict)]
    logs.sort(
        key=lambda x: (
            _parse_ts(str(x.get("timestamp", "9999-12-31 23:59:59"))),
            str(x.get("app_log_id", "")),
        )
    )
    return logs


class StateAbstractionAPI:
    def __init__(self, benchmark_path: Path):
        self.benchmark_path = Path(benchmark_path)
        self.benchmark = json.loads(self.benchmark_path.read_text(encoding="utf-8"))
        self.checkpoints = list(self.benchmark.get("checkpoints", []))

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        out = []
        for cp in self.checkpoints:
            as_of = cp.get("as_of", {})
            out.append(
                {
                    "checkpoint_id": cp.get("checkpoint_id"),
                    "timestamp": as_of.get("timestamp"),
                    "window_id": as_of.get("window_id"),
                    "domain": as_of.get("domain"),
                    "app_log_id": as_of.get("app_log_id"),
                }
            )
        return out

    def select_checkpoints_by_id(self, checkpoint_ids: Sequence[str]) -> List[Dict[str, Any]]:
        target = {str(x) for x in checkpoint_ids}
        return [cp for cp in self.checkpoints if str(cp.get("checkpoint_id")) in target]

    def select_checkpoints_by_date(self, dates: Sequence[str]) -> List[Dict[str, Any]]:
        # one checkpoint per date: latest checkpoint <= date 23:59:59
        cps = sorted(
            self.checkpoints,
            key=lambda cp: _parse_ts(str((cp.get("as_of") or {}).get("timestamp", "9999-12-31 23:59:59"))),
        )
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for d in dates:
            cutoff = _parse_ts(f"{d} 23:59:59")
            chosen = None
            for cp in cps:
                ts = _parse_ts(str((cp.get("as_of") or {}).get("timestamp", "9999-12-31 23:59:59")))
                if ts <= cutoff:
                    chosen = cp
                else:
                    break
            if chosen is not None:
                cid = str(chosen.get("checkpoint_id"))
                if cid not in selected_ids:
                    selected_ids.add(cid)
                    selected.append(chosen)
        return selected

    def export_tasks(
        self,
        checkpoints: Sequence[Dict[str, Any]],
        app_logs_path: Path,
        output_path: Path,
        include_targets: bool = False,
    ) -> Dict[str, Any]:
        logs = _normalize_app_logs(json.loads(Path(app_logs_path).read_text(encoding="utf-8")))
        tasks: List[Dict[str, Any]] = []

        for cp in checkpoints:
            cid = str(cp.get("checkpoint_id"))
            as_of = cp.get("as_of", {})
            cp_dt = _parse_ts(str(as_of.get("timestamp", "9999-12-31 23:59:59")))
            visible_logs = [x for x in logs if _parse_ts(str(x.get("timestamp", "9999-12-31 23:59:59"))) <= cp_dt]

            task = {
                "task_id": cid,
                "checkpoint_id": cid,
                "as_of": as_of,
                "input": {
                    "app_logs": visible_logs,
                },
                "output_schema": {
                    "snapshot_state": {
                        "<state_category>": {"<state_name>": "<state_value_json>"}
                    },
                    "delta_prediction": {
                        "added": {"<state_category>": {"<state_name>": "<state_value_json>"}},
                        "updated": {"<state_category>": {"<state_name>": {"before": "<state_value_json>", "after": "<state_value_json>"}}},
                        "removed": {"<state_category>": ["<state_name>"]},
                    },
                    "uncertainty": {
                        "<state_category>:<state_name>": "<float 0..1>",
                    },
                },
            }
            if include_targets:
                task["target"] = {
                    "expected_snapshot_state": cp.get("expected_snapshot_state", {}),
                    "expected_delta_from_previous": cp.get("expected_delta_from_previous", {}),
                }
            tasks.append(task)

        payload = {
            "user_id": self.benchmark.get("user_id"),
            "task_count": len(tasks),
            "tasks": tasks,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def export_subset_benchmark(self, checkpoints: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        subset = {
            "user_id": self.benchmark.get("user_id"),
            "source_total_app_logs": self.benchmark.get("source_total_app_logs"),
            "total_chains": self.benchmark.get("total_chains"),
            "total_checkpoints": len(checkpoints),
            "checkpoints": list(checkpoints),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        return subset

    def evaluate_predictions(self, prediction_path: Path, output_path: Path) -> Dict[str, Any]:
        raw_pred = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
        pred_by_id = normalize_predictions(raw_pred)

        rows, evaluated = evaluate_checkpoints(
            self.benchmark,
            pred_by_id,
            include_internal_payload=False,
        )
        summary = mean_numeric_fields(
            [{k: v for k, v in row.items() if k != "checkpoint_id"} for row in rows]
        )

        result = {
            "user_id": self.benchmark.get("user_id"),
            "total_checkpoints": len(self.checkpoints),
            "evaluated_checkpoints": evaluated,
            "skipped_checkpoints": max(len(self.checkpoints) - evaluated, 0),
            "summary": summary,
            "checkpoints": rows,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
