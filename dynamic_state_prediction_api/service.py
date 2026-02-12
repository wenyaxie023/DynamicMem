#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dynamic_state_prediction_core import (
    build_target_templates,
    evaluate_checkpoints,
    mean_numeric_fields,
    normalize_app_logs,
    normalize_predictions,
    observed_logs_for_checkpoint,
    parse_ts,
)


@dataclass
class CheckpointTask:
    task_id: str
    checkpoint_id: str
    as_of: Dict[str, Any]
    app_logs: List[Dict[str, Any]]


class DynamicStatePredictionAPI:
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
            key=lambda cp: parse_ts(str((cp.get("as_of") or {}).get("timestamp", "9999-12-31 23:59:59"))),
        )
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for d in dates:
            cutoff = parse_ts(f"{d} 23:59:59")
            chosen = None
            for cp in cps:
                ts = parse_ts(str((cp.get("as_of") or {}).get("timestamp", "9999-12-31 23:59:59")))
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
        logs = normalize_app_logs(json.loads(Path(app_logs_path).read_text(encoding="utf-8")))
        tasks: List[Dict[str, Any]] = []

        for cp in checkpoints:
            cid = str(cp.get("checkpoint_id"))
            as_of = cp.get("as_of", {})
            visible_logs, _cp_dt, cp_ts = observed_logs_for_checkpoint(cp, logs)
            target_keys, target_value_templates = build_target_templates(cp)
            target_evidence_templates = {k: [] for k in target_keys}

            task = {
                "task_id": cid,
                "checkpoint_id": cid,
                "as_of": as_of,
                "input": {
                    "app_logs": visible_logs,
                },
                "output_schema": {
                    "snapshot_state": target_value_templates,
                    "evidence": target_evidence_templates,
                },
                "target_keys": target_keys,
                "target_value_templates": target_value_templates,
                "target_evidence_templates": target_evidence_templates,
                "metadata": {
                    "checkpoint_timestamp": cp_ts,
                    "num_visible_logs": len(visible_logs),
                },
            }
            if include_targets:
                task["target"] = {
                    "expected_snapshot_state": cp.get("expected_snapshot_state", {}),
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
