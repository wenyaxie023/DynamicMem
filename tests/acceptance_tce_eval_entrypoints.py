#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_core.tce_evaluator import build_tce_result_payload, evaluate_tce_rows
from eval.eval_tce import evaluate
from tce_contracts import (
    CURRENT_TASK_CONTRACT_VERSION,
    LEGACY_TASK_CONTRACT_VERSION,
    RESEARCH_FRAME_VERSION_V2,
)


class TceEvalEntrypointAcceptance(unittest.TestCase):
    def _write(self, path: Path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_eval_and_bench_core_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            prediction_path = root / "prediction.json"

            benchmark_payload = {
                "user_id": "001_user_001",
                "task_contract_version": CURRENT_TASK_CONTRACT_VERSION,
                "research_frame_version": RESEARCH_FRAME_VERSION_V2,
                "total_checkpoints": 1,
                "checkpoints": [
                    {
                        "checkpoint_id": "cp_0001",
                        "as_of": {"timestamp": "2025-01-01 08:00:00"},
                        "expected_snapshot_state": {
                            "habits_state": {
                                "morning_walk": {"timing": {"start_time": "06:30"}}
                            }
                        },
                        "state_observability": {
                            "habits_state": {
                                "morning_walk": {
                                    "evidence_app_log_ids": ["log_0001"],
                                }
                            }
                        },
                    }
                ],
            }
            prediction_payload = {
                "predictions": [
                    {
                        "checkpoint_id": "legacy_cp_id",
                        "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                        "snapshot_state": {
                            "habits_state:morning_walk": {"timing": {"start_time": "06:30"}}
                        },
                        "evidence": {"habits_state:morning_walk": ["log_0001"]},
                    }
                ]
            }

            self._write(benchmark_path, benchmark_payload)
            self._write(prediction_path, prediction_payload)

            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            raw_pred = json.loads(prediction_path.read_text(encoding="utf-8"))
            eval_result, _align = evaluate(benchmark, raw_pred, enable_llm_judge=False, save_eyeball=False)
            rows, evaluated, _align_report = evaluate_tce_rows(
                benchmark,
                raw_pred,
                align_by_timestamp=True,
                include_internal_payload=False,
            )
            bench_core_result = build_tce_result_payload(
                benchmark,
                rows,
                evaluated,
                save_eyeball=False,
                strip_internal_payload=True,
            )

            self.assertEqual(eval_result["evaluated_checkpoints"], bench_core_result["evaluated_checkpoints"])
            self.assertEqual(
                eval_result["summary"]["snapshot_value_f1_mean_on_expected_mean"],
                bench_core_result["summary"]["snapshot_value_f1_mean_on_expected_mean"],
            )
            self.assertEqual(len(eval_result["checkpoints"]), len(bench_core_result["checkpoints"]))
            self.assertEqual(eval_result["task_contract_version"], CURRENT_TASK_CONTRACT_VERSION)
            self.assertEqual(bench_core_result["task_contract_version"], CURRENT_TASK_CONTRACT_VERSION)

    def test_eval_defaults_missing_contract_metadata_to_legacy_v1(self):
        result = build_tce_result_payload(
            {
                "user_id": "001_user_001",
                "total_checkpoints": 0,
                "checkpoints": [],
            },
            [],
            0,
            save_eyeball=False,
            strip_internal_payload=True,
        )
        self.assertEqual(result["task_contract_version"], LEGACY_TASK_CONTRACT_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
