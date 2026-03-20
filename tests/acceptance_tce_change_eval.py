#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_tce import evaluate


class TceChangeEvalAcceptance(unittest.TestCase):
    def test_change_metrics_present_and_computable(self):
        benchmark = {
            "user_id": "001_user_001",
            "total_checkpoints": 2,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-01-02 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "07:00"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0002"]}
                        }
                    },
                },
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "legacy_1",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {"habits_state:morning_walk": ["log_0001"]},
                    "change_analysis": {},
                },
                {
                    "checkpoint_id": "legacy_2",
                    "metadata": {"checkpoint_timestamp": "2025-01-02 08:00:00"},
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "07:00"}}},
                    "evidence": {"habits_state:morning_walk": ["log_0002"]},
                    "change_analysis": {
                        "habits_state:morning_walk": {
                            "before": {"timing": {"start_time": "06:30"}},
                            "after": {"timing": {"start_time": "07:00"}},
                            "change_reason": "Routine shifted later after schedule change.",
                            "evidence": ["log_0002"],
                        }
                    },
                },
            ]
        }

        result, _align = evaluate(
            benchmark,
            prediction,
            enable_llm_judge=False,
            save_eyeball=False,
        )
        summary = result.get("summary", {})
        self.assertIn("change_before_after_correctness_mean_on_changed_mean", summary)
        self.assertIn("change_reason_mean_on_changed_mean", summary)
        self.assertIn("change_evidence_recall_mean_on_changed_mean", summary)
        self.assertGreaterEqual(summary["change_before_after_f1_mean_on_changed_mean"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
