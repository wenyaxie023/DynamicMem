#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_tce import evaluate


class TceEvalFieldRemovalAcceptance(unittest.TestCase):
    def test_removed_llm_evidence_alignment_fields_absent(self):
        benchmark = {
            "user_id": "u",
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {"a": {"b": {"v": 1}}},
                    "state_observability": {"a": {"b": {"is_valid": True, "evidence_app_log_ids": ["log_1"]}}},
                }
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "cp_0001",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {"a:b": {"v": 1}},
                    "evidence": {"a:b": ["log_1"]},
                    "change_analysis": {},
                }
            ]
        }
        result, _ = evaluate(benchmark, prediction, enable_llm_judge=False, save_eyeball=False)
        row = (result.get("checkpoints") or [{}])[0]
        self.assertNotIn("llm_judge_evidence_alignment_avg_1_5", row)
        self.assertNotIn("change_evidence_alignment_llm_avg_1_5", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
