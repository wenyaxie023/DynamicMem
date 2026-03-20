#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_tce import evaluate


class TceRq3BreakingSchemaAcceptance(unittest.TestCase):
    def test_legacy_rq3_fields_raise(self):
        benchmark = {
            "checkpoints": [
                {
                    "checkpoint_id": "cp_1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {"a_state": {"x": 1}},
                    "state_observability": {"a_state": {"x": {"is_valid": True}}},
                    "rq3_know_apply": {"keys": {}},
                }
            ]
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "cp_1",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {"a_state:x": 1},
                    "rq3_answers": {},
                }
            ]
        }
        with self.assertRaises(ValueError):
            evaluate(benchmark, prediction, enable_llm_judge=False, save_eyeball=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)

