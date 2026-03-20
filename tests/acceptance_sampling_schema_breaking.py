#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.exposure_checkpoint_builder import _rebuild_checkpoints_from_export_items


class SamplingSchemaBreakingAcceptance(unittest.TestCase):
    def _write_json(self, path: Path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_sampling_meta_exposure_schema_no_legacy_anchor_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "tce_benchmark.json"
            benchmark_path.write_text("{}", encoding="utf-8")
            self._write_json(
                root / "app_logs_final.json",
                {
                    "app_logs": [
                        {
                            "app_log_id": "log_0001",
                            "timestamp": "2025-01-01 08:00:00",
                            "metadata": {"chain_id": "chain_1"},
                            "golden_evidence": [
                                {
                                    "state_category": "habits_state",
                                    "state_name": "morning_walk",
                                    "change_type": "update",
                                    "state_value": {"timing": {"start_time": "06:30"}},
                                    "evidenced_fields": ["timing.start_time"],
                                }
                            ],
                        }
                    ]
                },
            )
            self._write_json(root / "all_events_chains.json", [])

            with mock.patch(
                "tce_core.exposure_checkpoint_builder.build_chain_state_validity",
                return_value={"chain_1": {"habits_state:morning_walk": {"is_valid": True}}},
            ):
                _checkpoints, meta = _rebuild_checkpoints_from_export_items(
                    benchmark_path=benchmark_path,
                    export_items=[(0, {"anchor_percent": 10, "tokenizer_model": "gpt-5-mini"})],
                    sampling_mode="exposure_token",
                )

            self.assertIn("exp_010", meta)
            m = meta["exp_010"]
            self.assertEqual(m.get("sampling_mode"), "exposure_token")
            self.assertIn("sampling_params", m)
            self.assertNotIn("anchor_percent", m)
            self.assertNotIn("anchor_index", m)

    def test_sampling_meta_calendar_schema_contains_token_markers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "tce_benchmark.json"
            benchmark_path.write_text("{}", encoding="utf-8")
            self._write_json(
                root / "app_logs_final.json",
                {
                    "app_logs": [
                        {
                            "app_log_id": "log_0001",
                            "timestamp": "2025-01-01 08:00:00",
                            "metadata": {"chain_id": "chain_1"},
                            "golden_evidence": [
                                {
                                    "state_category": "habits_state",
                                    "state_name": "morning_walk",
                                    "change_type": "update",
                                    "state_value": {"timing": {"start_time": "06:30"}},
                                    "evidenced_fields": ["timing.start_time"],
                                }
                            ],
                        }
                    ]
                },
            )
            self._write_json(root / "all_events_chains.json", [])

            with mock.patch(
                "tce_core.exposure_checkpoint_builder.build_chain_state_validity",
                return_value={"chain_1": {"habits_state:morning_walk": {"is_valid": True}}},
            ):
                _checkpoints, meta = _rebuild_checkpoints_from_export_items(
                    benchmark_path=benchmark_path,
                    export_items=[
                        (
                            0,
                            {
                                "calendar_anchor_freq": "quarterly",
                                "anchor_index": 1,
                                "anchor_timestamp": "2025-01-01 08:00:00",
                                "actual_tokens_at_cutoff": 123,
                                "total_tokens": 456,
                                "cutoff_log_tokens": 12,
                                "tokenizer_model": "gpt-4o-mini",
                            },
                        )
                    ],
                    sampling_mode="calendar_time",
                )

            self.assertIn("cal_quarterly_001", meta)
            m = meta["cal_quarterly_001"]
            sp = m.get("sampling_params", {})
            self.assertEqual(m.get("sampling_mode"), "calendar_time")
            self.assertEqual(int(sp.get("actual_tokens_at_cutoff", 0)), 123)
            self.assertEqual(int(sp.get("total_tokens", 0)), 456)
            self.assertEqual(str(sp.get("tokenizer_model", "")), "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main(verbosity=2)
