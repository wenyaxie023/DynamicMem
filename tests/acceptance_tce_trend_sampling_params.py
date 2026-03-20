#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TceTrendSamplingParamsAcceptance(unittest.TestCase):
    def test_trend_allows_missing_exposure_percent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "benchmark.json"
            prediction = root / "prediction.json"
            output_dir = root / "out"

            benchmark.write_text(
                json.dumps(
                    {
                        "total_checkpoints": 1,
                        "checkpoints": [
                            {
                                "checkpoint_id": "cp_0001",
                                "as_of": {"timestamp": "2025-01-01 08:00:00"},
                                "expected_snapshot_state": {
                                    "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            prediction.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "checkpoint_id": "cp_0001",
                                "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                                "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    "-m",
                    "eval.analyze_tce_item_trends",
                    "--benchmark",
                    str(benchmark),
                    "--prediction",
                    str(prediction),
                    "--output-dir",
                    str(output_dir),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue((output_dir / "item_level_metrics.csv").exists())
            self.assertTrue((output_dir / "changed_vs_unchanged_daily.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
