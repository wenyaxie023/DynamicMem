#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.run_tce_batch import _ensure_benchmark_exists


class AutoBuildBenchmarkAcceptance(unittest.TestCase):
    def test_auto_build_invokes_builder_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = root / "tce_benchmark.json"
            app_logs_final = root / "app_logs_final.json"
            all_events_chains = root / "all_events_chains.json"
            app_logs_final.write_text('{"app_logs": []}', encoding='utf-8')
            all_events_chains.write_text('[]', encoding='utf-8')

            user_cfg = {
                "runtime": {"auto_build_benchmark": True},
                "data": {
                    "benchmark": str(benchmark),
                    "app_logs_final_path": str(app_logs_final),
                    "all_events_chains_path": str(all_events_chains),
                },
            }

            def _fake_call(_cmd):
                benchmark.write_text('{"checkpoints": []}', encoding='utf-8')
                return 0

            with mock.patch("subprocess.check_call", side_effect=_fake_call) as m:
                built_now = _ensure_benchmark_exists(user_cfg)

            self.assertTrue(built_now)
            self.assertTrue(benchmark.exists())
            self.assertEqual(m.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
