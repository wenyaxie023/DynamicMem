#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_core.loader import load_app_logs, load_benchmark, load_prediction


class LoaderContractsAcceptance(unittest.TestCase):
    def _write(self, path: Path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_loader_contracts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            logs_list_path = root / "app_log_list.json"
            logs_obj_path = root / "app_log_obj.json"
            pred_path = root / "pred.json"

            self._write(
                benchmark_path,
                {
                    "user_id": "001_user_001",
                    "checkpoints": [{"checkpoint_id": "cp_0001"}],
                },
            )
            self._write(logs_list_path, [{"app_log_id": "log_0001"}])
            self._write(logs_obj_path, {"app_logs": [{"app_log_id": "log_0001"}]})
            self._write(pred_path, {"predictions": [{"checkpoint_id": "cp_0001", "snapshot_state": {}}]})

            benchmark = load_benchmark(benchmark_path)
            logs_list = load_app_logs(logs_list_path)
            logs_obj = load_app_logs(logs_obj_path)
            pred = load_prediction(pred_path)

            self.assertEqual(benchmark.user_id, "001_user_001")
            self.assertEqual(benchmark.total_checkpoints, 1)
            self.assertEqual(len(logs_list.app_logs), 1)
            self.assertEqual(len(logs_obj.app_logs), 1)
            self.assertIn("cp_0001", pred.predictions_by_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
