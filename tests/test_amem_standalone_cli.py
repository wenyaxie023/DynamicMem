#!/usr/bin/env python3
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from baseline_prediction.Amem import amem as amem_builder
from baseline_prediction.Amem import tce as amem_tce


class AmemStandaloneCliTest(unittest.TestCase):
    def test_builder_main_rejects_checkpoint_dir_alias(self):
        argv = [
            "amem.py",
            "--user-id",
            "001_user_001",
            "--benchmark-path",
            "/tmp/bench.json",
            "--checkpoint-dir",
            "/tmp/legacy",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                amem_builder.main()
        self.assertEqual(exc.exception.code, 2)

    def test_builder_main_accepts_data_storage_path(self):
        argv = [
            "amem.py",
            "--user-id",
            "001_user_001",
            "--benchmark-path",
            "/tmp/bench.json",
            "--data-storage-path",
            "/tmp/runtime",
            "--snapshot-dir",
            "/tmp/snapshots",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            amem_builder, "evaluate_membench", return_value={"events_processed": 0}
        ) as evaluate_mock:
            amem_builder.main()
        self.assertEqual(evaluate_mock.call_args.kwargs["data_storage_path"], "/tmp/runtime")

    def test_tce_main_rejects_checkpoint_dir_flag(self):
        argv = [
            "tce.py",
            "--benchmark",
            "/tmp/bench.json",
            "--app-logs-path",
            "/tmp/app_logs.json",
            "--output",
            "/tmp/pred.json",
            "--user-id",
            "001_user_001",
            "--checkpoint-dir",
            "/tmp/legacy",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                amem_tce.main()
        self.assertEqual(exc.exception.code, 2)

    def test_tce_main_accepts_snapshot_root(self):
        argv = [
            "tce.py",
            "--benchmark",
            "/tmp/bench.json",
            "--app-logs-path",
            "/tmp/app_logs.json",
            "--output",
            "/tmp/pred.json",
            "--user-id",
            "001_user_001",
            "--snapshot-root",
            "/tmp/snapshots/001_user_001/large",
            "--linked-neighbor-top-k",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            amem_tce, "run_generation", return_value={"predictions": []}
        ) as run_mock:
            amem_tce.main()
        self.assertEqual(run_mock.call_args.kwargs["snapshot_root"], "/tmp/snapshots/001_user_001/large")
        self.assertEqual(run_mock.call_args.kwargs["linked_neighbor_top_k"], 0)
        self.assertEqual(run_mock.call_args.kwargs["output_path"], Path("/tmp/pred.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
