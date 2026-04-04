#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TceBatchAcceptance(unittest.TestCase):
    def test_batch_dry_run_user_template(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "rag_batch.yaml"
            cfg.write_text(
                """
users:
  - 001_user_001
  - 002_user_002
runtime:
  baseline: rag
data:
  benchmark: /tmp/bench/{user_id}/benchmark.json
  app_logs_path: /tmp/logs/{user_id}/app_log_large.json
output:
  prediction_path: /tmp/out/{user_id}/pred.json
llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1
retrieval:
  top_k: 5
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cmd = [
                "python3",
                "-m",
                "generation.run_tce_batch",
                "--config",
                str(cfg),
                "--dry-run",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["runtime"]["user_id"], "001_user_001")
            self.assertIn("/tmp/bench/001_user_001/benchmark.json", payload[0]["data"]["benchmark"])
            self.assertIn("/tmp/bench/002_user_002/benchmark.json", payload[1]["data"]["benchmark"])

    def test_batch_dry_run_run_id_and_run_name_templates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "letta_batch.yaml"
            cfg.write_text(
                """
users:
  - 001_user_001
runtime:
  baseline: letta
  experiment_name: letta_v14_user1_local_selfhost
  run_id: "review1"
data:
  benchmark: /tmp/bench/{user_id}/benchmark.json
  app_logs_path: /tmp/logs/{user_id}/app_log_large.json
output:
  prediction_path: /tmp/out/{user_id}/{run_name}/pred.json
llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1
baseline_params:
  checkpoint_agents_dir: /tmp/agents/{run_name}/{user_id}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cmd = [
                "python3",
                "-m",
                "generation.run_tce_batch",
                "--config",
                str(cfg),
                "--dry-run",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(len(payload), 1)
            resolved = payload[0]
            self.assertEqual(resolved["runtime"]["run_id"], "review1")
            self.assertEqual(
                resolved["runtime"]["run_name"],
                "letta_v14_user1_local_selfhost__review1",
            )
            self.assertEqual(
                resolved["output"]["prediction_path"],
                "/tmp/out/001_user_001/{}/pred.json".format(resolved["runtime"]["run_name"]),
            )
            self.assertEqual(
                resolved["baseline_params"]["checkpoint_agents_dir"],
                "/tmp/agents/{}/001_user_001".format(resolved["runtime"]["run_name"]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
