#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class QaRunnerDryRunAcceptance(unittest.TestCase):
    def test_run_qa_dry_run_with_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "qa.yaml"
            cfg.write_text(
                """
runtime:
  baseline: oracle
  command: generation
  subcommand: sample
  user_id: "12"

data:
  input_root_dir: /tmp/in
  output_root_dir: /tmp/out
  qa_dir: /tmp/qa

llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 2

sampling:
  sample_per_group: 3
  sample_seed: 7
  qtypes: "1,2"
  categories: "travel"

output:
  run_record_path: generation/{baseline}/results/{user_id}/prediction/qa_run.json
""".strip()
                + "\n",
                encoding="utf-8",
            )

            cmd = [
                "python3",
                "-m",
                "generation.run_qa",
                "--config",
                str(cfg),
                "--user-id",
                "001_user_001",
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
            self.assertEqual(payload["runtime"]["baseline"], "oracle")
            self.assertEqual(payload["runtime"]["command"], "generation")
            self.assertEqual(payload["runtime"]["subcommand"], "sample")
            self.assertEqual(payload["runtime"]["user_id"], "001_user_001")
            self.assertEqual(payload["data"]["qa_dir"], "/tmp/qa")
            self.assertEqual(payload["llm"]["provider"], "openai")
            self.assertEqual(payload["sampling"]["qtypes"], "1,2")
            self.assertTrue(
                payload["output"]["run_record_path"].endswith(
                    "generation/oracle/results/001_user_001/prediction/qa_run.json"
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
