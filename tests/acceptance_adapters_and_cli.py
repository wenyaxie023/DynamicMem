#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AdapterCliAcceptance(unittest.TestCase):
    def _write_cfg(self, path: Path, baseline: str, extra: str = ""):
        text = """
runtime:
  baseline: {baseline}
  resume: false
  debug: false
  save_prompt_and_raw: false

data:
  benchmark: {benchmark}
  app_logs_path: {logs}

output:
  prediction_path: {output}

llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1

baseline_params:
{extra}
""".strip().format(
            baseline=baseline,
            benchmark=str(path.parent / "benchmark.json"),
            logs=str(path.parent / "app_logs.json"),
            output=str(path.parent / (baseline + "_pred.json")),
            extra=extra or "  {}",
        )
        path.write_text(text + "\n", encoding="utf-8")

    def test_run_tce_dry_run_with_yaml(self):
        baselines = [
            "icl",
            "oracle",
            "rag",
            "hipporag2",
            "memoryos",
            "letta",
            "amem",
            "nemori",
            "zep",
            "mem0",
            "memgpt",
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "benchmark.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
            (root / "app_logs.json").write_text(json.dumps([]), encoding="utf-8")

            for baseline in baselines:
                cfg = root / (baseline + ".yaml")
                extra = ""
                if baseline == "rag":
                    extra = "  retrieval_top_k: 5\n"
                elif baseline == "hipporag2":
                    extra = "  hipporag_dir: /tmp/dummy_hipporag\n"
                elif baseline == "amem":
                    extra = "  user_id: 003_user_003\n  size: large\n"
                self._write_cfg(cfg, baseline, extra)

                cmd = [
                    "python3",
                    "-m",
                    "generation.run_tce",
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
                self.assertEqual(payload["runtime"]["baseline"], baseline)

    def test_unimplemented_baseline_fails_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "benchmark.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
            (root / "app_logs.json").write_text(json.dumps([]), encoding="utf-8")
            cfg = root / "mem0.yaml"
            self._write_cfg(cfg, "mem0")

            cmd = [
                "python3",
                "-m",
                "generation.run_tce",
                "--config",
                str(cfg),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not wired for tce", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
