#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AdapterCliAcceptance(unittest.TestCase):
    def _write_cfg(
        self,
        path: Path,
        baseline: str,
        extra: str = "",
        retrieval_extra: str = "",
        runtime_extra: str = "",
    ):
        text = """
runtime:
  baseline: {baseline}
  resume: false
  debug: false
  save_prompt_and_raw: false
{runtime_extra}

data:
  benchmark: {benchmark}
  app_logs_path: {logs}

output:
  prediction_path: {output}

llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1
{retrieval_section}

baseline_params:
{extra}
""".strip().format(
            baseline=baseline,
            runtime_extra=runtime_extra.rstrip(),
            benchmark=str(path.parent / "benchmark.json"),
            logs=str(path.parent / "app_logs.json"),
            output=str(path.parent / (baseline + "_pred.json")),
            retrieval_section=retrieval_extra.rstrip(),
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
            "simplemem",
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
                retrieval_extra = ""
                runtime_extra = ""
                if baseline == "rag":
                    retrieval_extra = "retrieval:\n  top_k: 5\n"
                elif baseline == "hipporag2":
                    extra = "  hipporag_dir: /tmp/dummy_hipporag\n"
                elif baseline == "amem":
                    runtime_extra = "  user_id: 003_user_003\n"
                    extra = "  size: large\n"
                elif baseline == "memoryos":
                    runtime_extra = "  user_id: 003_user_003\n"
                    extra = "  size: large\n"
                elif baseline == "mem0":
                    runtime_extra = "  user_id: 003_user_003\n"
                    retrieval_extra = "retrieval:\n  top_k: 5\n"
                self._write_cfg(cfg, baseline, extra, retrieval_extra, runtime_extra)

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

    def test_run_tce_batch_dry_run_for_snapshot_baselines(self):
        baselines = ["amem", "memoryos", "mem0"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for baseline in baselines:
                cfg = root / "{}_batch.yaml".format(baseline)
                cfg.write_text(
                    """
users:
  - 001_user_001
  - 002_user_002
runtime:
  baseline: {baseline}
data:
  benchmark: /tmp/bench/{{user_id}}/benchmark.json
  app_logs_path: /tmp/logs/{{user_id}}/app_log_large.json
output:
  prediction_path: /tmp/out/{{user_id}}/{baseline}.json
llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1
baseline_params:
  {{}}
""".strip().format(baseline=baseline)
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
                self.assertEqual(payload[0]["runtime"]["baseline"], baseline)
                self.assertEqual(payload[0]["runtime"]["user_id"], "001_user_001")
                self.assertEqual(payload[1]["runtime"]["user_id"], "002_user_002")

    def test_run_tce_dry_run_infers_runtime_user_id_from_single_users_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "amem.yaml"
            cfg.write_text(
                """
users:
  - 001_user_001
runtime:
  baseline: amem
data:
  benchmark: /tmp/bench/{user_id}/benchmark.json
  app_logs_path: /tmp/logs/{user_id}/app_log_large.json
output:
  prediction_path: /tmp/out/{user_id}/pred.json
llm:
  provider: openai
  model: gpt-5-mini
baseline_params:
  size: large
""".strip()
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["python3", "-m", "generation.run_tce", "--config", str(cfg), "--dry-run"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["runtime"]["user_id"], "001_user_001")
            self.assertEqual(payload["data"]["benchmark"], "/tmp/bench/001_user_001/benchmark.json")

    def test_run_tce_cli_resume_overrides_config_runtime_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "letta.yaml"
            (root / "benchmark.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
            (root / "app_logs.json").write_text(json.dumps([]), encoding="utf-8")
            self._write_cfg(cfg, "letta")

            proc = subprocess.run(
                ["python3", "-m", "generation.run_tce", "--config", str(cfg), "--dry-run", "--resume"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["runtime"]["resume"])

    def test_zep_formal_config_dry_run_uses_shared_sections(self):
        proc = subprocess.run(
            ["python3", "-m", "generation.run_tce", "--config", "configs/experiments/tce/zep.yaml", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["runtime"]["baseline"], "zep")
        self.assertEqual(payload["runtime"]["user_id"], "001_user_001")
        self.assertEqual(payload["retriever"]["model"], "text-embedding-3-large")
        self.assertEqual(payload["retrieval"]["top_k"], 5)
        self.assertEqual(payload["baseline_params"]["max_coroutines"], "10")

    def test_runtime_user_id_rejects_legacy_baseline_params_user_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "amem.yaml"
            cfg.write_text(
                """
runtime:
  baseline: amem
data:
  benchmark: /tmp/bench.json
  app_logs_path: /tmp/app_logs.json
output:
  prediction_path: /tmp/pred.json
llm:
  provider: openai
  model: gpt-5-mini
baseline_params:
  user_id: 001_user_001
  size: large
""".strip()
                + "\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["python3", "-m", "generation.run_tce", "--config", str(cfg), "--dry-run"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("legacy baseline_params keys", proc.stderr.lower())
            self.assertIn("user_id", proc.stderr)

    def test_unimplemented_baseline_fails_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "benchmark.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
            (root / "app_logs.json").write_text(json.dumps([]), encoding="utf-8")
            cfg = root / "nemori.yaml"
            self._write_cfg(cfg, "nemori")

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
