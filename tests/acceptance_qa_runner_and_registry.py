#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.qa_adapters.base import QaAdapterArgs
from baseline_prediction.qa_adapters.registry import list_adapters, run_adapter
from baseline_prediction.qa_config import resolved_payload, write_run_settings


class QaRunnerRegistryAcceptance(unittest.TestCase):
    def test_registry_and_settings(self):
        self.assertIn("qa_pipeline", list_adapters())
        self.assertIn("oracle", list_adapters())

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "qa_run.json"
            args = QaAdapterArgs(
                baseline="qa_pipeline",
                user_id="10",
                input_root_dir=None,
                output_root_dir=None,
                qa_dir=None,
                output_path=None,
                llm_provider="openai",
                llm_model="gpt-5-mini",
                llm_max_workers=1,
                command="pipeline",
                subcommand=None,
                retry_times=1,
                flush_every=1,
                sample_per_group=5,
                sample_seed=42,
                qtypes=None,
                categories=None,
                fix_links_inplace=False,
                run_record_path=out,
                experiment_name=None,
                extras={},
            )
            resolved = resolved_payload(args)
            p = write_run_settings(resolved, out)
            self.assertTrue(p.exists())
            text = p.read_text(encoding="utf-8")
            self.assertIn("saved_at_utc", text)
            self.assertIn("command: pipeline", text)

    def test_unknown_baseline_fails_clearly(self):
        args = QaAdapterArgs(
            baseline="not_exist",
            user_id="10",
            input_root_dir=None,
            output_root_dir=None,
            qa_dir=None,
            output_path=None,
            llm_provider=None,
            llm_model=None,
            llm_max_workers=None,
            command="pipeline",
            subcommand=None,
            retry_times=None,
            flush_every=None,
            sample_per_group=None,
            sample_seed=None,
            qtypes=None,
            categories=None,
            fix_links_inplace=False,
            run_record_path=Path("tmp.json"),
            experiment_name=None,
            extras={},
        )
        with self.assertRaises(ValueError) as ctx:
            run_adapter(args)
        self.assertIn("Unknown QA baseline", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
