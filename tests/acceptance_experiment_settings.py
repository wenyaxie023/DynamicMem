#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.tce_config import write_run_settings


class ExperimentSettingsAcceptance(unittest.TestCase):
    def test_settings_file_written(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "prediction" / "result.json"
            resolved = {
                "runtime": {"baseline": "rag"},
                "data": {"benchmark": "a", "app_logs_path": "b"},
                "output": {"prediction_path": str(out)},
                "llm": {"provider": "openai", "model": "gpt-5-mini", "max_workers": 1},
                "baseline_params": {"retrieval_top_k": "5"},
            }
            p = write_run_settings(resolved, out)
            self.assertTrue(p.exists())
            text = p.read_text(encoding="utf-8")
            self.assertIn("saved_at_utc", text)
            self.assertIn("baseline: rag", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
