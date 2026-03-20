#!/usr/bin/env python3
import unittest
from pathlib import Path


class RefactorReadmeAcceptance(unittest.TestCase):
    def test_step_readmes_exist(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "docs/refactor_steps/README.md",
            root / "docs/refactor_steps/STEP1_loader_contracts.md",
            root / "docs/refactor_steps/STEP2_eval_entry_unification.md",
            root / "docs/refactor_steps/STEP3_adapter_migration.md",
            root / "docs/refactor_steps/STEP4_scripts_and_config_convergence.md",
        ]
        for p in paths:
            self.assertTrue(p.exists(), msg="Missing {}".format(p))
            text = p.read_text(encoding="utf-8").strip()
            self.assertTrue(len(text) > 20, msg="README too short: {}".format(p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
