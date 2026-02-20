#!/usr/bin/env python3
import unittest
from pathlib import Path


class QaContractsAcceptance(unittest.TestCase):
    def test_single_source_contract_exists(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "docs/protocols/qa_generation_and_eval_contract.md"
        self.assertTrue(path.exists(), msg="Missing {}".format(path))
        text = path.read_text(encoding="utf-8")

        required_tokens = [
            "Single Source of Truth",
            "Contributor Boundary",
            "QA Prediction JSON Contract",
            "Unified QA Runner Contract",
            "python -m generation.run_qa",
            "acceptance_qa_contracts.py",
        ]
        for token in required_tokens:
            self.assertIn(token, text, msg="Missing token '{}' in {}".format(token, path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
