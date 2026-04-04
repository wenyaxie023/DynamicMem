#!/usr/bin/env python3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DOC = ROOT / "docs/protocols/tce_generation_and_adapter_contract.md"
MANUAL_DOC = ROOT / "docs/protocols/temporal_checkpoint_evaluation_developer_manual.md"
RUNBOOK_DOC = ROOT / "docs/runbooks/tce_execution_runbook.md"
README_DOC = ROOT / "generation/README.md"


class TceContractAcceptance(unittest.TestCase):
    def test_contributor_contract_exists_and_contains_required_sections(self):
        self.assertTrue(CONTRACT_DOC.exists())
        text = CONTRACT_DOC.read_text(encoding="utf-8")
        self.assertIn("run(args: TceAdapterArgs) -> dict", text)
        self.assertIn("generation/<baseline>/results/<user_id>/prediction/*.json", text)
        self.assertIn("unified tce baseline route", text.lower())
        self.assertIn("prepare_checkpoint_state", text)
        self.assertIn("retrieve_context_for_query", text)
        self.assertIn("answer_query", text)
        self.assertIn("amem", text)
        self.assertIn("memoryos", text)
        self.assertIn("mem0", text)
        self.assertIn("final-checkpoint qa", text.lower())

    def test_manual_runbook_and_readme_reference_contributor_contract(self):
        contract_relpath = "docs/protocols/tce_generation_and_adapter_contract.md"
        for path in (MANUAL_DOC, RUNBOOK_DOC, README_DOC):
            text = path.read_text(encoding="utf-8")
            self.assertIn(contract_relpath, text, msg=str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
