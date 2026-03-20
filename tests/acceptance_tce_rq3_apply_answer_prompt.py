#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.prompts import build_rq3_apply_answer_prompt


class TceRq3ApplyAnswerPromptAcceptance(unittest.TestCase):
    def test_prompt_uses_four_block_structure_with_scenario_and_evidence_objects(self) -> None:
        prompt = build_rq3_apply_answer_prompt(
            apply_scenario="The assistant must choose one communication policy during recurring briefings. Candidate actions: A. Interrupt immediately. B. Hold non-urgent requests until after the briefing. C. Disable all automated handling.",
            question_text="What reminder time should be used for the user's budget review on Sunday?",
            context_logs=[{"app_log_id": "log_00028", "timestamp": "2025-01-14 09:00:00"}],
            log_to_text=lambda x: str(x),
        )

        self.assertIn("[Task]", prompt)
        self.assertIn("Instruction:", prompt)
        self.assertIn("Query:", prompt)
        self.assertIn("Service scenario:", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("[User memory]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn("[Rules]", prompt)
        self.assertNotIn("[Checkpoint timestamp]", prompt)
        self.assertNotIn("[Example]", prompt)
        self.assertNotIn("[Definitions]", prompt)
        self.assertIn('"answer"', prompt)
        self.assertIn('"app_log_id"', prompt)
        self.assertIn('"evidence_content"', prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
