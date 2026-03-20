#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.prompts import build_rq3_apply_question_pack_prompt


class TceRq3QuestionPromptAcceptance(unittest.TestCase):
    def test_prompt_removes_internal_jargon_and_uses_new_service_schema(self):
        prompt = build_rq3_apply_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}},
            item_count=2,
        )
        self.assertNotIn("DSP", prompt)
        self.assertNotIn("RQ3", prompt)
        self.assertNotIn("Supporting user memory logs", prompt)
        self.assertIn("Generate personalized service free-form QA items", prompt)
        self.assertIn("[Definitions]", prompt)
        self.assertIn("[Example Input 1]", prompt)
        self.assertIn("[Example Output 1]", prompt)
        self.assertIn("service_category", prompt)
        self.assertNotIn('"qa_id"', prompt)
        self.assertIn('"question"', prompt)
        self.assertIn('"reference_answer"', prompt)
        self.assertNotIn('"rubric"', prompt)
        self.assertIn("[Constraints]\n1.", prompt)
        self.assertIn("state changed to a reasonable alternative", prompt)
        self.assertIn("schedule-grounded service operations are valid when the question asks what the assistant should do around the routine", prompt)
        self.assertIn("Do not turn the item into a pure lookup, direct state restatement, trivial extraction, or simple calendar arithmetic problem with no real assistant-action layer.", prompt)
        self.assertIn("Reject questions whose scenario merely echoes the same wording, label, or semantic category already present in the state", prompt)
        self.assertIn("Do not include explicit answer options in the question, and do not frame the question as a named A/B menu of candidate actions.", prompt)
        self.assertIn("The reference_answer must not introduce concrete facts, parameters, product specifications, thresholds, or background knowledge", prompt)
        self.assertIn("The reference_answer must not introduce extra methods, tactics, metrics, escalation mechanisms, capacity calculations", prompt)
        self.assertIn("Prefer bounded policy or service-decision questions such as:", prompt)
        self.assertIn("defer vs escalate policy", prompt)
        self.assertIn("route / prioritize policy", prompt)
        self.assertIn("notification strategy", prompt)
        self.assertIn("support routing", prompt)
        self.assertIn('Actual Input:\n- checkpoint_timestamp: 2025-01-01 08:00:00', prompt)
        self.assertIn('- state_key: "habits_state:morning_walk"', prompt)
        self.assertIn('"start_time": "06:30"', prompt)
        self.assertIn("Do not generate `qa_id` or any other internal identifier field.", prompt)
        self.assertIn("Bad question:", prompt)
        self.assertIn("What specific lot-selection strategy should the assistant utilize when the user requests to liquidate a portion of the portfolio for an upcoming expense?", prompt)
        self.assertIn("What type of care recommendation should the assistant prioritize when the user reports a minor, non-acute health issue?", prompt)
        self.assertIn("How should the assistant handle non-urgent notifications during the user's recurring Sunday family dinner block?", prompt)
        self.assertIn("How should the assistant route this week's incoming client advisory work for the user?", prompt)
        self.assertIn("Route environmental compliance advisory requests that match the user's certification to the user, and route generic relationship-management follow-ups elsewhere.", prompt)
        self.assertIn("\n26. Output JSON only", prompt)
        self.assertNotIn("apply_scenario", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
