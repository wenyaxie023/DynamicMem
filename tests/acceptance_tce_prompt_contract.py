#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.prompts import (
    build_apply_answer_scoring_points_prompt,
    build_apply_rubric_rewrite_prompt,
    build_apply_rubric_validation_prompt,
    build_change_reasoning_prompt,
    build_generation_prompt,
    build_value_micro_points_prompt,
)


class TcePromptContractAcceptance(unittest.TestCase):
    def test_state_completion_prompt_uses_four_block_structure(self) -> None:
        prompt = build_generation_prompt(
            context_logs=[{"app_log_id": "log_0001"}],
            task_instruction="Predict the user's current state values.",
            task_query="As of 2025-01-01 08:00:00, infer the user's current state for all items in this template: {\"habits_state:morning_walk\": {\"timing\": {\"start_time\": \"<fill the blank>\"}}}.",
            target_keys=["habits_state:morning_walk"],
            target_value_templates={"habits_state:morning_walk": {"timing": {"start_time": "<fill the blank>"}}},
            log_to_text=lambda x: str(x),
        )
        self.assertIn("[Task]", prompt)
        self.assertIn("Instruction:", prompt)
        self.assertIn("Query:", prompt)
        self.assertIn("[User memory]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn("[Rules]", prompt)
        self.assertIn('"evidence_content"', prompt)
        self.assertNotIn("Answering task:", prompt)

    def test_change_prompt_uses_evidence_objects(self) -> None:
        prompt = build_change_reasoning_prompt(
            context_logs=[{"app_log_id": "log_0002"}],
            task_instruction="Infer the latest change details.",
            task_query="As of 2025-02-01 08:00:00, for state items that changed since 2025-01-01 08:00:00, infer the latest change details.",
            changed_keys=["habits_state:morning_walk"],
            changed_value_templates={
                "habits_state:morning_walk": {
                    "before": {"timing": {"start_time": "<fill the blank>"}},
                    "after": {"timing": {"start_time": "<fill the blank>"}},
                    "change_reason": "<fill the blank>",
                    "evidence": [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}],
                }
            },
            log_to_text=lambda x: str(x),
        )
        self.assertIn("[Task]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn('"app_log_id"', prompt)
        self.assertIn('"evidence_content"', prompt)
        self.assertNotIn("Answering task:", prompt)

    def test_value_micro_points_prompt_uses_clear_rubric_definitions(self) -> None:
        prompt = build_value_micro_points_prompt(
            state_key="preferences_state:learning_modality",
            text_value="Prefers self-paced white papers and webinars over large conferences.",
            max_points=5,
        )
        self.assertLess(prompt.index("[Task Instruction]"), prompt.index("[Definitions]"))
        self.assertLess(prompt.index("[Definitions]"), prompt.index("[Constraints]"))
        self.assertLess(prompt.index("[Constraints]"), prompt.index("[Example]"))
        self.assertLess(prompt.index("[Example]"), prompt.index("[Input/Output Format]"))
        self.assertIn("Generate 1 to 3 atomic facts.", prompt)
        self.assertIn("shared binary `0/1` hit scale", prompt)
        self.assertIn("Prefer stable semantic requirements over brittle exact surface forms.", prompt)
        self.assertIn("Do not generate multiple points that all hinge on the same narrow identifier", prompt)
        self.assertIn('"rubric": [', prompt)
        self.assertNotIn("reference_value", prompt)
        self.assertNotIn("target_path", prompt)

    def test_apply_atomic_fact_prompts_include_state_grounding_and_over_specific(self) -> None:
        gen_prompt = build_apply_answer_scoring_points_prompt(
            state_key="user_attributes_state:home_media_server",
            state_value="Synology DS923+ NAS for local 4K movie collection",
            apply_scenario="A high-bitrate movie is available from two playback sources.",
            apply_question="How should the assistant prioritize playback sources?",
            apply_reference_answer="Select the local NAS as the source for the 4K digital version.",
            max_points=3,
        )
        self.assertIn("grounding context", gen_prompt)
        self.assertIn("state_value", gen_prompt)
        self.assertIn("not just the question wording", gen_prompt)
        self.assertIn("Do not generate multiple points that all hinge on the same narrow identifier", gen_prompt)

        validation_prompt = build_apply_rubric_validation_prompt(
            state_key="user_attributes_state:home_media_server",
            state_value="Synology DS923+ NAS for local 4K movie collection",
            service_category="playback policy",
            question="How should the assistant prioritize playback sources?",
            reference_answer="Select the local NAS as the source for the 4K digital version.",
            points=[],
        )
        self.assertIn("over_specific", validation_prompt)
        self.assertIn("Treat `state_value` as grounding context", validation_prompt)

        rewrite_prompt = build_apply_rubric_rewrite_prompt(
            state_key="user_attributes_state:home_media_server",
            state_value="Synology DS923+ NAS for local 4K movie collection",
            service_category="playback policy",
            question="How should the assistant prioritize playback sources?",
            reference_answer="Select the local NAS as the source for the 4K digital version.",
            points=[],
            validation_points=[],
            set_failures=[],
            max_points=3,
        )
        self.assertIn("over_specific", rewrite_prompt)
        self.assertIn("broader, more stable semantic fact", rewrite_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
