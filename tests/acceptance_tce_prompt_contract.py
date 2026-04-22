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
    build_change_reasoning_prompt_with_agent_memory,
    build_change_reasoning_prompt_with_inline_memory,
    build_task_c_prompt_with_agent_memory,
    build_task_c_prompt_with_inline_memory,
    build_structured_service_completion_prompt_with_agent_memory,
    build_structured_service_completion_prompt_with_inline_memory,
    build_service_application_prompt_with_agent_memory,
    build_service_application_prompt_with_inline_memory,
    build_state_completion_prompt_with_agent_memory,
    build_state_completion_prompt_with_inline_memory,
    build_task_c_v2_question_pack_prompt,
    build_task_c_v2_user_communication_prompt_with_inline_memory,
    build_task_c_v2_rewrite_prompt,
    build_task_c_v2_validation_prompt,
    build_value_micro_points_prompt,
)
from tce_core.final_checkpoint_qa import (
    build_final_qa_prompt_with_agent_memory,
    build_final_qa_prompt_with_inline_memory,
)


class TcePromptContractAcceptance(unittest.TestCase):
    def test_state_completion_prompt_uses_direct_question_plus_memory_blocks(self) -> None:
        prompt = build_state_completion_prompt_with_inline_memory(
            context_logs=[{"app_log_id": "log_0001"}],
            task_query="As of 2025-01-01 08:00:00, infer the user's current state for all items in this template: {\"habits_state:morning_walk\": {\"timing\": {\"start_time\": \"<fill the blank>\"}}}.",
            target_keys=["habits_state:morning_walk"],
            target_value_templates={"habits_state:morning_walk": {"timing": {"start_time": "<fill the blank>"}}},
            log_to_text=lambda x: str(x),
        )
        self.assertTrue(prompt.startswith("As of 2025-01-01 08:00:00"))
        self.assertIn("[Memory]", prompt)
        self.assertIn("[Instructions]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn('"evidence_content"', prompt)
        self.assertNotIn("[Task]", prompt)
        self.assertNotIn("Instruction:", prompt)
        self.assertNotIn("Query:", prompt)
        self.assertNotIn("Checkpoint time:", prompt)
        self.assertNotIn("[Rules]", prompt)

    def test_change_prompt_uses_evidence_objects(self) -> None:
        prompt = build_change_reasoning_prompt_with_inline_memory(
            context_logs=[{"app_log_id": "log_0002"}],
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
        self.assertTrue(prompt.startswith("As of 2025-02-01 08:00:00"))
        self.assertIn("[Memory]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn('"app_log_id"', prompt)
        self.assertIn('"evidence_content"', prompt)
        self.assertNotIn("[Task]", prompt)
        self.assertNotIn("Instruction:", prompt)
        self.assertNotIn("Query:", prompt)
        self.assertNotIn("Checkpoint time:", prompt)

    def test_agent_memory_prompt_variants_do_not_require_inline_user_memory(self) -> None:
        state_prompt = build_state_completion_prompt_with_agent_memory(
            context_logs=[{"app_log_id": "log_0001"}],
            task_query="As of 2025-01-01 08:00:00, infer the user's current state for all items in this template.",
            target_keys=["habits_state:morning_walk"],
            target_value_templates={"habits_state:morning_walk": {"timing": {"start_time": "<fill the blank>"}}},
            log_to_text=lambda x: str(x),
        )
        change_prompt = build_change_reasoning_prompt_with_agent_memory(
            context_logs=[{"app_log_id": "log_0002"}],
            task_query="As of 2025-02-01 08:00:00, infer the latest change details.",
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
        apply_prompt = build_service_application_prompt_with_agent_memory(
            question_text="What should the assistant recommend?",
            context_logs=[{"app_log_id": "log_0003"}],
            log_to_text=lambda x: str(x),
        )

        for prompt in (state_prompt, change_prompt, apply_prompt):
            self.assertIn("[Memory]", prompt)
            self.assertNotIn("[User memory]\n{'app_log_id':", prompt)
        self.assertTrue(state_prompt.startswith("As of 2025-01-01 08:00:00"))
        self.assertTrue(change_prompt.startswith("As of 2025-02-01 08:00:00"))
        self.assertTrue(apply_prompt.startswith("What should the assistant recommend?"))
        self.assertNotIn("[Task]", state_prompt)
        self.assertNotIn("Instruction:", state_prompt)
        self.assertNotIn("Query:", state_prompt)
        self.assertNotIn("Checkpoint time:", state_prompt)
        self.assertNotIn("[Task]", change_prompt)
        self.assertNotIn("Instruction:", change_prompt)
        self.assertNotIn("Query:", change_prompt)
        self.assertNotIn("Checkpoint time:", change_prompt)
        self.assertNotIn("[Task]", apply_prompt)
        self.assertNotIn("Instruction:", apply_prompt)
        self.assertNotIn("Query:", apply_prompt)
        self.assertNotIn("Checkpoint time:", apply_prompt)

    def test_final_qa_prompt_variants_match_memory_modes(self) -> None:
        inline_prompt = build_final_qa_prompt_with_inline_memory(
            question="What drink should the assistant recommend?",
            context_text='{"app_log_id": "log_0002", "response": {"favorite_coffee": "espresso"}}',
        )
        agent_prompt = build_final_qa_prompt_with_agent_memory(
            question="What drink should the assistant recommend?"
        )
        self.assertIn("# Retrieved Context", inline_prompt)
        self.assertIn('"answer": string', inline_prompt)
        self.assertIn('"evidence": array', inline_prompt)
        self.assertIn("# Agent memory", agent_prompt)
        self.assertIn("already stored in the agent", agent_prompt)
        self.assertNotIn("# Retrieved Context", agent_prompt)

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

    def test_task_c_v2_generation_and_validation_prompts_follow_checklist_and_structured_schema(self) -> None:
        generation_prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}},
            service_family="user_communication",
        )
        self.assertLess(generation_prompt.index("[Task]"), generation_prompt.index("[What the answering assistant must do]"))
        self.assertLess(generation_prompt.index("[What the answering assistant must do]"), generation_prompt.index("[Definitions]"))
        self.assertLess(generation_prompt.index("[Definitions]"), generation_prompt.index("[Hard Constraints]"))
        self.assertLess(generation_prompt.index("[Hard Constraints]"), generation_prompt.index("[Good Example A Input]"))
        self.assertIn("Generate exactly one low-leakage benchmark item for a habit-conditioned User Communication task.", generation_prompt)
        self.assertNotIn('"service_family": "user_communication"', generation_prompt)
        self.assertIn('"reference_answer":', generation_prompt)
        self.assertIn('"scoring_rubric": {', generation_prompt)
        self.assertIn('"criteria": [', generation_prompt)
        self.assertIn('"item": {', generation_prompt)
        self.assertNotIn('"output_template": {', generation_prompt)
        self.assertNotIn('"reference_output": {', generation_prompt)
        self.assertNotIn('"service_category"', generation_prompt)
        self.assertNotIn("state_type", generation_prompt)
        self.assertNotIn("service_family =", generation_prompt)
        self.assertNotIn('"service_family":', generation_prompt)
        self.assertIn("The answering assistant will see:", generation_prompt)
        self.assertIn("leakage", generation_prompt)
        self.assertIn("free-form natural language", generation_prompt)
        self.assertIn("[Good Example B Input]", generation_prompt)
        self.assertIn("It is Wednesday at 06:50. Nothing has been started yet this morning.", generation_prompt)
        self.assertIn("As the assistant, what single message should be sent to the user right now?", generation_prompt)
        self.assertIn("It is Sunday at 16:45. Everyone is home", generation_prompt)
        self.assertIn("[Bad Example", generation_prompt)

        info_prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="preferences_state:learning_modality",
            state_value={
                "statement": "prefers self-paced webinars",
                "signals": ["downloaded a report", "joined a webinar"],
            },
            service_family="information_request_construction",
        )
        self.assertNotIn("state_type", info_prompt)
        self.assertNotIn('"service_family":', info_prompt)
        self.assertIn("downstream search, retrieval, or recommendation system", info_prompt)
        self.assertIn("resource search is about to run", info_prompt)
        self.assertIn("As the assistant, complete the structured information request below", info_prompt)
        self.assertIn("portfolio-screening request is about to run", info_prompt)
        self.assertIn("[Good Example B Input]", info_prompt)
        self.assertNotIn("user-facing communication action", info_prompt)
        self.assertNotIn("supporting_signals", info_prompt)
        self.assertNotIn('"signals": [', info_prompt)

        action_prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="user_attributes_state:travel_booking_profile",
            state_value={"departure_airport": "ORD"},
            service_family="action_configuration",
        )
        self.assertNotIn("state_type", action_prompt)
        self.assertNotIn('"service_family":', action_prompt)
        self.assertIn("downstream tool or workflow", action_prompt)
        self.assertIn("device-registration step is about to run", action_prompt)
        self.assertIn("Complete the sync settings the assistant should use for this registration step.", action_prompt)
        self.assertIn("profile-update step is about to run", action_prompt)
        self.assertIn("top-level JSON objects", action_prompt)
        self.assertNotIn("resource-search system", action_prompt)

        validation_prompt = build_task_c_v2_validation_prompt(
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}},
            service_family="user_communication",
            scenario="It is 06:10. Nothing has been logged yet today.",
            task_instruction="Write the short reminder message the assistant should send right now.",
            output_template=None,
            reference_output=None,
            reference_answer="Send a reminder that the walk starts at 06:30.",
        )
        self.assertIn('"criterion": "service_completion_quality"', validation_prompt)
        self.assertIn('"criterion": "full_field_dependency"', validation_prompt)
        self.assertIn('"criterion": "low_leakage"', validation_prompt)
        self.assertIn('"criterion": "answer_groundedness"', validation_prompt)
        self.assertNotIn('"criterion": "schema_groundedness"', validation_prompt)
        self.assertNotIn('"criterion": "point_pairability"', validation_prompt)
        self.assertIn("short natural-language assistant response", validation_prompt)
        self.assertIn("restates or paraphrases the habit action", validation_prompt)

    def test_task_c_v2_structured_answer_prompts_use_output_envelope(self) -> None:
        inline_prompt = build_structured_service_completion_prompt_with_inline_memory(
            service_family="information_request_construction",
            scenario="The assistant is preparing a structured information request before ordering coffee.",
            task_instruction="Fill the structured request payload.",
            output_template={"statement": "<fill>"},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )
        agent_prompt = build_structured_service_completion_prompt_with_agent_memory(
            service_family="information_request_construction",
            scenario="The assistant is preparing a structured information request before ordering coffee.",
            task_instruction="Fill the structured request payload.",
            output_template={"statement": "<fill>"},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )

        for prompt in (inline_prompt, agent_prompt):
            self.assertIn('"output": {', prompt)
            self.assertIn('"evidence": [', prompt)
            self.assertNotIn('"answer":', prompt)
            self.assertIn("[Required Output Object]", prompt)
            self.assertNotIn("[Task Type]", prompt)
            self.assertNotIn("information_request_construction", prompt)
        self.assertIn("[Memory]", inline_prompt)
        self.assertIn("[Memory]", agent_prompt)

        user_comm_inline = build_task_c_v2_user_communication_prompt_with_inline_memory(
            scenario="It is 06:10. Nothing has been logged yet today.",
            task_instruction="Write the short reminder message the assistant should send right now.",
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )
        self.assertIn('"answer":', user_comm_inline)
        self.assertNotIn('"output": {', user_comm_inline)
        self.assertIn("one short natural-language assistant response", user_comm_inline)

    def test_generic_task_c_runtime_prompt_builder_supports_text_and_structured_modes(self) -> None:
        text_prompt = build_task_c_prompt_with_inline_memory(
            response_mode="text",
            scenario="It is 06:10. Nothing has been logged yet today.",
            task_instruction="Write the short reminder message the assistant should send right now.",
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )
        structured_prompt = build_task_c_prompt_with_agent_memory(
            response_mode="structured",
            service_family="action_configuration",
            scenario="A device setup flow is being completed before sync starts.",
            task_instruction="Fill the setup payload.",
            output_template={"setup": {"device_model": "<fill>"}},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )

        self.assertIn("[Scenario]", text_prompt)
        self.assertIn("[Task Instruction]", text_prompt)
        self.assertIn('"answer":', text_prompt)
        self.assertNotIn('"output": {', text_prompt)

        self.assertIn("[Required Output Object]", structured_prompt)
        self.assertNotIn("[Task Type]", structured_prompt)
        self.assertIn('"output": {', structured_prompt)
        self.assertNotIn('"answer": "<one short assistant response>"', structured_prompt)

    def test_task_c_v2_rewrite_prompt_preserves_structured_contract(self) -> None:
        rewrite_prompt = build_task_c_v2_rewrite_prompt(
            state_key="preferences_state:learning_modality",
            state_value={"statement": "prefers self-paced webinars"},
            service_family="information_request_construction",
            scenario="The assistant is preparing the request payload before a training-resource search.",
            task_instruction="Fill the structured information-request payload.",
            output_template={"request_profile": {"preferred_profile": "<fill>"}},
            reference_output={"request_profile": {"preferred_profile": "prefers self-paced webinars"}},
            failed_rules=["full_field_dependency", "llm_invalid"],
            semantic_criteria=[
                {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                {"criterion": "full_field_dependency", "pass": False, "analysis": "The state field was treated as optional."},
                {"criterion": "schema_groundedness", "pass": True, "analysis": "ok"},
                {"criterion": "point_pairability", "pass": True, "analysis": "ok"},
            ],
        )
        self.assertIn('"criterion": "full_field_dependency"', rewrite_prompt)
        self.assertIn('"output_template": {', rewrite_prompt)
        self.assertIn('"reference_output": {', rewrite_prompt)
        self.assertNotIn('"service_category"', rewrite_prompt)
        self.assertIn("family-appropriate service object", rewrite_prompt)

        user_comm_rewrite_prompt = build_task_c_v2_rewrite_prompt(
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}},
            service_family="user_communication",
            scenario="It is 06:10. Nothing has been logged yet today.",
            task_instruction="Write the short reminder message the assistant should send right now.",
            output_template=None,
            reference_output=None,
            reference_answer="Send a reminder that the walk starts at 06:30.",
            failed_rules=["low_leakage"],
            semantic_criteria=[
                {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                {"criterion": "low_leakage", "pass": False, "analysis": "The scenario restates the routine."},
                {"criterion": "answer_groundedness", "pass": True, "analysis": "ok"},
            ],
        )
        self.assertIn('"reference_answer":', user_comm_rewrite_prompt)
        self.assertNotIn('"output_template": {', user_comm_rewrite_prompt)
        self.assertIn("natural-language assistant-response form", user_comm_rewrite_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
