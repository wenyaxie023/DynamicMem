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
    build_task_c_task_body,
    build_change_reasoning_prompt_with_agent_memory,
    build_change_reasoning_prompt_with_inline_memory,
    build_task_c_prompt_with_agent_memory,
    build_task_c_prompt_with_inline_memory,
    build_structured_service_completion_prompt_with_agent_memory,
    build_structured_service_completion_prompt_with_inline_memory,
    build_state_completion_prompt_with_agent_memory,
    build_state_completion_prompt_with_inline_memory,
    build_task_c_v2_question_pack_prompt,
    build_task_c_v2_user_communication_prompt_with_inline_memory,
    build_task_c_v2_rewrite_prompt,
    build_task_c_v2_validation_prompt,
    build_value_micro_points_prompt,
    build_value_rubric_rewrite_prompt,
    build_value_rubric_validation_prompt,
)
from tce_core.final_checkpoint_qa import (
    build_final_qa_prompt_with_agent_memory,
    build_final_qa_prompt_with_inline_memory,
)


class TcePromptContractAcceptance(unittest.TestCase):
    def assertNoTaskCV2ScoringAuthoringLanguage(self, prompt: str) -> None:
        forbidden_terms = [
            "scoring_rubric",
            "answer_scoring_points",
            "answer scoring points",
            "scoring point",
            "scoring points",
            "scoring criteria",
            "scoring ids",
            "rubric",
        ]
        lowered_prompt = prompt.lower()
        for term in forbidden_terms:
            self.assertNotIn(term, lowered_prompt)

    def test_state_completion_prompt_uses_direct_question_plus_memory_blocks(self) -> None:
        prompt = build_state_completion_prompt_with_inline_memory(
            context_logs=[{"app_log_id": "log_0001"}],
            task_query="As of 2025-01-01 08:00:00, infer the user's current state for all items in this template: {\"habits_state:morning_walk\": {\"timing\": {\"start_time\": \"<fill the blank>\"}}}.",
            target_keys=["habits_state:morning_walk"],
            target_value_templates={"habits_state:morning_walk": {"timing": {"start_time": "<fill the blank>"}}},
            log_to_text=lambda x: str(x),
        )
        self.assertTrue(prompt.startswith("[Instructions]"))
        self.assertLess(prompt.index("[Instructions]"), prompt.index("As of 2025-01-01 08:00:00"))
        self.assertLess(prompt.index("As of 2025-01-01 08:00:00"), prompt.rindex("[Memory]"))
        self.assertIn("[Memory]", prompt)
        self.assertIn("[Instructions]", prompt)
        self.assertIn("[Output format]", prompt)
        self.assertIn("system memory about the user's trajectory", prompt)
        self.assertIn('"user_state"', prompt)
        self.assertNotIn('"snapshot_state"', prompt)
        self.assertIn("Make each answer value as detailed and accurate", prompt)
        self.assertIn("Habit schedule format vocabulary", prompt)
        self.assertIn("every month on specific calendar date", prompt)
        self.assertIn("every month on an ordinal weekday", prompt)
        self.assertIn('"frequency_type": "monthly_nth_weekday"', prompt)
        self.assertIn('"evidence_content"', prompt)
        self.assertIn("0=Monday", prompt)
        self.assertIn("1=Tuesday", prompt)
        self.assertNotIn("[Task]", prompt)
        self.assertNotIn("Instruction:", prompt)
        self.assertNotIn("Query:", prompt)
        self.assertNotIn("Checkpoint time:", prompt)
        self.assertNotIn("[Rules]", prompt)

    def test_state_completion_prompt_omits_schedule_supplement_for_non_habits(self) -> None:
        prompt = build_state_completion_prompt_with_inline_memory(
            context_logs=[{"app_log_id": "log_0001"}],
            task_query="Infer the user's current state for preferences learning modality.",
            target_keys=["preferences_state:learning_modality"],
            target_value_templates={"preferences_state:learning_modality": {"statement": "<fill the blank>"}},
            log_to_text=lambda x: str(x),
        )
        self.assertIn('"user_state"', prompt)
        self.assertNotIn("Schedule date encoding:", prompt)
        self.assertNotIn("Habit schedule format vocabulary", prompt)

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

        for prompt in (state_prompt, change_prompt):
            self.assertIn("[Memory]", prompt)
            self.assertNotIn("[User memory]\n{'app_log_id':", prompt)
        self.assertTrue(state_prompt.startswith("[Instructions]"))
        self.assertTrue(change_prompt.startswith("As of 2025-02-01 08:00:00"))
        self.assertIn("system's stored checkpoint-bounded memory about the user's trajectory", state_prompt)
        self.assertNotIn("[Task]", state_prompt)
        self.assertNotIn("Instruction:", state_prompt)
        self.assertNotIn("Query:", state_prompt)
        self.assertNotIn("Checkpoint time:", state_prompt)
        self.assertNotIn("[Task]", change_prompt)
        self.assertNotIn("Instruction:", change_prompt)
        self.assertNotIn("Query:", change_prompt)
        self.assertNotIn("Checkpoint time:", change_prompt)

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
        self.assertIn("Generate 1 to 3 scoring points.", prompt)
        self.assertIn("shared binary `0/1` hit scale", prompt)
        self.assertIn("stable: prefer stable semantic requirements over brittle exact surface forms", prompt)
        self.assertIn("distinct: points in the same set must not repeat the same meaning", prompt)
        self.assertIn("coverage_gap", prompt)
        self.assertIn('"rubric": [', prompt)
        self.assertNotIn("reference_value", prompt)
        self.assertNotIn("target_path", prompt)

        validation_prompt = build_value_rubric_validation_prompt(
            state_key="preferences_state:learning_modality",
            text_value="Prefers self-paced white papers and webinars over large conferences.",
            points=[],
        )
        rewrite_prompt = build_value_rubric_rewrite_prompt(
            state_key="preferences_state:learning_modality",
            text_value="Prefers self-paced white papers and webinars over large conferences.",
            validation_feedback={"points": [], "set_analysis": "", "set_failures": []},
            max_points=5,
        )
        for value_prompt in (prompt, validation_prompt, rewrite_prompt):
            self.assertIn("unsupported", value_prompt)
            self.assertIn("drift", value_prompt)
            self.assertIn("not_atomic", value_prompt)
            self.assertIn("redundant", value_prompt)
            self.assertIn("over_specific", value_prompt)
        self.assertIn('"set_analysis":', validation_prompt)

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
        self.assertLess(generation_prompt.index("[Task]"), generation_prompt.index("[Design Principle]"))
        self.assertLess(generation_prompt.index("[Design Principle]"), generation_prompt.index("[Definitions]"))
        self.assertLess(generation_prompt.index("[Definitions]"), generation_prompt.index("[Hard Constraints]"))
        self.assertLess(generation_prompt.index("[Hard Constraints]"), generation_prompt.index("[Good Example A Input]"))
        self.assertIn("Generate exactly one habit-conditioned communication task for a user-facing assistant.", generation_prompt)
        self.assertNotIn('"service_family": "user_communication"', generation_prompt)
        self.assertIn('"reference_answer":', generation_prompt)
        self.assertNotIn("scoring_rubric", generation_prompt)
        self.assertNotIn("scoring criteria", generation_prompt)
        self.assertIn('"item": {', generation_prompt)
        self.assertNotIn('"output_template": {', generation_prompt)
        self.assertNotIn('"reference_output": {', generation_prompt)
        self.assertNotIn('"service_category"', generation_prompt)
        self.assertNotIn("state_type", generation_prompt)
        self.assertNotIn("service_family =", generation_prompt)
        self.assertNotIn('"service_family":', generation_prompt)
        self.assertIn("Each item contains:", generation_prompt)
        self.assertIn("`task_instruction`: copy the fixed string exactly.", generation_prompt)
        self.assertIn("leakage", generation_prompt)
        self.assertIn("current-world anchor", generation_prompt)
        self.assertIn("0=Monday", generation_prompt)
        self.assertIn("days_of_week: [1] means Tuesday, not Monday", generation_prompt)
        self.assertIn("a clock time alone is not enough", generation_prompt)
        self.assertIn("Bad Example 2", generation_prompt)
        self.assertIn("free-form natural language", generation_prompt)
        self.assertNotIn('[Required Rubric Skeleton]', generation_prompt)
        self.assertNotIn('"id": "timing.start_time"', generation_prompt)
        self.assertIn("silently confirm the item would later pass the same five semantic validation criteria", generation_prompt)
        self.assertIn("answerability", generation_prompt)
        self.assertNotIn("checkpoint_timestamp", generation_prompt)
        self.assertIn("service_realism", generation_prompt)
        self.assertIn("full_field_dependency", generation_prompt)
        self.assertIn("low_leakage", generation_prompt)
        self.assertIn("output_groundedness", generation_prompt)
        self.assertIn("[Good Example B Input]", generation_prompt)
        self.assertIn("It is Monday at 09:20. Nothing has been started yet this morning.", generation_prompt)
        self.assertIn("Draft a concise but complete reminder text for the user in this scenario.", generation_prompt)
        self.assertIn("[Bad Example", generation_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(generation_prompt)

        info_prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="preferences_state:learning_modality",
            state_value={"statement": "prefers self-paced webinars"},
            service_family="information_request_construction",
        )
        self.assertNotIn("state_type", info_prompt)
        self.assertNotIn('"service_family":', info_prompt)
        self.assertIn("structured search/filter object for the assistant to apply before showing matching options", info_prompt)
        self.assertIn("Fill the search filters the assistant should apply now.", info_prompt)
        self.assertIn("Generate exactly one preference-conditioned search-filter task", info_prompt)
        self.assertIn("0=Monday", info_prompt)
        self.assertIn("The user is reviewing investment options for an upcoming planning session.", info_prompt)
        self.assertIn("The assistant is setting strategy filters before showing matching options.", info_prompt)
        self.assertNotIn("checkpoint_timestamp", info_prompt)
        self.assertIn("[Good Example B Input]", info_prompt)
        self.assertIn("silently confirm the item would later pass the same five semantic validation criteria", info_prompt)
        self.assertIn("service_realism", info_prompt)
        self.assertIn("full_field_dependency", info_prompt)
        self.assertIn("output_groundedness", info_prompt)
        self.assertNotIn("user-facing communication action", info_prompt)
        self.assertNotIn("supporting_signals", info_prompt)
        self.assertNotIn('"signals": [', info_prompt)
        self.assertNotIn('"scoring_rubric": {', info_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(info_prompt)

        action_prompt = build_task_c_v2_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="user_attributes_state:travel_booking_profile",
            state_value={"departure_airport": "ORD"},
            service_family="action_configuration",
        )
        self.assertNotIn("state_type", action_prompt)
        self.assertNotIn('"service_family":', action_prompt)
        self.assertIn("Generate exactly one attribute-conditioned action-configuration task", action_prompt)
        self.assertIn("0=Monday", action_prompt)
        self.assertIn("structured action-configuration object for a user-facing tool", action_prompt)
        self.assertIn("Use the user's known attributes to fill the setup or form fields", action_prompt)
        self.assertNotIn("checkpoint_timestamp", action_prompt)
        self.assertIn("the user is connecting a device or account", action_prompt)
        self.assertIn("the user is preparing a profile or form before submission", action_prompt)
        self.assertIn("top-level JSON objects", action_prompt)
        self.assertIn("silently confirm the item would later pass the same five semantic validation criteria", action_prompt)
        self.assertIn("service_realism", action_prompt)
        self.assertIn("full_field_dependency", action_prompt)
        self.assertIn("output_groundedness", action_prompt)
        self.assertNotIn("resource-search system", action_prompt)
        self.assertNotIn('"scoring_rubric": {', action_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(action_prompt)

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
        self.assertIn('"criterion": "answerability"', validation_prompt)
        self.assertIn('"criterion": "service_realism"', validation_prompt)
        self.assertIn('"criterion": "full_field_dependency"', validation_prompt)
        self.assertIn('"criterion": "low_leakage"', validation_prompt)
        self.assertIn('"criterion": "output_groundedness"', validation_prompt)
        self.assertIn("short natural-language assistant response", validation_prompt)
        self.assertIn("field paths in `state_value`", validation_prompt)
        self.assertIn("a clock time alone is not enough", validation_prompt)
        self.assertIn("0=Monday", validation_prompt)
        self.assertIn("1=Tuesday", validation_prompt)
        self.assertIn("conflicts with encoded schedule fields", validation_prompt)
        self.assertIn("Current weekday/date/time are allowed", validation_prompt)
        self.assertIn("restates or paraphrases the habit action/identity", validation_prompt)
        self.assertIn('state_key: "habits_state:evening_walk"', validation_prompt)
        self.assertIn("It is 18:50. The evening is quiet and nothing has been started.", validation_prompt)
        self.assertIn('"pass": false', validation_prompt)
        self.assertIn("The scenario gives only a clock time", validation_prompt)
        self.assertNotIn("preferences_state:learning_modality", validation_prompt)
        self.assertNotIn("Task C v2", validation_prompt)
        self.assertNotIn("service_family", validation_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(validation_prompt)

    def test_task_c_v2_structured_answer_prompts_use_output_envelope(self) -> None:
        inline_prompt = build_structured_service_completion_prompt_with_inline_memory(
            scenario="The assistant is preparing a structured information request before ordering coffee.",
            task_instruction="Fill the structured request payload.",
            output_template={"statement": "<fill>"},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )
        agent_prompt = build_structured_service_completion_prompt_with_agent_memory(
            scenario="The assistant is preparing a structured information request before ordering coffee.",
            task_instruction="Fill the structured request payload.",
            output_template={"statement": "<fill>"},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )

        for prompt in (inline_prompt, agent_prompt):
            self.assertIn('"answer": {', prompt)
            self.assertIn('"evidence": [', prompt)
            self.assertNotIn('"output": {', prompt)
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
        self.assertIn("Put the completed task result in `answer`.", user_comm_inline)
        self.assertIn("concise but complete assistant message", user_comm_inline)
        self.assertIn('"answer": "<concise but complete assistant message>"', user_comm_inline)
        self.assertNotIn("one short natural-language assistant response", user_comm_inline)
        task_body_idx = user_comm_inline.index("\n\n[Assistant Task]\n")
        memory_idx = user_comm_inline.index("\n\n[Memory]\n", task_body_idx)
        self.assertLess(user_comm_inline.index("[Instructions]"), task_body_idx)
        self.assertLess(task_body_idx, memory_idx)

    def test_generic_task_c_runtime_prompt_builder_supports_text_and_structured_modes(self) -> None:
        text_prompt = build_task_c_prompt_with_inline_memory(
            response_mode="text",
            task_body=build_task_c_task_body(
                scenario="It is 06:10. Nothing has been logged yet today.",
                task_instruction="Write the short reminder message the assistant should send right now.",
            ),
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )
        structured_prompt = build_task_c_prompt_with_agent_memory(
            response_mode="structured",
            task_body=build_task_c_task_body(
                scenario="A device setup flow is being completed before sync starts.",
                task_instruction="Fill the setup payload.",
                output_template={"setup": {"device_model": "<fill>"}},
            ),
            output_template={"setup": {"device_model": "<fill>"}},
            context_logs=[{"app_log_id": "log_0001"}],
            log_to_text=lambda x: str(x),
        )

        self.assertIn("[Scenario]", text_prompt)
        self.assertIn("[Task Instruction]", text_prompt)
        self.assertIn('"answer":', text_prompt)
        self.assertIn("Return JSON only; do not write any text outside the JSON object.", text_prompt)
        self.assertIn("Put the completed task result in `answer`.", text_prompt)
        self.assertNotIn('"output": {', text_prompt)
        text_task_body_idx = text_prompt.index("\n\n[Assistant Task]\n")
        text_memory_idx = text_prompt.index("\n\n[Memory]\n", text_task_body_idx)
        self.assertLess(text_prompt.index("[Instructions]"), text_task_body_idx)
        self.assertLess(text_task_body_idx, text_memory_idx)

        self.assertIn("[Required Output Object]", structured_prompt)
        self.assertNotIn("[Task Type]", structured_prompt)
        self.assertIn('"answer": {', structured_prompt)
        self.assertNotIn('"output": {', structured_prompt)
        self.assertNotIn('"answer": "<one short assistant response>"', structured_prompt)
        structured_task_body_idx = structured_prompt.index("\n\n[Assistant Task]\n")
        structured_memory_idx = structured_prompt.index("\n\n[Memory]\n", structured_task_body_idx)
        self.assertLess(structured_prompt.index("[Instructions]"), structured_task_body_idx)
        self.assertLess(structured_task_body_idx, structured_memory_idx)

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
                {"criterion": "answerability", "pass": True, "analysis": "ok"},
                {"criterion": "service_realism", "pass": True, "analysis": "ok"},
                {"criterion": "full_field_dependency", "pass": False, "analysis": "The state field was treated as optional."},
                {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
            ],
        )
        self.assertIn('"criterion": "full_field_dependency"', rewrite_prompt)
        self.assertIn('"output_template": {', rewrite_prompt)
        self.assertIn('"reference_output": {', rewrite_prompt)
        self.assertNotIn('"scoring_rubric": {', rewrite_prompt)
        self.assertNotIn('"path": "request_profile.preferred_profile"', rewrite_prompt)
        self.assertNotIn('"service_category"', rewrite_prompt)
        self.assertIn("task-appropriate service object", rewrite_prompt)
        self.assertNotIn("source leaves", rewrite_prompt)
        self.assertNotIn("scoring_rubric", rewrite_prompt)
        self.assertNotIn("scoring criteria", rewrite_prompt)
        self.assertIn("delta patch over mutable fields only", rewrite_prompt)
        self.assertIn("Include only the fields you actually changed", rewrite_prompt)
        self.assertIn("validation_feedback", rewrite_prompt)
        self.assertIn("structural_issues", rewrite_prompt)
        self.assertIn("[Repair Instructions]", rewrite_prompt)
        self.assertIn("preference-conditioned search-filter completion", rewrite_prompt)
        self.assertIn("0=Monday", rewrite_prompt)
        self.assertIn("content_search_filters", rewrite_prompt)
        self.assertIn("preference_statement", rewrite_prompt)
        self.assertNotIn("llm_invalid", rewrite_prompt)
        self.assertNotIn("wearable_sync_setup", rewrite_prompt)
        self.assertNotIn("service_family", rewrite_prompt)
        self.assertNotIn('"service_family": "information_request_construction"', rewrite_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(rewrite_prompt)

        action_rewrite_prompt = build_task_c_v2_rewrite_prompt(
            state_key="user_attributes_state:fitness_technology",
            state_value=[
                "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
                "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)",
            ],
            service_family="action_configuration",
            scenario="The user has an Apple Watch Series 9 and an Oura Ring Gen3, so connected devices are being configured.",
            task_instruction="Fill the structured action configuration before the sync setup is saved.",
            output_template={"devices": ["<fill>", "<fill>"]},
            reference_output={
                "devices": [
                    "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
                    "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)",
                ]
            },
            failed_rules=["service_realism", "low_leakage", "output_groundedness"],
            semantic_criteria=[
                {"criterion": "answerability", "pass": True, "analysis": "ok"},
                {"criterion": "service_realism", "pass": False, "analysis": "raw list"},
                {"criterion": "full_field_dependency", "pass": False, "analysis": "collapsed fields"},
                {"criterion": "low_leakage", "pass": False, "analysis": "leaks devices"},
                {"criterion": "output_groundedness", "pass": False, "analysis": "raw mirror"},
            ],
        )
        self.assertIn("attribute-conditioned action-configuration", action_rewrite_prompt)
        self.assertIn("0=Monday", action_rewrite_prompt)
        self.assertIn("sports_equipment_profile", action_rewrite_prompt)
        self.assertIn("family sports equipment inventory", action_rewrite_prompt)
        self.assertIn("execution-ready configuration object", action_rewrite_prompt)
        self.assertNotIn("preferences_state:learning_modality", action_rewrite_prompt)
        self.assertNotIn("content_search_filters", action_rewrite_prompt)
        self.assertNotIn("service_family", action_rewrite_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(action_rewrite_prompt)

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
                {"criterion": "answerability", "pass": True, "analysis": "ok"},
                {"criterion": "service_realism", "pass": True, "analysis": "ok"},
                {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                {"criterion": "low_leakage", "pass": False, "analysis": "The scenario restates the routine."},
                {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
            ],
        )
        self.assertIn('"reference_answer":', user_comm_rewrite_prompt)
        self.assertNotIn('"output_template": {', user_comm_rewrite_prompt)
        self.assertNotIn('"scoring_rubric": {', user_comm_rewrite_prompt)
        self.assertNotIn('"id": "timing.start_time"', user_comm_rewrite_prompt)
        self.assertIn("natural-language assistant-response form", user_comm_rewrite_prompt)
        self.assertNotIn("scoring_rubric", user_comm_rewrite_prompt)
        self.assertNotIn("scoring criteria", user_comm_rewrite_prompt)
        self.assertIn("delta patch over mutable fields only", user_comm_rewrite_prompt)
        self.assertIn("Include only the fields you actually changed", user_comm_rewrite_prompt)
        self.assertIn("validation_feedback", user_comm_rewrite_prompt)
        self.assertIn("[Repair Instructions]", user_comm_rewrite_prompt)
        self.assertIn("keep current weekday/date/time", user_comm_rewrite_prompt)
        self.assertIn("0=Monday", user_comm_rewrite_prompt)
        self.assertIn("conflicts with encoded schedule fields", user_comm_rewrite_prompt)
        self.assertIn("It is Sunday at 16:45", user_comm_rewrite_prompt)
        self.assertNotIn("service_family", user_comm_rewrite_prompt)
        self.assertNotIn('"service_family": "user_communication"', user_comm_rewrite_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(user_comm_rewrite_prompt)

        structured_validation_prompt = build_task_c_v2_validation_prompt(
            state_key="preferences_state:learning_modality",
            state_value={"statement": "prefers self-paced webinars"},
            service_family="information_request_construction",
            scenario="A training-resource search request is about to run.",
            task_instruction="Fill the structured request payload before the search is sent.",
            output_template={"request_profile": {"preferred_format": "<fill>"}},
            reference_output={"request_profile": {"preferred_format": "self-paced webinars"}},
            reference_answer="",
        )
        self.assertIn('"criterion": "answerability"', structured_validation_prompt)
        self.assertIn('"criterion": "low_leakage"', structured_validation_prompt)
        self.assertIn('"criterion": "output_groundedness"', structured_validation_prompt)
        self.assertIn("preference-conditioned search-filter completion task", structured_validation_prompt)
        self.assertIn("0=Monday", structured_validation_prompt)
        self.assertIn("content_search_filters", structured_validation_prompt)
        self.assertIn("preference_statement", structured_validation_prompt)
        self.assertIn("The scenario restates the user's actual coffee-shop preference.", structured_validation_prompt)
        self.assertIn('"pass": false', structured_validation_prompt)
        self.assertNotIn('"scoring_rubric": {', structured_validation_prompt)
        self.assertNotIn("preserve every source leaf", structured_validation_prompt)
        self.assertNotIn("source-leaf order", structured_validation_prompt)
        self.assertIn("task-appropriate search/filter object", structured_validation_prompt)
        self.assertIn("field paths in `state_value`", structured_validation_prompt)
        self.assertNotIn("wearable_sync_setup", structured_validation_prompt)
        self.assertNotIn("Task C v2", structured_validation_prompt)
        self.assertNotIn("service_family", structured_validation_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(structured_validation_prompt)

        action_validation_prompt = build_task_c_v2_validation_prompt(
            state_key="user_attributes_state:fitness_technology",
            state_value=[
                "Apple Watch Series 9 (Midnight aluminum, used for daily heart rate and step tracking)",
                "Oura Ring Gen3 (Stealth finish, primarily for sleep staging and recovery metrics)",
            ],
            service_family="action_configuration",
            scenario="A wellness app setup is being finalized. Connected-device sources are being configured before health data syncing starts.",
            task_instruction="Fill the structured action configuration before the sync setup is saved.",
            output_template={"wearable_sync_setup": {"connected_sources": []}},
            reference_output={"wearable_sync_setup": {"connected_sources": []}},
            reference_answer="",
        )
        self.assertIn("attribute-conditioned action-configuration task", action_validation_prompt)
        self.assertIn("0=Monday", action_validation_prompt)
        self.assertIn("wearable_sync_setup", action_validation_prompt)
        self.assertIn("Connected-device sources", action_validation_prompt)
        self.assertIn("executable sync configuration", action_validation_prompt)
        self.assertIn("not that all 10 balls and cones should be donated", action_validation_prompt)
        self.assertNotIn("preferences_state:learning_modality", action_validation_prompt)
        self.assertNotIn("content_search_filters", action_validation_prompt)
        self.assertNotIn("Task C v2", action_validation_prompt)
        self.assertNotIn("service_family", action_validation_prompt)
        self.assertNoTaskCV2ScoringAuthoringLanguage(action_validation_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
