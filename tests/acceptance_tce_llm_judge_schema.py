#!/usr/bin/env python3
import argparse
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prompts_tce import (
    build_apply_holistic_judge_prompt,
    build_apply_slot_judge_prompt,
    build_change_slot_judge_prompt,
    build_snapshot_holistic_judge_prompt,
    build_snapshot_slot_judge_prompt,
)
from eval.eval_tce import (
    _apply_eval_config,
    _apply_holistic_fields,
    _load_eval_config,
    _normalize_snapshot_holistic_judgments,
    _normalize_slot_judgments,
    _run_apply_holistic_judge,
    _run_apply_slot_judge,
    _run_change_slot_judge,
    _run_snapshot_holistic_judge,
    _run_snapshot_slot_judge,
    evaluate,
)


class TceSlotLlmJudgeSchemaAcceptance(unittest.TestCase):
    def setUp(self):
        text_format_patch = patch("eval.eval_tce._build_slot_judge_text_format", return_value=dict)
        text_format_patch.start()
        self.addCleanup(text_format_patch.stop)
        holistic_text_format_patch = patch("eval.eval_tce._build_snapshot_holistic_judge_text_format", return_value=dict)
        holistic_text_format_patch.start()
        self.addCleanup(holistic_text_format_patch.stop)

    def test_slot_prompts_use_boolean_correctness(self):
        slots = [
            {
                "point_id": "p1",
                "point_type": "field",
                "polarity": "positive",
                "reference_value": "06:30",
                "predicted_value": "06:30",
            }
        ]
        snapshot_prompt = build_snapshot_slot_judge_prompt(
            state_key="habits_state:morning_walk",
            slots=slots,
        )
        change_prompt = build_change_slot_judge_prompt(
            state_key="habits_state:morning_walk",
            slots=slots,
        )
        apply_prompt = build_apply_slot_judge_prompt(
            state_key="habits_state:morning_walk",
            qa_id="q1",
            question="What should the assistant do?",
            reference_answer="Keep the walk at 06:30.",
            predicted_answer="Keep it at 06:30.",
            slots=slots,
        )
        for prompt in (snapshot_prompt, change_prompt, apply_prompt):
            self.assertIn('"correct": "<bool>"', prompt)
            self.assertIn('"analysis": "<brief checklist-item judgment>"', prompt)
            self.assertNotIn("core_value_match", prompt)
            self.assertNotIn("constraint_coverage", prompt)
            self.assertNotIn("value_precision", prompt)
            self.assertNotIn("unsupported_inference_control", prompt)
            self.assertIn("[Prediction]", prompt)
            self.assertIn("[Checklist]", prompt)
            self.assertIn("Treat the checklist as unordered", prompt)
            self.assertIn("short checklist identifier", prompt)
            self.assertIn("core idea or practical value", prompt)
            self.assertIn("do not require exact wording", prompt)
            self.assertIn("semantic equivalents", prompt)
            self.assertIn("weekday name/index encodings where 0=Monday", prompt)
            self.assertIn("6=Sunday", prompt)
            self.assertIn("do not move a value from a different field or indexed position", prompt)
            self.assertNotIn("TCE Task", prompt)
            self.assertNotIn("Task C v2", prompt)
            self.assertNotIn("[Unit Context]", prompt)
            self.assertNotIn("[Item Context]", prompt)
        self.assertNotIn('"reference_value"', snapshot_prompt)
        self.assertNotIn('"predicted_value"', snapshot_prompt)
        self.assertNotIn("if omitted, treat it as `positive`", snapshot_prompt)
        self.assertNotIn('"polarity": "positive"', snapshot_prompt)
        self.assertNotIn("polarity =", change_prompt)
        self.assertNotIn("polarity =", apply_prompt)
        self.assertNotIn('"point_text": "The value is 06:30."', snapshot_prompt)

    def test_snapshot_holistic_prompt_uses_core_detail_schema_without_points(self):
        prompt = build_snapshot_holistic_judge_prompt(
            state_key="habits_state:morning_walk",
            golden_state_value={
                "timing": {"start_time": "06:30"},
                "schedule": {"frequency_type": "monthly_nth_weekday", "week_of_month": "last", "day_of_week": 4},
            },
            predicted_state_value="Morning walk on the last Friday of each month at 06:30.",
            fields_to_judge=[
                {"field_path": "schedule.frequency_type", "golden_field_value": "monthly_nth_weekday"},
                {"field_path": "schedule.week_of_month", "golden_field_value": "last"},
                {"field_path": "schedule.day_of_week", "golden_field_value": 4},
                {"field_path": "timing.start_time", "golden_field_value": "06:30"},
            ],
        )
        self.assertIn("predicted personal profile entry matches a reference profile entry", prompt)
        self.assertIn("fields_to_judge", prompt)
        self.assertIn("field_judgments", prompt)
        self.assertIn('"field_path": "<field_path>"', prompt)
        self.assertIn('"golden"', prompt)
        self.assertIn('"predicted"', prompt)
        self.assertIn('"analysis": "<brief analysis before the labels>"', prompt)
        self.assertIn('"core_correct": "<bool>"', prompt)
        self.assertIn('"detail_quality": "<0|1|2 integer>"', prompt)
        self.assertIn("Use `state_key` to identify the routine", prompt)
        self.assertIn("For `timing.*` fields", prompt)
        self.assertIn("core meaning", prompt)
        self.assertIn("supporting precision beyond the core", prompt)
        self.assertIn("habit core_correct examples", prompt)
        self.assertIn("habit detail_quality examples", prompt)
        self.assertIn("weekday names versus weekday indexes", prompt)
        self.assertIn("field path `value` means the entire profile entry", prompt)
        self.assertNotIn("golden_state_value", prompt)
        self.assertNotIn("predicted_state_value", prompt)
        self.assertNotIn("golden_field_value", prompt)
        self.assertNotIn("TCE Task", prompt)
        self.assertNotIn("Task A", prompt)
        self.assertNotIn("state_family", prompt)
        self.assertNotIn("scoring_points", prompt)
        self.assertNotIn("checklist", prompt.lower())
        self.assertNotIn("point_id", prompt)
        self.assertLess(prompt.index('"analysis"'), prompt.index('"core_correct"'))

    def test_snapshot_holistic_normalization_computes_weighted_score(self):
        normalized = _normalize_snapshot_holistic_judgments(
            {
                "field_judgments": [
                    {
                        "field_path": "statement",
                        "analysis": "Core is right, one detail is missing.",
                        "core_correct": True,
                        "detail_quality": 1,
                    }
                ]
            },
            [{"field_path": "statement"}],
        )
        self.assertEqual(normalized["statement"]["analysis"], "Core is right, one detail is missing.")
        self.assertTrue(normalized["statement"]["core_correct"])
        self.assertEqual(normalized["statement"]["detail_quality"], 1)
        self.assertAlmostEqual(normalized["statement"]["score_0_1"], 0.9)

    def test_apply_holistic_prompt_uses_core_detail_schema_without_points(self):
        prompt = build_apply_holistic_judge_prompt(
            state_key="preferences_state:exercise_environment",
            qa_id="q1",
            service_family="information_request_construction",
            scenario="The user is exploring local fitness facilities and workout classes.",
            task_instruction="Fill the search filters the assistant should apply now before showing matches.",
            reference_output={
                "fitness_search_criteria": {
                    "preferred_environment": "climate-controlled indoor exercise environments",
                    "excluded_environment": "outdoor activities",
                }
            },
            predicted_output={
                "fitness_search_criteria": {
                    "preferred_environment": "Home Gym",
                    "excluded_environment": "",
                }
            },
            fields_to_judge=[
                {"field_path": "fitness_search_criteria.preferred_environment"},
                {"field_path": "fitness_search_criteria.excluded_environment"},
            ],
        )
        self.assertIn("Core + Detail field evaluation method", prompt)
        self.assertIn("predicted assistant response matches the reference response", prompt)
        self.assertIn("core belongs to the service output field being judged", prompt)
        self.assertIn("Do not penalize the prediction for not restating source records", prompt)
        self.assertNotIn("service_topic", prompt)
        self.assertNotIn("user memory", prompt)
        self.assertNotIn("memory_topic", prompt)
        self.assertNotIn("memory record", prompt)
        self.assertNotIn("memory records", prompt)
        self.assertNotIn("Task C", prompt)
        self.assertNotIn("Task A", prompt)
        self.assertNotIn("service_type", prompt)
        self.assertNotIn('"state_key"', prompt)
        self.assertIn("field_judgments", prompt)
        self.assertIn('"core_correct": "<bool>"', prompt)
        self.assertIn('"detail_quality": "<0|1|2 integer>"', prompt)
        self.assertIn("structured search/filter core_correct examples", prompt)
        self.assertIn('"preferred_environment": "climate-controlled indoor exercise environments"', prompt)
        self.assertNotIn("preferred_formats", prompt)
        self.assertNotIn("[\"self-paced webinars\"", prompt)
        self.assertIn("Do not use scoring points", prompt)
        self.assertNotIn("answer_scoring_points", prompt)
        self.assertNotIn("point_id", prompt)
        self.assertLess(prompt.index('"analysis"'), prompt.index('"core_correct"'))

    def test_apply_holistic_user_communication_uses_checklist_fields(self):
        fields = _apply_holistic_fields(
            {
                "service_family": "user_communication",
                "reference_answer": "Your morning walk starts at 06:30 on the lakefront trail.",
                "predicted_answer": "Your morning walk starts at 06:30.",
                "slots": [
                    {
                        "point_id": "identity",
                        "point_type": "micro",
                        "point_role": "identity_gate",
                        "point_text": "The message is clearly about the morning walk routine itself.",
                        "predicted_value": "Your morning walk starts at 06:30.",
                    },
                    {
                        "point_id": "start",
                        "point_type": "micro",
                        "source_field_path": "timing.start_time",
                        "point_text": "The message correctly uses timing.start_time with value 06:30.",
                        "reference_value": "06:30",
                        "predicted_value": "Your morning walk starts at 06:30.",
                    },
                    {
                        "point_id": "place",
                        "point_type": "micro",
                        "source_field_path": "location",
                        "point_text": "The message correctly uses location with value lakefront trail.",
                        "reference_value": "lakefront trail",
                        "predicted_value": "Your morning walk starts at 06:30.",
                    },
                ],
            }
        )
        self.assertEqual([field["field_path"] for field in fields], ["identity_gate", "timing.start_time", "location"])
        prompt = build_apply_holistic_judge_prompt(
            state_key="habits_state:morning_walk",
            qa_id="q1",
            service_family="user_communication",
            scenario="It is Monday at 06:10.",
            task_instruction="Draft the reminder text.",
            reference_answer="Your morning walk starts at 06:30 on the lakefront trail.",
            predicted_answer="Your morning walk starts at 06:30.",
            fields_to_judge=fields,
        )
        self.assertIn('"field_path": "timing.start_time"', prompt)
        self.assertIn('"field_path": "location"', prompt)
        self.assertIn('"criterion": "The message correctly uses location with value lakefront trail."', prompt)
        self.assertIn('"reference": "Your morning walk starts at 06:30 on the lakefront trail."', prompt)
        self.assertNotIn('"field_path": "answer"', prompt)
        self.assertNotIn("service_topic", prompt)
        self.assertNotIn("user memory", prompt)

    def test_apply_slot_prompt_uses_user_communication_v2_context(self):
        slots = [
            {
                "point_id": "p1",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The message should mention that the walk starts at 06:30.",
                "predicted_value": "Morning walk starts at 06:30 on the lakefront trail.",
            }
        ]
        prompt = build_apply_slot_judge_prompt(
            state_key="habits_state:morning_walk",
            qa_id="q1",
            reference_answer="Your morning walk starts at 06:30 on the lakefront trail.",
            predicted_answer="Morning walk starts at 06:30 on the lakefront trail.",
            service_family="user_communication",
            scenario="It is Wednesday at 06:10. Nothing has been logged yet this morning.",
            task_instruction="As the assistant, what single message should be sent to the user right now?",
            slots=slots,
        )
        self.assertIn("You are evaluating an assistant message.", prompt)
        self.assertNotIn("TCE Task C user communication", prompt)
        self.assertNotIn('"service_family": "user_communication"', prompt)
        self.assertNotIn('"qa_id": "q1"', prompt)
        self.assertNotIn('"scenario": "It is Wednesday at 06:10. Nothing has been logged yet this morning."', prompt)
        self.assertNotIn('"assistant_task": "As the assistant, what single message should be sent to the user right now?"', prompt)
        self.assertIn("Judge each checklist item independently", prompt)
        self.assertIn("do not require wording overlap", prompt)
        self.assertNotIn("reference_information", prompt)
        self.assertNotIn("if omitted, treat it as `positive`", prompt)
        self.assertNotIn('"polarity": "positive"', prompt)

    def test_structured_task_c_families_use_family_specific_examples(self):
        slots = [
            {
                "point_id": "p1",
                "point_type": "field",
                "point_text": "The structured output includes the required value.",
                "predicted_value": "ok",
            }
        ]
        info_prompt = build_apply_slot_judge_prompt(
            state_key="preferences_state:investment_style",
            qa_id="q1",
            service_family="information_request_construction",
            predicted_output={"allocation_parameters": {"time_frame": "long-term"}},
            slots=slots,
        )
        action_prompt = build_apply_slot_judge_prompt(
            state_key="user_attributes_state:community_memberships",
            qa_id="q1",
            service_family="action_configuration",
            predicted_output={
                "membership_directory_submission": {
                    "affiliation": {"organization_name": "American Coatings Association"}
                }
            },
            slots=slots,
        )
        self.assertIn("structured information request output", info_prompt)
        self.assertIn("allocation_parameters", info_prompt)
        self.assertNotIn("membership_directory_submission", info_prompt)
        self.assertIn("structured action setup output", action_prompt)
        self.assertIn("membership_directory_submission", action_prompt)
        self.assertNotIn("allocation_parameters", action_prompt)
        self.assertNotIn('"service_family":', info_prompt)
        self.assertNotIn('"service_family":', action_prompt)

    def test_slot_parser_defaults_missing_points_to_false(self):
        normalized = _normalize_slot_judgments(
            {
                "judgments": [
                    {
                        "point_id": "p1",
                        "analysis": "matches",
                        "correct": True,
                    }
                ]
            },
            ["p1", "p2"],
        )
        self.assertTrue(normalized["p1"]["correct"])
        self.assertFalse(normalized["p2"]["correct"])
        self.assertIn("missing", normalized["p2"]["analysis"])

    def test_slot_parser_maps_short_prompt_point_ids_to_canonical_ids(self):
        normalized = _normalize_slot_judgments(
            {
                "judgments": [
                    {
                        "point_id": "1",
                        "analysis": "matches",
                        "correct": True,
                    }
                ]
            },
            ["scp_long_internal_id"],
            ["1"],
        )
        self.assertTrue(normalized["scp_long_internal_id"]["correct"])
        self.assertEqual(normalized["scp_long_internal_id"]["analysis"], "matches")

    def test_snapshot_slot_judge_requests_are_key_level(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "p1", "analysis": "ok", "correct": True},
                            {"point_id": "p2", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_snapshot_slots_by_key": {
                    "a:x": [
                        {"point_id": "p1", "point_type": "field", "reference_value": 1, "predicted_value": 1},
                    ],
                    "a:y": [
                        {"point_id": "p2", "point_type": "field", "reference_value": 2, "predicted_value": 2},
                    ],
                }
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, audit_records = _run_snapshot_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 2)
        self.assertEqual(len(audit_records), 2)
        self.assertEqual(len(FakeClient.prompts), 2)
        self.assertTrue(all('"state_key": "a:' not in prompt for prompt in FakeClient.prompts))
        self.assertTrue(all("short checklist identifier" in prompt for prompt in FakeClient.prompts))
        self.assertIn("snapshot_slot_eval_by_key", rows[0])
        slot_eval = rows[0]["snapshot_slot_eval_by_key"]["a:x"]
        self.assertIn("slot_context", slot_eval)
        self.assertIn("judgments", slot_eval)

    def test_snapshot_slot_judge_resume_skips_completed_keys(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "p2", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_snapshot_slots_by_key": {
                    "a:x": [
                        {"point_id": "p1", "point_type": "field", "reference_value": 1, "predicted_value": 1},
                    ],
                    "a:y": [
                        {"point_id": "p2", "point_type": "field", "reference_value": 2, "predicted_value": 2},
                    ],
                },
                "snapshot_slot_eval_by_key": {
                    "a:x": {
                        "score_0_1": 1.0,
                        "slot_count": 1,
                        "slot_context": [{"point_id": "p1"}],
                        "judgments": [{"point_id": "p1", "analysis": "cached", "correct": True}],
                    }
                },
                "snapshot_slot_judge_reason": "cached",
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, _audit_records = _run_snapshot_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"state_key": "a:y"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "p2"', FakeClient.prompts[0])
        self.assertIn("a:x", rows[0]["snapshot_slot_eval_by_key"])
        self.assertIn("a:y", rows[0]["snapshot_slot_eval_by_key"])
        self.assertEqual(rows[0]["snapshot_slot_eval_by_key"]["a:x"]["judgments"][0]["analysis"], "cached")
        self.assertEqual(rows[0]["snapshot_point_score_mean_on_expected"], 1.0)

    def test_snapshot_holistic_judge_requests_are_key_level(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "field_judgments": [
                            {
                                "field_path": "timing.start_time",
                                "analysis": "The start time matches.",
                                "core_correct": True,
                                "detail_quality": 2,
                            },
                            {
                                "field_path": "current_value",
                                "analysis": "The preference value matches.",
                                "core_correct": True,
                                "detail_quality": 2,
                            }
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "checkpoint_id": "cp_0001",
                "_expected_snapshot": {
                    "habits_state:morning_walk": {"timing": {"start_time": "06:30"}},
                    "preferences:learning_format": {"current_value": "webinar"},
                },
                "_pred_snapshot": {
                    "habits_state:morning_walk": "Morning walk at 06:30.",
                    "preferences:learning_format": "Webinar.",
                },
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, audit_records = _run_snapshot_holistic_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 2)
        self.assertEqual(len(audit_records), 2)
        self.assertEqual(len(FakeClient.prompts), 2)
        self.assertTrue(all("scoring_points" not in prompt for prompt in FakeClient.prompts))
        self.assertTrue(all("checklist" not in prompt.lower() for prompt in FakeClient.prompts))
        self.assertIn("snapshot_holistic_eval_by_key", rows[0])
        record = rows[0]["snapshot_holistic_eval_by_key"]["habits_state:morning_walk"]
        self.assertEqual(record["field_count"], 1)
        self.assertEqual(record["field_judgments"][0]["analysis"], "The start time matches.")
        self.assertTrue(record["field_judgments"][0]["core_correct"])
        self.assertEqual(record["field_judgments"][0]["detail_quality"], 2)
        self.assertAlmostEqual(record["score_0_1"], 1.0)
        self.assertAlmostEqual(rows[0]["snapshot_holistic_score_mean_on_expected"], 1.0)

    def test_snapshot_holistic_judge_resume_skips_completed_keys(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "field_judgments": [
                            {
                                "field_path": "current_value",
                                "analysis": "Complete match.",
                                "core_correct": True,
                                "detail_quality": 2,
                            }
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_expected_snapshot": {
                    "habits_state:morning_walk": {"timing": {"start_time": "06:30"}},
                    "preferences:learning_format": {"current_value": "webinar"},
                },
                "_pred_snapshot": {
                    "habits_state:morning_walk": "Morning walk at 06:30.",
                    "preferences:learning_format": "Webinar.",
                },
                "snapshot_holistic_eval_by_key": {
                    "habits_state:morning_walk": {
                        "score_0_1": 1.0,
                        "field_count": 1,
                        "field_judgments": [
                            {
                                "field_path": "timing.start_time",
                                "analysis": "cached",
                                "core_correct": True,
                                "detail_quality": 2,
                                "score_0_1": 1.0,
                            }
                        ],
                    }
                },
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, _audit_records = _run_snapshot_holistic_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertIn("preferences:learning_format", FakeClient.prompts[0])
        self.assertIn("habits_state:morning_walk", rows[0]["snapshot_holistic_eval_by_key"])
        self.assertIn("preferences:learning_format", rows[0]["snapshot_holistic_eval_by_key"])
        self.assertEqual(rows[0]["snapshot_holistic_eval_by_key"]["habits_state:morning_walk"]["field_judgments"][0]["analysis"], "cached")
        self.assertAlmostEqual(rows[0]["snapshot_holistic_score_mean_on_expected"], 1.0)

    def test_change_slot_judge_requests_are_key_level(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "b1", "analysis": "ok", "correct": True},
                            {"point_id": "a1", "analysis": "ok", "correct": True},
                            {"point_id": "r1", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_change_slots_by_key": {
                    "a:x": {
                        "before": [{"point_id": "b1", "point_type": "field", "reference_value": 1, "predicted_value": 1}],
                        "after": [{"point_id": "a1", "point_type": "field", "reference_value": 2, "predicted_value": 2}],
                        "change_reason": [{"point_id": "r1", "point_type": "micro", "point_text": "The reason says the routine shifted later.", "predicted_value": "later"}],
                    }
                }
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, audit_records = _run_change_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(audit_records), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"state_key": "a:x"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertIn('"point_id": "2"', FakeClient.prompts[0])
        self.assertIn('"point_id": "3"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "b1"', FakeClient.prompts[0])
        self.assertIn("change_slot_eval_by_key", rows[0])
        self.assertIn("before", rows[0]["change_slot_eval_by_key"]["a:x"])

    def test_change_slot_judge_resume_skips_completed_keys(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "b2", "analysis": "ok", "correct": True},
                            {"point_id": "a2", "analysis": "ok", "correct": True},
                            {"point_id": "r2", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_change_slots_by_key": {
                    "a:x": {
                        "before": [{"point_id": "b1", "point_type": "field"}],
                        "after": [{"point_id": "a1", "point_type": "field"}],
                        "change_reason": [{"point_id": "r1", "point_type": "micro"}],
                    },
                    "a:y": {
                        "before": [{"point_id": "b2", "point_type": "field"}],
                        "after": [{"point_id": "a2", "point_type": "field"}],
                        "change_reason": [{"point_id": "r2", "point_type": "micro"}],
                    },
                },
                "change_slot_eval_by_key": {
                    "a:x": {
                        "before": {"score_0_1": 1.0, "slot_count": 1, "slot_context": [{"point_id": "b1"}], "judgments": [{"point_id": "b1", "analysis": "cached", "correct": True}]},
                        "after": {"score_0_1": 1.0, "slot_count": 1, "slot_context": [{"point_id": "a1"}], "judgments": [{"point_id": "a1", "analysis": "cached", "correct": True}]},
                        "state_predict": {"score_0_1": 1.0, "slot_count": 2, "slot_context": [{"point_id": "b1"}, {"point_id": "a1"}], "judgments": [{"point_id": "b1", "analysis": "cached", "correct": True}, {"point_id": "a1", "analysis": "cached", "correct": True}]},
                        "change_reason": {"score_0_1": 1.0, "slot_count": 1, "slot_context": [{"point_id": "r1"}], "judgments": [{"point_id": "r1", "analysis": "cached", "correct": True}]},
                    }
                },
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, _audit_records = _run_change_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"state_key": "a:y"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertIn('"point_id": "2"', FakeClient.prompts[0])
        self.assertIn('"point_id": "3"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "b2"', FakeClient.prompts[0])
        self.assertIn("a:x", rows[0]["change_slot_eval_by_key"])
        self.assertIn("a:y", rows[0]["change_slot_eval_by_key"])
        self.assertEqual(rows[0]["change_slot_eval_by_key"]["a:x"]["before"]["judgments"][0]["analysis"], "cached")
        self.assertEqual(rows[0]["change_state_predict_point_score_mean_on_changed"], 1.0)

    def test_apply_slot_judge_requests_are_item_level(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "a1", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_rq3_apply_slots_by_item": {
                    "a:x::q1": {
                        "state_key": "a:x",
                        "qa_id": "q1",
                        "question": "What should the assistant do?",
                        "reference_answer": "Do X.",
                        "predicted_answer": "Do X.",
                        "slots": [
                            {"point_id": "a1", "point_type": "micro", "point_text": "The answer says Do X.", "predicted_value": "Do X"},
                        ],
                    }
                }
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, audit_records = _run_apply_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(audit_records), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"qa_id": "q1"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "a1"', FakeClient.prompts[0])
        self.assertIn("rq3_apply_slot_eval_by_item", rows[0])
        self.assertIn("slot_context", rows[0]["rq3_apply_slot_eval_by_item"]["a:x::q1"])

    def test_apply_holistic_judge_requests_are_item_level(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "field_judgments": [
                            {
                                "field_path": "fitness_search_criteria.preferred_environment",
                                "analysis": "The indoor environment core is present with one missing qualifier.",
                                "core_correct": True,
                                "detail_quality": 1,
                            },
                            {
                                "field_path": "fitness_search_criteria.excluded_environment",
                                "analysis": "The outdoor-activity exclusion is missing.",
                                "core_correct": False,
                                "detail_quality": 0,
                            },
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "checkpoint_id": "cp_0001",
                "_rq3_apply_slots_by_item": {
                    "preferences_state:exercise_environment::q1": {
                        "state_key": "preferences_state:exercise_environment",
                        "qa_id": "q1",
                        "service_family": "information_request_construction",
                        "scenario": "The user is exploring local fitness facilities and workout classes.",
                        "task_instruction": "Fill the search filters the assistant should apply now before showing matches.",
                        "reference_output": {
                            "fitness_search_criteria": {
                                "preferred_environment": "climate-controlled indoor exercise environments",
                                "excluded_environment": "outdoor activities",
                            }
                        },
                        "predicted_output": {
                            "fitness_search_criteria": {
                                "preferred_environment": "Home Gym",
                                "excluded_environment": "",
                            }
                        },
                        "slots": [
                            {"point_id": "a1", "point_type": "field"},
                        ],
                    }
                },
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, audit_records = _run_apply_holistic_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(audit_records), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"qa_id": "q1"', FakeClient.prompts[0])
        self.assertIn("Core + Detail field evaluation method", FakeClient.prompts[0])
        self.assertIn("rq3_apply_holistic_eval_by_item", rows[0])
        record = rows[0]["rq3_apply_holistic_eval_by_item"]["preferences_state:exercise_environment::q1"]
        self.assertAlmostEqual(record["score_0_1"], 0.45)
        self.assertEqual(rows[0]["rq3_apply_holistic_score_mean"], record["score_0_1"])

    def test_apply_slot_judge_resume_skips_completed_items(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "a2", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_rq3_apply_slots_by_item": {
                    "a:x::q1": {
                        "state_key": "a:x",
                        "qa_id": "q1",
                        "question": "What should the assistant do?",
                        "reference_answer": "Do X.",
                        "predicted_answer": "Do X.",
                        "slots": [
                            {"point_id": "a1", "point_type": "micro"},
                        ],
                    },
                    "a:y::q2": {
                        "state_key": "a:y",
                        "qa_id": "q2",
                        "question": "What should the assistant do?",
                        "reference_answer": "Do Y.",
                        "predicted_answer": "Do Y.",
                        "slots": [
                            {"point_id": "a2", "point_type": "micro"},
                        ],
                    },
                },
                "rq3_apply_slot_eval_by_item": {
                    "a:x::q1": {
                        "state_key": "a:x",
                        "qa_id": "q1",
                        "score_0_1": 1.0,
                        "slot_count": 1,
                        "slot_context": [{"point_id": "a1"}],
                        "judgments": [{"point_id": "a1", "analysis": "cached", "correct": True}],
                    }
                },
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, _audit_records = _run_apply_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"qa_id": "q2"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "a2"', FakeClient.prompts[0])
        self.assertIn("a:x::q1", rows[0]["rq3_apply_slot_eval_by_item"])
        self.assertIn("a:y::q2", rows[0]["rq3_apply_slot_eval_by_item"])
        self.assertEqual(rows[0]["rq3_apply_slot_eval_by_item"]["a:x::q1"]["judgments"][0]["analysis"], "cached")
        self.assertEqual(rows[0]["rq3_apply_answer_point_score_mean"], 1.0)

    def test_apply_slot_judge_resume_retries_items_with_judge_error(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "a1", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        rows = [
            {
                "_rq3_apply_slots_by_item": {
                    "a:x::q1": {
                        "state_key": "a:x",
                        "qa_id": "q1",
                        "question": "What should the assistant do?",
                        "reference_answer": "Do X.",
                        "predicted_answer": "Do X.",
                        "slots": [
                            {"point_id": "a1", "point_type": "micro"},
                        ],
                    }
                },
                "rq3_apply_slot_eval_by_item": {
                    "a:x::q1": {
                        "state_key": "a:x",
                        "qa_id": "q1",
                        "score_0_1": 0.0,
                        "slot_count": 1,
                        "slot_context": [{"point_id": "a1"}],
                        "judgments": [
                            {
                                "point_id": "a1",
                                "analysis": "judge_error: Error code: 429",
                                "correct": False,
                            }
                        ],
                    }
                },
                "rq3_apply_answer_slot_judge_reason": "a:x::q1: Error code: 429",
            }
        ]

        with patch("eval.client.LLMClient", FakeClient):
            reqs, _audit_records = _run_apply_slot_judge(rows, "openai", "gpt-5-mini", 2)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"qa_id": "q1"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "a1"', FakeClient.prompts[0])
        self.assertEqual(rows[0]["rq3_apply_slot_eval_by_item"]["a:x::q1"]["judgments"][0]["analysis"], "ok")
        self.assertEqual(rows[0]["rq3_apply_answer_point_score_mean"], 1.0)

    def test_evaluate_resume_merges_existing_slot_results_by_checkpoint_id(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result({"judgments": []})
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        benchmark = {
            "user_id": "001_user_001",
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_completion_pack": {
                        "keys": {
                            "habits_state:morning_walk": {
                                "scoring_points": [
                                    {
                                        "point_id": "scp_walk_p1",
                                        "point_type": "field",
                                        "polarity": "positive",
                                        "target_path": "timing.start_time",
                                        "reference_value": "06:30",
                                    }
                                ]
                            }
                        }
                    },
                }
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "legacy",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                        ]
                    },
                    "change_analysis": {},
                    "rq3_apply_answers": {},
                }
            ]
        }
        existing_result = {
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "snapshot_slot_eval_by_key": {
                        "habits_state:morning_walk": {
                            "score_0_1": 1.0,
                            "slot_count": 1,
                            "slot_context": [{"point_id": "scp_walk_p1"}],
                            "judgments": [
                                {"point_id": "scp_walk_p1", "analysis": "cached", "correct": True}
                            ],
                        }
                    },
                    "snapshot_slot_judge_reason": "cached",
                    "snapshot_point_score_mean_on_expected": 1.0,
                    "snapshot_holistic_eval_by_key": {
                        "habits_state:morning_walk": {
                            "score_0_1": 1.0,
                            "field_count": 1,
                            "field_judgments": [
                                {
                                    "field_path": "timing.start_time",
                                    "analysis": "cached holistic",
                                    "core_correct": True,
                                    "detail_quality": 2,
                                    "score_0_1": 1.0,
                                }
                            ],
                        }
                    },
                    "snapshot_holistic_score_mean_on_expected": 1.0,
                }
            ]
        }

        with patch("eval.client.LLMClient", FakeClient):
            result, _align = evaluate(
                benchmark,
                prediction,
                enable_llm_judge=True,
                enable_snapshot_slot_judge=True,
                save_eyeball=False,
                existing_result=existing_result,
            )

        self.assertEqual(FakeClient.prompts, [])
        checkpoint = result["checkpoints"][0]
        self.assertEqual(
            checkpoint["snapshot_slot_eval_by_key"]["habits_state:morning_walk"]["judgments"][0]["analysis"],
            "cached",
        )
        self.assertEqual(
            checkpoint["snapshot_holistic_eval_by_key"]["habits_state:morning_walk"]["field_judgments"][0]["analysis"],
            "cached holistic",
        )

    def test_evaluate_resume_retries_cached_task_c_judge_errors(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "asp_q1_1", "analysis": "ok", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        benchmark = {
            "user_id": "001_user_001",
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {},
                    "validated_snapshot_state": {},
                    "state_observability": {},
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "pair_count_per_key": 1,
                        "keys": {
                            "habits_state:morning_walk": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_category": "calendar_assistant",
                                        "question": "When should the user leave home for the walk?",
                                        "reference_answer": "Leave before 06:30 for the walk.",
                                        "retrieval_query": "Question: When should the user leave home for the walk?",
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "asp_q1_1",
                                                "point_type": "micro",
                                                "polarity": "positive",
                                                "point_text": "The answer says the user should leave before 06:30.",
                                            }
                                        ],
                                    }
                                ]
                            }
                        },
                    },
                }
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "legacy_1",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {},
                    "evidence": {},
                    "change_analysis": {},
                    "rq3_apply_answers": {
                        "habits_state:morning_walk": {
                            "items": [
                                {
                                    "qa_id": "q1",
                                    "answer": "Leave before 06:30.",
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                                    ],
                                }
                            ]
                        }
                    },
                }
            ]
        }
        existing_result = {
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "rq3_apply_slot_eval_by_item": {
                        "habits_state:morning_walk::q1": {
                            "state_key": "habits_state:morning_walk",
                            "qa_id": "q1",
                            "score_0_1": 0.0,
                            "slot_count": 1,
                            "slot_context": [{"point_id": "asp_q1_1"}],
                            "judgments": [
                                {
                                    "point_id": "asp_q1_1",
                                    "analysis": "judge_error: Error code: 429",
                                    "correct": False,
                                }
                            ],
                        }
                    },
                    "rq3_apply_answer_slot_judge_reason": "habits_state:morning_walk::q1: Error code: 429",
                    "rq3_apply_answer_point_score_mean": 0.0,
                    "rq3_apply_holistic_eval_by_item": {
                        "habits_state:morning_walk::q1": {
                            "score_0_1": 1.0,
                            "field_count": 1,
                            "field_judgments": [
                                {
                                    "field_path": "value",
                                    "analysis": "cached holistic",
                                    "core_correct": True,
                                    "detail_quality": 2,
                                    "score_0_1": 1.0,
                                }
                            ],
                        }
                    },
                    "rq3_apply_holistic_score_mean": 1.0,
                }
            ]
        }

        with patch("eval.client.LLMClient", FakeClient):
            result, _align = evaluate(
                benchmark,
                prediction,
                enable_llm_judge=True,
                enable_apply_slot_judge=True,
                save_eyeball=False,
                existing_result=existing_result,
            )

        self.assertEqual(len(FakeClient.prompts), 1)
        self.assertNotIn('"qa_id": "q1"', FakeClient.prompts[0])
        self.assertIn('"point_id": "1"', FakeClient.prompts[0])
        self.assertNotIn('"point_id": "asp_q1_1"', FakeClient.prompts[0])
        checkpoint = result["checkpoints"][0]
        self.assertEqual(
            checkpoint["rq3_apply_slot_eval_by_item"]["habits_state:morning_walk::q1"]["judgments"][0]["analysis"],
            "ok",
        )
        self.assertEqual(checkpoint["rq3_apply_answer_point_score_mean"], 1.0)

    def test_evaluate_defaults_to_task_c_holistic_and_skips_apply_slot_judge(self):
        class FakeClient:
            prompts = []

            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                self.__class__.prompts.append(prompt)
                future = Future()
                future.set_result(
                    {
                        "field_judgments": [
                            {
                                "field_path": "search.format",
                                "analysis": "The predicted format preserves the webinar core but misses self-paced detail.",
                                "core_correct": True,
                                "detail_quality": 1,
                            }
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        benchmark = {
            "user_id": "001_user_001",
            "task_contract_version": "taskabc_v2",
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {},
                    "validated_snapshot_state": {},
                    "state_observability": {},
                    "rq3_apply_service_qa": {
                        "keys": {
                            "preferences_state:learning_format": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_family": "information_request_construction",
                                        "scenario": "The user is searching for professional learning options.",
                                        "task_instruction": "Fill the search object.",
                                        "retrieval_query": "Fill the search object.",
                                        "reference_output": {"search": {"format": "self-paced webinar"}},
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "asp_q1_1",
                                                "point_type": "field",
                                                "target_path": "search.format",
                                                "reference_value": "self-paced webinar",
                                            }
                                        ],
                                    }
                                ]
                            }
                        }
                    },
                }
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "legacy_1",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {},
                    "evidence": {},
                    "change_analysis": {},
                    "rq3_apply_answers": {
                        "preferences_state:learning_format": {
                            "items": [
                                {
                                    "qa_id": "q1",
                                    "answer": {"search": {"format": "webinar"}},
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "prefers webinars"}
                                    ],
                                }
                            ]
                        }
                    },
                }
            ]
        }

        with patch("eval.client.LLMClient", FakeClient):
            result, _align, audit = evaluate(
                benchmark,
                prediction,
                enable_llm_judge=True,
                save_eyeball=False,
                return_audit=True,
            )

        checkpoint = result["checkpoints"][0]
        self.assertNotIn("rq3_apply_slot_eval_by_item", checkpoint)
        self.assertNotIn("rq3_apply_answer_point_score_mean", checkpoint)
        self.assertIn("rq3_apply_holistic_eval_by_item", checkpoint)
        self.assertAlmostEqual(checkpoint["rq3_apply_holistic_score_mean"], 0.9)
        self.assertEqual(audit["apply"], [])
        self.assertEqual(len(audit["apply_holistic"]), 1)
        self.assertIn("Core + Detail field evaluation method", FakeClient.prompts[0])

    def test_evaluate_keeps_slot_judge_audit_out_of_main_result(self):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                future = Future()
                if "field_judgments" in prompt:
                    future.set_result(
                        {
                            "field_judgments": [
                                {
                                    "field_path": "timing.start_time",
                                    "analysis": "The timing core and detail match.",
                                    "core_correct": True,
                                    "detail_quality": 2,
                                }
                            ]
                        }
                    )
                    return future
                future.set_result(
                    {
                        "judgments": [
                            {"point_id": "scp_walk_p1", "analysis": "The predicted value matches 06:30.", "correct": True},
                        ]
                    }
                )
                return future

            def ask_async(self, prompt, response_type="json"):
                raise AssertionError("unexpected non-structured call")

            def close(self):
                return None

        benchmark = {
            "user_id": "001_user_001",
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_completion_pack": {
                        "keys": {
                            "habits_state:morning_walk": {
                                "scoring_points": [
                                    {
                                        "point_id": "scp_walk_p1",
                                        "point_type": "field",
                                        "polarity": "positive",
                                        "target_path": "timing.start_time",
                                        "reference_value": "06:30",
                                    }
                                ]
                            }
                        }
                    },
                }
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "legacy",
                    "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                        ]
                    },
                    "change_analysis": {},
                    "rq3_apply_answers": {},
                }
            ]
        }

        with patch("eval.client.LLMClient", FakeClient):
            result, _align, audit = evaluate(
                benchmark,
                prediction,
                enable_llm_judge=True,
                save_eyeball=False,
                judge_experiment_name="promptclean_0501",
                return_audit=True,
            )

        checkpoint = result["checkpoints"][0]
        self.assertEqual(result["judge_experiment_name"], "promptclean_0501")
        self.assertEqual(audit["judge_experiment_name"], "promptclean_0501")
        self.assertNotIn("snapshot_slot_eval_by_key", checkpoint)
        self.assertIn("snapshot_holistic_eval_by_key", checkpoint)
        self.assertNotIn("snapshot_slot_judge_inputs", result)
        self.assertNotIn("snapshot_slot_judge_prompt_by_key", checkpoint)
        self.assertNotIn("snapshot_slot_judge_raw_output_by_key", checkpoint)
        self.assertIn("snapshot", audit)
        self.assertEqual(len(audit["snapshot"]), 0)
        self.assertEqual(len(audit["snapshot_holistic"]), 1)
        self.assertIn("prompt", audit["snapshot_holistic"][0])
        self.assertIn("raw_output", audit["snapshot_holistic"][0])

    def test_eval_config_loads_nested_runtime_shape(self):
        config = _load_eval_config(ROOT / "configs/experiments/tce/amem_eval_user001_predict_0430.yaml")
        self.assertEqual(config["benchmark"], "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/tce_benchmark_vnext_20260429_formal_task_packs.json")
        self.assertEqual(config["prediction"], "results/Amem/results/001_user_001/prediction/predict_0430/tce_results.json")
        self.assertEqual(config["output"], "results/Amem/results/001_user_001/eval/predict_0430/tce_eval.json")
        self.assertTrue(config["enable_llm_judge"])
        self.assertNotIn("enable_apply_slot_judge", config)
        self.assertTrue(config["save_eyeball"])
        self.assertTrue(config["resume"])
        self.assertEqual(config["llm_provider"], "azure")
        self.assertEqual(config["llm_model"], "gpt-5.4")
        self.assertEqual(config["llm_max_workers"], 4)

    def test_eval_config_cli_values_override_config(self):
        args = argparse.Namespace(
            config=None,
            benchmark=None,
            prediction=Path("override_prediction.json"),
            output=None,
            enable_llm_judge=None,
            enable_snapshot_slot_judge=None,
            enable_apply_slot_judge=None,
            save_eyeball=None,
            llm_provider=None,
            llm_model="override-model",
            llm_max_workers=None,
            resume=None,
            llm_judge_input_output=None,
            judge_experiment_name="cli-tag",
        )
        resolved = _apply_eval_config(
            args,
            {
                "benchmark": "benchmark.json",
                "prediction": "config_prediction.json",
                "output": "eval.json",
                "enable_llm_judge": True,
                "llm_model": "config-model",
                "llm_max_workers": 1,
                "judge_experiment_name": "config-tag",
            },
        )
        self.assertEqual(resolved.benchmark, Path("benchmark.json"))
        self.assertEqual(resolved.prediction, Path("override_prediction.json"))
        self.assertEqual(resolved.output, Path("eval.json"))
        self.assertTrue(resolved.enable_llm_judge)
        self.assertFalse(resolved.enable_apply_slot_judge)
        self.assertEqual(resolved.llm_model, "override-model")
        self.assertEqual(resolved.llm_max_workers, 1)
        self.assertEqual(resolved.judge_experiment_name, "cli-tag")


if __name__ == "__main__":
    unittest.main(verbosity=2)
