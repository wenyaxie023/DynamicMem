#!/usr/bin/env python3
import copy
import json
import unittest
from unittest.mock import patch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.task_packs import _rewrite_apply_item, _validate_apply_item, build_task_packs
from tce_contracts import (
    CANONICAL_RESEARCH_DOC_V2,
    CURRENT_TASK_CONTRACT_VERSION,
    LEGACY_TASK_CONTRACT_VERSION,
    RESEARCH_FRAME_VERSION_V2,
)


class TceTaskPackAcceptance(unittest.TestCase):
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

    def test_v2_validation_and_rewrite_prompts_do_not_receive_scoring_fields(self):
        item = {
            "qa_id": "q1",
            "service_family": "user_communication",
            "scenario": "It is 06:10. Nothing has been logged yet today.",
            "task_instruction": "Write the short reminder message the assistant should send right now.",
            "reference_answer": "Send a reminder that the walk starts at 06:30.",
            "scoring_rubric": {"criteria": [{"id": "timing.start_time", "description": "old"}]},
            "answer_scoring_points": [{"point_id": "old", "point_text": "old"}],
        }
        state_value = {"timing": {"start_time": "06:30"}}

        class _CaptureValidatorClient:
            prompts: list[str] = []

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                self.prompts.append(prompt)
                return {
                    "criteria": [
                        {"criterion": "answerability", "pass": True, "analysis": "ok"},
                        {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                        {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                        {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                        {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                    ]
                }

        class _CaptureRewriteClient:
            prompts: list[str] = []

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                self.prompts.append(prompt)
                return {"scenario": "It is 06:10. Nothing has been logged yet, and the user is idle."}

        validator = _CaptureValidatorClient()
        _validate_apply_item(
            validator_client=validator,
            state_key="habits_state:morning_walk",
            state_value=state_value,
            item=item,
            task_contract_version=CURRENT_TASK_CONTRACT_VERSION,
        )
        rewriter = _CaptureRewriteClient()
        _rewrite_apply_item(
            generator_client=rewriter,
            state_key="habits_state:morning_walk",
            state_value=state_value,
            item=item,
            validation_payload={
                "failed_rules": ["low_leakage"],
                "semantic_criteria": [
                    {"criterion": "low_leakage", "pass": False, "analysis": "scenario leaks"},
                ],
            },
            task_contract_version=CURRENT_TASK_CONTRACT_VERSION,
        )

        self.assertEqual(len(validator.prompts), 1)
        self.assertEqual(len(rewriter.prompts), 1)
        self.assertNoTaskCV2ScoringAuthoringLanguage(validator.prompts[0])
        self.assertNoTaskCV2ScoringAuthoringLanguage(rewriter.prompts[0])

    def test_v2_rewrite_apply_item_merges_delta_without_dropping_unchanged_fields(self):
        item = {
            "qa_id": "q1",
            "service_family": "user_communication",
            "scenario": "It is 06:10. Nothing has been logged yet today.",
            "task_instruction": "Write the short reminder message the assistant should send right now.",
            "reference_answer": "Send a reminder that the walk starts at 06:30.",
        }

        class _FakeGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "scenario": "It is 06:10. Nothing has been logged yet, and the user is idle.",
                }

        rewritten = _rewrite_apply_item(
            generator_client=_FakeGeneratorClient(),
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}},
            item=item,
            validation_payload={
                "failed_rules": ["low_leakage"],
                "semantic_criteria": [
                    {"criterion": "answerability", "pass": True, "analysis": "ok"},
                    {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                    {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                    {"criterion": "low_leakage", "pass": False, "analysis": "scenario leaks"},
                    {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                ],
            },
            task_contract_version=CURRENT_TASK_CONTRACT_VERSION,
        )

        self.assertEqual(rewritten["scenario"], "It is 06:10. Nothing has been logged yet, and the user is idle.")
        self.assertEqual(rewritten["task_instruction"], item["task_instruction"])
        self.assertEqual(rewritten["reference_answer"], item["reference_answer"])
        self.assertNotIn("scoring_rubric", rewritten)

    def test_v2_rewrite_apply_item_does_not_carry_structured_scoring_rubric(self):
        item = {
            "qa_id": "q1",
            "service_family": "information_request_construction",
            "scenario": "A training-resource search request is about to run.",
            "task_instruction": "Fill the structured request payload before the search is sent.",
            "output_template": {"request_profile": {"preferred_profile": "<fill>"}},
            "reference_output": {"request_profile": {"preferred_profile": "prefers self-paced webinars"}},
        }

        class _FakeGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "reference_output": {"request_profile": {"preferred_profile": "self-paced webinars"}},
                }

        rewritten = _rewrite_apply_item(
            generator_client=_FakeGeneratorClient(),
            state_key="preferences_state:learning_modality",
            state_value={"statement": "prefers self-paced webinars"},
            item=item,
            validation_payload={
                "failed_rules": ["output_groundedness"],
                "semantic_criteria": [
                    {"criterion": "answerability", "pass": True, "analysis": "ok"},
                    {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                    {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                    {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                    {"criterion": "output_groundedness", "pass": False, "analysis": "output needs a more task-appropriate fill."},
                ],
            },
            task_contract_version=CURRENT_TASK_CONTRACT_VERSION,
        )

        self.assertEqual(
            rewritten["reference_output"],
            {"request_profile": {"preferred_profile": "self-paced webinars"}},
        )
        self.assertNotIn("scoring_rubric", rewritten)

    def test_build_task_packs_stamps_current_contract_metadata_when_missing(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "profile_state:favorite_coffee": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "profile_state": {"favorite_coffee": "latte"}
                    },
                }
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["state_completion"],
        )

        self.assertEqual(result["task_contract_version"], CURRENT_TASK_CONTRACT_VERSION)
        self.assertEqual(result["research_frame_version"], RESEARCH_FRAME_VERSION_V2)
        self.assertEqual(result["canonical_research_doc"], CANONICAL_RESEARCH_DOC_V2)

    def test_v2_all_builds_only_task_a_and_task_c(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "preferences_state:favorite_coffee": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "preferences_state": {"favorite_coffee": {"statement": "latte"}}
                    },
                    "state_observability": {
                        "preferences_state": {
                            "favorite_coffee": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate exactly one item for a preference-conditioned structured filtering task." in prompt:
                    return {
                        "items": [
                            {
                                "scenario": "The assistant is preparing the structured information-request object for the user's coffee-ordering preferences before a recurring office order.",
                                "task_instruction": "Fill the structured information-request payload.",
                                "output_template": {"request_profile": {"preferred_profile": "<fill>"}},
                                "reference_output": {"request_profile": {"preferred_profile": "latte"}},
                            }
                        ]
                    }
                return {}

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate whether this item is a strong structured completion task." in prompt:
                    return {
                        "criteria": [
                            {"criterion": "answerability", "pass": True, "analysis": "ok"},
                            {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                            {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                            {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                            {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                        ]
                    }
                return {}

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["all"],
            generator_client=_FakeApplyGeneratorClient(),
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        cp = result["checkpoints"][0]
        self.assertIn("state_completion_pack", cp)
        self.assertIn("rq3_apply_service_qa", cp)
        self.assertNotIn("change_tracking_pack", cp)
        item = cp["rq3_apply_service_qa"]["keys"]["preferences_state:favorite_coffee"]["items"][0]
        self.assertEqual(cp["rq3_apply_service_qa"]["version"], "v9")
        self.assertEqual(cp["rq3_apply_service_qa"]["pair_count_per_key"], 1)
        self.assertEqual(item["service_family"], "information_request_construction")
        self.assertEqual(item["output_template"], {"request_profile": {"preferred_profile": "<fill>"}})
        self.assertEqual(item["reference_output"], {"request_profile": {"preferred_profile": "latte"}})
        self.assertNotIn("scoring_rubric", item)
        self.assertEqual(len(item["answer_scoring_points"]), 1)
        point = item["answer_scoring_points"][0]
        self.assertEqual(point["point_id"], "aqp_preferences_state_favorite_coffee_q1_p1")
        self.assertEqual(point["point_type"], "field")
        self.assertNotIn("polarity", point)
        self.assertEqual(point["source_field_path"], "statement")
        self.assertEqual(point["output_field_path"], "request_profile.preferred_profile")
        self.assertEqual(point["target_path"], "request_profile.preferred_profile")
        self.assertEqual(point["reference_value"], "latte")
        self.assertTrue(item["item_validation"]["is_valid"])
        self.assertTrue(item["scoring_validation"]["is_valid"])

    def test_v2_user_communication_materializes_points_without_scoring_rubric(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "schedule": {
                                    "frequency_type": "weekly",
                                    "days_of_week": [2],
                                },
                                "timing": {"start_time": "06:30"},
                                "location": "lakefront trail",
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate exactly one item for a habit-conditioned assistant-message task." in prompt:
                    return {
                        "items": [
                            {
                                "scenario": "It is Wednesday at 06:10. Nothing has been logged yet this morning.",
                                "task_instruction": "As the assistant, what single message should be sent to the user right now? Make it complete for this moment by using the user's routine details, not a generic reminder.",
                                "reference_answer": "Your weekly morning walk is scheduled for Wednesday at 06:30 on the lakefront trail. It is almost time to head out.",
                            }
                        ]
                    }
                return {}

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate whether this item is a strong current-moment assistant message task." in prompt:
                    return {
                        "criteria": [
                            {"criterion": "answerability", "pass": True, "analysis": "ok"},
                            {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                            {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                            {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                            {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                        ]
                    }
                return {}

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=_FakeApplyGeneratorClient(),
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["items"][0]
        self.assertEqual(item["service_family"], "user_communication")
        self.assertNotIn("scoring_rubric", item)
        self.assertEqual(
            item["reference_answer"],
            "Your weekly morning walk is scheduled for Wednesday at 06:30 on the lakefront trail. It is almost time to head out.",
        )
        self.assertEqual(len(item["answer_scoring_points"]), 5)
        self.assertEqual(
            [point["point_type"] for point in item["answer_scoring_points"]],
            ["micro", "micro", "micro", "micro", "micro"],
        )
        self.assertEqual(
            item["answer_scoring_points"][0],
            {
                "point_id": "aqp_habits_state_morning_walk_q1_identity",
                "point_type": "micro",
                "point_role": "identity_gate",
                "point_text": "The message is clearly about the morning walk routine itself, not a different routine or unrelated task.",
            },
        )
        self.assertEqual(
            [point["point_text"] for point in item["answer_scoring_points"][1:]],
            [
                'The message correctly uses the state field schedule.frequency_type with value "weekly".',
                "The message correctly uses the state field schedule.days_of_week with value [2 (Wednesday)].",
                'The message correctly uses the state field timing.start_time with value "06:30".',
                'The message correctly uses the state field location with value "lakefront trail".',
            ],
        )
        self.assertEqual(
            [point["source_field_path"] for point in item["answer_scoring_points"][1:]],
            [
                "schedule.frequency_type",
                "schedule.days_of_week",
                "timing.start_time",
                "location",
            ],
        )
        self.assertTrue(all("polarity" not in point for point in item["answer_scoring_points"]))
        self.assertTrue(item["item_validation"]["is_valid"])
        self.assertTrue(item["scoring_validation"]["is_valid"])
        self.assertTrue(item["scoring_validation"]["uses_identity_gate"])

    def test_v2_preference_task_c_contracts_state_to_statement_only(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "preferences_state:learning_modality": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "preferences_state": {
                            "learning_modality": {
                                "statement": "prefers self-paced webinars",
                                "signals": [
                                    "downloaded a white paper",
                                    "joined a webinar",
                                ],
                            }
                        }
                    },
                    "state_observability": {
                        "preferences_state": {
                            "learning_modality": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            prompts: list[str] = []

            def ask(self, prompt: str, response_type: str = "json"):
                self.prompts.append(prompt)
                return {
                    "items": [
                        {
                            "scenario": "A training-resource lookup is about to run for the next study step.",
                            "task_instruction": "As the assistant, complete the structured information request below so it can be sent to the downstream information system.",
                            "output_template": {"request_profile": {"primary_preference": "<fill>"}},
                            "reference_output": {"request_profile": {"primary_preference": "prefers self-paced webinars"}},
                        }
                    ]
                }

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                return {
                    "criteria": [
                        {"criterion": "answerability", "pass": True, "analysis": "ok"},
                        {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                        {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                        {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                        {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                    ]
                }

        generator = _FakeApplyGeneratorClient()
        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=generator,
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        prompt_text = generator.prompts[0]
        self.assertIn("'statement': 'prefers self-paced webinars'", prompt_text)
        self.assertNotIn("'signals': [", prompt_text)

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["preferences_state:learning_modality"]["items"][0]
        self.assertEqual(len(item["answer_scoring_points"]), 1)
        point = item["answer_scoring_points"][0]
        self.assertEqual(point["source_field_path"], "statement")
        self.assertEqual(point["reference_value"], "prefers self-paced webinars")

    def test_v2_structured_pack_allows_multi_slot_reference_output_without_source_leaf_pairing(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "preferences_state:community_involvement_type": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "preferences_state": {
                            "community_involvement_type": {
                                "statement": "Strongly prefers youth-focused civic projects over general social mixers"
                            }
                        }
                    },
                    "state_observability": {
                        "preferences_state": {
                            "community_involvement_type": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "items": [
                        {
                            "scenario": "The user is reviewing volunteering options for the next free weekend. Candidate opportunities are being narrowed before any shortlist is shown.",
                            "task_instruction": "As the assistant, complete the structured information request below so it can be sent to the downstream information system.",
                            "output_template": {
                                "engagement_filters": {
                                    "preferred_domains": ["<fill>", "<fill>"],
                                    "deprioritized_formats": ["<fill>"],
                                }
                            },
                            "reference_output": {
                                "engagement_filters": {
                                    "preferred_domains": ["youth-focused projects", "civic impact work"],
                                    "deprioritized_formats": ["general social mixers"],
                                }
                            },
                        }
                    ]
                }

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "criteria": [
                        {"criterion": "answerability", "pass": True, "analysis": "ok"},
                        {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                        {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                        {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                        {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                    ]
                }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=_FakeApplyGeneratorClient(),
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["preferences_state:community_involvement_type"]["items"][0]
        self.assertTrue(item["item_validation"]["is_valid"])
        self.assertTrue(item["scoring_validation"]["is_valid"])
        self.assertEqual(
            [point["target_path"] for point in item["answer_scoring_points"]],
            [
                "engagement_filters.preferred_domains.0",
                "engagement_filters.preferred_domains.1",
                "engagement_filters.deprioritized_formats.0",
            ],
        )
        self.assertEqual(
            [point["reference_value"] for point in item["answer_scoring_points"]],
            [
                "youth-focused projects",
                "civic impact work",
                "general social mixers",
            ],
        )

    def test_v2_apply_reuses_authored_items_when_state_is_unchanged_across_checkpoints(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "06:30"},
                                "location": "lakefront trail",
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "06:30"},
                                "location": "lakefront trail",
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0001", "log_0002"]}
                        }
                    },
                },
            ],
        }

        class _FakeApplyGeneratorClient:
            generation_calls = 0

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Generate exactly one item for a habit-conditioned assistant-message task." in prompt:
                    type(self).generation_calls += 1
                    return {
                        "items": [
                            {
                                "scenario": "It is Wednesday at 06:10. Nothing has been logged yet this morning.",
                                "task_instruction": "As the assistant, what single message should be sent to the user right now?",
                                "reference_answer": "Send a reminder that the morning walk starts at 06:30 on the lakefront trail.",
                            }
                        ]
                    }
                return {}

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Validate whether this item is a strong current-moment assistant message task." in prompt:
                    return {
                        "criteria": [
                            {"criterion": "answerability", "pass": True, "analysis": "ok"},
                            {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                            {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                            {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                            {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                        ]
                    }
                if "Validate one generated atomic-fact set for an apply-service QA item." in prompt:
                    return {
                        "points": [
                            {
                                "point_id": "aqp_habits_state_morning_walk_q1_p1",
                                "pass": True,
                                "analysis": "ok",
                                "fail_reasons": [],
                            }
                        ],
                        "set_pass": True,
                        "set_failures": [],
                    }
                return {}

        generator = _FakeApplyGeneratorClient()
        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=generator,
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        cp1_item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["items"][0]
        cp2_key_node = result["checkpoints"][1]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]
        cp2_item = cp2_key_node["items"][0]

        self.assertEqual(generator.generation_calls, 1)
        self.assertEqual(cp2_key_node["pack_source"], "reused")
        self.assertEqual(cp1_item["scenario"], cp2_item["scenario"])
        self.assertEqual(cp1_item["task_instruction"], cp2_item["task_instruction"])
        self.assertEqual(cp2_item["gold_memory_evidence_app_log_ids"], ["log_0001", "log_0002"])

    def test_v2_user_communication_ignores_generated_rubric_descriptions(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "06:30"},
                                "location": "lakefront trail",
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "items": [
                        {
                            "scenario": "It is Wednesday at 06:10. Nothing has been logged yet this morning.",
                            "task_instruction": "As the assistant, what single message should be sent to the user right now?",
                            "reference_answer": "Send a reminder that the morning walk starts soon.",
                        }
                    ]
                }

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "criteria": [
                        {"criterion": "answerability", "pass": True, "analysis": "ok"},
                        {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                        {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                        {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                        {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                    ]
                }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=_FakeApplyGeneratorClient(),
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["items"][0]
        self.assertTrue(item["scoring_validation"]["is_valid"])
        self.assertFalse(item["scoring_validation"]["manual_review_required"])
        self.assertFalse(item["item_validation"]["manual_review_required"])

    def test_v2_apply_accepts_single_item_output_shape(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "preferences_state:favorite_coffee": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "preferences_state": {
                            "favorite_coffee": {
                                "statement": "latte"
                            }
                        }
                    },
                    "state_observability": {
                        "preferences_state": {
                            "favorite_coffee": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Generate exactly one item for a preference-conditioned structured filtering task." not in prompt:
                    return {}
                return {
                    "item": {
                        "scenario": "A coffee-order shortlist is being prepared before the menu is shown.",
                        "task_instruction": "As the assistant, complete the filtering parameters that should be sent right now. Use the user's preference statement to shape the filters, and do not write the final recommendation.",
                        "output_template": {
                            "drink_filters": {
                                "preferred_drink": "<fill>"
                            }
                        },
                        "reference_output": {
                            "drink_filters": {
                                "preferred_drink": "latte"
                            }
                        },
                    }
                }

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                del prompt, response_type
                return {
                    "criteria": [
                        {"criterion": "answerability", "pass": True, "analysis": "ok"},
                        {"criterion": "service_completion_quality", "pass": True, "analysis": "ok"},
                        {"criterion": "full_field_dependency", "pass": True, "analysis": "ok"},
                        {"criterion": "low_leakage", "pass": True, "analysis": "ok"},
                        {"criterion": "output_groundedness", "pass": True, "analysis": "ok"},
                    ]
                }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["apply"],
            generator_client=_FakeApplyGeneratorClient(),
            validator_client=_FakeApplyValidatorClient(),
            provider="test",
            model="generator",
            validator_provider="test",
            validator_model="validator",
            item_count_per_key=1,
            max_rewrites=0,
            apply_workers=1,
        )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["preferences_state:favorite_coffee"]["items"][0]
        self.assertEqual(item["service_family"], "information_request_construction")
        self.assertEqual(item["scenario"], "A coffee-order shortlist is being prepared before the menu is shown.")
        self.assertEqual(
            item["task_instruction"],
            "As the assistant, complete the filtering parameters that should be sent right now. Use the user's preference statement to shape the filters, and do not write the final recommendation.",
        )
        self.assertEqual(
            item["output_template"],
            {"drink_filters": {"preferred_drink": "<fill>"}},
        )
        self.assertEqual(
            item["reference_output"],
            {"drink_filters": {"preferred_drink": "latte"}},
        )
        self.assertTrue(item["item_validation"]["is_valid"])
        self.assertTrue(item["scoring_validation"]["is_valid"])

    def test_v2_task_a_filters_schedule_date_like_fields_from_template_and_scoring(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:weekend_woodworking_session": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "weekend_woodworking_session": {
                                "location": "basement workshop",
                                "priority": "high",
                                "schedule": {
                                    "days_of_week": [5],
                                    "frequency_type": "weekly",
                                },
                                "schedule_dates": ["2025-01-04", "2025-01-11"],
                            }
                        }
                    },
                }
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["state_completion"],
        )

        item = result["checkpoints"][0]["state_completion_pack"]["keys"]["habits_state:weekend_woodworking_session"]
        self.assertEqual(
            item["answer_template"],
            {
                "location": "<fill the blank>",
                "schedule": {
                    "days_of_week": ["<fill the blank>"],
                    "frequency_type": "<fill the blank>",
                },
            },
        )
        serialized_question = json.dumps(item["answer_template"], ensure_ascii=False, sort_keys=True)
        self.assertNotIn("schedule_dates", serialized_question)
        self.assertNotIn("priority", serialized_question)
        scoring_paths = {str(point.get("target_path") or "") for point in item["scoring_points"]}
        self.assertFalse(any("schedule_dates" in path for path in scoring_paths))
        self.assertFalse(any("priority" in path for path in scoring_paths))

    def test_state_completion_reuses_and_change_tracking_scopes_to_validated_intersection(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"evidence_app_log_ids": ["log_0002"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp3",
                    "as_of": {"timestamp": "2025-03-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "change_reason_validation": {
                                "exists": True,
                                "reason_analysis": "The evidence supports the routine shifting later.",
                                "is_valid": True,
                                "reason_codes": [],
                            },
                        },
                        "profile_state:favorite_coffee": {"is_questionable": True},
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "07:00"}}},
                        "profile_state": {"favorite_coffee": "latte"},
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "evidence_app_log_ids": ["log_0003"],
                                "last_change_reason": "Routine shifted later after a schedule change.",
                            }
                        },
                        "profile_state": {
                            "favorite_coffee": {"evidence_app_log_ids": ["log_0004"]}
                        },
                    },
                },
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["state_completion", "change_tracking"],
        )

        cp1, cp2, cp3 = result["checkpoints"]
        cp1_state_item = cp1["state_completion_pack"]["keys"]["habits_state:morning_walk"]
        cp2_state_item = cp2["state_completion_pack"]["keys"]["habits_state:morning_walk"]
        self.assertEqual(cp1_state_item["pack_source"], "computed")
        self.assertEqual(cp2_state_item["pack_source"], "reused")
        self.assertEqual(cp1_state_item["item_id"], cp2_state_item["item_id"])
        self.assertTrue(cp1_state_item["scoring_points"])

        self.assertEqual(cp1["change_tracking_pack"]["keys"], {})
        self.assertEqual(cp2["change_tracking_pack"]["keys"], {})
        cp3_change_keys = cp3["change_tracking_pack"]["keys"]
        self.assertEqual(sorted(cp3_change_keys.keys()), ["habits_state:morning_walk"])
        self.assertEqual(
            cp3_change_keys["habits_state:morning_walk"]["pack_identity"]["state_key"],
            "habits_state:morning_walk",
        )
        self.assertEqual(
            cp3_change_keys["habits_state:morning_walk"]["reference_change_reason"],
            "Routine shifted later after a schedule change.",
        )
        self.assertTrue(cp3_change_keys["habits_state:morning_walk"]["before_scoring_points"])
        self.assertTrue(cp3_change_keys["habits_state:morning_walk"]["after_scoring_points"])
        self.assertTrue(cp3_change_keys["habits_state:morning_walk"]["change_reason_scoring_points"])

    def test_state_completion_rewrites_invalid_generated_rubric_points_for_complex_statement(self):
        class _FakeGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Break one complex state field into 1 to" in prompt:
                    return {
                        "rubric": [
                            "The answer should recommend one-on-one private tutoring."
                        ]
                    }
                if "Rewrite an invalid scoring-point set for one complex state field." in prompt:
                    return {
                        "rubric": [
                            "The answer states that self-paced white papers or webinars are the preferred format.",
                            "The answer does not recommend large conferences as the preferred format.",
                        ]
                    }
                return {}

        class _FakeValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate one generated scoring-point set for a complex state field." not in prompt:
                    return {}
                if "one-on-one private tutoring" in prompt:
                    return {
                        "points": [
                            {
                                "point_id": "scp_preferences_state_learning_modality_statement_p1",
                                "pass": False,
                                "analysis": "The point reverses the preference and adds an unsupported exclusive claim.",
                                "fail_reasons": ["drift", "unsupported"],
                            }
                        ],
                        "set_pass": False,
                        "set_failures": ["coverage_gap"],
                    }
                return {
                    "points": [
                        {
                            "point_id": "scp_preferences_state_learning_modality_statement_p1",
                            "pass": True,
                            "analysis": "This point is directly grounded in the source text.",
                            "fail_reasons": [],
                        },
                        {
                            "point_id": "scp_preferences_state_learning_modality_statement_p2",
                            "pass": True,
                            "analysis": "This point stays aligned with the source comparison.",
                            "fail_reasons": [],
                        },
                    ],
                    "set_pass": True,
                    "set_failures": [],
                }

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "preferences_state:learning_modality": {"is_questionable": True}
                    },
                    "validated_snapshot_state": {
                        "preferences_state": {
                            "learning_modality": {
                                "statement": "Prefers self-paced white papers and webinars over large conferences."
                            }
                        }
                    },
                }
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["state_completion"],
            generator_client=_FakeGeneratorClient(),
            validator_client=_FakeValidatorClient(),
        )

        item = result["checkpoints"][0]["state_completion_pack"]["keys"]["preferences_state:learning_modality"]
        self.assertEqual(item["pack_source"], "computed")
        self.assertEqual(len(item["scoring_points"]), 2)
        self.assertEqual(
            [point["point_text"] for point in item["scoring_points"]],
            [
                "The answer states that self-paced white papers or webinars are the preferred format.",
                "The answer does not recommend large conferences as the preferred format.",
            ],
        )
        self.assertTrue(all("reference_value" not in point for point in item["scoring_points"]))

    def test_change_tracking_rewrites_invalid_generated_change_reason_rubric_points(self):
        class _FakeGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate atomic facts for evaluating a predicted change reason." in prompt:
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The user permanently rejected all car travel in every situation.",
                                "reference_value": "rejected all car travel",
                            }
                        ]
                    }
                if "Rewrite an invalid atomic-fact set for a state change reason." in prompt:
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The answer explains that commuting changed from car to train.",
                                "reference_value": "changed from car to train",
                            },
                            {
                                "polarity": "positive",
                                "point_text": "The answer focuses on the observed commuting transition without adding broader unsupported motives.",
                                "reference_value": "observed commuting transition",
                            },
                        ]
                    }
                return {}

        class _FakeValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate one generated atomic-fact set for a state change reason." not in prompt:
                    return {}
                if "rejected all car travel" in prompt:
                    return {
                        "points": [
                            {
                                "point_id": "ctp_reason_profile_state_commute_mode_p1",
                                "pass": False,
                                "analysis": "The point overstates the observed transition from car to train.",
                                "fail_reasons": ["drift", "unsupported"],
                            }
                        ],
                        "set_pass": False,
                        "set_failures": ["coverage_gap"],
                    }
                return {
                    "points": [
                        {
                            "point_id": "ctp_reason_profile_state_commute_mode_p1",
                            "pass": True,
                            "analysis": "This point is grounded in the before/after transition.",
                            "fail_reasons": [],
                        },
                        {
                            "point_id": "ctp_reason_profile_state_commute_mode_p2",
                            "pass": True,
                            "analysis": "This point stays focused on the actual transition.",
                            "fail_reasons": [],
                        },
                    ],
                    "set_pass": True,
                    "set_failures": [],
                }

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {"profile_state:commute_mode": {"is_questionable": True}},
                    "validated_snapshot_state": {"profile_state": {"commute_mode": "car"}},
                    "state_observability": {
                        "profile_state": {
                            "commute_mode": {"evidence_app_log_ids": ["log_001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "profile_state:commute_mode": {
                            "is_questionable": True,
                            "change_reason_validation": {
                                "exists": True,
                                "reason_analysis": "The evidence supports the switch from car to train.",
                                "is_valid": True,
                                "reason_codes": [],
                            },
                        }
                    },
                    "validated_snapshot_state": {"profile_state": {"commute_mode": "train"}},
                    "state_observability": {
                        "profile_state": {
                            "commute_mode": {
                                "evidence_app_log_ids": ["log_002"],
                                "last_change_reason": "Parking costs increased, so commuting switched to train.",
                            }
                        }
                    },
                },
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["change_tracking"],
            generator_client=_FakeGeneratorClient(),
            validator_client=_FakeValidatorClient(),
        )

        item = result["checkpoints"][1]["change_tracking_pack"]["keys"]["profile_state:commute_mode"]
        self.assertEqual(
            item["reference_change_reason"],
            "Parking costs increased, so commuting switched to train.",
        )
        self.assertEqual(
            [point["point_text"] for point in item["change_reason_scoring_points"]],
            [
                "The answer explains that commuting changed from car to train.",
                "The answer focuses on the observed commuting transition without adding broader unsupported motives.",
            ],
        )

    def test_change_tracking_filters_reference_change_reason_when_stage1_validation_fails(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {"profile_state:commute_mode": {"is_questionable": True}},
                    "validated_snapshot_state": {"profile_state": {"commute_mode": "car"}},
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "profile_state:commute_mode": {
                            "is_questionable": True,
                            "change_reason_validation": {
                                "exists": True,
                                "reason_analysis": "The evidence does not justify the claimed motive.",
                                "is_valid": False,
                                "reason_codes": ["change_reason_not_supported"],
                            },
                        }
                    },
                    "validated_snapshot_state": {"profile_state": {"commute_mode": "train"}},
                    "state_observability": {
                        "profile_state": {
                            "commute_mode": {
                                "evidence_app_log_ids": ["log_002"],
                                "last_change_reason": "Switched to train because the user permanently rejected all car travel.",
                            }
                        }
                    },
                },
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["change_tracking"],
        )

        cp2 = result["checkpoints"][1]["change_tracking_pack"]
        self.assertEqual(cp2["keys"], {})
        self.assertEqual(
            cp2["filtered_keys"]["profile_state:commute_mode"]["reason_codes"],
            ["reference_change_reason_invalid"],
        )

    def test_task_level_ground_truth_resolution_projects_transition_state_and_filters_unresolved_task_a_and_b(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "user_attributes_state:primary_vehicle": {"is_questionable": True},
                        "user_attributes_state:backup_vehicle": {"is_questionable": True},
                    },
                    "validated_snapshot_state": {
                        "user_attributes_state": {
                            "primary_vehicle": "2022 Audi Q7",
                            "backup_vehicle": "2018 Honda Civic",
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "user_attributes_state:primary_vehicle": {
                            "is_questionable": True,
                            "change_reason_validation": {
                                "exists": True,
                                "reason_analysis": "The evidence supports the Audi upgrade.",
                                "is_valid": True,
                                "reason_codes": [],
                            },
                        },
                        "user_attributes_state:backup_vehicle": {"is_questionable": True},
                    },
                    "validated_snapshot_state": {
                        "user_attributes_state": {
                            "primary_vehicle": {
                                "from": "2022 Audi Q7",
                                "to": "2024 Audi Q8",
                            },
                            "backup_vehicle": {
                                "from": "2018 Honda Civic",
                            },
                        }
                    },
                    "state_observability": {
                        "user_attributes_state": {
                            "primary_vehicle": {
                                "evidence_app_log_ids": ["log_010"],
                                "last_change_reason": "Upgraded to a newer Audi model.",
                            },
                            "backup_vehicle": {
                                "evidence_app_log_ids": ["log_011"],
                            },
                        }
                    },
                },
            ],
        }

        result = build_task_packs(
            benchmark=copy.deepcopy(benchmark),
            tasks=["state_completion", "change_tracking"],
        )

        cp2 = result["checkpoints"][1]
        task_a_keys = cp2["state_completion_pack"]["keys"]
        self.assertIn("user_attributes_state:primary_vehicle", task_a_keys)
        self.assertNotIn("user_attributes_state:backup_vehicle", task_a_keys)
        self.assertEqual(
            task_a_keys["user_attributes_state:primary_vehicle"]["answer_template"],
            "<fill the blank>",
        )
        primary_points = task_a_keys["user_attributes_state:primary_vehicle"]["scoring_points"]
        self.assertEqual(len(primary_points), 1)
        self.assertEqual(primary_points[0]["target_path"], "current_value")
        self.assertEqual(primary_points[0]["reference_value"], "2024 Audi Q8")
        self.assertEqual(
            cp2["state_completion_pack"]["filtered_keys"]["user_attributes_state:backup_vehicle"]["reason_codes"],
            ["current_state_unresolved_missing_to"],
        )

        task_b_keys = cp2["change_tracking_pack"]["keys"]
        self.assertIn("user_attributes_state:primary_vehicle", task_b_keys)
        self.assertNotIn("user_attributes_state:backup_vehicle", task_b_keys)
        change_item = task_b_keys["user_attributes_state:primary_vehicle"]
        self.assertEqual(change_item["reference_change_reason"], "Upgraded to a newer Audi model.")
        self.assertEqual(
            change_item["before_scoring_points"][0]["reference_value"],
            "2022 Audi Q7",
        )
        self.assertEqual(
            change_item["after_scoring_points"][0]["reference_value"],
            "2024 Audi Q8",
        )
        self.assertEqual(
            cp2["change_tracking_pack"]["filtered_keys"]["user_attributes_state:backup_vehicle"]["reason_codes"],
            ["after_state_unresolved"],
        )

    def test_apply_pack_reuses_task_a_current_state_projection_and_filters_unresolved_transition_keys(self):
        class _FakeApplyGeneratorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate personalized service free-form QA items" in prompt:
                    self.prompts.append(prompt)
                    return {
                        "items": [
                            {
                                "service_category": "transport planning",
                                "question": "What vehicle-related planning policy should the assistant follow for upcoming travel?",
                                "reference_answer": "Plan future vehicle-related travel support around the user's current primary vehicle.",
                            }
                        ]
                    }
                if "Generate scoreable atomic facts for one service-decision item." in prompt:
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The answer treats the current primary vehicle as the planning anchor.",
                                "reference_value": "current primary vehicle planning anchor",
                            },
                            {
                                "polarity": "positive",
                                "point_text": "The answer proposes one concrete transport-planning policy.",
                                "reference_value": "one concrete transport-planning policy",
                            },
                        ]
                    }
                return {}

            def __init__(self):
                self.prompts = []

        class _FakeApplyValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate whether this service-decision item is strong." in prompt:
                    return {
                        "criteria": [
                            {
                                "criterion": "personalization_necessity",
                                "pass": True,
                                "analysis": "The decision depends on the user's current vehicle state.",
                            },
                            {
                                "criterion": "service_decision_quality",
                                "pass": True,
                                "analysis": "The item asks for one transport-planning policy rather than recall or a menu choice.",
                            },
                            {
                                "criterion": "answer_groundedness",
                                "pass": True,
                                "analysis": "The reference answer defines one bounded policy without importing new details.",
                            },
                        ]
                    }
                if "Validate one generated atomic-fact set for a service-decision item." in prompt:
                    return {
                        "points": [
                            {
                                "point_id": "aqp_user_attributes_state_primary_vehicle_q1_p1",
                                "pass": True,
                                "analysis": "This point is grounded in the answer.",
                                "fail_reasons": [],
                            },
                            {
                                "point_id": "aqp_user_attributes_state_primary_vehicle_q1_p2",
                                "pass": True,
                                "analysis": "This point is independently scoreable.",
                                "fail_reasons": [],
                            },
                        ],
                        "set_pass": True,
                        "set_failures": [],
                    }
                return {}

        generator_client = _FakeApplyGeneratorClient()
        validator_client = _FakeApplyValidatorClient()
        prompt_calls = []

        def _fake_prompt_builder(*, checkpoint_timestamp, state_key, state_value, item_count):
            prompt_calls.append(
                {
                    "checkpoint_timestamp": checkpoint_timestamp,
                    "state_key": state_key,
                    "state_value": copy.deepcopy(state_value),
                    "item_count": item_count,
                }
            )
            return (
                "Generate personalized service free-form QA items\n"
                f"state_key={state_key}\n"
                f"state_value={state_value}"
            )

        benchmark = {
            "user_id": "001_user_001",
            "task_contract_version": LEGACY_TASK_CONTRACT_VERSION,
            "checkpoints": [
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "state_questionability": {
                        "user_attributes_state:primary_vehicle": {"is_questionable": True},
                        "user_attributes_state:backup_vehicle": {"is_questionable": True},
                    },
                    "validated_snapshot_state": {
                        "user_attributes_state": {
                            "primary_vehicle": {
                                "from": "2022 Audi Q7",
                                "to": "2024 Audi Q8",
                            },
                            "backup_vehicle": {
                                "from": "2018 Honda Civic",
                            },
                        }
                    },
                    "state_observability": {
                        "user_attributes_state": {
                            "primary_vehicle": {
                                "evidence_app_log_ids": ["log_001"],
                            },
                            "backup_vehicle": {
                                "evidence_app_log_ids": ["log_002"],
                            },
                        }
                    },
                }
            ],
        }

        with patch("tce_core.task_packs.build_rq3_apply_question_pack_prompt", _fake_prompt_builder):
            result = build_task_packs(
                benchmark=copy.deepcopy(benchmark),
                tasks=["apply"],
                generator_client=generator_client,
                validator_client=validator_client,
                provider="test",
                model="generator",
                validator_provider="test",
                validator_model="validator",
                item_count_per_key=1,
                max_rewrites=0,
                apply_workers=1,
            )

        cp = result["checkpoints"][0]
        self.assertIn("user_attributes_state:primary_vehicle", cp["rq3_apply_service_qa"]["keys"])
        self.assertNotIn("user_attributes_state:backup_vehicle", cp["rq3_apply_service_qa"]["keys"])
        self.assertEqual(
            cp["rq3_apply_service_qa"]["filtered_keys"]["user_attributes_state:backup_vehicle"]["reason_codes"],
            ["current_state_unresolved_missing_to"],
        )
        self.assertEqual(len(prompt_calls), 1)
        self.assertEqual(prompt_calls[0]["state_key"], "user_attributes_state:primary_vehicle")
        self.assertEqual(prompt_calls[0]["state_value"], "2024 Audi Q8")
        self.assertTrue(generator_client.prompts)
        item = cp["rq3_apply_service_qa"]["keys"]["user_attributes_state:primary_vehicle"]["items"][0]
        self.assertEqual(item["gold_memory_evidence_app_log_ids"], ["log_001"])
        self.assertEqual(len(item["answer_scoring_points"]), 2)
        self.assertTrue(item["qa_validation"]["is_valid"])
        self.assertTrue(item["atomic_fact_validation"]["is_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
