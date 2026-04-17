#!/usr/bin/env python3
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_construction.build_tce_rq3_apply_pack import build_apply_pack
from tce_core.prompts import (
    build_apply_answer_scoring_points_prompt,
    build_rq3_apply_question_pack_prompt,
    build_rq3_apply_rewrite_prompt,
    build_rq3_apply_validation_prompt,
)
from tce_core.task_packs import _validate_apply_item, build_task_packs
from tce_contracts import LEGACY_TASK_CONTRACT_VERSION


def _with_legacy_contract(payload):
    payload = json.loads(json.dumps(payload))
    payload.setdefault("task_contract_version", LEGACY_TASK_CONTRACT_VERSION)
    return payload


class _FakeStateValidatorClient:
    def __init__(self, provider: str, model_name: str, max_workers: int = 1):
        self.provider = provider
        self.model_name = model_name
        self.max_workers = max_workers

    def ask(self, prompt: str, response_type: str = "json"):
        if "Validate whether this state is inferable from evidence" in prompt:
            return {
                "is_questionable": True,
                "reason_codes": [],
                "llm_reason": "state is inferable",
                "field_verdicts": [
                    {
                        "field_name": "timing.start_time",
                        "reason_analysis": "supported by logs",
                        "is_valid": True,
                    }
                ],
            }
        return {}

    def close(self):
        return None


class _FakeApplyPackClient:
    def __init__(self, provider: str, model_name: str, max_workers: int = 1):
        self.provider = provider
        self.model_name = model_name
        self.max_workers = max_workers

    def ask(self, prompt: str, response_type: str = "json"):
        if "Generate personalized service free-form QA items" in prompt:
            return {
                "items": [
                    {
                        "service_category": "notification strategy",
                        "question": "What communication policy should the assistant adopt for the user during recurring work blocks?",
                        "reference_answer": "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests.",
                    }
                ]
            }
        if "Generate scoreable atomic facts for one apply-service QA item." in prompt:
            return {
                "points": [
                    {
                        "polarity": "positive",
                        "point_text": "The answer delays non-urgent briefing-related requests until after the recurring work block finishes.",
                        "reference_value": "delay non-urgent requests until after recurring work block",
                    },
                    {
                        "polarity": "positive",
                        "point_text": "The answer still allows urgent escalation requests to interrupt immediately.",
                        "reference_value": "urgent escalation requests interrupt immediately",
                    },
                ]
            }
        if "Validate whether this apply-service QA item is a strong personalized service decision item." in prompt:
            return {
                "criteria": [
                    {
                        "criterion": "personalization_necessity",
                        "pass": True,
                        "analysis": "The answer depends on the user's recurring work block state.",
                    },
                    {
                        "criterion": "service_decision_quality",
                        "pass": True,
                        "analysis": "The item asks for one service policy choice rather than recall or a menu.",
                    },
                    {
                        "criterion": "answer_groundedness",
                        "pass": True,
                        "analysis": "One service action is clearly preferable and the answer stays within the given information.",
                    },
                ]
            }
        if "Validate one generated atomic-fact set for an apply-service QA item." in prompt:
            return {
                "points": [
                    {
                        "point_id": "aqp_habits_state_morning_walk_q1_p1",
                        "pass": True,
                        "analysis": "This point is grounded in the reference answer.",
                        "fail_reasons": [],
                    },
                    {
                        "point_id": "aqp_habits_state_morning_walk_q1_p2",
                        "pass": True,
                        "analysis": "This point is atomic and scoreable.",
                        "fail_reasons": [],
                    },
                ],
                "set_pass": True,
                "set_failures": [],
            }
        return {}

    def close(self):
        return None


class TceApplyPackContractAcceptance(unittest.TestCase):
    def test_apply_generation_prompt_uses_checklist_structure_and_richer_example(self):
        prompt = build_rq3_apply_question_pack_prompt(
            checkpoint_timestamp="2025-01-01 08:00:00",
            state_key="preferences_state:learning_modality",
            state_value={
                "statement": "Prefers in-depth, self-paced technical white papers and webinars over large conferences"
            },
            item_count=1,
        )

        self.assertLess(prompt.index("[Task Instruction]"), prompt.index("[Definitions]"))
        self.assertLess(prompt.index("[Definitions]"), prompt.index("[Constraints]"))
        self.assertLess(prompt.index("[Constraints]"), prompt.index("[Example]"))
        self.assertLess(prompt.index("[Example]"), prompt.index("[Input/Output Format]"))
        self.assertIn("personalized service", prompt)
        self.assertIn("one service_category", prompt)
        self.assertIn("one best service action", prompt)
        self.assertNotIn("1 to 3 atomic facts", prompt)
        self.assertIn("Prefer bounded policy or service-decision questions such as:", prompt)
        self.assertIn("Reject questions whose scenario merely echoes the same wording, label, or semantic category already present in the state", prompt)
        self.assertIn("The reference_answer must not introduce extra methods, tactics, metrics, escalation mechanisms, capacity calculations", prompt)
        self.assertIn("schedule-grounded service operations are valid when the question asks what the assistant should do around the routine", prompt)
        self.assertIn('- state_key: one validated state item key', prompt)
        self.assertIn('Actual Input:\n- checkpoint_timestamp: 2025-01-01 08:00:00', prompt)
        self.assertIn('- state_key: "preferences_state:learning_modality"', prompt)
        self.assertNotIn('"qa_id"', prompt)
        self.assertIn("What training format should the assistant arrange for the user's upcoming technical upskilling plan?", prompt)
        self.assertIn("How should the assistant handle non-urgent notifications during the user's recurring Sunday family dinner block?", prompt)
        self.assertIn("What specific lot-selection strategy should the assistant utilize when the user requests to liquidate a portion of the portfolio for an upcoming expense?", prompt)
        self.assertIn("What type of care recommendation should the assistant prioritize when the user reports a minor, non-acute health issue?", prompt)

    def test_apply_validation_prompt_is_programmatic_and_rewrite_prompt_includes_feedback(self):
        validation_prompt = build_rq3_apply_validation_prompt(
            state_key="habits_state:budget_review",
            state_value={"timing": {"start_time": "09:30"}},
            service_category="scheduling",
            question="What time should be reserved for the user's budget review?",
            reference_answer="Reserve the usual 09:30 slot.",
        )
        self.assertNotIn('"is_valid": true', validation_prompt)
        self.assertIn('"criterion": "personalization_necessity"', validation_prompt)
        self.assertIn('"criterion": "service_decision_quality"', validation_prompt)
        self.assertIn('"criterion": "answer_groundedness"', validation_prompt)
        self.assertIn('"pass": true', validation_prompt)
        self.assertIn('"pass": "<bool>"', validation_prompt)
        self.assertNotIn('"llm_verdict": "pass|fail"', validation_prompt)
        self.assertIn('"service_category": "scheduling"', validation_prompt)
        self.assertNotIn('"rubric": [', validation_prompt)
        self.assertIn("introduces new concrete facts, thresholds, tactics, metrics, product specs, or background knowledge", validation_prompt)
        self.assertIn("explicit A/B or named-option menu", validation_prompt)
        self.assertIn("For habit states, allow schedule-grounded service operations", validation_prompt)

        rewrite_prompt = build_rq3_apply_rewrite_prompt(
            state_key="habits_state:budget_review",
            state_value={"timing": {"start_time": "09:30"}},
            service_category="scheduling",
            question="What time is the review?",
            reference_answer="09:30",
            failed_rules=["service_decision_quality", "llm_invalid"],
            semantic_criteria=[
                {
                    "criterion": "personalization_necessity",
                    "pass": True,
                    "analysis": "The state matters.",
                },
                {
                    "criterion": "service_decision_quality",
                    "pass": False,
                    "analysis": "The item is direct state restatement rather than a real service decision.",
                },
                {
                    "criterion": "answer_groundedness",
                    "pass": True,
                    "analysis": "The revised answer should be bounded.",
                },
            ],
        )
        self.assertIn('"criterion": "service_decision_quality"', rewrite_prompt)
        self.assertIn('"analysis": "The item is direct state restatement rather than a real service decision."', rewrite_prompt)
        self.assertIn('"service_decision_quality"', rewrite_prompt)
        self.assertNotIn('"rubric": [', rewrite_prompt)

    def test_state_validate_only_runs_stage1_and_writes_validation_identity(self):
        payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench.json"
            out = td_path / "bench_validated.json"
            inp.write_text(json.dumps(_with_legacy_contract(payload), ensure_ascii=False, indent=2), encoding="utf-8")

            with patch("data_construction.build_tce_state_validation.LLMClient", _FakeStateValidatorClient):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=2,
                    questionability_mode="filter",
                    state_validate_only=True,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        cp = result["checkpoints"][0]
        sq = cp["state_questionability"]["habits_state:morning_walk"]
        self.assertIn("validated_snapshot_state", cp)
        self.assertTrue(sq["is_questionable"])
        self.assertEqual(sq["validation_source"], "computed")
        self.assertEqual(sq["validation_identity"]["prompt_version"], "state_validate_prompt_v2")
        self.assertEqual(
            cp["validated_snapshot_state"]["habits_state"]["morning_walk"],
            {"timing": {"start_time": "06:30"}},
        )

    def test_apply_pack_builds_from_validated_benchmark_only(self):
        validated_payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench_validated.json"
            out = td_path / "bench_apply.json"
            inp.write_text(json.dumps(_with_legacy_contract(validated_payload), ensure_ascii=False, indent=2), encoding="utf-8")

            with patch("data_construction.build_tce_task_packs.LLMClient", _FakeApplyPackClient):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=2,
                    questionability_mode="filter",
                    state_validate_only=False,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        cp = result["checkpoints"][0]
        key_node = cp["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]
        self.assertEqual(key_node["pack_source"], "computed")
        self.assertIn("pack_identity", key_node)
        self.assertEqual(
            key_node["pack_identity"]["prompt_version"],
            "apply_pack_prompt_v13",
        )
        self.assertEqual(key_node["items"][0]["qa_id"], "q1")
        self.assertIn("retrieval_query", key_node["items"][0])
        self.assertTrue(key_node["items"][0]["qa_validation"]["is_valid"])
        self.assertTrue(key_node["items"][0]["atomic_fact_validation"]["is_valid"])
        self.assertIn("answer_scoring_points", key_node["items"][0])
        self.assertEqual(key_node["items"][0]["service_category"], "notification strategy")
        self.assertEqual(
            key_node["items"][0]["question"],
            "What communication policy should the assistant adopt for the user during recurring work blocks?",
        )
        self.assertEqual(
            key_node["items"][0]["reference_answer"],
            "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests.",
        )
        self.assertEqual(
            key_node["items"][0]["gold_memory_evidence_app_log_ids"],
            ["log_0001"],
        )
        self.assertNotIn("rubric_invalid", key_node["items"][0]["qa_validation"]["failed_rules"])
        self.assertNotIn("missing_gold_memory_evidence", key_node["items"][0]["qa_validation"]["failed_rules"])
        self.assertEqual(
            [item["criterion"] for item in key_node["items"][0]["qa_validation"]["semantic_criteria"]],
            [
                "personalization_necessity",
                "service_decision_quality",
                "answer_groundedness",
            ],
        )
        self.assertFalse(key_node["items"][0]["atomic_fact_validation"]["used_safe_fallback"])

    def test_apply_pack_invalid_when_validator_schema_missing_criterion(self):
        class _BrokenValidatorClient(_FakeApplyPackClient):
            atomic_fact_prompt_count = 0

            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate personalized service free-form QA items" in prompt:
                    return super().ask(prompt, response_type=response_type)
                if "Generate scoreable atomic facts for one apply-service QA item." in prompt:
                    type(self).atomic_fact_prompt_count += 1
                    return super().ask(prompt, response_type=response_type)
                if "Validate whether this apply-service QA item is a strong personalized service decision item." in prompt:
                    return {
                        "criteria": [
                            {
                                "criterion": "personalization_necessity",
                                "pass": True,
                                "analysis": "ok",
                            },
                            {
                                "criterion": "service_decision_quality",
                                "pass": True,
                                "analysis": "ok",
                            },
                        ]
                    }
                return {}

        validated_payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench_validated.json"
            out = td_path / "bench_apply.json"
            inp.write_text(json.dumps(_with_legacy_contract(validated_payload), ensure_ascii=False, indent=2), encoding="utf-8")

            with patch("data_construction.build_tce_task_packs.LLMClient", _BrokenValidatorClient):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=0,
                    questionability_mode="filter",
                    state_validate_only=False,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        discarded = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["discarded_items"][0]
        self.assertIn("llm_invalid", discarded["qa_validation"]["failed_rules"])
        self.assertTrue(discarded["qa_validation"]["manual_review_required"])
        self.assertIsNone(discarded["atomic_fact_validation"])
        self.assertEqual(_BrokenValidatorClient.atomic_fact_prompt_count, 0)

    def test_apply_pack_rewrites_invalid_rubric_before_accepting_item(self):
        class _AtomicFactRewriteClient(_FakeApplyPackClient):
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate personalized service free-form QA items" in prompt:
                    return {
                        "items": [
                            {
                                "service_category": "notification strategy",
                                "question": "What communication policy should the assistant adopt for the user during recurring work blocks?",
                                "reference_answer": "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests.",
                            }
                        ]
                    }
                if "Generate scoreable atomic facts for one apply-service QA item." in prompt:
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The answer routes every request into a weekly postal mail digest.",
                                "reference_value": "weekly postal mail digest",
                            },
                            {
                                "polarity": "negative",
                                "point_text": "The answer never allows urgent escalation requests to interrupt.",
                                "reference_value": "never allows urgent escalation",
                            },
                        ]
                    }
                if "Rewrite an invalid atomic-fact set for one apply-service QA item." in prompt:
                    return {
                        "rubric": [
                            "The answer delays non-urgent requests until after the recurring work block finishes.",
                            "The answer still allows urgent escalation requests to interrupt immediately.",
                        ],
                    }
                if "Validate whether this apply-service QA item is a strong personalized service decision item." in prompt:
                    return {
                        "criteria": [
                            {"criterion": "personalization_necessity", "pass": True, "analysis": "The policy depends on the recurring work block state."},
                            {"criterion": "service_decision_quality", "pass": True, "analysis": "The question asks for one service policy rather than recall or a menu."},
                            {"criterion": "answer_groundedness", "pass": True, "analysis": "The answer is specific, bounded, and grounded."},
                        ]
                    }
                if "Validate one generated atomic-fact set for an apply-service QA item." in prompt:
                    if "weekly postal mail digest" in prompt:
                        return {
                            "points": [
                                {
                                    "point_id": "aqp_habits_state_morning_walk_q1_p1",
                                    "pass": False,
                                    "analysis": "This point overfits an unnecessary delivery mechanism and conflicts with the reference answer.",
                                    "fail_reasons": ["drift", "unsupported", "over_specific"],
                                },
                                {
                                    "point_id": "aqp_habits_state_morning_walk_q1_p2",
                                    "pass": False,
                                    "analysis": "This point repeats the same narrow postal-mail framing instead of capturing a distinct semantic requirement.",
                                    "fail_reasons": ["redundant", "over_specific"],
                                },
                            ],
                            "set_pass": False,
                            "set_failures": ["coverage_gap"],
                        }
                    return {
                        "points": [
                            {
                                "point_id": "aqp_habits_state_morning_walk_q1_p1",
                                "pass": True,
                                "analysis": "This point is grounded in the reference answer.",
                                "fail_reasons": [],
                            },
                            {
                                "point_id": "aqp_habits_state_morning_walk_q1_p2",
                                "pass": True,
                                "analysis": "This point captures the urgency exception cleanly.",
                                "fail_reasons": [],
                            },
                        ],
                        "set_pass": True,
                        "set_failures": [],
                    }
                return super().ask(prompt, response_type=response_type)

        validated_payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench_validated.json"
            out = td_path / "bench_apply.json"
            inp.write_text(json.dumps(_with_legacy_contract(validated_payload), ensure_ascii=False, indent=2), encoding="utf-8")
            with patch("data_construction.build_tce_task_packs.LLMClient", _AtomicFactRewriteClient):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=2,
                    questionability_mode="filter",
                    state_validate_only=False,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["items"][0]
        self.assertTrue(item["qa_validation"]["is_valid"])
        self.assertTrue(item["atomic_fact_validation"]["is_valid"])
        self.assertEqual(item["atomic_fact_validation"]["rewrite_attempts"], 1)
        self.assertFalse(item["atomic_fact_validation"]["used_safe_fallback"])
        self.assertEqual(
            item["question"],
            "What communication policy should the assistant adopt for the user during recurring work blocks?",
        )
        self.assertEqual(
            [point["point_text"] for point in item["answer_scoring_points"]],
            [
                "The answer delays non-urgent requests until after the recurring work block finishes.",
                "The answer still allows urgent escalation requests to interrupt immediately.",
            ],
        )
        self.assertEqual(
            item["rubric"],
            [
                "The answer delays non-urgent requests until after the recurring work block finishes.",
                "The answer still allows urgent escalation requests to interrupt immediately.",
            ],
        )
        self.assertEqual(item["atomic_fact_validation"]["failed_rules"], [])

    def test_apply_pack_rewrites_over_specific_device_slots_into_stable_semantic_facts(self):
        class _HomeMediaRewriteClient(_FakeApplyPackClient):
            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate personalized service free-form QA items" in prompt:
                    return {
                        "items": [
                            {
                                "service_category": "playback policy",
                                "question": "How should the assistant prioritize playback sources when a 4K film is available through both a commercial streaming platform and the local repository?",
                                "reference_answer": "Select the local Synology DS923+ NAS as the source to ensure the 4K digital version is used.",
                            }
                        ]
                    }
                if "Generate scoreable atomic facts for one apply-service QA item." in prompt:
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The answer specifies that the Synology DS923+ NAS should be selected as the source for movie playback.",
                                "reference_value": "Synology DS923+ NAS",
                            },
                            {
                                "polarity": "positive",
                                "point_text": "The answer states that selecting the Synology DS923+ NAS ensures the 4K digital version is utilized.",
                                "reference_value": "Synology DS923+ NAS ensures 4K digital version",
                            },
                        ]
                    }
                if "Rewrite an invalid atomic-fact set for one apply-service QA item." in prompt:
                    return {
                        "rubric": [
                            "The answer prioritizes the local NAS or local repository as the playback source.",
                            "The answer prefers the local 4K digital copy over the commercial streaming source.",
                        ]
                    }
                if "Validate whether this apply-service QA item is a strong personalized service decision item." in prompt:
                    return {
                        "criteria": [
                            {"criterion": "personalization_necessity", "pass": True, "analysis": "The best playback choice depends on the user's media-server state."},
                            {"criterion": "service_decision_quality", "pass": True, "analysis": "The item asks for one playback-source policy rather than recall or a menu."},
                            {"criterion": "answer_groundedness", "pass": True, "analysis": "The answer is bounded by the state and question."},
                        ]
                    }
                if "Validate one generated atomic-fact set for an apply-service QA item." in prompt:
                    if "Synology DS923+ NAS should be selected" in prompt:
                        return {
                            "points": [
                                {
                                    "point_id": "aqp_user_attributes_state_home_media_server_q1_p1",
                                    "pass": False,
                                    "analysis": "This point depends on an unnecessary exact device model and is too brittle.",
                                    "fail_reasons": ["over_specific"],
                                },
                                {
                                    "point_id": "aqp_user_attributes_state_home_media_server_q1_p2",
                                    "pass": False,
                                    "analysis": "This point repeats the same narrow device anchor instead of covering a distinct semantic requirement.",
                                    "fail_reasons": ["redundant", "over_specific"],
                                },
                            ],
                            "set_pass": False,
                            "set_failures": ["coverage_gap"],
                        }
                    return {
                        "points": [
                            {
                                "point_id": "aqp_user_attributes_state_home_media_server_q1_p1",
                                "pass": True,
                                "analysis": "This point captures the source-selection decision without brittle device overfitting.",
                                "fail_reasons": [],
                            },
                            {
                                "point_id": "aqp_user_attributes_state_home_media_server_q1_p2",
                                "pass": True,
                                "analysis": "This point captures the 4K-local-copy preference as a distinct semantic requirement.",
                                "fail_reasons": [],
                            },
                        ],
                        "set_pass": True,
                        "set_failures": [],
                    }
                return super().ask(prompt, response_type=response_type)

        validated_payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_observability": {
                        "user_attributes_state": {
                            "home_media_server": {"is_valid": True, "evidence_app_log_ids": ["log_00582"]}
                        }
                    },
                    "state_questionability": {
                        "user_attributes_state:home_media_server": {
                            "is_questionable": True,
                            "validated_state_value": "Synology DS923+ NAS for local 4K movie collection",
                        }
                    },
                    "validated_snapshot_state": {
                        "user_attributes_state": {
                            "home_media_server": "Synology DS923+ NAS for local 4K movie collection"
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench_validated.json"
            out = td_path / "bench_apply.json"
            inp.write_text(json.dumps(_with_legacy_contract(validated_payload), ensure_ascii=False, indent=2), encoding="utf-8")
            with patch("data_construction.build_tce_task_packs.LLMClient", _HomeMediaRewriteClient):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=2,
                    questionability_mode="filter",
                    state_validate_only=False,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["user_attributes_state:home_media_server"]["items"][0]
        self.assertTrue(item["atomic_fact_validation"]["is_valid"])
        self.assertEqual(item["atomic_fact_validation"]["rewrite_attempts"], 1)
        self.assertEqual(item["atomic_fact_validation"]["failed_rules"], [])
        self.assertEqual(
            [point["point_text"] for point in item["answer_scoring_points"]],
            [
                "The answer prioritizes the local NAS or local repository as the playback source.",
                "The answer prefers the local 4K digital copy over the commercial streaming source.",
            ],
        )

    def test_apply_atomic_fact_prompt_uses_state_value_as_grounding_context(self):
        prompt = build_apply_answer_scoring_points_prompt(
            state_key="user_attributes_state:home_media_server",
            state_value="Synology DS923+ NAS for local 4K movie collection",
            apply_scenario="A high-bitrate movie is available through two playback sources.",
            apply_question="How should the assistant prioritize playback sources?",
            apply_reference_answer="Select the local NAS as the source for the 4K digital version.",
            max_points=3,
        )
        self.assertIn("Use `state_value` together with `apply_question` and `apply_reference_answer` as grounding context.", prompt)
        self.assertIn("Do not generate multiple points that all hinge on the same narrow identifier", prompt)

    def test_apply_pack_uses_safe_atomic_fact_fallback_without_rewriting_qa(self):
        class _AtomicFactFallbackClient(_FakeApplyPackClient):
            def __init__(self, provider: str, model_name: str, max_workers: int = 1):
                super().__init__(provider, model_name, max_workers=max_workers)
                self.atomic_generation_prompts = []
                self.atomic_rewrite_prompts = []

            def ask(self, prompt: str, response_type: str = "json"):
                if "Generate personalized service free-form QA items" in prompt:
                    return {
                        "items": [
                            {
                                "service_category": "notification strategy",
                                "question": "What communication policy should the assistant adopt for the user during recurring work blocks?",
                                "reference_answer": "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests.",
                            }
                        ]
                    }
                if "Generate scoreable atomic facts for one apply-service QA item." in prompt:
                    self.atomic_generation_prompts.append(prompt)
                    return {
                        "points": [
                            {
                                "polarity": "positive",
                                "point_text": "The answer routes every request into a weekly postal mail digest.",
                                "reference_value": "weekly postal mail digest",
                            }
                        ]
                    }
                if "Rewrite an invalid atomic-fact set for one apply-service QA item." in prompt:
                    self.atomic_rewrite_prompts.append(prompt)
                    return {
                        "rubric": [
                            "The answer still routes every request into a weekly postal mail digest."
                        ]
                    }
                if "Validate whether this apply-service QA item is a strong personalized service decision item." in prompt:
                    return super().ask(prompt, response_type=response_type)
                if "Validate one generated atomic-fact set for an apply-service QA item." in prompt:
                    return {
                        "points": [
                            {
                                "point_id": "aqp_habits_state_morning_walk_q1_p1",
                                "pass": False,
                                "analysis": "The point conflicts with the reference answer.",
                                "fail_reasons": ["drift", "unsupported"],
                            }
                        ],
                        "set_pass": False,
                        "set_failures": ["coverage_gap"],
                    }
                return super().ask(prompt, response_type=response_type)

        validated_payload = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}}
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "bench_validated.json"
            out = td_path / "bench_apply.json"
            inp.write_text(json.dumps(_with_legacy_contract(validated_payload), ensure_ascii=False, indent=2), encoding="utf-8")
            client = _AtomicFactFallbackClient("azure", "gpt-5.1")
            with patch("data_construction.build_tce_task_packs.LLMClient", lambda *args, **kwargs: client):
                result = build_apply_pack(
                    benchmark_path=inp,
                    output_path=out,
                    provider="azure",
                    model="gpt-5.1",
                    item_count_per_key=1,
                    apply_workers=1,
                    reuse_scope="key_value_signature",
                    max_checkpoints=1,
                    save_raw=False,
                    validator_provider="azure",
                    validator_model="gpt-5-mini",
                    max_rewrites=1,
                    questionability_mode="filter",
                    state_validate_only=False,
                    app_logs_path=None,
                    l2_evidence_top_k=0,
                )

        item = result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]["habits_state:morning_walk"]["items"][0]
        self.assertTrue(item["qa_validation"]["is_valid"])
        self.assertEqual(item["qa_validation"]["rewrite_attempts"], 0)
        self.assertTrue(item["atomic_fact_validation"]["used_safe_fallback"])
        self.assertEqual(item["atomic_fact_validation"]["rewrite_attempts"], 1)
        self.assertEqual(
            item["question"],
            "What communication policy should the assistant adopt for the user during recurring work blocks?",
        )
        self.assertEqual(len(client.atomic_generation_prompts), 1)
        self.assertEqual(len(client.atomic_rewrite_prompts), 1)
        self.assertEqual(
            [point["point_text"] for point in item["answer_scoring_points"]],
            [
                "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests."
            ],
        )
        self.assertEqual(
            item["rubric"],
            [
                "Hold non-urgent briefing-related requests until after the user's recurring work block finishes, but interrupt immediately for urgent escalation requests."
            ],
        )

    def test_validate_apply_item_fails_on_failed_criterion_and_empty_analysis(self):
        class _CriterionFailClient:
            def ask(self, prompt: str, response_type: str = "json"):
                return {
                    "criteria": [
                        {
                            "criterion": "personalization_necessity",
                            "pass": False,
                            "analysis": "The scenario alone decides the answer.",
                        },
                        {
                            "criterion": "service_decision_quality",
                            "pass": True,
                            "analysis": "",
                        },
                        {
                            "criterion": "answer_groundedness",
                            "pass": True,
                            "analysis": "One option is uniquely best.",
                        },
                    ]
                }

        is_valid, payload = _validate_apply_item(
            validator_client=_CriterionFailClient(),
            state_key="habits_state:budget_review",
            state_value={"timing": {"start_time": "09:30"}},
            item={
                "apply_scenario": "The assistant must choose one policy. Candidate actions: A. Keep immediate alerts. B. Delay non-urgent alerts until after the review block. C. Disable all alerts.",
                "apply_question": "Which candidate action should the assistant adopt?",
                "apply_reference_answer": "B. Delay non-urgent alerts until after the review block.",
            },
        )
        self.assertFalse(is_valid)
        self.assertIn("personalization_necessity", payload["failed_rules"])
        self.assertIn("llm_invalid", payload["failed_rules"])
        self.assertEqual(payload["semantic_criteria"][1]["criterion"], "service_decision_quality")
        self.assertEqual(payload["semantic_criteria"][1]["analysis"], "")

    def test_validate_apply_item_ignores_surface_style_rules_when_semantics_pass(self):
        class _PassingValidatorClient:
            def ask(self, prompt: str, response_type: str = "json"):
                return {
                    "criteria": [
                        {
                            "criterion": "personalization_necessity",
                            "pass": True,
                            "analysis": "The best action still depends on the user's state.",
                        },
                        {
                            "criterion": "service_decision_quality",
                            "pass": True,
                            "analysis": "The item asks for one concrete assistant decision rather than recall or a menu.",
                        },
                        {
                            "criterion": "answer_groundedness",
                            "pass": True,
                            "analysis": "The answer is bounded and grounded enough for scoring.",
                        },
                    ]
                }

        is_valid, payload = _validate_apply_item(
            validator_client=_PassingValidatorClient(),
            state_key="habits_state:budget_review",
            state_value={"timing": {"start_time": "09:30"}},
            item={
                "service_category": "notification strategy",
                "question": "Should you delay non-urgent update requests until after the recurring budget review block even when weekly project summaries arrive during that period",
                "reference_answer": "Delay non-urgent requests until after the recurring budget review block, but allow urgent escalation requests immediately.",
            },
        )
        self.assertTrue(is_valid)
        self.assertNotIn("question_too_long", payload["failed_rules"])
        self.assertNotIn("second_person_wording", payload["failed_rules"])
        self.assertNotIn("missing_question_mark", payload["failed_rules"])

    def test_apply_pack_can_incrementally_save_every_n_keys(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        },
                        "habits_state:budget_review": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "09:30"}},
                        },
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}},
                            "budget_review": {"timing": {"start_time": "09:30"}},
                        }
                    },
                }
            ],
        }
        snapshots = []

        def _save_snapshot(payload):
            cp = payload["checkpoints"][0]
            snapshots.append(
                len((cp.get("rq3_apply_service_qa") or {}).get("keys") or {})
            )

        result = build_task_packs(
            benchmark=benchmark,
            tasks=["apply"],
            generator_client=_FakeApplyPackClient("gemini", "gemini-3-flash-preview"),
            validator_client=_FakeApplyPackClient("gemini", "gemini-3-flash-preview"),
            provider="gemini",
            model="gemini-3-flash-preview",
            validator_provider="gemini",
            validator_model="gemini-3-flash-preview",
            item_count_per_key=1,
            max_rewrites=0,
            save_raw=False,
            reuse_scope="key_value_signature",
            save_every_apply_keys=1,
            save_callback=_save_snapshot,
            show_progress=False,
        )
        self.assertGreaterEqual(len(snapshots), 3)
        self.assertIn(1, snapshots)
        self.assertEqual(snapshots[-1], 2)
        self.assertEqual(
            len(result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]),
            2,
        )

    def test_apply_pack_processes_keys_concurrently_when_workers_gt_one(self):
        class _SlowApplyClient(_FakeApplyPackClient):
            def ask(self, prompt: str, response_type: str = "json"):
                time.sleep(0.2)
                return super().ask(prompt, response_type=response_type)

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        },
                        "habits_state:budget_review": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "09:30"}},
                        },
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}},
                            "budget_review": {"timing": {"start_time": "09:30"}},
                        }
                    },
                }
            ],
        }

        start = time.perf_counter()
        result = build_task_packs(
            benchmark=benchmark,
            tasks=["apply"],
            generator_client=_SlowApplyClient("gemini", "gemini-3-flash-preview", max_workers=2),
            validator_client=_SlowApplyClient("gemini", "gemini-3-flash-preview", max_workers=2),
            provider="gemini",
            model="gemini-3-flash-preview",
            validator_provider="gemini",
            validator_model="gemini-3-flash-preview",
            item_count_per_key=1,
            max_rewrites=0,
            save_raw=False,
            reuse_scope="key_value_signature",
            apply_workers=2,
            show_progress=False,
        )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)
        self.assertEqual(
            len(result["checkpoints"][0]["rq3_apply_service_qa"]["keys"]),
            2,
        )

    def test_apply_pack_rejects_reuse_scope_none(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "validated_state_value": {"timing": {"start_time": "06:30"}},
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {"timing": {"start_time": "06:30"}},
                        }
                    },
                }
            ],
        }
        with self.assertRaises(ValueError):
            build_task_packs(
                benchmark=benchmark,
                tasks=["apply"],
                generator_client=_FakeApplyPackClient("gemini", "gemini-3-flash-preview"),
                validator_client=_FakeApplyPackClient("gemini", "gemini-3-flash-preview"),
                provider="gemini",
                model="gemini-3-flash-preview",
                validator_provider="gemini",
                validator_model="gemini-3-flash-preview",
                item_count_per_key=1,
                max_rewrites=0,
                save_raw=False,
                reuse_scope="none",
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
