#!/usr/bin/env python3
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prompts_tce import (
    build_apply_slot_judge_prompt,
    build_change_slot_judge_prompt,
    build_snapshot_slot_judge_prompt,
)
from eval.eval_tce import (
    _normalize_slot_judgments,
    _run_apply_slot_judge,
    _run_change_slot_judge,
    _run_snapshot_slot_judge,
    evaluate,
)


class TceSlotLlmJudgeSchemaAcceptance(unittest.TestCase):
    def setUp(self):
        text_format_patch = patch("eval.eval_tce._build_slot_judge_text_format", return_value=dict)
        text_format_patch.start()
        self.addCleanup(text_format_patch.stop)

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
            self.assertIn('"analysis": "<brief slot-level judgment>"', prompt)
            self.assertNotIn("core_value_match", prompt)
            self.assertNotIn("constraint_coverage", prompt)
            self.assertNotIn("value_precision", prompt)
            self.assertNotIn("unsupported_inference_control", prompt)
            self.assertIn("indexed list paths", prompt)
        self.assertIn("reference_value", snapshot_prompt)
        self.assertNotIn('"point_text": "The value is 06:30."', snapshot_prompt)

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
        self.assertTrue(all('"state_key": "a:' in prompt for prompt in FakeClient.prompts))
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
        self.assertIn('"state_key": "a:y"', FakeClient.prompts[0])
        self.assertIn("a:x", rows[0]["snapshot_slot_eval_by_key"])
        self.assertIn("a:y", rows[0]["snapshot_slot_eval_by_key"])
        self.assertEqual(rows[0]["snapshot_slot_eval_by_key"]["a:x"]["judgments"][0]["analysis"], "cached")
        self.assertEqual(rows[0]["snapshot_point_score_mean_on_expected"], 1.0)

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
        self.assertIn('"state_key": "a:x"', FakeClient.prompts[0])
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
        self.assertIn('"state_key": "a:y"', FakeClient.prompts[0])
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
        self.assertIn('"qa_id": "q1"', FakeClient.prompts[0])
        self.assertIn("rq3_apply_slot_eval_by_item", rows[0])
        self.assertIn("slot_context", rows[0]["rq3_apply_slot_eval_by_item"]["a:x::q1"])

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
        self.assertIn('"qa_id": "q2"', FakeClient.prompts[0])
        self.assertIn("a:x::q1", rows[0]["rq3_apply_slot_eval_by_item"])
        self.assertIn("a:y::q2", rows[0]["rq3_apply_slot_eval_by_item"])
        self.assertEqual(rows[0]["rq3_apply_slot_eval_by_item"]["a:x::q1"]["judgments"][0]["analysis"], "cached")
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
                }
            ]
        }

        with patch("eval.client.LLMClient", FakeClient):
            result, _align = evaluate(
                benchmark,
                prediction,
                enable_llm_judge=True,
                save_eyeball=False,
                existing_result=existing_result,
            )

        self.assertEqual(FakeClient.prompts, [])
        checkpoint = result["checkpoints"][0]
        self.assertEqual(
            checkpoint["snapshot_slot_eval_by_key"]["habits_state:morning_walk"]["judgments"][0]["analysis"],
            "cached",
        )

    def test_evaluate_keeps_slot_judge_audit_out_of_main_result(self):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def supports_structured_response(self):
                return True

            def ask_structured_async(self, prompt, text_format=None):
                future = Future()
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
                return_audit=True,
            )

        checkpoint = result["checkpoints"][0]
        self.assertIn("snapshot_slot_eval_by_key", checkpoint)
        self.assertNotIn("snapshot_slot_judge_inputs", result)
        self.assertNotIn("snapshot_slot_judge_prompt_by_key", checkpoint)
        self.assertNotIn("snapshot_slot_judge_raw_output_by_key", checkpoint)
        self.assertIn("snapshot", audit)
        self.assertEqual(len(audit["snapshot"]), 1)
        self.assertIn("prompt", audit["snapshot"][0])
        self.assertIn("raw_output", audit["snapshot"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
