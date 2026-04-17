#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_tce import evaluate
from bench_core.tce_evaluator import evaluate_tce_rows
from tce_contracts import CURRENT_TASK_CONTRACT_VERSION, RESEARCH_FRAME_VERSION_V2


class TceRq3EvalAcceptance(unittest.TestCase):
    def test_rq3_structured_fields_present(self):
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
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "reason_codes": [],
                            "askable_fields": ["start_time"],
                            "validator_version": "qv1",
                        }
                    },
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "pair_count_per_key": 2,
                        "keys": {
                            "habits_state:morning_walk": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_category": "calendar_assistant",
                                        "question": "When should the user leave home for the walk?",
                                        "reference_answer": "Leave before 06:30 for the walk.",
                                        "retrieval_query": "Service scenario:\nThe user must schedule a morning departure now.\n\nQuestion:\nWhen should the user leave home for the walk?",
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "asp_q1_1",
                                                "point_type": "micro",
                                                "polarity": "positive",
                                                "point_text": "The answer says the user should leave before 06:30.",
                                            }
                                        ],
                                    },
                                    {
                                        "qa_id": "q2",
                                        "service_category": "alarm_assistant",
                                        "question": "What alarm time supports this walk?",
                                        "reference_answer": "Set the alarm for 06:00.",
                                        "retrieval_query": "Service scenario:\nThe user is setting tomorrow morning alarm now.\n\nQuestion:\nWhat alarm time supports this walk?",
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "asp_q2_1",
                                                "point_type": "micro",
                                                "polarity": "positive",
                                                "point_text": "The answer recommends a 06:00 alarm.",
                                            }
                                        ],
                                    },
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
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                        ]
                    },
                    "rq3_apply_answers": {
                        "habits_state:morning_walk": {
                            "items": [
                                {
                                    "qa_id": "q1",
                                    "answer": "Leave before 06:30.",
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                                    ],
                                },
                                {
                                    "qa_id": "q2",
                                    "answer": "Set the alarm for 06:00.",
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                                    ],
                                },
                            ]
                        }
                    },
                }
            ]
        }

        result, _ = evaluate(benchmark, prediction, enable_llm_judge=False, save_eyeball=False)
        summary = result.get("summary", {})
        self.assertIn("rq3_apply_item_count_mean", summary)
        self.assertIn("rq3_apply_key_coverage_mean", summary)
        self.assertIn("rq3_apply_evidence_content_nonempty_rate_mean", summary)
        self.assertNotIn("rq3_apply_option_extractable_item_count_mean", summary)
        self.assertNotIn("rq3_apply_option_prediction_coverage_on_extractable_mean", summary)
        self.assertNotIn("rq3_apply_option_accuracy_on_extractable_mean", summary)
        self.assertEqual(summary["rq3_apply_item_count_mean"], 2.0)
        self.assertEqual(summary["rq3_apply_key_coverage_mean"], 1.0)

    def test_rq3_missing_answer_scoring_points_raise(self):
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
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "reason_codes": [],
                            "askable_fields": ["start_time"],
                            "validator_version": "qv1",
                        }
                    },
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "pair_count_per_key": 1,
                        "keys": {
                            "habits_state:morning_walk": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "question": "When should the user leave home for the walk?",
                                        "reference_answer": "Leave before 06:30 for the walk.",
                                        "retrieval_query": "When should the user leave home for the walk?",
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
                    "snapshot_state": {"habits_state:morning_walk": {"timing": {"start_time": "06:30"}}},
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                        ]
                    },
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

        with self.assertRaisesRegex(ValueError, "missing non-empty answer_scoring_points"):
            evaluate(benchmark, prediction, enable_llm_judge=False, save_eyeball=False)

    def test_rq3_v2_structured_output_materializes_field_slots(self):
        benchmark = {
            "user_id": "001_user_001",
            "task_contract_version": CURRENT_TASK_CONTRACT_VERSION,
            "research_frame_version": RESEARCH_FRAME_VERSION_V2,
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "preferences_state": {"learning_modality": {"statement": "prefers self-paced webinars"}}
                    },
                    "state_observability": {
                        "preferences_state": {
                            "learning_modality": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "preferences_state:learning_modality": {
                            "is_questionable": True,
                            "reason_codes": [],
                            "askable_fields": ["statement"],
                            "validator_version": "qv1",
                        }
                    },
                    "rq3_apply_service_qa": {
                        "version": "v9",
                        "pair_count_per_key": 1,
                        "keys": {
                            "preferences_state:learning_modality": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_family": "information_request_construction",
                                        "scenario": "The assistant is preparing the structured information-request object before a training-resource search.",
                                        "task_instruction": "Fill the structured information-request payload.",
                                        "output_template": {"request_profile": {"preferred_profile": "<fill>"}},
                                        "reference_output": {"request_profile": {"preferred_profile": "prefers self-paced webinars"}},
                                        "retrieval_query": "Service family:\ninformation_request_construction",
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "aqp_q1_p1",
                                                "point_type": "field",
                                                "polarity": "positive",
                                                "source_field_path": "statement",
                                                "output_field_path": "request_profile.preferred_profile",
                                                "target_path": "request_profile.preferred_profile",
                                                "point_text": "The structured service output correctly fills request_profile.preferred_profile using the value grounded in source field statement.",
                                                "reference_value": "prefers self-paced webinars",
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
                    "snapshot_state": {
                        "preferences_state:learning_modality": {"statement": "prefers self-paced webinars"}
                    },
                    "evidence": {
                        "preferences_state:learning_modality": [
                            {"app_log_id": "log_0001", "evidence_content": "prefers self-paced webinars"}
                        ]
                    },
                    "rq3_apply_answers": {
                        "preferences_state:learning_modality": {
                            "items": [
                                {
                                    "qa_id": "q1",
                                    "service_family": "information_request_construction",
                                    "output": {"request_profile": {"preferred_profile": "prefers self-paced webinars"}},
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "prefers self-paced webinars"}
                                    ],
                                }
                            ]
                        }
                    },
                }
            ]
        }

        rows, evaluated, _ = evaluate_tce_rows(
            benchmark,
            prediction,
            align_by_timestamp=True,
            include_internal_payload=True,
        )

        self.assertEqual(evaluated, 1)
        row = rows[0]
        item = row["_rq3_apply_slots_by_item"]["preferences_state:learning_modality::q1"]
        self.assertEqual(item["service_family"], "information_request_construction")
        self.assertEqual(item["reference_output"], {"request_profile": {"preferred_profile": "prefers self-paced webinars"}})
        self.assertEqual(item["predicted_output"], {"request_profile": {"preferred_profile": "prefers self-paced webinars"}})
        self.assertEqual(len(item["slots"]), 1)
        slot = item["slots"][0]
        self.assertEqual(slot["point_id"], "aqp_q1_p1")
        self.assertEqual(slot["point_type"], "field")
        self.assertEqual(slot["polarity"], "positive")
        self.assertEqual(slot["target_path"], "request_profile.preferred_profile")
        self.assertEqual(slot["reference_value"], "prefers self-paced webinars")
        self.assertEqual(slot["predicted_value"], "prefers self-paced webinars")

    def test_rq3_v2_user_communication_uses_natural_language_answer_points(self):
        benchmark = {
            "user_id": "001_user_001",
            "task_contract_version": CURRENT_TASK_CONTRACT_VERSION,
            "research_frame_version": RESEARCH_FRAME_VERSION_V2,
            "total_checkpoints": 1,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "is_questionable": True,
                            "reason_codes": [],
                            "askable_fields": ["timing.start_time"],
                            "validator_version": "qv1",
                        }
                    },
                    "rq3_apply_service_qa": {
                        "version": "v9",
                        "pair_count_per_key": 1,
                        "keys": {
                            "habits_state:morning_walk": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_family": "user_communication",
                                        "scenario": "It is 06:10. Nothing has been logged yet today.",
                                        "task_instruction": "Write the short reminder message the assistant should send right now.",
                                        "reference_answer": "Send a reminder that the walk starts at 06:30.",
                                        "retrieval_query": "Service family:\nuser_communication",
                                        "gold_memory_evidence_app_log_ids": ["log_0001"],
                                        "answer_scoring_points": [
                                            {
                                                "point_id": "aqp_q1_p1",
                                                "point_type": "micro",
                                                "polarity": "positive",
                                                "point_text": "The answer mentions that the reminder is for the walk.",
                                                "reference_value": "Send a reminder that the walk starts at 06:30.",
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
                    "snapshot_state": {
                        "habits_state:morning_walk": {"timing": {"start_time": "06:30"}}
                    },
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "walk starts at 06:30"}
                        ]
                    },
                    "rq3_apply_answers": {
                        "habits_state:morning_walk": {
                            "items": [
                                {
                                    "qa_id": "q1",
                                    "service_family": "user_communication",
                                    "answer": "Send a reminder that the walk starts at 06:30.",
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "walk starts at 06:30"}
                                    ],
                                }
                            ]
                        }
                    },
                }
            ]
        }

        rows, evaluated, _ = evaluate_tce_rows(
            benchmark,
            prediction,
            align_by_timestamp=True,
            include_internal_payload=True,
        )

        self.assertEqual(evaluated, 1)
        row = rows[0]
        item = row["_rq3_apply_slots_by_item"]["habits_state:morning_walk::q1"]
        self.assertEqual(item["service_family"], "user_communication")
        self.assertEqual(item["reference_answer"], "Send a reminder that the walk starts at 06:30.")
        self.assertEqual(item["predicted_answer"], "Send a reminder that the walk starts at 06:30.")
        self.assertIsNone(item["reference_output"])
        self.assertIsNone(item["predicted_output"])
        self.assertEqual(len(item["slots"]), 1)
        slot = item["slots"][0]
        self.assertEqual(slot["point_id"], "aqp_q1_p1")
        self.assertEqual(slot["point_type"], "micro")
        self.assertEqual(slot["polarity"], "positive")
        self.assertEqual(slot["predicted_value"], "Send a reminder that the walk starts at 06:30.")

    def test_eval_prefers_prebuilt_task_pack_scope(self):
        benchmark = {
            "user_id": "001_user_001",
            "total_checkpoints": 2,
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}},
                        "profile_state": {"favorite_coffee": "latte"},
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}}
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "previous_checkpoint_id": "",
                        "previous_cutoff_ts": "",
                        "keys": {},
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                    "rq3_apply_service_qa": {"version": "v1", "keys": {}},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "07:00"}}},
                        "profile_state": {"favorite_coffee": "espresso"},
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "07:00"}}}
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:morning_walk": {
                                "answer_template": {"timing": {"start_time": "<fill the blank>"}}
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "previous_checkpoint_id": "cp_0001",
                        "previous_cutoff_ts": "2025-01-01 08:00:00",
                        "keys": {
                            "habits_state:morning_walk": {
                                "before_template": {"timing": {"start_time": "<fill the blank>"}},
                                "after_template": {"timing": {"start_time": "<fill the blank>"}},
                            }
                        },
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0002"]}
                        }
                    },
                    "rq3_apply_service_qa": {"version": "v1", "keys": {}},
                },
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "cp_0001",
                    "snapshot_state": {
                        "habits_state:morning_walk": {"timing": {"start_time": "06:30"}},
                        "profile_state:favorite_coffee": "wrong",
                    },
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                        ]
                    },
                    "change_analysis": {},
                    "rq3_apply_answers": {},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "snapshot_state": {
                        "habits_state:morning_walk": {"timing": {"start_time": "07:00"}},
                        "profile_state:favorite_coffee": "wrong_again",
                    },
                    "evidence": {
                        "habits_state:morning_walk": [
                            {"app_log_id": "log_0002", "evidence_content": "start_time 07:00"}
                        ]
                    },
                    "change_analysis": {
                        "habits_state:morning_walk": {
                            "before": {"timing": {"start_time": "06:30"}},
                            "after": {"timing": {"start_time": "07:00"}},
                            "change_reason": "schedule moved later",
                            "evidence": [
                                {"app_log_id": "log_0002", "evidence_content": "start_time 07:00"}
                            ],
                        }
                    },
                    "rq3_apply_answers": {},
                },
            ]
        }

        result, _ = evaluate(benchmark, prediction, enable_llm_judge=False, save_eyeball=False)
        checkpoints = result.get("checkpoints", [])
        cp1 = checkpoints[0]
        cp2 = checkpoints[1]
        self.assertEqual(cp1["snapshot_value_accuracy_on_expected"], 1.0)
        self.assertEqual(cp2["snapshot_value_accuracy_on_expected"], 1.0)
        self.assertEqual(cp2["change_item_count"], 1.0)
        self.assertEqual(cp2["change_before_after_f1_mean_on_changed"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
