#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_tce import evaluate


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
                                        "apply_scenario": "The user must schedule a morning departure now.",
                                        "apply_question": "When should the user leave home for the walk?",
                                        "apply_reference_answer": "A. Leave before 06:30",
                                        "retrieval_query": "Service scenario:\nThe user must schedule a morning departure now.\n\nQuestion:\nWhen should the user leave home for the walk?",
                                    },
                                    {
                                        "qa_id": "q2",
                                        "apply_scenario": "The user is setting tomorrow morning alarm now.",
                                        "apply_question": "What alarm time supports this walk?",
                                        "apply_reference_answer": "B. Set alarm for 06:00",
                                        "retrieval_query": "Service scenario:\nThe user is setting tomorrow morning alarm now.\n\nQuestion:\nWhat alarm time supports this walk?",
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
                                    "answer": "A. Leave before 06:30",
                                    "evidence": [
                                        {"app_log_id": "log_0001", "evidence_content": "start_time 06:30"}
                                    ],
                                },
                                {
                                    "qa_id": "q2",
                                    "answer": "B. Set alarm for 06:00",
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
        self.assertIn("rq3_apply_option_extractable_item_count_mean", summary)
        self.assertIn("rq3_apply_option_prediction_coverage_on_extractable_mean", summary)
        self.assertIn("rq3_apply_option_accuracy_on_extractable_mean", summary)
        self.assertEqual(summary["rq3_apply_item_count_mean"], 2.0)
        self.assertEqual(summary["rq3_apply_key_coverage_mean"], 1.0)
        self.assertEqual(summary["rq3_apply_option_extractable_item_count_mean"], 2.0)
        self.assertEqual(summary["rq3_apply_option_prediction_coverage_on_extractable_mean"], 1.0)
        self.assertEqual(summary["rq3_apply_option_accuracy_on_extractable_mean"], 1.0)

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
