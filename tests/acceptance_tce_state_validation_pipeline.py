#!/usr/bin/env python3
import copy
import json
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.state_validation import build_state_validation


class _CountingValidatorClient:
    calls = 0

    def ask(self, prompt: str, response_type: str = "json"):
        if "Validate whether this state is inferable from evidence" in prompt:
            _CountingValidatorClient.calls += 1
            return {
                "field_verdicts": [
                    {
                        "reason_analysis": "mock evidence",
                        "is_valid": True,
                    }
                ],
            }
        return {}


class TceStateValidationPipelineAcceptance(unittest.TestCase):
    def test_reuses_only_when_evidence_signature_is_unchanged(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_app_log_id": "log_0001",
                            }
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_app_log_id": "log_0001",
                            }
                        }
                    },
                },
                {
                    "checkpoint_id": "cp3",
                    "as_of": {"timestamp": "2025-03-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001", "log_0002"],
                                "last_app_log_id": "log_0002",
                            }
                        }
                    },
                },
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        self.assertEqual(_CountingValidatorClient.calls, 2)
        cp1, cp2, cp3 = result["checkpoints"]
        self.assertEqual(cp1["state_validation_summary"]["computed_count"], 1)
        self.assertEqual(cp1["state_validation_summary"]["reused_count"], 0)
        self.assertEqual(cp2["state_validation_summary"]["computed_count"], 0)
        self.assertEqual(cp2["state_validation_summary"]["reused_count"], 1)
        self.assertEqual(cp3["state_validation_summary"]["computed_count"], 1)
        self.assertEqual(cp3["state_validation_summary"]["reused_count"], 0)
        self.assertEqual(
            cp2["state_questionability"]["habits_state:morning_walk"]["validation_source"],
            "reused",
        )
        self.assertEqual(
            cp3["state_questionability"]["habits_state:morning_walk"]["validation_source"],
            "computed",
        )

    def test_resume_skips_saved_states_and_seeds_reuse_cache(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_app_log_id": "log_0001",
                            }
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_app_log_id": "log_0001",
                            }
                        }
                    },
                },
            ],
        }

        first_pass = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )
        self.assertEqual(_CountingValidatorClient.calls, 1)

        resumed_benchmark = copy.deepcopy(benchmark)
        resumed_benchmark["checkpoints"][0]["state_questionability"] = copy.deepcopy(
            first_pass["checkpoints"][0]["state_questionability"]
        )
        resumed_benchmark["checkpoints"][0]["validated_snapshot_state"] = copy.deepcopy(
            first_pass["checkpoints"][0]["validated_snapshot_state"]
        )
        resumed_benchmark["checkpoints"][0]["state_validation_summary"] = copy.deepcopy(
            first_pass["checkpoints"][0]["state_validation_summary"]
        )

        _CountingValidatorClient.calls = 0
        resumed = build_state_validation(
            benchmark=resumed_benchmark,
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        self.assertEqual(_CountingValidatorClient.calls, 0)
        self.assertEqual(
            resumed["checkpoints"][1]["state_questionability"]["habits_state:morning_walk"]["validation_source"],
            "reused",
        )
        self.assertEqual(resumed["checkpoints"][1]["state_validation_summary"]["reused_count"], 1)

    def test_save_callback_runs_every_n_new_states(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "expected_snapshot_state": {
                        "work_state": {"office_days": {"schedule_dates": ["Mon", "Tue"]}}
                    },
                    "state_observability": {
                        "work_state": {
                            "office_days": {"is_valid": True, "evidence_app_log_ids": ["log_0002"]}
                        }
                    },
                },
            ],
        }
        save_snapshots = []

        build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
            save_every_states=1,
            save_callback=lambda payload: save_snapshots.append(
                sum(
                    len(cp.get("state_questionability") or {})
                    for cp in payload.get("checkpoints", [])
                    if isinstance(cp, dict)
                )
            ),
        )

        self.assertEqual(save_snapshots, [1, 2])

    def test_partial_payload_is_stage2_compatible_before_all_checkpoints_finish(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": {"timing": {"start_time": "06:30"}}}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "expected_snapshot_state": {
                        "work_state": {"office_days": {"schedule_dates": ["Mon", "Tue"]}}
                    },
                    "state_observability": {
                        "work_state": {
                            "office_days": {"is_valid": True, "evidence_app_log_ids": ["log_0002"]}
                        }
                    },
                },
            ],
        }
        saved_payloads = []

        build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
            save_every_states=1,
            save_callback=lambda payload: saved_payloads.append(copy.deepcopy(payload)),
        )

        first_saved = saved_payloads[0]
        cp2 = first_saved["checkpoints"][1]
        self.assertIn("state_questionability", cp2)
        self.assertIn("validated_snapshot_state", cp2)
        self.assertIn("state_validation_summary", cp2)
        self.assertEqual(cp2["state_questionability"], {})
        self.assertEqual(cp2["validated_snapshot_state"], {})

    def test_scalar_state_uses_current_value_as_candidate_field(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "user_attributes_state": {"eldest_child": "Leo (15), high school sophomore"}
                    },
                    "state_observability": {
                        "user_attributes_state": {
                            "eldest_child": {"is_valid": True, "evidence_app_log_ids": ["log_0001"]}
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["user_attributes_state:eldest_child"]
        self.assertEqual(qmeta["askable_fields"], ["current_value"])
        self.assertEqual(qmeta["validated_field_paths"], ["current_value"])
        self.assertNotIn("l2_is_questionable", qmeta)
        self.assertEqual(
            result["checkpoints"][0]["validated_snapshot_state"]["user_attributes_state"]["eldest_child"],
            "Leo (15), high school sophomore",
        )

    def test_l2_accepts_keyed_field_verdicts_from_prompt_shape(self):
        class _KeyedValidatorClient:
            prompt = ""

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Validate whether this state is inferable from evidence" in prompt:
                    _KeyedValidatorClient.prompt = prompt
                    return {
                        "field_verdicts": {
                            "timing.start_time": {
                                "reason_analysis": "mock start evidence",
                                "is_valid": True,
                            },
                            "made_up.path": {
                                "reason_analysis": "should be ignored",
                                "is_valid": True,
                            },
                        }
                    }
                return {}

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {
                                    "start_time": "06:30",
                                    "end_time": "07:00",
                                }
                            }
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

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_KeyedValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:morning_walk"]
        self.assertIn('"timing.start_time"', _KeyedValidatorClient.prompt)
        self.assertIn('"timing.end_time"', _KeyedValidatorClient.prompt)
        self.assertIn("0=Monday", _KeyedValidatorClient.prompt)
        self.assertIn("6=Sunday", _KeyedValidatorClient.prompt)
        self.assertIn("days_of_month", _KeyedValidatorClient.prompt)
        self.assertIn("semantic alignment", _KeyedValidatorClient.prompt)
        self.assertIn("implicit behavioral support", _KeyedValidatorClient.prompt)
        self.assertIn("Do not require exact wording", _KeyedValidatorClient.prompt)
        self.assertIn("core user-specific claims", _KeyedValidatorClient.prompt)
        self.assertIn("preferences_state:community_involvement_type", _KeyedValidatorClient.prompt)
        self.assertIn("user_attributes_state:industry_software_skills", _KeyedValidatorClient.prompt)
        self.assertIn("advanced Microsoft Project skill", _KeyedValidatorClient.prompt)
        self.assertEqual(qmeta["validated_field_paths"], ["timing.start_time"])
        self.assertEqual(qmeta["dropped_field_paths"], ["timing.end_time"])
        self.assertEqual(
            result["checkpoints"][0]["validated_snapshot_state"]["habits_state"]["morning_walk"],
            {"timing": {"start_time": "06:30"}},
        )

    def test_stage1_pre_excludes_schedule_dates_before_questionability(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-06 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "daily_walk": {
                                "timing": {"start_time": "06:30"},
                                "schedule": {"frequency_type": "daily"},
                                "schedule_dates": [
                                    "2025-01-01",
                                    "2025-01-02",
                                    "2025-01-03",
                                    "2025-01-04",
                                    "2025-01-05",
                                    "2025-01-06",
                                ],
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "daily_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001", "log_0002"],
                                "last_app_log_id": "log_0002",
                            }
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={
                "log_0001": {"timestamp": "2025-01-01 08:00:00"},
                "log_0002": {"timestamp": "2025-01-04 08:00:00"},
            },
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:daily_walk"]
        self.assertTrue(qmeta["l1_is_questionable"])
        self.assertNotIn("l2_is_questionable", qmeta)
        self.assertNotIn("reason_codes", qmeta)
        self.assertFalse(any("schedule_dates" in path for path in qmeta["askable_fields"]))
        self.assertFalse(any("schedule_dates" in path for path in qmeta["validated_field_paths"]))
        self.assertEqual(_CountingValidatorClient.calls, 1)
        self.assertNotIn(
            "schedule_dates",
            json.dumps(result["checkpoints"][0]["validated_snapshot_state"]["habits_state"]["daily_walk"]),
        )

    def test_stage1_skips_l2_when_only_excluded_schedule_dates_remain(self):
        _CountingValidatorClient.calls = 0
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-06 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "daily_walk": {
                                "schedule": {
                                    "schedule_dates": [
                                        "2025-01-01",
                                        "2025-01-02",
                                    ]
                                }
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "daily_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                            }
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_CountingValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:daily_walk"]
        self.assertEqual(_CountingValidatorClient.calls, 0)
        self.assertEqual(qmeta["askable_fields"], [])
        self.assertEqual(qmeta["validated_field_paths"], [])
        self.assertEqual(qmeta["validated_state_value"], {})
        self.assertEqual(result["checkpoints"][0]["validated_snapshot_state"], {})

    def test_stage1_pre_excludes_preference_signals_from_validated_snapshot(self):
        class _PreferenceValidatorClient:
            calls = 0

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Validate whether this state is inferable from evidence" in prompt:
                    _PreferenceValidatorClient.calls += 1
                    return {
                        "field_verdicts": [
                            {
                                "reason_analysis": "mock evidence",
                                "is_valid": True,
                            }
                        ],
                    }
                return {}

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-06 08:00:00"},
                    "expected_snapshot_state": {
                        "preferences_state": {
                            "learning_modality": {
                                "statement": "prefers self-paced webinars",
                                "signals": ["downloaded a report", "joined a webinar"],
                            }
                        }
                    },
                    "state_observability": {
                        "preferences_state": {
                            "learning_modality": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                            }
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_PreferenceValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["preferences_state:learning_modality"]
        self.assertNotIn("l2_is_questionable", qmeta)
        self.assertEqual(qmeta["askable_fields"], ["statement"])
        self.assertEqual(qmeta["validated_field_paths"], ["statement"])
        self.assertEqual(qmeta["validated_state_value"], {"statement": "prefers self-paced webinars"})
        self.assertEqual(
            result["checkpoints"][0]["validated_snapshot_state"]["preferences_state"]["learning_modality"],
            {"statement": "prefers self-paced webinars"},
        )
        self.assertEqual(_PreferenceValidatorClient.calls, 1)

    def test_stage1_pre_excludes_priority_from_validated_snapshot(self):
        class _PriorityValidatorClient:
            calls = 0

            def ask(self, prompt: str, response_type: str = "json"):
                del response_type
                if "Validate whether this state is inferable from evidence" in prompt:
                    _PriorityValidatorClient.calls += 1
                    return {
                        "field_verdicts": [
                            {
                                "reason_analysis": "mock evidence",
                                "is_valid": True,
                            }
                        ],
                    }
                return {}

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-06 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {
                            "daily_walk": {
                                "timing": {"start_time": "06:30"},
                                "priority": "high",
                            }
                        }
                    },
                    "state_observability": {
                        "habits_state": {
                            "daily_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                            }
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_PriorityValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:daily_walk"]
        self.assertNotIn("l2_is_questionable", qmeta)
        self.assertFalse(any("priority" in path for path in qmeta["askable_fields"]))
        self.assertFalse(any("priority" in path for path in qmeta["validated_field_paths"]))
        self.assertEqual(
            result["checkpoints"][0]["validated_snapshot_state"]["habits_state"]["daily_walk"],
            {"timing": {"start_time": "06:30"}},
        )
        self.assertEqual(_PriorityValidatorClient.calls, 1)

    def test_stage1_adds_change_reason_validation_when_last_change_reason_exists(self):
        class _ChangeReasonValidatorClient:
            calls = 0

            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate whether this state is inferable from evidence" in prompt:
                    return {
                        "field_verdicts": [
                            {
                                "reason_analysis": "mock evidence",
                                "is_valid": True,
                            }
                        ],
                    }
                if "Validate whether the provided change reason is supported by evidence" in prompt:
                    _ChangeReasonValidatorClient.calls += 1
                    return {
                        "change_reason_verdict": {
                            "reason_analysis": "The evidence supports the later schedule shift.",
                            "is_valid": True,
                        },
                    }
                return {}

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": "07:00"}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_change_reason": "Routine shifted later after a schedule change.",
                            }
                        }
                    },
                }
            ],
        }

        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_ChangeReasonValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:morning_walk"]
        self.assertEqual(_ChangeReasonValidatorClient.calls, 1)
        self.assertEqual(
            qmeta["change_reason_validation"],
            {
                "reason_analysis": "The evidence supports the later schedule shift.",
                "is_valid": True,
            },
        )
        self.assertEqual(
            qmeta["validation_identity"]["change_reason_prompt_version"],
            "change_reason_validate_prompt_v2_verdict_only",
        )

    def test_resume_backfills_missing_change_reason_validation(self):
        class _BackfillChangeReasonValidatorClient:
            state_calls = 0
            change_reason_calls = 0

            def ask(self, prompt: str, response_type: str = "json"):
                if "Validate whether this state is inferable from evidence" in prompt:
                    _BackfillChangeReasonValidatorClient.state_calls += 1
                    return {
                        "field_verdicts": [
                            {
                                "reason_analysis": "mock evidence",
                                "is_valid": True,
                            }
                        ],
                    }
                if "Validate whether the provided change reason is supported by evidence" in prompt:
                    _BackfillChangeReasonValidatorClient.change_reason_calls += 1
                    return {
                        "change_reason_verdict": {
                            "reason_analysis": "The evidence supports the change reason.",
                            "is_valid": True,
                        },
                    }
                return {}

        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "expected_snapshot_state": {
                        "habits_state": {"morning_walk": "07:00"}
                    },
                    "state_observability": {
                        "habits_state": {
                            "morning_walk": {
                                "is_valid": True,
                                "evidence_app_log_ids": ["log_0001"],
                                "last_change_reason": "Routine shifted later after a schedule change.",
                            }
                        }
                    },
                    "state_questionability": {
                        "habits_state:morning_walk": {
                            "l1_is_questionable": True,
                            "askable_fields": ["current_value"],
                            "validated_field_paths": ["current_value"],
                            "dropped_field_paths": [],
                            "validated_state_value": "07:00",
                            "field_verdicts": [
                                {
                                    "field_name": "current_value",
                                    "reason_analysis": "mock evidence",
                                    "is_valid": True,
                                }
                            ],
                            "validator_version": "qv3_l1_l2_preexclude_derived",
                            "validation_source": "computed",
                            "validation_identity": {
                                "state_key": "habits_state:morning_walk",
                                "validated_state_value_signature": "\"07:00\"",
                                "evidence_signature": "[seeded]",
                                "validator_version": "qv3_l1_l2_preexclude_derived",
                                "prompt_version": "state_validate_prompt_v8_field_keyed_implicit_semantic_alignment",
                            },
                        }
                    },
                    "validated_snapshot_state": {
                        "habits_state": {"morning_walk": "07:00"}
                    },
                    "state_validation_summary": {
                        "pre_validate_count": 1,
                        "after_l1_count": 1,
                        "after_l2_count": 1,
                        "after_l1_l2_count": 1,
                        "reused_count": 0,
                        "computed_count": 1,
                    },
                }
            ],
        }

        _BackfillChangeReasonValidatorClient.state_calls = 0
        _BackfillChangeReasonValidatorClient.change_reason_calls = 0
        result = build_state_validation(
            benchmark=copy.deepcopy(benchmark),
            validator_client=_BackfillChangeReasonValidatorClient(),
            app_logs_by_id={},
            max_checkpoints=None,
            l2_evidence_top_k=0,
        )

        qmeta = result["checkpoints"][0]["state_questionability"]["habits_state:morning_walk"]
        self.assertEqual(_BackfillChangeReasonValidatorClient.state_calls, 0)
        self.assertEqual(_BackfillChangeReasonValidatorClient.change_reason_calls, 1)
        self.assertEqual(
            qmeta["change_reason_validation"],
            {
                "reason_analysis": "The evidence supports the change reason.",
                "is_valid": True,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
