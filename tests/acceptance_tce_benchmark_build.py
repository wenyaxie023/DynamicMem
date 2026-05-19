#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_construction.build_tce_benchmark import (
    build_chain_change_reasons,
    build_chain_state_validity,
    build_checkpoints,
)


class TceBenchmarkBuildAcceptance(unittest.TestCase):
    def test_build_checkpoints_preserves_last_change_reason_in_state_observability(self):
        all_events_chains = [
            {
                "chain_id": "chain_001",
                "events": [
                    {
                        "time_specification": {},
                        "evidence_for_states": [
                            {
                                "state_name": "commute_mode",
                                "evidenced_fields": ["delta.to", "change_reason"],
                            }
                        ],
                    }
                ],
                "state_refs": [
                    {
                        "state_category": "profile_state",
                        "state_name": "commute_mode",
                        "resolved_items": [
                            {
                                "name": "commute_mode",
                                "change_type": "modify",
                                "delta": {"from": "car", "to": "train"},
                                "change_reason": "Switched to train after downtown parking fees increased.",
                                "required_observable_fields": ["delta.to", "change_reason"],
                            }
                        ],
                    }
                ],
            }
        ]
        app_logs_final = {
            "user_id": "001_user_001",
            "app_logs": [
                {
                    "app_log_id": "log_0001",
                    "timestamp": "2025-01-01 08:00:00",
                    "metadata": {"chain_id": "chain_001"},
                    "golden_evidence": [
                        {
                            "state_category": "profile_state",
                            "state_name": "commute_mode",
                            "change_type": "modify",
                            "evidenced_fields": ["delta.to", "change_reason"],
                            "state_value": "train",
                        }
                    ],
                }
            ],
        }

        chain_state_validity = build_chain_state_validity(all_events_chains)
        chain_change_reasons = build_chain_change_reasons(all_events_chains)
        benchmark = build_checkpoints(app_logs_final, chain_state_validity, chain_change_reasons)

        checkpoint = benchmark["checkpoints"][0]
        observability = checkpoint["state_observability"]["profile_state"]["commute_mode"]
        self.assertEqual(
            observability["last_change_reason"],
            "Switched to train after downtown parking fees increased.",
        )

    def test_build_checkpoints_keeps_last_change_reason_after_later_unchanged_evidence(self):
        all_events_chains = {
            "Work & Education": [
                {
                    "window_id": "w1",
                    "event_chains": [
                        {
                            "chain_id": "chain_change",
                            "events": [
                                {
                                    "time_specification": {},
                                    "evidence_for_states": [
                                        {
                                            "state_name": "commute_mode",
                                            "evidenced_fields": ["delta.to", "change_reason"],
                                        }
                                    ],
                                }
                            ],
                            "state_refs": [
                                {
                                    "state_category": "profile_state",
                                    "state_name": "commute_mode",
                                    "resolved_items": [
                                        {
                                            "name": "commute_mode",
                                            "change_type": "modify",
                                            "delta": {"from": "car", "to": "train"},
                                            "change_reason": "Switched to train after downtown parking fees increased.",
                                            "required_observable_fields": ["delta.to", "change_reason"],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "chain_id": "chain_followup",
                            "events": [
                                {
                                    "time_specification": {},
                                    "evidence_for_states": [
                                        {
                                            "state_name": "commute_mode",
                                            "evidenced_fields": ["current_value"],
                                        }
                                    ],
                                }
                            ],
                            "state_refs": [
                                {
                                    "state_category": "profile_state",
                                    "state_name": "commute_mode",
                                    "resolved_items": [
                                        {
                                            "name": "commute_mode",
                                            "change_type": "unchanged",
                                            "current_value": "train",
                                            "required_observable_fields": ["current_value"],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ]
        }
        app_logs_final = {
            "user_id": "001_user_001",
            "app_logs": [
                {
                    "app_log_id": "log_0001",
                    "timestamp": "2025-01-01 08:00:00",
                    "metadata": {"chain_id": "chain_change"},
                    "golden_evidence": [
                        {
                            "state_category": "profile_state",
                            "state_name": "commute_mode",
                            "change_type": "modify",
                            "evidenced_fields": ["delta.to", "change_reason"],
                            "state_value": "train",
                        }
                    ],
                },
                {
                    "app_log_id": "log_0002",
                    "timestamp": "2025-01-02 08:00:00",
                    "metadata": {"chain_id": "chain_followup"},
                    "golden_evidence": [
                        {
                            "state_category": "profile_state",
                            "state_name": "commute_mode",
                            "change_type": "unchanged",
                            "evidenced_fields": ["current_value"],
                            "state_value": "train",
                        }
                    ],
                },
            ],
        }

        chain_state_validity = build_chain_state_validity(all_events_chains)
        chain_change_reasons = build_chain_change_reasons(all_events_chains)
        benchmark = build_checkpoints(app_logs_final, chain_state_validity, chain_change_reasons)

        checkpoint = benchmark["checkpoints"][0]
        observability = checkpoint["state_observability"]["profile_state"]["commute_mode"]
        self.assertEqual(
            observability["last_change_reason"],
            "Switched to train after downtown parking fees increased.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
