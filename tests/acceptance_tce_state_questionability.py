#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.questionability import evaluate_state_questionability


class TceStateQuestionabilityAcceptance(unittest.TestCase):
    def test_filters_empty_value(self):
        out = evaluate_state_questionability(
            state_key="profile_state:note",
            state_value={"note": ""},
        )
        self.assertFalse(out["is_questionable"])
        self.assertIn("empty_value", out["reason_codes"])
        self.assertNotIn("llm_reason", out)

    def test_allows_non_action_hint_state_if_not_empty(self):
        out = evaluate_state_questionability(
            state_key="preferences_state:fitness_philosophy",
            state_value="progress over perfection",
        )
        self.assertTrue(out["is_questionable"])
        self.assertNotIn("no_actionable_fields", out["reason_codes"])
        self.assertEqual(out["askable_fields"], ["current_value"])
        self.assertNotIn("llm_reason", out)

    def test_accepts_actionable_schedule_like_value(self):
        out = evaluate_state_questionability(
            state_key="habits_state:morning_walk",
            state_value={"timing": {"start_time": "06:30"}, "frequency": "weekly"},
        )
        self.assertTrue(out["is_questionable"])
        self.assertGreater(len(out["askable_fields"]), 0)
        self.assertNotIn("llm_reason", out)

    def test_does_not_filter_large_value_by_size(self):
        out = evaluate_state_questionability(
            state_key="profile_state:long_note",
            state_value={"note": "x" * 6000},
        )
        self.assertTrue(out["is_questionable"])
        self.assertNotIn("value_too_large", out["reason_codes"])

    def test_filters_sparse_schedule_date_evidence_for_habit(self):
        out = evaluate_state_questionability(
            state_key="habits_state:daily_walk",
            state_value={
                "schedule": {"frequency_type": "daily"},
                "schedule_dates": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                ],
            },
            state_observability={"evidence_app_log_ids": ["log_1", "log_2"]},
            app_logs_by_id={
                "log_1": {"timestamp": "2025-01-01 08:00:00"},
                "log_2": {"timestamp": "2025-01-04 08:00:00"},
            },
        )
        self.assertFalse(out["is_questionable"])
        self.assertIn("schedule_dates_evidence_undercoverage", out["reason_codes"])

    def test_accepts_sufficient_schedule_date_evidence_for_habit(self):
        out = evaluate_state_questionability(
            state_key="habits_state:daily_walk",
            state_value={
                "schedule": {"frequency_type": "daily"},
                "schedule_dates": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
            },
            state_observability={"evidence_app_log_ids": ["log_1", "log_2", "log_3", "log_4", "log_5"]},
            app_logs_by_id={
                "log_1": {"timestamp": "2025-01-01 08:00:00"},
                "log_2": {"timestamp": "2025-01-02 08:00:00"},
                "log_3": {"timestamp": "2025-01-03 08:00:00"},
                "log_4": {"timestamp": "2025-01-04 08:00:00"},
                "log_5": {"timestamp": "2025-01-05 08:00:00"},
            },
        )
        self.assertTrue(out["is_questionable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
