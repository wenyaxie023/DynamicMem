#!/usr/bin/env python3
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.scoring_points import (
    build_value_scoring_points,
    score_points,
    validate_apply_answer_rubric_points,
)


class TceScoringPointsAcceptance(unittest.TestCase):
    def test_apply_rubric_validation_accepts_over_specific_fail_reason(self):
        class _Validator:
            def ask(self, prompt: str, response_type: str = "json"):
                return {
                    "points": [
                        {
                            "point_id": "aqp_home_media_q1_p1",
                            "pass": False,
                            "analysis": "The point depends on an unnecessary exact device model and is too brittle.",
                            "fail_reasons": ["over_specific"],
                        }
                    ],
                    "set_pass": False,
                    "set_analysis": "The set-level rubric is acceptable, but the single point is too over-specific.",
                    "set_failures": [],
                }

        validation = validate_apply_answer_rubric_points(
            state_key="user_attributes_state:home_media_server",
            state_value="Synology DS923+ NAS for local 4K movie collection",
            service_category="playback policy",
            question="How should the assistant prioritize playback sources?",
            reference_answer="Select the local NAS as the source for the 4K digital version.",
            points=[
                {
                    "point_id": "aqp_home_media_q1_p1",
                    "point_type": "micro",
                    "polarity": "positive",
                    "point_text": "The answer specifies that the Synology DS923+ NAS should be selected as the source.",
                    "reference_value": "Synology DS923+ NAS",
                }
            ],
            validator_client=_Validator(),
        )

        self.assertFalse(validation["is_valid"])
        self.assertIn("over_specific", validation["failed_rules"])
        self.assertIn("set-level rubric", validation["set_analysis"])

    def test_schedule_date_mismatch_gets_zero_score(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "list_item",
                "target_path": "schedule_dates",
                "reference_value": "2024-10-27",
                "point_text": "schedule_dates should include this item",
            }
        ]
        predicted = {"schedule_dates": ["2024-07-14", None, None]}

        score, judgments = score_points(points, predicted)

        self.assertEqual(score, 0.0)
        self.assertEqual(judgments[0]["score_0_1"], 0.0)
        self.assertIn("similarity=0.00", judgments[0]["reason"])
        self.assertIn("predicted=2024-07-14", judgments[0]["reason"])

    def test_schedule_date_ordered_matching_does_not_reuse_first_item(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "list_item",
                "target_path": "schedule_dates",
                "reference_value": "2024-07-14",
                "point_text": "schedule_dates should include this item",
            },
            {
                "point_id": "p2",
                "point_type": "list_item",
                "target_path": "schedule_dates",
                "reference_value": "2024-10-27",
                "point_text": "schedule_dates should include this item",
            },
        ]
        predicted = {"schedule_dates": ["2024-07-14", None, None]}

        score, judgments = score_points(points, predicted)

        self.assertEqual(score, 0.5)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)
        self.assertIn("predicted=2024-07-14", judgments[0]["reason"])
        self.assertEqual(judgments[1]["score_0_1"], 0.0)
        self.assertTrue(judgments[1]["reason"].endswith("predicted="))

    def test_schedule_weekday_numeric_and_name_match(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "list_item",
                "target_path": "schedule.days_of_week",
                "reference_value": 6,
                "point_text": "schedule.days_of_week should include this item",
            }
        ]
        predicted = {"schedule": {"days_of_week": ["Sunday"]}}

        score, judgments = score_points(points, predicted)

        self.assertEqual(score, 1.0)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)
        self.assertIn("similarity=1.00", judgments[0]["reason"])
        self.assertIn("predicted=Sunday", judgments[0]["reason"])

    def test_time_field_normalizes_seconds(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "field",
                "target_path": "timing.start_time",
                "reference_value": "09:30",
                "point_text": "timing.start_time should match the validated value",
            }
        ]
        predicted = {"timing": {"start_time": "09:30:00"}}

        score, judgments = score_points(points, predicted)

        self.assertEqual(score, 1.0)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)
        self.assertIn("similarity=1.00", judgments[0]["reason"])

    def test_generic_text_uses_binary_hit_after_threshold(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "field",
                "target_path": "current_value",
                "reference_value": "Amazon Prime Video",
                "point_text": "subscription name should match",
            }
        ]

        score, judgments = score_points(points, {"current_value": "Prime Video"})

        self.assertEqual(score, 1.0)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)

    def test_scalar_root_matches_current_value_alias(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "field",
                "target_path": "current_value",
                "reference_value": "State Farm Umbrella Policy ($1M liability coverage for personal asset protection)",
                "point_text": "current_value should match the validated value",
            }
        ]

        score, judgments = score_points(
            points,
            "State Farm Umbrella Policy ($1M liability coverage for personal asset protection)",
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)

    def test_root_level_descriptive_string_uses_micro_points(self):
        points = build_value_scoring_points(
            state_key="user_attributes_state:insurance_portfolio",
            value="State Farm Umbrella Policy ($1M liability coverage for personal asset protection)",
            generator_client=None,
            validator_client=None,
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["point_type"], "micro")
        self.assertIn("State Farm Umbrella Policy", points[0]["point_text"])

    def test_list_is_flattened_into_indexed_field_points(self):
        points = build_value_scoring_points(
            state_key="habits_state:industry_news_review",
            value={"schedule_dates": ["2024-07-14", "2024-10-27"]},
            generator_client=None,
            validator_client=None,
        )

        self.assertEqual(
            [(point["point_type"], point["target_path"], point["reference_value"]) for point in points],
            [
                ("field", "schedule_dates.0", "2024-07-14"),
                ("field", "schedule_dates.1", "2024-10-27"),
            ],
        )

    def test_complex_list_item_becomes_path_bound_micro_points(self):
        points = build_value_scoring_points(
            state_key="preferences_state:learning_modality",
            value={
                "signals": [
                    "Declined a large conference in favor of webinar-based continuing education"
                ]
            },
            generator_client=None,
            validator_client=None,
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["point_type"], "micro")
        self.assertEqual(points[0]["target_path"], "signals.0")
        self.assertIn("webinar-based continuing education", points[0]["point_text"])

    def test_micro_points_score_only_against_target_path_and_average(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "micro",
                "target_path": "statement",
                "point_text": "prefers webinars",
            },
            {
                "point_id": "p2",
                "point_type": "micro",
                "target_path": "statement",
                "point_text": "avoids large conferences",
            },
            {
                "point_id": "p3",
                "point_type": "micro",
                "target_path": "statement",
                "point_text": "likes self-paced reading",
            },
        ]
        predicted = {
            "statement": "prefers webinars",
            "notes": "avoids large conferences and likes self-paced reading",
        }

        score, judgments = score_points(points, predicted)

        self.assertAlmostEqual(score, 1.0 / 3.0)
        self.assertEqual([judgment["score_0_1"] for judgment in judgments], [1.0, 0.0, 0.0])

    def test_indexed_field_path_reads_from_list_position(self):
        points = [
            {
                "point_id": "p1",
                "point_type": "field",
                "target_path": "schedule_dates.1",
                "reference_value": "2024-10-27",
                "point_text": "schedule_dates.1 should match the validated value",
            }
        ]

        score, judgments = score_points(points, {"schedule_dates": ["2024-07-14", "2024-10-27"]})

        self.assertEqual(score, 1.0)
        self.assertEqual(judgments[0]["score_0_1"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
