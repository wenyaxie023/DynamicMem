#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.build_tce_analysis_pack import build_analysis_pack
from tce_contracts import CURRENT_TASK_CONTRACT_VERSION, RESEARCH_FRAME_VERSION_V2


class TceAnalysisPackAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark_path = Path(
            "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/"
            "tce_benchmark_vnext_20260319_formal_crv_task_packs_v14.json"
        )
        self.prediction_path = Path(
            "generation/rag/results/001_user_001/prediction/"
            "tce_results_vnext_20260319_formal_topk20_gpt5mini_v14_taskabc.json"
        )
        self.eval_path = Path(
            "generation/rag/results/001_user_001/eval/"
            "tce_eval_vnext_20260319_formal_topk20_gpt5mini_v14_taskabc_slotjudge_azure.json"
        )
        self.benchmark = json.loads(self.benchmark_path.read_text(encoding="utf-8"))
        self.prediction = json.loads(self.prediction_path.read_text(encoding="utf-8"))
        self.eval_payload = json.loads(self.eval_path.read_text(encoding="utf-8"))

    def test_builds_expected_outputs_from_v14_artifacts(self):
        expected_files = {
            "checkpoint_summary.csv",
            "rq1_task_scores_by_checkpoint.csv",
            "rq2_taskb_attribution_by_checkpoint.csv",
            "rq3_know_apply_gap_by_checkpoint.csv",
            "correlation_summary.csv",
            "task_a_units.csv",
            "task_b_units.csv",
            "task_c_units.csv",
            "rq1_task_scores_vs_checkpoint.png",
            "rq2_taskb_attribution_vs_checkpoint.png",
            "rq3_know_apply_gap_vs_checkpoint.png",
            "task_a_state_heatmap.png",
            "task_b_state_predict_heatmap.png",
            "task_b_change_reason_heatmap.png",
            "task_c_state_heatmap.png",
            "rq3_gap_heatmap.png",
            "task_a_state_lines_vs_checkpoint.png",
            "task_b_state_predict_lines_vs_checkpoint.png",
            "task_b_change_reason_lines_vs_checkpoint.png",
            "task_c_state_lines_vs_checkpoint.png",
            "task_a_evidence_vs_score_scatter.png",
            "task_b_state_evidence_vs_score_scatter.png",
            "task_b_reason_evidence_vs_score_scatter.png",
            "task_c_evidence_vs_score_scatter.png",
            "task_a_top_variable_states.csv",
            "task_b_state_predict_top_variable_states.csv",
            "task_b_change_reason_top_variable_states.csv",
            "task_c_top_variable_states.csv",
            "rq3_top_positive_gap_states.csv",
            "rq3_top_negative_gap_states.csv",
            "milestone1_analysis_summary.md",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "analysis_pack"
            result = build_analysis_pack(
                benchmark_payload=self.benchmark,
                prediction_payload=self.prediction,
                eval_payload=self.eval_payload,
                output_dir=outdir,
                artifact_paths={
                    "benchmark": str(self.benchmark_path),
                    "prediction": str(self.prediction_path),
                    "eval": str(self.eval_path),
                    "output_dir": str(outdir),
                },
            )

            self.assertEqual(result["alignment_report"]["aligned_by_timestamp"], 5)
            self.assertEqual({p.name for p in outdir.iterdir()}, expected_files)

            with (outdir / "checkpoint_summary.csv").open() as f:
                checkpoint_rows = list(csv.DictReader(f))
            self.assertEqual(len(checkpoint_rows), 5)
            self.assertTrue(all(row["checkpoint_id"] for row in checkpoint_rows))
            self.assertTrue(any(row["rq3_overlap_state_count"] for row in checkpoint_rows))

            eval_by_checkpoint = {
                cp["checkpoint_id"]: cp
                for cp in self.eval_payload["checkpoints"]
            }
            for row in checkpoint_rows:
                cp = eval_by_checkpoint[row["checkpoint_id"]]
                self.assertAlmostEqual(
                    float(row["task_a_point_score"]),
                    cp["snapshot_point_score_mean_on_expected"],
                    places=6,
                )
                self.assertAlmostEqual(
                    float(row["task_c_answer_score"]),
                    cp["rq3_apply_answer_point_score_mean"],
                    places=6,
                )
                expected_b_state = cp.get("change_state_predict_point_score_mean_on_changed")
                if expected_b_state is None:
                    self.assertEqual(row["task_b_state_score"], "")
                else:
                    self.assertAlmostEqual(float(row["task_b_state_score"]), expected_b_state, places=6)
                expected_b_reason = cp.get("change_reason_point_score_mean_on_changed")
                if expected_b_reason is None:
                    self.assertEqual(row["task_b_reason_score"], "")
                else:
                    self.assertAlmostEqual(float(row["task_b_reason_score"]), expected_b_reason, places=6)
            self.assertEqual(checkpoint_rows[0]["task_b_state_score"], "")
            self.assertEqual(checkpoint_rows[0]["task_b_reason_score"], "")

            with (outdir / "correlation_summary.csv").open() as f:
                corr_rows = list(csv.DictReader(f))
            self.assertEqual(
                {(row["task"], row["score_variant"]) for row in corr_rows},
                {
                    ("task_a", "score"),
                    ("task_b", "state_predict"),
                    ("task_b", "change_reason"),
                    ("task_c", "score"),
                },
            )
            self.assertTrue(all(int(row["unit_count"]) > 0 for row in corr_rows))

            with (outdir / "task_a_units.csv").open() as f:
                task_a_units = list(csv.DictReader(f))
            with (outdir / "task_b_units.csv").open() as f:
                task_b_units = list(csv.DictReader(f))
            with (outdir / "task_c_units.csv").open() as f:
                task_c_units = list(csv.DictReader(f))
            self.assertTrue(task_a_units)
            self.assertTrue(task_b_units)
            self.assertTrue(task_c_units)

            expected_task_c_state_means = defaultdict(list)
            for row in task_c_units:
                expected_task_c_state_means[(row["checkpoint_id"], row["state_key"])].append(float(row["score"]))
            actual_task_c_state_means = {
                (row["checkpoint_id"], row["state_key"]): row["state_score"]
                for row in result["task_c_state_rows"]
            }
            self.assertEqual(set(actual_task_c_state_means.keys()), set(expected_task_c_state_means.keys()))
            for key, values in expected_task_c_state_means.items():
                self.assertAlmostEqual(
                    float(actual_task_c_state_means[key]),
                    sum(values) / len(values),
                    places=6,
                )

            with (outdir / "task_c_top_variable_states.csv").open() as f:
                task_c_rank_rows = list(csv.DictReader(f))
            self.assertTrue(task_c_rank_rows)
            self.assertLessEqual(len(task_c_rank_rows), 20)

            with (outdir / "rq3_know_apply_gap_by_checkpoint.csv").open() as f:
                rq3_rows = list(csv.DictReader(f))
            self.assertEqual(len(rq3_rows), 5)
            self.assertTrue(any(int(row["overlap_state_count"] or 0) > 0 for row in rq3_rows))

            with (outdir / "rq3_top_positive_gap_states.csv").open() as f:
                positive_gap_rows = list(csv.DictReader(f))
            with (outdir / "rq3_top_negative_gap_states.csv").open() as f:
                negative_gap_rows = list(csv.DictReader(f))
            self.assertTrue(positive_gap_rows)
            self.assertTrue(negative_gap_rows)

            self.assertTrue(result["rq3_gap_state_rows"])
            task_a_pairs = {(row["checkpoint_id"], row["state_key"]) for row in task_a_units}
            task_c_pairs = {(row["checkpoint_id"], row["state_key"]) for row in result["task_c_state_rows"]}
            self.assertTrue(
                all(
                    (row["checkpoint_id"], row["state_key"]) in task_a_pairs
                    and (row["checkpoint_id"], row["state_key"]) in task_c_pairs
                    for row in result["rq3_gap_state_rows"]
                )
            )

    def test_v2_analysis_pack_omits_task_b_and_builds_rq2_from_task_a_slices(self):
        benchmark = {
            "user_id": "001_user_001",
            "task_contract_version": CURRENT_TASK_CONTRACT_VERSION,
            "research_frame_version": RESEARCH_FRAME_VERSION_V2,
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00"},
                    "sampling": {"params": {"actual_tokens_at_cutoff": 100, "total_tokens": 200}},
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "06:30"},
                                "schedule_dates": ["2025-01-01"],
                                "priority": "high",
                            }
                        },
                        "preferences_state": {"favorite_coffee": "latte"},
                    },
                    "expected_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "06:30"},
                                "schedule_dates": ["2025-01-01"],
                                "priority": "high",
                            }
                        },
                        "preferences_state": {"favorite_coffee": "latte"},
                    },
                    "state_observability": {
                        "habits_state": {"morning_walk": {"evidence_app_log_ids": ["log_0001"]}},
                        "preferences_state": {"favorite_coffee": {"evidence_app_log_ids": ["log_0002"]}},
                    },
                    "rq3_apply_service_qa": {
                        "keys": {
                            "preferences_state:favorite_coffee": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "question": "Which coffee should the assistant remember?",
                                        "reference_answer": "Remember latte.",
                                        "gold_memory_evidence_app_log_ids": ["log_0002"],
                                    }
                                ]
                            }
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-02-01 08:00:00"},
                    "sampling": {"params": {"actual_tokens_at_cutoff": 150, "total_tokens": 200}},
                    "validated_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "07:00"},
                                "schedule_dates": ["2025-02-01"],
                                "priority": "low",
                            }
                        },
                        "preferences_state": {"favorite_coffee": "latte"},
                    },
                    "expected_snapshot_state": {
                        "habits_state": {
                            "morning_walk": {
                                "timing": {"start_time": "07:00"},
                                "schedule_dates": ["2025-02-01"],
                                "priority": "low",
                            }
                        },
                        "preferences_state": {"favorite_coffee": "latte"},
                    },
                    "state_observability": {
                        "habits_state": {"morning_walk": {"evidence_app_log_ids": ["log_0003"]}},
                        "preferences_state": {"favorite_coffee": {"evidence_app_log_ids": ["log_0002"]}},
                    },
                    "rq3_apply_service_qa": {
                        "keys": {
                            "preferences_state:favorite_coffee": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "question": "Which coffee should the assistant remember?",
                                        "reference_answer": "Remember latte.",
                                        "gold_memory_evidence_app_log_ids": ["log_0002"],
                                    }
                                ]
                            }
                        }
                    },
                },
            ],
        }
        prediction = {
            "predictions": [
                {
                    "checkpoint_id": "cp1",
                    "snapshot_state": {
                        "habits_state:morning_walk": {"timing": {"start_time": "06:30"}},
                        "preferences_state:favorite_coffee": "latte",
                    },
                    "evidence": {
                        "habits_state:morning_walk": ["log_0001"],
                        "preferences_state:favorite_coffee": ["log_0002"],
                    },
                    "rq3_apply_answers": {
                        "preferences_state:favorite_coffee": {
                            "items": [{"qa_id": "q1", "answer": "latte", "evidence": ["log_0002"]}]
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "snapshot_state": {
                        "habits_state:morning_walk": {"timing": {"start_time": "06:45"}},
                        "preferences_state:favorite_coffee": "latte",
                    },
                    "evidence": {
                        "habits_state:morning_walk": ["log_0003"],
                        "preferences_state:favorite_coffee": ["log_0002"],
                    },
                    "rq3_apply_answers": {
                        "preferences_state:favorite_coffee": {
                            "items": [{"qa_id": "q1", "answer": "latte", "evidence": ["log_0002"]}]
                        }
                    },
                },
            ]
        }
        eval_payload = {
            "task_contract_version": CURRENT_TASK_CONTRACT_VERSION,
            "research_frame_version": RESEARCH_FRAME_VERSION_V2,
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "snapshot_point_score_mean_on_expected": 1.0,
                    "snapshot_evidence_f1_mean_on_expected": 1.0,
                    "snapshot_slot_eval_by_key": {
                        "habits_state:morning_walk": {"score_0_1": 1.0, "slot_count": 1},
                        "preferences_state:favorite_coffee": {"score_0_1": 1.0, "slot_count": 1},
                    },
                    "rq3_apply_answer_point_score_mean": 1.0,
                    "rq3_apply_evidence_f1": 1.0,
                    "rq3_apply_slot_eval_by_item": {
                        "preferences_state:favorite_coffee::q1": {
                            "state_key": "preferences_state:favorite_coffee",
                            "qa_id": "q1",
                            "score_0_1": 1.0,
                            "slot_count": 1,
                        }
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "snapshot_point_score_mean_on_expected": 0.75,
                    "snapshot_evidence_f1_mean_on_expected": 1.0,
                    "snapshot_slot_eval_by_key": {
                        "habits_state:morning_walk": {"score_0_1": 0.5, "slot_count": 1},
                        "preferences_state:favorite_coffee": {"score_0_1": 1.0, "slot_count": 1},
                    },
                    "rq3_apply_answer_point_score_mean": 1.0,
                    "rq3_apply_evidence_f1": 1.0,
                    "rq3_apply_slot_eval_by_item": {
                        "preferences_state:favorite_coffee::q1": {
                            "state_key": "preferences_state:favorite_coffee",
                            "qa_id": "q1",
                            "score_0_1": 1.0,
                            "slot_count": 1,
                        }
                    },
                },
            ],
        }

        expected_files = {
            "checkpoint_summary.csv",
            "rq1_task_scores_by_checkpoint.csv",
            "rq2_taska_changed_vs_unchanged_by_checkpoint.csv",
            "rq2_transition_units.csv",
            "rq3_know_apply_gap_by_checkpoint.csv",
            "correlation_summary.csv",
            "task_a_units.csv",
            "task_c_units.csv",
            "rq1_task_scores_vs_checkpoint.png",
            "rq2_taska_changed_vs_unchanged_vs_checkpoint.png",
            "rq3_know_apply_gap_vs_checkpoint.png",
            "task_a_state_heatmap.png",
            "task_c_state_heatmap.png",
            "rq3_gap_heatmap.png",
            "task_a_state_lines_vs_checkpoint.png",
            "task_c_state_lines_vs_checkpoint.png",
            "task_a_evidence_vs_score_scatter.png",
            "task_c_evidence_vs_score_scatter.png",
            "task_a_top_variable_states.csv",
            "task_c_top_variable_states.csv",
            "rq3_top_positive_gap_states.csv",
            "rq3_top_negative_gap_states.csv",
            "milestone1_analysis_summary.md",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "analysis_pack_v2"
            result = build_analysis_pack(
                benchmark_payload=benchmark,
                prediction_payload=prediction,
                eval_payload=eval_payload,
                output_dir=outdir,
                artifact_paths={
                    "benchmark": "benchmark.json",
                    "prediction": "prediction.json",
                    "eval": "eval.json",
                    "output_dir": str(outdir),
                },
            )

            self.assertEqual({p.name for p in outdir.iterdir()}, expected_files)
            self.assertEqual(result["task_b_units"], [])
            self.assertEqual(len(result["rq2_transition_units"]), 2)

            with (outdir / "checkpoint_summary.csv").open() as f:
                checkpoint_rows = list(csv.DictReader(f))
            self.assertEqual(checkpoint_rows[0]["task_b_state_score"], "")
            self.assertEqual(checkpoint_rows[1]["task_b_state_score"], "")
            self.assertEqual(checkpoint_rows[1]["rq2_changed_state_count"], "1")
            self.assertEqual(checkpoint_rows[1]["rq2_unchanged_state_count"], "1")
            self.assertAlmostEqual(float(checkpoint_rows[1]["rq2_changed_task_a_score"]), 0.5, places=6)
            self.assertAlmostEqual(float(checkpoint_rows[1]["rq2_unchanged_task_a_score"]), 1.0, places=6)
            self.assertAlmostEqual(float(checkpoint_rows[1]["rq2_update_gap"]), -0.5, places=6)

            with (outdir / "rq2_transition_units.csv").open() as f:
                transition_rows = list(csv.DictReader(f))
            status_by_key = {row["state_key"]: row["change_status"] for row in transition_rows}
            self.assertEqual(status_by_key["habits_state:morning_walk"], "changed")
            self.assertEqual(status_by_key["preferences_state:favorite_coffee"], "unchanged")

            summary_text = (outdir / "milestone1_analysis_summary.md").read_text(encoding="utf-8")
            self.assertIn("`RQ2` is not a standalone Task B", summary_text)
            self.assertIn("omitted Task B CSV/PNG artifacts are intentional", summary_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
