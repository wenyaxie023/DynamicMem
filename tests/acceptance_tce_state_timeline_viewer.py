#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.state_timeline_viewer import (
    _build_task_a_input_map,
    build_viewer_payload,
    write_viewer_payload,
)


class TceStateTimelineViewerAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        base = Path("data_construction/generated_outputs/gemini_3_flash_preview/001_user_001")
        pred_base = Path("generation/rag/results/001_user_001")
        self.benchmark = json.loads(
            (base / "tce_benchmark_task_packs_20260308_applycrit_all.json").read_text(encoding="utf-8")
        )
        self.prediction = json.loads(
            (
                pred_base
                / "prediction/tce_results_topk20_perkey_time_quarterly_applycrit_taskabc_azure.json"
            ).read_text(encoding="utf-8")
        )
        self.eval_payload = json.loads(
            (
                pred_base
                / "eval/tce_eval_topk20_perkey_time_quarterly_applycrit_taskabc_azure_rubric4.json"
            ).read_text(encoding="utf-8")
        )
        self.app_logs = json.loads((base / "app_log_large.json").read_text(encoding="utf-8"))

    def test_builds_viewer_payload_from_real_artifacts(self):
        payload = build_viewer_payload(
            benchmark_payload=self.benchmark,
            prediction_payload=self.prediction,
            eval_payload=self.eval_payload,
            app_logs_payload=self.app_logs,
            source_files={
                "benchmark": "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/tce_benchmark_task_packs_20260308_applycrit_all.json",
                "prediction": "generation/rag/results/001_user_001/prediction/tce_results_topk20_perkey_time_quarterly_applycrit_taskabc_azure.json",
                "eval": "generation/rag/results/001_user_001/eval/tce_eval_topk20_perkey_time_quarterly_applycrit_taskabc_azure_rubric4.json",
                "app_logs": "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json",
            },
        )
        self.assertEqual(payload["meta"]["user_id"], "001_user_001")
        self.assertEqual(len(payload["meta"]["checkpoint_ids"]), 5)
        self.assertIn("states", payload)
        self.assertIn("timeline", payload)
        self.assertIn("app_logs_by_id", payload)

        state_key = "habits_state:industry_news_review"
        cp1 = "cal_quarterly_001"
        self.assertIn(state_key, payload["timeline"])
        self.assertIn(cp1, payload["timeline"][state_key])
        cell = payload["timeline"][state_key][cp1]
        self.assertIn("expected_state_value", cell)
        self.assertIn("validated_state_value", cell)
        self.assertIn("predicted_task_a_value", cell)
        self.assertIn("task_a_eval", cell)
        self.assertIn("task_b_payload", cell)
        self.assertIn("task_c_items", cell)
        self.assertIn("model_inputs", cell)
        self.assertIn("task_a", cell["model_inputs"])

        task_a = cell["model_inputs"]["task_a"]
        self.assertTrue(task_a["user_memory_logs"])
        self.assertTrue(all("app_log_id" in item for item in task_a["user_memory_logs"]))

        task_c_items = cell["task_c_items"]
        self.assertTrue(task_c_items)
        self.assertIn("scenario", task_c_items[0])
        self.assertIn("question", task_c_items[0])
        self.assertIn("predicted_answer", task_c_items[0])
        self.assertIn("eval", task_c_items[0])
        self.assertIn("evidence_id_metrics", task_c_items[0]["eval"])
        self.assertNotIn("legacy_metrics", task_c_items[0]["eval"])
        self.assertNotIn("predicted_option", task_c_items[0]["eval"])

        bbq = payload["timeline"]["habits_state:backyard_bbq_hosting"]
        self.assertEqual(bbq["cal_quarterly_004"]["transition_from_previous"]["status"], "appeared_and_validated")
        self.assertEqual(bbq["cal_quarterly_005"]["transition_from_previous"]["status"], "dropped_from_expected")
        self.assertTrue(bbq["cal_quarterly_004"]["task_a_eval"]["judge"]["reason"])

        spouse = payload["timeline"]["habits_state:spouse_date_night"]["cal_quarterly_001"]
        self.assertEqual(spouse["presence_status"], "expected_but_filtered")
        self.assertFalse(spouse["exists_in_validated"])
        self.assertIsNone(spouse["task_a_eval"])

        budget_change = payload["timeline"]["habits_state:budget_review"]["cal_quarterly_002"]["task_b_payload"]
        self.assertTrue(budget_change["applicable"])
        self.assertTrue(budget_change["eval"]["evidence_id_metrics"]["expected_ids"])

    def test_supports_eval_without_task_c_legacy_judge(self):
        eval_copy = json.loads(json.dumps(self.eval_payload))
        for checkpoint in eval_copy.get("checkpoints", []):
            if isinstance(checkpoint, dict):
                checkpoint.pop("rq3_llm_judge_prompt", None)
                checkpoint.pop("rq3_llm_judge_raw_output", None)
                checkpoint.pop("rq3_llm_judge_judgments", None)
                checkpoint.pop("rq3_apply_llm_mean_1_5", None)
                checkpoint.pop("rq3_llm_judge_score", None)
        payload = build_viewer_payload(
            benchmark_payload=self.benchmark,
            prediction_payload=self.prediction,
            eval_payload=eval_copy,
            app_logs_payload=self.app_logs,
        )
        self.assertEqual(payload["meta"]["eval_protocol_version"], "legacy_task_c_deterministic_only")
        state_key = "habits_state:industry_news_review"
        cp1 = "cal_quarterly_001"
        item = payload["timeline"][state_key][cp1]["task_c_items"][0]
        self.assertNotIn("legacy_metrics", item["eval"])

    def test_writes_payload_and_viewer_static_files_exist(self):
        payload = build_viewer_payload(
            benchmark_payload=self.benchmark,
            prediction_payload=self.prediction,
            eval_payload=self.eval_payload,
            app_logs_payload=self.app_logs,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "state_timeline_viewer_data.json"
            write_viewer_payload(payload, out)
            self.assertTrue(out.exists())
            self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["meta"]["checkpoint_ids"])

        viewer_dir = Path("analysis_tools/tce_state_timeline_viewer")
        index_text = (viewer_dir / "index.html").read_text(encoding="utf-8")
        app_text = (viewer_dir / "app.js").read_text(encoding="utf-8")
        self.assertIn("matrix-container", index_text)
        self.assertIn("matrix-view-filter", index_text)
        self.assertIn("Task C quality", index_text)
        self.assertIn("timeline-cards", index_text)
        self.assertIn("state_timeline_viewer_data.json", index_text)
        self.assertIn("task_c_items", app_text)
        self.assertIn("user_memory_logs", app_text)
        self.assertIn("taskBScoreValue", app_text)
        self.assertIn("taskCScoreValue", app_text)
        self.assertIn("taskCEvidenceMetrics", app_text)
        self.assertIn("ground truth", app_text)
        self.assertIn("model prediction", app_text)
        self.assertIn('matrixMode === "task_b"', app_text)
        self.assertIn('matrixMode === "task_c"', app_text)

    def test_compact_mode_drops_heavy_fields(self):
        payload = build_viewer_payload(
            benchmark_payload=self.benchmark,
            prediction_payload=self.prediction,
            eval_payload=self.eval_payload,
            app_logs_payload=self.app_logs,
            compact=True,
        )
        self.assertTrue(payload["meta"]["compact_mode"])
        state_key = "habits_state:industry_news_review"
        cp1 = "cal_quarterly_001"
        task_a = payload["timeline"][state_key][cp1]["model_inputs"]["task_a"]
        self.assertIsNone(task_a["prompt"])
        self.assertIn("raw_model_output", task_a)
        first_task_c = payload["timeline"][state_key][cp1]["task_c_items"][0]
        self.assertIsNotNone((first_task_c.get("model_inputs") or {}).get("raw_model_output"))
        any_log = next(iter(payload["app_logs_by_id"].values()))
        self.assertNotIn("request", any_log)
        self.assertNotIn("response", any_log)

    def test_task_a_input_map_reads_raw_output_records_from_dict_payload(self):
        input_map = _build_task_a_input_map(
            {
                "metadata": {
                    "target_keys": ["habits_state:morning_walk"],
                    "prompt": ["Prompt A"],
                    "raw_model_output": {
                        "mode": "per_key",
                        "records": [
                            {
                                "key": "habits_state:morning_walk",
                                "prompt": "Prompt A",
                                "raw_model_output": {"snapshot_state": {"habits_state:morning_walk": "06:30"}},
                            }
                        ],
                    },
                    "per_key_retrieval": [
                        {
                            "key": "habits_state:morning_walk",
                            "retrieval_query": "Infer morning walk state.",
                            "retrieval_metadata": {"retrieved_app_log_ids": ["log_0001"]},
                            "context_log_ids": ["log_0001"],
                        }
                    ],
                }
            }
        )
        task_a = input_map["habits_state:morning_walk"]
        self.assertEqual(task_a["prompt"], "Prompt A")
        self.assertEqual(
            task_a["raw_model_output"],
            {"snapshot_state": {"habits_state:morning_walk": "06:30"}},
        )
        self.assertEqual(task_a["retrieval_query"], "Infer morning walk state.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
