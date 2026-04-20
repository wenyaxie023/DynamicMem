#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from generation.Amem import export_memory_viewer
from generation.Amem import usage_sidecar


class AmemBuildUsageSidecarTest(unittest.TestCase):
    def test_build_usage_sidecar_paths_are_user_scoped(self):
        root = Path("/tmp/amem")
        state_path = root / "membench_amem_001_user_001_large.pkl"

        final_path, live_path = usage_sidecar.build_usage_sidecar_paths(state_path)

        self.assertEqual(final_path, root / "membench_amem_001_user_001_large_usage_cost.json")
        self.assertEqual(live_path, root / "membench_amem_001_user_001_large_usage_cost_live.json")

    def test_write_build_usage_sidecar_records_build_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecar_path = root / "usage_cost_live.json"
            state_path = root / "membench_amem_001_user_001_large.pkl"
            checkpoint_path = root / "membench_amem_001_user_001_large.json"

            usage_sidecar.write_build_usage_sidecar(
                sidecar_path=sidecar_path,
                state_path=state_path,
                checkpoint_path=checkpoint_path,
                llm_provider="azure",
                llm_model="gpt-5-mini",
                retriever_provider="azure",
                embedding_model_name="text-embedding-3-large",
                build_duration_s=12.5,
                build_memory_usage={
                    "request_count": 3,
                    "chat_request_count": 1,
                    "embedding_request_count": 2,
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "reasoning_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "context_tokens": 0,
                    "total_tokens": 150,
                    "turn_count": 3,
                    "by_model": [],
                },
                is_live=True,
            )

            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["llm_provider"], "azure")
            self.assertEqual(payload["llm_model"], "gpt-5-mini")
            self.assertEqual(payload["retriever_provider"], "azure")
            self.assertEqual(payload["embedding_model"], "text-embedding-3-large")
            self.assertEqual(payload["usage"]["build_memory"]["request_count"], 3)
            self.assertEqual(payload["usage"]["build_memory"]["total_tokens"], 150)
            self.assertEqual(payload["timing"]["build_memory_duration_s"], 12.5)
            self.assertEqual(payload["timing"]["generation_duration_s"], 0.0)
            self.assertEqual(payload["timing"]["total_duration_s"], 12.5)
            self.assertTrue(payload["is_live"])

    def test_export_memory_viewer_falls_back_to_checkpoint_usage_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "run_x" / "membench_amem_001_user_001_large.pkl"
            checkpoint_path = root / "run_x" / "membench_amem_001_user_001_large.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_bytes(b"placeholder")
            checkpoint_path.write_text("{}", encoding="utf-8")
            usage_sidecar_path = state_path.parent / "membench_amem_001_user_001_large_usage_cost_live.json"
            usage_sidecar_path.write_text(
                json.dumps(
                    {
                        "usage": {"build_memory": {"request_count": 7, "total_tokens": 700}},
                        "timing": {"build_memory_duration_s": 21.0, "generation_duration_s": 0.0, "total_duration_s": 21.0},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch.object(export_memory_viewer, "_load_state", return_value={}), mock.patch.object(
                export_memory_viewer,
                "_load_checkpoint",
                return_value={"last_event_idx": 4, "events_processed": 5, "saved_at": "2026-04-15T22:16:30"},
            ):
                payload = export_memory_viewer._build_payload(state_path, checkpoint_path)

            self.assertEqual(payload["meta"]["usage_cost_path"], str(usage_sidecar_path))
            self.assertEqual(payload["usage_cost"]["usage"]["build_memory"]["request_count"], 7)
            self.assertEqual(payload["usage_cost"]["timing"]["build_memory_duration_s"], 21.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
