#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from generation.adapters import rag as rag_adapter
from generation.adapters.base import TceAdapterArgs
from generation.rag import rag_tce


class RagTceClientConfigTest(unittest.TestCase):
    def test_build_openai_client_uses_openai_env_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_BASE_URL": "https://azure.example/openai/v1",
            },
            clear=False,
        ), mock.patch("generation.rag.rag_tce.OpenAI") as openai_cls:
            rag_tce._build_openai_client("openai")

        openai_cls.assert_called_once_with(
            api_key="openai-key",
            base_url="https://openai.example/v1",
        )

    def test_build_openai_client_uses_azure_env_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_BASE_URL": "https://azure.example/openai/v1",
            },
            clear=False,
        ), mock.patch("generation.rag.rag_tce.OpenAI") as openai_cls:
            rag_tce._build_openai_client("azure")

        openai_cls.assert_called_once_with(
            api_key="azure-key",
            base_url="https://azure.example/openai/v1",
            default_headers={"api-key": "azure-key"},
        )

    def test_embedding_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_path = root / "memory" / "embeddings.npz"
            meta_path = root / "memory" / "embeddings.meta.json"
            embeddings = np.array([[3.0, 4.0], [0.0, 2.0]], dtype="float32")

            rag_tce._write_embedding_cache(
                cache_path=cache_path,
                meta_path=meta_path,
                embeddings=embeddings,
                metadata={"num_logs": 2},
            )

            loaded = rag_tce._load_embedding_cache(cache_path, expected_rows=2)
            meta_exists = meta_path.exists()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.shape, (2, 2))
        np.testing.assert_allclose(
            loaded,
            np.array([[0.6, 0.8], [0.0, 1.0]], dtype="float32"),
            atol=1e-6,
        )
        self.assertTrue(meta_exists)

    def test_embedding_cache_key_changes_when_logs_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_path = root / "results" / "001_user_001" / "prediction" / "exp1" / "prediction.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            app_logs_path = root / "app_logs.json"
            app_logs_path.write_text("[]", encoding="utf-8")
            first_cache_path, _ = rag_tce._embedding_cache_paths(
                app_logs_path=app_logs_path,
                output_path=output_path,
                retriever_provider="openai",
                retriever_model="text-embedding-3-large",
            )

            app_logs_path.write_text('[{"app_log_id":"log_1"}]', encoding="utf-8")
            second_cache_path, _ = rag_tce._embedding_cache_paths(
                app_logs_path=app_logs_path,
                output_path=output_path,
                retriever_provider="openai",
                retriever_model="text-embedding-3-large",
            )

        self.assertNotEqual(first_cache_path, second_cache_path)
        self.assertEqual(first_cache_path.parent, root / "results" / "001_user_001" / "memory")

    def test_adapter_passes_build_only_to_rag_generation(self) -> None:
        captured = {}

        def _fake_run_generation(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        args = TceAdapterArgs(
            baseline="rag",
            user_id="001_user_001",
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/out.json"),
            llm_provider="azure",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="azure",
            retriever_model="text-embedding-3-large",
            retriever_batch_size=16,
            retrieval_top_k=20,
            extras={
                "predict_per_key": "true",
                "build_only": "true",
                "__task_selection__": "task_c_only",
            },
        )
        with mock.patch("generation.rag.rag_tce.run_generation", side_effect=_fake_run_generation):
            rag_adapter.run(args)

        self.assertTrue(captured["build_only"])
        self.assertEqual(captured["task_selection"], "task_c_only")

    def test_run_generation_build_only_skips_pipeline_and_writes_embedding_cache(self) -> None:
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {"checkpoint_id": "cp_0001"},
                {"checkpoint_id": "cp_0002"},
            ],
        }
        app_logs = [
            {
                "app_log_id": "log_0001",
                "timestamp": "2025-01-01 08:00:00",
                "app_name": "coffee_app",
                "api_name": "save_preference",
                "request": {},
                "response": {"favorite_coffee": "latte"},
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "build_only.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch(
                "generation.rag.rag_tce.LLMClient",
                side_effect=AssertionError("LLMClient should not be constructed in build_only mode"),
            ), mock.patch(
                "generation.rag.rag_tce.run_pipeline",
                side_effect=AssertionError("run_pipeline should not be called in build_only mode"),
            ), mock.patch(
                "generation.rag.rag_tce._build_openai_client",
                lambda provider: object(),
            ), mock.patch(
                "generation.rag.rag_tce._embed_texts",
                return_value=np.array([[1.0, 0.0]], dtype="float32"),
            ):
                result = rag_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="azure",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=64,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    build_only=True,
                )

            cache_path = Path(result["build_only"]["embedding_cache_path"])
            meta_path = Path(result["build_only"]["embedding_cache_meta_path"])
            cache_exists = cache_path.exists()
            meta_exists = meta_path.exists()
            output_exists = output_path.exists()

        self.assertEqual(result["predictions"], [])
        self.assertTrue(result["build_only"]["enabled"])
        self.assertEqual(result["build_only"]["requested_checkpoint_ids"], ["cp_0001"])
        self.assertEqual(result["build_only"]["num_logs"], 1)
        self.assertTrue(cache_exists)
        self.assertTrue(meta_exists)
        self.assertFalse(output_exists)


if __name__ == "__main__":
    unittest.main(verbosity=2)
