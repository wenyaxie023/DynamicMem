#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import generation.common.provider_config as provider_config
from generation.adapters import amem as amem_adapter
from generation.adapters.base import TceAdapterArgs


class AmemAdapterChainTest(unittest.TestCase):
    def test_amem_adapter_runs_builder_then_tce_test_with_shared_inputs(
        self,
    ):
        evaluate_membench_mock = mock.Mock(return_value={"events_processed": 7, "snapshots_written": 2})
        run_generation_mock = mock.Mock()

        def test_side_effect(*args, **kwargs):
            return {"predictions": []}

        run_generation_mock.side_effect = test_side_effect

        fake_amem_builder = ModuleType("generation.Amem.amem")
        fake_amem_builder.evaluate_membench = evaluate_membench_mock

        fake_amem_tce = ModuleType("generation.Amem.tce")
        fake_amem_tce.run_generation = run_generation_mock

        fake_usage_module = ModuleType("generation.Amem.agentic_memory.llm_controller")
        fake_usage_module.reset_usage_tracker = mock.Mock()
        fake_usage_module.get_usage_summary = mock.Mock(return_value={})

        args = TceAdapterArgs(
            baseline="amem",
            user_id="001_user_001",
            benchmark=Path("/tmp/bench.json"),
            app_logs_path=Path("/tmp/app_log_large.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retrieval_top_k=20,
            enable_change_reasoning=True,
            enable_rq3_apply_service_qa=True,
            enable_final_qa=True,
            extras={
                "size": "large",
                "save_every": 5,
                "linked_neighbor_top_k": 0,
                "memory_action": "build_then_predict",
                "data_storage_path": "/tmp/amem-runtime",
                "snapshot_dir": "/tmp/amem-snapshots",
            },
        )

        with mock.patch.object(provider_config, "load_repo_dotenv") as load_repo_dotenv_mock, mock.patch.object(
            provider_config, "resolve_openai_compatible_credentials"
        ) as resolve_creds_mock, mock.patch.dict(
            sys.modules,
            {
                "generation.Amem.amem": fake_amem_builder,
                "generation.Amem.tce": fake_amem_tce,
                "generation.Amem.agentic_memory.llm_controller": fake_usage_module,
            },
        ):
            resolve_creds_mock.side_effect = [
                ("retriever-key", "https://retriever.example"),
                ("llm-key", "https://llm.example"),
            ]

            result = amem_adapter.run(args)

            self.assertEqual(result, {"predictions": []})
            load_repo_dotenv_mock.assert_called_once()

            evaluate_membench_mock.assert_called_once()
            builder_kwargs = evaluate_membench_mock.call_args[1]
            self.assertEqual(builder_kwargs["user_id"], "001_user_001")
            self.assertEqual(builder_kwargs["app_log_path"], Path("/tmp/app_log_large.json"))
            self.assertEqual(builder_kwargs["benchmark_path"], "/tmp/bench.json")
            self.assertEqual(builder_kwargs["size"], "large")
            self.assertEqual(builder_kwargs["save_every"], 5)
            self.assertEqual(builder_kwargs["data_storage_path"], "/tmp/amem-runtime")
            self.assertEqual(builder_kwargs["embedding_api_key"], "retriever-key")
            self.assertEqual(builder_kwargs["embedding_api_base_url"], "https://retriever.example")
            self.assertEqual(builder_kwargs["llm_controller_api_key"], "llm-key")
            self.assertEqual(builder_kwargs["llm_controller_api_base_url"], "https://llm.example")

            run_generation_mock.assert_called_once()
            test_kwargs = run_generation_mock.call_args[1]
            self.assertEqual(test_kwargs["benchmark_path"], Path("/tmp/bench.json"))
            self.assertEqual(test_kwargs["app_logs_path"], Path("/tmp/app_log_large.json"))
            self.assertEqual(test_kwargs["user_id"], "001_user_001")
            self.assertEqual(test_kwargs["size"], "large")
            self.assertEqual(test_kwargs["snapshot_dir"], "/tmp/amem-snapshots")
            self.assertEqual(test_kwargs["linked_neighbor_top_k"], 0)
            self.assertEqual(test_kwargs["embedding_api_key"], "retriever-key")
            self.assertEqual(test_kwargs["embedding_api_base_url"], "https://retriever.example")

    def test_amem_adapter_writes_usage_sidecar_and_build_timing(self):
        evaluate_membench_mock = mock.Mock(return_value={"events_processed": 3, "snapshots_written": 1})
        run_generation_mock = mock.Mock(
            return_value={
                "predictions": [],
                "answer_llm_usage": {
                    "request_count": 1,
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "reasoning_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "context_tokens": 0,
                    "total_tokens": 18,
                    "turn_count": 1,
                    "by_model": [],
                },
            }
        )

        fake_amem_builder = ModuleType("generation.Amem.amem")
        fake_amem_builder.evaluate_membench = evaluate_membench_mock

        fake_amem_tce = ModuleType("generation.Amem.tce")
        fake_amem_tce.run_generation = run_generation_mock

        fake_usage_module = ModuleType("generation.Amem.agentic_memory.llm_controller")
        fake_usage_module.reset_usage_tracker = mock.Mock()
        usage_summary_mock = mock.Mock(
            side_effect=[
                {
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
                {
                    "request_count": 1,
                    "chat_request_count": 0,
                    "embedding_request_count": 1,
                    "prompt_tokens": 25,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "context_tokens": 0,
                    "total_tokens": 25,
                    "turn_count": 1,
                    "by_model": [],
                },
            ]
        )
        fake_usage_module.get_usage_summary = usage_summary_mock

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "prediction.json"
            args = TceAdapterArgs(
                baseline="amem",
                user_id="001_user_001",
                benchmark=Path("/tmp/bench.json"),
                app_logs_path=Path("/tmp/app_log_large.json"),
                output=output_path,
                llm_provider="azure",
                llm_model="gpt-5-mini",
                llm_max_workers=1,
                retriever_provider="azure",
                retriever_model="text-embedding-3-large",
                retrieval_top_k=20,
                extras={
                    "size": "large",
                    "save_every": 5,
                    "memory_action": "build_then_predict",
                    "data_storage_path": "/tmp/amem-runtime",
                    "snapshot_dir": "/tmp/amem-snapshots",
                },
            )

            with mock.patch.object(provider_config, "load_repo_dotenv") as load_repo_dotenv_mock, mock.patch.object(
                provider_config, "resolve_openai_compatible_credentials"
            ) as resolve_creds_mock, mock.patch.dict(
                sys.modules,
                {
                    "generation.Amem.amem": fake_amem_builder,
                    "generation.Amem.tce": fake_amem_tce,
                    "generation.Amem.agentic_memory.llm_controller": fake_usage_module,
                },
            ):
                resolve_creds_mock.side_effect = [
                    ("retriever-key", "https://retriever.example"),
                    ("llm-key", "https://llm.example"),
                ]

                result = amem_adapter.run(args)

            self.assertEqual(result["predictions"], [])
            load_repo_dotenv_mock.assert_called_once()
            self.assertEqual(fake_usage_module.reset_usage_tracker.call_count, 2)
            self.assertEqual(usage_summary_mock.call_count, 2)

            sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["llm_provider"], "azure")
            self.assertEqual(sidecar["llm_model"], "gpt-5-mini")
            self.assertEqual(sidecar["retriever_provider"], "azure")
            self.assertEqual(sidecar["embedding_model"], "text-embedding-3-large")
            self.assertGreaterEqual(float(sidecar["timing"]["build_memory_duration_s"]), 0.0)
            self.assertGreaterEqual(float(sidecar["timing"]["generation_duration_s"]), 0.0)
            self.assertEqual(sidecar["usage"]["build_memory"]["request_count"], 3)
            self.assertEqual(sidecar["usage"]["retrieval"]["request_count"], 1)
            self.assertEqual(sidecar["usage"]["answer_llm"]["prompt_tokens"], 11)
            self.assertEqual(sidecar["usage"]["answer_llm"]["completion_tokens"], 7)

    def test_amem_adapter_build_only_skips_generation(self):
        evaluate_membench_mock = mock.Mock(return_value={"events_processed": 3, "snapshots_written": 1})
        run_generation_mock = mock.Mock()

        fake_amem_builder = ModuleType("generation.Amem.amem")
        fake_amem_builder.evaluate_membench = evaluate_membench_mock

        fake_amem_tce = ModuleType("generation.Amem.tce")
        fake_amem_tce.run_generation = run_generation_mock

        fake_usage_module = ModuleType("generation.Amem.agentic_memory.llm_controller")
        fake_usage_module.reset_usage_tracker = mock.Mock()
        fake_usage_module.get_usage_summary = mock.Mock(
            return_value={
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
            }
        )

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "build_only.json"
            args = TceAdapterArgs(
                baseline="amem",
                user_id="001_user_001",
                benchmark=Path("/tmp/bench.json"),
                app_logs_path=Path("/tmp/app_log_large.json"),
                output=output_path,
                llm_provider="azure",
                llm_model="gpt-5-mini",
                llm_max_workers=1,
                retriever_provider="azure",
                retriever_model="text-embedding-3-large",
                retrieval_top_k=20,
                extras={
                    "size": "large",
                    "save_every": 5,
                    "memory_action": "build_only",
                    "data_storage_path": "/tmp/amem-runtime",
                    "snapshot_dir": "/tmp/amem-snapshots",
                },
            )

            with mock.patch.object(provider_config, "load_repo_dotenv"), mock.patch.object(
                provider_config, "resolve_openai_compatible_credentials"
            ) as resolve_creds_mock, mock.patch.dict(
                sys.modules,
                {
                    "generation.Amem.amem": fake_amem_builder,
                    "generation.Amem.tce": fake_amem_tce,
                    "generation.Amem.agentic_memory.llm_controller": fake_usage_module,
                },
            ):
                resolve_creds_mock.side_effect = [
                    ("retriever-key", "https://retriever.example"),
                    ("llm-key", "https://llm.example"),
                ]
                result = amem_adapter.run(args)

            self.assertEqual(result["predictions"], [])
            self.assertTrue(result["build_only"]["enabled"])
            evaluate_membench_mock.assert_called_once()
            run_generation_mock.assert_not_called()
            sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["usage"]["build_memory"]["request_count"], 3)
            self.assertEqual(sidecar["usage"]["retrieval"]["request_count"], 0)

    def test_amem_adapter_predict_from_prebuilt_skips_builder(self):
        evaluate_membench_mock = mock.Mock(side_effect=AssertionError("builder should not run"))
        run_generation_mock = mock.Mock(return_value={"predictions": []})

        fake_amem_builder = ModuleType("generation.Amem.amem")
        fake_amem_builder.evaluate_membench = evaluate_membench_mock

        fake_amem_tce = ModuleType("generation.Amem.tce")
        fake_amem_tce.run_generation = run_generation_mock

        fake_usage_module = ModuleType("generation.Amem.agentic_memory.llm_controller")
        fake_usage_module.reset_usage_tracker = mock.Mock()
        fake_usage_module.get_usage_summary = mock.Mock(return_value={})

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot_dir = root / "snapshots"
            manifest_path = snapshot_dir / "001_user_001" / "large" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({"latest_snapshot": None, "snapshots": []}), encoding="utf-8")
            output_path = root / "prediction.json"
            args = TceAdapterArgs(
                baseline="amem",
                user_id="001_user_001",
                benchmark=Path("/tmp/bench.json"),
                app_logs_path=Path("/tmp/app_log_large.json"),
                output=output_path,
                llm_provider="azure",
                llm_model="gpt-5-mini",
                llm_max_workers=1,
                retriever_provider="azure",
                retriever_model="text-embedding-3-large",
                retrieval_top_k=20,
                extras={
                    "size": "large",
                    "save_every": 5,
                    "memory_action": "predict_from_prebuilt",
                    "data_storage_path": str(root / "runtime"),
                    "snapshot_dir": str(snapshot_dir),
                },
            )

            with mock.patch.object(provider_config, "load_repo_dotenv"), mock.patch.object(
                provider_config, "resolve_openai_compatible_credentials"
            ) as resolve_creds_mock, mock.patch.dict(
                sys.modules,
                {
                    "generation.Amem.amem": fake_amem_builder,
                    "generation.Amem.tce": fake_amem_tce,
                    "generation.Amem.agentic_memory.llm_controller": fake_usage_module,
                },
            ):
                resolve_creds_mock.side_effect = [
                    ("retriever-key", "https://retriever.example"),
                    ("llm-key", "https://llm.example"),
                ]
                result = amem_adapter.run(args)

            self.assertEqual(result["predictions"], [])
            evaluate_membench_mock.assert_not_called()
            run_generation_mock.assert_called_once()
            self.assertEqual(resolve_creds_mock.call_count, 1)
            test_kwargs = run_generation_mock.call_args[1]
            self.assertEqual(test_kwargs["snapshot_dir"], str(snapshot_dir))
            self.assertEqual(test_kwargs["embedding_api_key"], "retriever-key")
            sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["timing"]["build_memory_duration_s"], 0.0)

    def test_amem_adapter_legacy_skip_build_alias_uses_prebuilt_snapshots(self):
        evaluate_membench_mock = mock.Mock(side_effect=AssertionError("builder should not run"))
        run_generation_mock = mock.Mock(return_value={"predictions": []})

        fake_amem_builder = ModuleType("generation.Amem.amem")
        fake_amem_builder.evaluate_membench = evaluate_membench_mock

        fake_amem_tce = ModuleType("generation.Amem.tce")
        fake_amem_tce.run_generation = run_generation_mock

        fake_usage_module = ModuleType("generation.Amem.agentic_memory.llm_controller")
        fake_usage_module.reset_usage_tracker = mock.Mock()
        fake_usage_module.get_usage_summary = mock.Mock(return_value={})

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot_dir = root / "snapshots"
            manifest_path = snapshot_dir / "001_user_001" / "large" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({"latest_snapshot": None, "snapshots": []}), encoding="utf-8")
            args = TceAdapterArgs(
                baseline="amem",
                user_id="001_user_001",
                benchmark=Path("/tmp/bench.json"),
                app_logs_path=Path("/tmp/app_log_large.json"),
                output=root / "prediction.json",
                llm_provider="azure",
                llm_model="gpt-5-mini",
                llm_max_workers=1,
                retriever_provider="azure",
                retriever_model="text-embedding-3-large",
                retrieval_top_k=20,
                extras={
                    "size": "large",
                    "skip_build": "true",
                    "snapshot_dir": str(snapshot_dir),
                },
            )

            with mock.patch.object(provider_config, "load_repo_dotenv"), mock.patch.object(
                provider_config, "resolve_openai_compatible_credentials"
            ) as resolve_creds_mock, mock.patch.dict(
                sys.modules,
                {
                    "generation.Amem.amem": fake_amem_builder,
                    "generation.Amem.tce": fake_amem_tce,
                    "generation.Amem.agentic_memory.llm_controller": fake_usage_module,
                },
            ):
                resolve_creds_mock.side_effect = [
                    ("retriever-key", "https://retriever.example"),
                    ("llm-key", "https://llm.example"),
                ]
                amem_adapter.run(args)

            evaluate_membench_mock.assert_not_called()
            run_generation_mock.assert_called_once()
            self.assertEqual(resolve_creds_mock.call_count, 1)

    def test_amem_adapter_rejects_conflicting_memory_action_flags(self):
        args = TceAdapterArgs(
            baseline="amem",
            user_id="001_user_001",
            benchmark=Path("/tmp/bench.json"),
            app_logs_path=Path("/tmp/app_log_large.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retrieval_top_k=20,
            extras={
                "memory_action": "predict_from_prebuilt",
                "build_only": "true",
                "snapshot_dir": "/tmp/amem-snapshots",
            },
        )

        with self.assertRaisesRegex(ValueError, "Conflicting amem memory controls"):
            amem_adapter.run(args)

    def test_amem_adapter_rejects_negative_linked_neighbor_top_k(self):
        args = TceAdapterArgs(
            baseline="amem",
            user_id="001_user_001",
            benchmark=Path("/tmp/bench.json"),
            app_logs_path=Path("/tmp/app_log_large.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retrieval_top_k=20,
            extras={
                "memory_action": "predict_from_prebuilt",
                "linked_neighbor_top_k": "-1",
                "snapshot_dir": "/tmp/amem-snapshots",
            },
        )

        with self.assertRaisesRegex(ValueError, "linked_neighbor_top_k"):
            amem_adapter.run(args)

    def test_amem_adapter_rejects_legacy_checkpoint_dir_extra(self):
        args = TceAdapterArgs(
            baseline="amem",
            user_id="001_user_001",
            benchmark=Path("/tmp/bench.json"),
            app_logs_path=Path("/tmp/app_log_large.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retrieval_top_k=20,
            extras={
                "size": "large",
                "data_storage_path": "/tmp/amem-runtime",
                "snapshot_dir": "/tmp/amem-snapshots",
                "checkpoint_dir": "/tmp/legacy-checkpoints",
            },
        )

        with self.assertRaisesRegex(ValueError, "baseline_params\\.checkpoint_dir"):
            amem_adapter.run(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
