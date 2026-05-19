#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.letta.agent_loop import LettaAgentLoop, estimate_usage_cost_usd, sort_logs
from baseline_prediction.letta import tce as letta_tce
from baseline_prediction.adapters.base import (
    AGENT_LOOP_ROUTE,
    SHARED_PIPELINE_SNAPSHOT_ROUTE,
    TceAdapterArgs,
    get_baseline_route_class,
)
from baseline_prediction.adapters import letta as letta_adapter
from baseline_prediction.tce_config import build_from_config
from tce_core.orchestrator_protocol import AnswerExecutionResult, CheckpointHandle, RetrievalResult
from tce_core.pipeline import run_pipeline


class _FakeLettaAgentLoop:
    instances = []
    agent_records_by_id = {}

    def __init__(self, *args, **kwargs):
        self._fallback = False
        self.agent = None
        self.ingested_ids = []
        self.saved_paths = []
        self.imported_payloads = []
        self.deleted_ids = []
        self.loaded_paths = []
        self.attached_ids = []
        self.temp_counter = 0
        _FakeLettaAgentLoop.instances.append(self)

    def _active_agent_id(self) -> str:
        return "" if self.agent is None else str(self.agent.id)

    def effective_backend(self):
        return "sdk"

    def create_agent(self):
        self.agent = SimpleNamespace(id="builder-agent", name="fake-builder")
        _FakeLettaAgentLoop.agent_records_by_id.setdefault(self.agent.id, [])
        return self.agent.id

    def attach_agent(self, agent_id: str):
        self.agent = SimpleNamespace(id=str(agent_id), name="attached-builder")
        self.attached_ids.append(str(agent_id))
        _FakeLettaAgentLoop.agent_records_by_id.setdefault(str(agent_id), [])
        return self.agent.id

    def active_agent_id(self):
        return None if self.agent is None else str(self.agent.id)

    def active_agent_name(self):
        return None if self.agent is None else getattr(self.agent, "name", None)

    def ingest_log(self, log):
        self.ingested_ids.append(str(log.get("app_log_id", "")))
        _FakeLettaAgentLoop.agent_records_by_id.setdefault(self._active_agent_id(), []).append(dict(log))

    def save_agent_file(self, output_path: Path, agent_id=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_agent_id = str(agent_id or self._active_agent_id())
        output_path.write_text(
            json.dumps(
                {"ingested_logs": _FakeLettaAgentLoop.agent_records_by_id.get(target_agent_id, [])},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.saved_paths.append(str(output_path))
        return output_path

    def load_agent_file(self, file_path: Path, activate: bool = False):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        ingested_logs = payload.get("ingested_logs", [])
        self.ingested_ids = [str(log.get("app_log_id", "")) for log in ingested_logs if isinstance(log, dict)]
        self.loaded_paths.append(str(file_path))
        if activate:
            self.agent = SimpleNamespace(id="loaded-builder-agent", name="loaded-builder")
            _FakeLettaAgentLoop.agent_records_by_id[self.agent.id] = list(ingested_logs)
        return "loaded-builder-agent"

    def import_agent_file(self, file_bytes: bytes, activate: bool = False):
        payload = json.loads(file_bytes.decode("utf-8"))
        self.imported_payloads.append(payload)
        self.temp_counter += 1
        return f"temp-agent-{self.temp_counter}"

    def ask_json_with_agent(self, agent_id: str, question: str):
        if '"change_analysis"' in question:
            return {
                "change_analysis": {
                    "profile_state:favorite_coffee": {
                        "before": "latte",
                        "after": "espresso",
                        "change_reason": "updated coffee preference",
                        "evidence": [
                            {
                                "app_log_id": "log_0002",
                                "evidence_content": "favorite_coffee espresso",
                            }
                        ],
                    }
                }
            }
        if '"answer"' in question and '"snapshot_state"' not in question and '"user_state"' not in question:
            return {
                "answer": "Recommend espresso",
                "evidence": [
                    {
                        "app_log_id": "log_0002",
                        "evidence_content": "favorite_coffee espresso",
                    }
                ],
            }
        if "habits_state:budget_review" in question:
            return {
                "snapshot_state": {"habits_state:budget_review": "monthly review"},
                "evidence": {"habits_state:budget_review": ["log_0001"]},
            }
        if "preferences_state:learning_modality" in question:
            return {
                "snapshot_state": {"preferences_state:learning_modality": {"statement": "self-paced webinars"}},
                "evidence": {"preferences_state:learning_modality": ["log_0001"]},
            }
        return {"snapshot_state": {}, "evidence": {}}

    def ask_json(self, question: str):
        if '"change_analysis"' in question:
            return {
                "change_analysis": {
                    "profile_state:favorite_coffee": {
                        "before": "latte",
                        "after": "espresso",
                        "change_reason": "updated coffee preference",
                        "evidence": [
                            {
                                "app_log_id": "log_0002",
                                "evidence_content": "favorite_coffee espresso",
                            }
                        ],
                    }
                }
            }
        if '"answer"' in question and '"snapshot_state"' not in question and '"user_state"' not in question:
            return {
                "answer": "Recommend espresso",
                "evidence": [
                    {
                        "app_log_id": "log_0002",
                        "evidence_content": "favorite_coffee espresso",
                    }
                ],
            }
        return {"snapshot_state": {}, "evidence": {}}

    def delete_agent(self, agent_id: str, ignore_missing: bool = True):
        self.deleted_ids.append(str(agent_id))
        return True

    def close(self):
        return None

    def usage_summary(self):
        return {
            "turn_count": 3,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "reasoning_tokens": 12,
            "cached_input_tokens": 40,
            "cache_write_tokens": 0,
            "context_tokens": 150,
            "total_tokens": 150,
            "step_count": 1,
            "run_ids": ["run-fake-1"],
        }

    def usage_records(self):
        return []


class _UncertainFailFakeLettaAgentLoop(_FakeLettaAgentLoop):
    def ingest_log(self, log):
        super().ingest_log(log)
        if str(log.get("app_log_id", "")) == "log_0002":
            raise RuntimeError("simulated ambiguous ingest failure")


class LettaTceMinimalAcceptance(unittest.TestCase):
    def setUp(self):
        _FakeLettaAgentLoop.instances = []
        _FakeLettaAgentLoop.agent_records_by_id = {}

    def test_sort_logs_orders_by_timestamp_then_id(self):
        logs = [
            {"app_log_id": "log_0002", "timestamp": "2025-01-01 09:00:00"},
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 08:00:00"},
            {"app_log_id": "log_0003", "timestamp": "2025-01-01 09:00:00"},
        ]
        ordered = sort_logs(logs)
        self.assertEqual(
            [x["app_log_id"] for x in ordered],
            ["log_0001", "log_0002", "log_0003"],
        )

    def test_local_fallback_produces_task_a_json_shape(self):
        agent = LettaAgentLoop(mode="local", allow_local_fallback=True)
        agent.ingest_log({"app_log_id": "log_0001", "timestamp": "2025-01-01 08:00:00"})
        prompt = """Using ONLY your memory built from previously ingested app logs, predict values.
Return JSON only with this exact top-level shape:
{"snapshot_state": {"k1": "<value>", "k2": "<value>"}, "evidence": {"k1": ["<app_log_id>"], "k2": ["<app_log_id>"]}}
Rules:
1. test
"""
        out = agent.ask_json(prompt)
        self.assertEqual(sorted(out.keys()), ["evidence", "snapshot_state"])
        self.assertEqual(sorted(out["snapshot_state"].keys()), ["k1", "k2"])
        self.assertEqual(out["snapshot_state"]["k1"], None)
        self.assertEqual(out["evidence"]["k1"], [])

    def test_sdk_backend_passes_base_url_when_configured(self):
        captured = {}

        class _FakeLetta:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}), mock.patch.dict(
            os.environ,
            {
                "LETTA_API_KEY": "test-key",
                "LETTA_BASE_URL": "http://127.0.0.1:8283",
            },
            clear=False,
        ):
            agent = LettaAgentLoop(mode="sdk", allow_local_fallback=False, create_agent=False)

        self.assertFalse(agent._fallback)
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "http://127.0.0.1:8283")

    def test_sdk_backend_uses_explicit_embedding_when_provided(self):
        captured_create = {}

        class _FakeAgents:
            def create(self, **kwargs):
                captured_create.update(kwargs)
                return SimpleNamespace(id="agent-1")

        class _FakeLetta:
            def __init__(self, **kwargs):
                self.agents = _FakeAgents()

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            LettaAgentLoop(
                mode="sdk",
                allow_local_fallback=False,
                llm_model="azure/gpt-5-mini",
                embedding="azure/text-embedding-3-large",
                answer_temperature=0.0,
                answer_top_p=1.0,
                answer_top_k=25,
            )

        self.assertEqual(captured_create["embedding"], "azure/text-embedding-3-large")
        self.assertEqual(captured_create["model_settings"]["temperature"], 0.0)
        self.assertEqual(captured_create["model_settings"]["top_p"], 1.0)
        self.assertEqual(captured_create["model_settings"]["top_k"], 25)

    def test_sdk_backend_uses_explicit_context_window_when_provided(self):
        captured_create = {}

        class _FakeAgents:
            def create(self, **kwargs):
                captured_create.update(kwargs)
                return SimpleNamespace(id="agent-1")

        class _FakeLetta:
            def __init__(self, **kwargs):
                self.agents = _FakeAgents()

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            LettaAgentLoop(
                mode="sdk",
                allow_local_fallback=False,
                llm_model="azure/gpt-5-mini",
                embedding="azure/text-embedding-3-large",
                context_window_limit=16000,
            )

        self.assertEqual(captured_create["context_window_limit"], 16000)

    def test_sdk_backend_uses_explicit_human_block_limit_when_provided(self):
        captured_create = {}

        class _FakeAgents:
            def create(self, **kwargs):
                captured_create.update(kwargs)
                return SimpleNamespace(id="agent-1")

        class _FakeLetta:
            def __init__(self, **kwargs):
                self.agents = _FakeAgents()

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            LettaAgentLoop(
                mode="sdk",
                allow_local_fallback=False,
                llm_model="azure/gpt-5-mini",
                embedding="azure/text-embedding-3-large",
                human_block_limit_chars=8000,
            )

        human_block = next(x for x in captured_create["memory_blocks"] if x["label"] == "human")
        self.assertEqual(human_block["limit"], 8000)

    def test_sdk_backend_ignores_human_block_limit_env_without_explicit_config(self):
        captured_create = {}

        class _FakeAgents:
            def create(self, **kwargs):
                captured_create.update(kwargs)
                return SimpleNamespace(id="agent-1")

        class _FakeLetta:
            def __init__(self, **kwargs):
                self.agents = _FakeAgents()

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            with mock.patch.dict(os.environ, {"LETTA_HUMAN_BLOCK_LIMIT_CHARS": "8000"}, clear=False):
                LettaAgentLoop(
                    mode="sdk",
                    allow_local_fallback=False,
                    llm_model="azure/gpt-5-mini",
                    embedding="azure/text-embedding-3-large",
                )

        human_block = next(x for x in captured_create["memory_blocks"] if x["label"] == "human")
        self.assertEqual(human_block["limit"], 100000)

    def test_sdk_backend_passes_explicit_client_timeout_when_provided(self):
        captured = {}

        class _FakeLetta:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.agents = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(id="agent-1"))

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            LettaAgentLoop(
                mode="sdk",
                allow_local_fallback=False,
                llm_model="azure/gpt-5-mini",
                embedding="azure/text-embedding-3-large",
                client_timeout_seconds=600,
            )

        self.assertEqual(captured["timeout"], 600.0)

    def test_sdk_backend_records_usage_from_messages_create(self):
        class _FakeResponse:
            def __init__(self):
                self.messages = [
                    {
                        "message_type": "assistant_message",
                        "content": [{"type": "text", "text": "{\"answer\": \"ok\"}"}],
                    }
                ]

            def to_dict(self):
                return {
                    "messages": self.messages,
                    "stop_reason": {"stop_reason": "end_turn"},
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "reasoning_tokens": 12,
                        "cached_input_tokens": 40,
                        "cache_write_tokens": 0,
                        "context_tokens": 150,
                        "total_tokens": 150,
                        "step_count": 1,
                        "run_ids": ["run-1"],
                    },
                }

        class _FakeAgents:
            def create(self, **kwargs):
                return SimpleNamespace(id="agent-1")

            class messages:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse()

        class _FakeLetta:
            def __init__(self, **kwargs):
                self.agents = _FakeAgents()

        fake_module = type(sys)("letta_client")
        fake_module.Letta = _FakeLetta

        with mock.patch.dict(sys.modules, {"letta_client": fake_module}):
            agent = LettaAgentLoop(
                mode="sdk",
                allow_local_fallback=False,
                llm_model="azure/gpt-5-mini",
                embedding="azure/text-embedding-3-large",
            )
            out = agent.ask_json("Reply with JSON only.")

        self.assertEqual(out, {"answer": "ok"})
        self.assertEqual(agent.usage_summary()["turn_count"], 1)
        self.assertEqual(agent.usage_summary()["prompt_tokens"], 120)
        self.assertEqual(agent.usage_summary()["completion_tokens"], 30)
        self.assertEqual(agent.usage_summary()["run_ids"], ["run-1"])

    def test_estimate_usage_cost_uses_uncached_and_completion_tokens(self):
        cost = estimate_usage_cost_usd(
            {
                "prompt_tokens": 3457,
                "completion_tokens": 104,
                "cached_input_tokens": 2816,
                "cache_write_tokens": 0,
            },
            prompt_cost_per_1m=0.275,
            completion_cost_per_1m=2.2,
        )
        self.assertAlmostEqual(cost["prompt_cost_usd"], 641 / 1_000_000 * 0.275, places=12)
        self.assertAlmostEqual(cost["completion_cost_usd"], 104 / 1_000_000 * 2.2, places=12)
        self.assertAlmostEqual(
            cost["total_cost_usd"],
            (641 / 1_000_000 * 0.275) + (104 / 1_000_000 * 2.2),
            places=12,
        )

    def test_v14_user1_config_dry_run_resolves(self):
        cmd = [
            "python3",
            "-m",
            "generation.run_tce_batch",
            "--config",
            "configs/experiments/tce/letta_v14_user1.yaml",
            "--dry-run",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), 1)
        resolved = payload[0]
        self.assertEqual(resolved["runtime"]["baseline"], "letta")
        self.assertEqual(resolved["runtime"]["experiment_name"], "letta_v14_user1_local_selfhost_ctx16k_h8k")
        self.assertEqual(resolved["runtime"]["run_id"], "main")
        self.assertEqual(resolved["runtime"]["run_name"], "letta_v14_user1_local_selfhost_ctx16k_h8k")
        self.assertEqual(resolved["runtime"]["user_id"], "001_user_001")
        self.assertTrue(resolved["runtime"]["enable_change_reasoning"])
        self.assertTrue(resolved["runtime"]["enable_rq3_apply_service_qa"])
        self.assertTrue(
            resolved["data"]["benchmark"].endswith(
                "tce_benchmark_vnext_20260319_formal_crv_task_packs_v14.json"
            )
        )
        self.assertEqual(resolved["llm"]["provider"], "azure")
        self.assertEqual(resolved["llm"]["model"], "azure/gpt-5-mini")
        self.assertEqual(resolved["llm"]["temperature"], 0.0)
        self.assertEqual(resolved["llm"]["top_p"], 1.0)
        self.assertEqual(resolved["llm"]["top_k"], None)
        self.assertTrue(
            resolved["output"]["prediction_path"].endswith(
                "/generation/letta/results/001_user_001/prediction/{}/tce_results_v14_taskabc.json".format(
                    resolved["runtime"]["run_name"]
                )
            )
        )
        self.assertFalse(resolved["final_qa"]["enabled"])
        self.assertFalse(resolved["final_qa"]["save_prompt_and_raw"])
        self.assertEqual(resolved["runtime"]["checkpoint_workers"], 1)
        self.assertEqual(resolved["runtime"]["within_checkpoint_workers"], 1)
        self.assertEqual(resolved["baseline_params"]["query_isolation_mode"], "checkpoint_snapshot")
        self.assertEqual(resolved["baseline_params"]["allow_local_fallback"], "False")
        self.assertEqual(resolved["baseline_params"]["embedding"], "azure/text-embedding-3-large")
        self.assertEqual(resolved["baseline_params"]["context_window_limit"], "16000")
        self.assertEqual(resolved["baseline_params"]["human_block_limit_chars"], "8000")
        self.assertEqual(resolved["baseline_params"]["client_timeout_seconds"], "600")
        self.assertTrue(
            resolved["output"]["prediction_path"].endswith(
                "/prediction/{}/tce_results_v14_taskabc.json".format(resolved["runtime"]["experiment_name"])
            )
        )
        self.assertTrue(
            resolved["baseline_params"]["checkpoint_agents_dir"].endswith(
                "/generation/letta/agents/{}/001_user_001".format(resolved["runtime"]["run_name"])
            )
        )

    def test_memgpt_v14_config_dry_run_resolves(self):
        cmd = [
            "python3",
            "-m",
            "generation.run_tce_batch",
            "--config",
            "configs/experiments/tce/memgpt_v14.yaml",
            "--dry-run",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), 1)
        resolved = payload[0]
        self.assertEqual(resolved["runtime"]["baseline"], "memgpt")
        self.assertEqual(resolved["runtime"]["experiment_name"], "memgpt_v14_user1_local_selfhost")
        self.assertEqual(resolved["runtime"]["run_id"], "main")
        self.assertEqual(resolved["runtime"]["run_name"], "memgpt_v14_user1_local_selfhost")
        self.assertEqual(resolved["runtime"]["user_id"], "001_user_001")
        self.assertEqual(resolved["llm"]["provider"], "azure")
        self.assertEqual(resolved["llm"]["model"], "azure/gpt-5-mini")
        self.assertEqual(resolved["llm"]["temperature"], 0.0)
        self.assertEqual(resolved["llm"]["top_p"], 1.0)
        self.assertEqual(resolved["llm"]["top_k"], None)
        self.assertTrue(resolved["runtime"]["enable_change_reasoning"])
        self.assertTrue(resolved["runtime"]["enable_rq3_apply_service_qa"])
        self.assertFalse(resolved["final_qa"]["enabled"])
        self.assertFalse(resolved["final_qa"]["save_prompt_and_raw"])
        self.assertEqual(resolved["baseline_params"]["allow_local_fallback"], "False")
        self.assertEqual(resolved["baseline_params"]["embedding"], "azure/text-embedding-3-large")
        self.assertTrue(
            resolved["baseline_params"]["checkpoint_agents_dir"].endswith(
                "/generation/memgpt/agents/{}/001_user_001".format(resolved["runtime"]["run_name"])
            )
        )
        self.assertNotIn("persona", resolved["baseline_params"])
        self.assertNotIn("human", resolved["baseline_params"])

    def test_baseline_route_class_splits_shared_pipeline_and_agent_loop(self):
        self.assertEqual(get_baseline_route_class("rag"), SHARED_PIPELINE_SNAPSHOT_ROUTE)
        self.assertEqual(get_baseline_route_class("letta"), AGENT_LOOP_ROUTE)
        self.assertEqual(get_baseline_route_class("memgpt"), AGENT_LOOP_ROUTE)

    def test_unified_pipeline_accepts_agent_loop_baseline_with_explicit_hooks(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "checkpoint_id": "cp1",
                                "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                                "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                                "state_completion_pack": {
                                    "version": "v1",
                                    "pack_authoring": "deterministic",
                                    "keys": {
                                        "habits_state:budget_review": {
                                            "item_id": "scp1",
                                            "state_key": "habits_state:budget_review",
                                            "question_text": "Infer budget review habit.",
                                            "answer_template": "<fill>",
                                            "retrieval_query": "Retrieve budget review habit.",
                                        }
                                    },
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            app_logs_path.write_text(
                json.dumps(
                    [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_pipeline(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                ask_json=lambda _: {},
                ask_structured=None,
                use_structured_response=False,
                close=lambda: None,
                prepare_checkpoint_state=lambda cp, memory_pool: CheckpointHandle(
                    checkpoint_id=str(cp.get("checkpoint_id") or ""),
                    state_kind="agent_snapshot",
                    state_ref={"checkpoint_id": cp.get("checkpoint_id"), "memory_pool_size": len(memory_pool)},
                    metadata={"retrieval_mode": "agent_memory"},
                ),
                retrieve_context_for_query=lambda checkpoint_handle, query_spec, retrieval_options, memory_pool: RetrievalResult(
                    mode="agent_memory",
                    inline_memory_blocks=[],
                    debug_metadata={
                        "retrieval_query": query_spec.retrieval_query_text,
                        "checkpoint_state_kind": checkpoint_handle.state_kind,
                    },
                ),
                answer_query=lambda checkpoint_handle, query_spec, retrieval_result: AnswerExecutionResult(
                    raw_output={
                        "snapshot_state": {"habits_state:budget_review": "monthly review"},
                        "evidence": {
                            "habits_state:budget_review": [
                                {
                                    "app_log_id": "log_0001",
                                    "evidence_content": "budget review monthly",
                                }
                            ]
                        },
                    },
                    prompt="agent-memory prompt",
                ),
                baseline_name="letta",
                resume=False,
                max_checkpoints=None,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
            )

        prediction = result["predictions"][0]
        self.assertEqual(prediction["snapshot_state"]["habits_state:budget_review"], "monthly review")
        self.assertEqual(
            prediction["metadata"]["per_key_retrieval"][0]["retrieval_metadata"]["checkpoint_state_kind"],
            "agent_snapshot",
        )
        self.assertEqual(prediction["metadata"]["prompt"], ["agent-memory prompt"])

    def test_checkpoint_snapshot_isolates_each_key(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {
                        "habits_state": {"budget_review": "monthly review"},
                        "preferences_state": {"learning_modality": {"statement": "self-paced webinars"}},
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            },
                            "preferences_state:learning_modality": {
                                "item_id": "scp2",
                                "state_key": "preferences_state:learning_modality",
                                "question_text": "Infer learning modality.",
                                "answer_template": {"statement": "<fill>"},
                                "retrieval_query": "Infer learning modality.",
                            },
                        },
                    },
                }
            ],
        }
        app_logs = [{"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"}]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            result = letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )

            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            usage_cost = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))

            builder = _FakeLettaAgentLoop.instances[-1]
            self.assertEqual(builder.ingested_ids, ["log_0001"])
            self.assertEqual(len(builder.imported_payloads), 2)
            self.assertEqual(len(builder.deleted_ids), 2)
            self.assertTrue(any(path.endswith("cp1.af") for path in builder.saved_paths))
            self.assertTrue(any(path.endswith("final_1.af") for path in builder.saved_paths))
            self.assertEqual(progress["builder_agent_id"], "builder-agent")
            self.assertEqual(progress["ingested_until_log_index"], 0)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0001")
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(usage_cost["llm_provider"], "azure")
            self.assertEqual(usage_cost["llm_model"], "myazure/gpt-5-mini")
            self.assertEqual(usage_cost["usage_summary"]["prompt_tokens"], 120)
            self.assertEqual(usage_cost["usage_summary"]["completion_tokens"], 30)
            self.assertIsNotNone(usage_cost["estimated_cost_usd"])
            self.assertIn("total_cost_usd", usage_cost["estimated_cost_usd"])
        pred = result["predictions"][0]
        self.assertEqual(pred["metadata"]["retrieval_mode"], "per_key_isolated")
        self.assertEqual(
            pred["metadata"]["per_key_retrieval"][0]["retrieval_metadata"]["retrieval_mode"],
            "letta_checkpoint_snapshot",
        )
        self.assertEqual(
            pred["metadata"]["per_key_retrieval"][0]["retrieval_metadata"]["query_isolation_mode"],
            "checkpoint_snapshot",
        )
        self.assertEqual(pred["snapshot_state"]["habits_state:budget_review"], "monthly review")
        self.assertEqual(
            pred["snapshot_state"]["preferences_state:learning_modality"],
            {"statement": "self-paced webinars"},
        )

    def test_resume_loads_latest_checkpoint_snapshot_when_local_builder_progress_is_unavailable(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                },
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-02 07:00:00"},
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=False,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )
            first_builder = _FakeLettaAgentLoop.instances[-1]
            self.assertEqual(first_builder.ingested_ids, ["log_0001"])

            (agents_dir / "builder_progress.json").write_text(
                json.dumps(
                    {
                        "builder_agent_id": None,
                        "ingested_until_log_index": -1,
                        "last_ingested_app_log_id": None,
                        "ingested_log_count": 0,
                        "status": "initialized",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=True,
                max_checkpoints=2,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )
            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            second_builder = _FakeLettaAgentLoop.instances[-1]
            self.assertTrue(any(path.endswith(".af") for path in second_builder.loaded_paths))
            self.assertEqual(second_builder.attached_ids, [])
            self.assertEqual(second_builder.ingested_ids, ["log_0001", "log_0002"])
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["ingested_until_log_index"], 1)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0002")

    def test_resume_reuses_existing_builder_from_progress_without_snapshot(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 1},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-01 07:30:00"},
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            agents_dir.mkdir(parents=True, exist_ok=True)
            _FakeLettaAgentLoop.agent_records_by_id["builder-existing"] = [dict(app_logs[0])]
            (agents_dir / "builder_progress.json").write_text(
                json.dumps(
                    {
                        "builder_agent_id": "builder-existing",
                        "ingested_until_log_index": 0,
                        "last_ingested_app_log_id": "log_0001",
                        "ingested_log_count": 1,
                        "status": "ingesting",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=True,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )
            builder = _FakeLettaAgentLoop.instances[-1]
            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(builder.attached_ids, ["builder-existing"])
            self.assertEqual(builder.ingested_ids, ["log_0002"])
            self.assertEqual(progress["builder_agent_id"], "builder-existing")
            self.assertEqual(progress["ingested_until_log_index"], 1)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0002")

    def test_ingest_failure_marks_progress_uncertain_without_regressing_confirmed_index(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 1},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-01 07:30:00"},
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "generation.letta.tce.LettaAgentLoop",
            _UncertainFailFakeLettaAgentLoop,
        ):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "simulated ambiguous ingest failure"):
                letta_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="myazure/gpt-5-mini",
                    letta_mode="sdk",
                    allow_local_fallback=False,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    query_isolation_mode="checkpoint_snapshot",
                    checkpoint_agents_dir=agents_dir,
                )

            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "ingest_uncertain")
            self.assertEqual(progress["ingested_until_log_index"], 0)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0001")
            self.assertEqual(progress["uncertain_log_index"], 1)
            self.assertEqual(progress["uncertain_app_log_id"], "log_0002")

    def test_resume_prefers_local_progress_over_snapshot(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                },
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-02 07:00:00"},
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=False,
                max_checkpoints=2,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )

            (agents_dir / "builder_progress.json").write_text(
                json.dumps(
                    {
                        "builder_agent_id": "builder-stale",
                        "ingested_until_log_index": 0,
                        "last_ingested_app_log_id": "log_0001",
                        "ingested_log_count": 1,
                        "status": "ingesting",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _FakeLettaAgentLoop.agent_records_by_id["builder-stale"] = [dict(app_logs[0])]
            output_path.unlink()

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=True,
                max_checkpoints=2,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )

            builder = _FakeLettaAgentLoop.instances[-1]
            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(builder.attached_ids, ["builder-stale"])
            self.assertEqual(builder.loaded_paths, [])
            self.assertEqual(builder.ingested_ids, ["log_0002"])
            self.assertEqual(progress["ingested_until_log_index"], 1)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0002")

    def test_resume_attaches_recorded_builder_without_live_verification(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 1},
                    "validated_snapshot_state": {"habits_state": {"budget_review": "monthly review"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "habits_state:budget_review": {
                                "item_id": "scp1",
                                "state_key": "habits_state:budget_review",
                                "question_text": "Infer budget review habit.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer budget review habit.",
                            }
                        },
                    },
                }
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-01 07:30:00"},
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            agents_dir.mkdir(parents=True, exist_ok=True)
            _FakeLettaAgentLoop.agent_records_by_id["builder-existing"] = [dict(app_logs[1])]
            (agents_dir / "builder_progress.json").write_text(
                json.dumps(
                    {
                        "builder_agent_id": "builder-existing",
                        "ingested_until_log_index": 0,
                        "last_ingested_app_log_id": "log_0001",
                        "ingested_log_count": 1,
                        "status": "ingesting",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=True,
                max_checkpoints=1,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )

            builder = _FakeLettaAgentLoop.instances[-1]
            progress = json.loads((agents_dir / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(builder.attached_ids, ["builder-existing"])
            self.assertEqual(builder.ingested_ids, ["log_0002"])
            self.assertEqual(progress["ingested_until_log_index"], 1)
            self.assertEqual(progress["last_ingested_app_log_id"], "log_0002")

    def test_builder_lock_blocks_concurrent_reuse(self):
        benchmark = {"user_id": "001_user_001", "checkpoints": []}
        app_logs = []

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            (agents_dir / "builder.lock.json").write_text(
                json.dumps(
                    {
                        "session_token": "other-session",
                        "owner_pid": os.getpid(),
                        "owner_hostname": socket.gethostname(),
                        "owner_started_at": "2025-01-01T00:00:00Z",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "builder lock already held"):
                letta_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="myazure/gpt-5-mini",
                    letta_mode="sdk",
                    allow_local_fallback=False,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    checkpoint_agents_dir=agents_dir,
                )

    def test_adapter_forwards_change_reasoning_and_final_qa_options(self):
        captured = {}

        def _fake_run_generation(**kwargs):
            captured.update(kwargs)
            return {"predictions": []}

        args = TceAdapterArgs(
            baseline="memgpt",
            user_id=None,
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/prediction.json"),
            llm_provider="azure",
            llm_model="azure/gpt-5-mini",
            llm_max_workers=1,
            enable_change_reasoning=True,
            enable_final_qa=True,
            final_qa_path="/tmp/qa.json",
            final_qa_output_path="/tmp/final_qa.json",
            final_qa_save_prompt_and_raw=True,
        )

        with mock.patch("generation.letta.tce.run_generation", side_effect=_fake_run_generation):
            letta_adapter.run(args)

        self.assertTrue(captured["enable_change_reasoning"])
        self.assertTrue(captured["enable_final_qa"])
        self.assertEqual(captured["final_qa_path"], "/tmp/qa.json")
        self.assertEqual(captured["final_qa_output_path"], "/tmp/final_qa.json")
        self.assertTrue(captured["final_qa_save_prompt_and_raw"])
        self.assertEqual(captured["answer_temperature"], 0.0)
        self.assertEqual(captured["answer_top_p"], 1.0)
        self.assertEqual(captured["answer_top_k"], None)
        self.assertEqual(captured["baseline_name"], "memgpt")
        self.assertNotIn("rq3_apply_retrieval_top_k", captured)
        self.assertNotIn("final_qa_retrieval_top_k", captured)
        self.assertNotIn("llm_max_workers", captured)
        self.assertNotIn("persona", captured)
        self.assertNotIn("human", captured)

    def test_build_from_config_reads_runtime_change_reasoning_and_rejects_legacy_baseline_param(self):
        args = build_from_config(
            {
                "runtime": {
                    "baseline": "letta",
                    "enable_change_reasoning": True,
                },
                "data": {
                    "benchmark": "/tmp/benchmark.json",
                    "app_logs_path": "/tmp/app_logs.json",
                },
                "output": {
                    "prediction_path": "/tmp/prediction.json",
                },
                "llm": {
                    "provider": "azure",
                    "model": "azure/gpt-5-mini",
                    "max_workers": 1,
                },
                "baseline_params": {},
            }
        )
        self.assertTrue(args.enable_change_reasoning)

        with self.assertRaisesRegex(
            ValueError,
            "Legacy baseline_params keys are no longer supported: enable_change_reasoning",
        ):
            build_from_config(
                {
                    "runtime": {
                        "baseline": "letta",
                    },
                    "data": {
                        "benchmark": "/tmp/benchmark.json",
                        "app_logs_path": "/tmp/app_logs.json",
                    },
                    "output": {
                        "prediction_path": "/tmp/prediction.json",
                    },
                    "baseline_params": {
                        "enable_change_reasoning": True,
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "rq3_apply_fail_on_missing_pack is no longer supported"):
            build_from_config(
                {
                    "runtime": {
                        "baseline": "letta",
                        "rq3_apply_fail_on_missing_pack": True,
                    },
                    "data": {
                        "benchmark": "/tmp/benchmark.json",
                        "app_logs_path": "/tmp/app_logs.json",
                    },
                    "output": {
                        "prediction_path": "/tmp/prediction.json",
                    },
                }
            )

    def test_checkpoint_snapshot_supports_final_qa(self):
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp1",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0},
                    "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp1",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer favorite coffee.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer favorite coffee.",
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "previous_cutoff_ts": "",
                        "keys": {},
                    },
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_category": "drink recommendation",
                                        "question": "What drink should the assistant recommend?",
                                        "reference_answer": "Recommend latte",
                                        "retrieval_query": "What drink should the assistant recommend?",
                                    }
                                ]
                            }
                        },
                    },
                },
                {
                    "checkpoint_id": "cp2",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1, "app_log_id": "log_0002"},
                    "validated_snapshot_state": {"profile_state": {"favorite_coffee": "espresso"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "pack_authoring": "deterministic",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp2",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer favorite coffee.",
                                "answer_template": "<fill>",
                                "retrieval_query": "Infer favorite coffee.",
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "previous_cutoff_ts": "2025-01-01 08:00:00",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "ctp1",
                                "question_text": "How did favorite coffee change?",
                                "retrieval_query": "How did favorite coffee change?",
                                "before_template": "<fill>",
                                "after_template": "<fill>",
                            }
                        },
                    },
                    "rq3_apply_service_qa": {
                        "version": "v1",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "items": [
                                    {
                                        "qa_id": "q1",
                                        "service_category": "drink recommendation",
                                        "question": "What drink should the assistant recommend?",
                                        "reference_answer": "Recommend espresso",
                                        "retrieval_query": "What drink should the assistant recommend?",
                                    }
                                ]
                            }
                        },
                    },
                },
            ],
        }
        app_logs = [
            {"app_log_id": "log_0001", "timestamp": "2025-01-01 07:00:00"},
            {"app_log_id": "log_0002", "timestamp": "2025-01-02 07:00:00"},
        ]
        qa_list = [
            {
                "id": "qa1",
                "query": "What drink should the assistant recommend?",
                "reference": "Recommend espresso",
                "metadata": {"app_log_ids": ["log_0002"]},
            }
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch("generation.letta.tce.LettaAgentLoop", _FakeLettaAgentLoop):
            td_path = Path(td)
            benchmark_path = td_path / "benchmark.json"
            app_logs_path = td_path / "app_logs.json"
            qa_path = td_path / "qa.json"
            output_path = td_path / "prediction.json"
            agents_dir = td_path / "agents"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

            result = letta_tce.run_generation(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                max_visible_logs=None,
                llm_provider="azure",
                llm_model="myazure/gpt-5-mini",
                letta_mode="sdk",
                allow_local_fallback=False,
                resume=False,
                max_checkpoints=2,
                debug=False,
                debug_dir=None,
                save_prompt_and_raw=True,
                enable_change_reasoning=True,
                enable_rq3_apply_service_qa=True,
                rq3_apply_save_prompt_and_raw=True,
                checkpoint_workers=1,
                within_checkpoint_workers=1,
                save_every_generation_keys=1,
                enable_final_qa=True,
                final_qa_path=str(qa_path),
                query_isolation_mode="checkpoint_snapshot",
                checkpoint_agents_dir=agents_dir,
            )

            final_qa_results = json.loads(output_path.with_name("prediction_final_qa.json").read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"][-1]["change_analysis"]["profile_state:favorite_coffee"]["after"], "espresso")
        self.assertEqual(
            result["predictions"][-1]["rq3_apply_answers"]["profile_state:favorite_coffee"]["items"][0]["answer"],
            "Recommend espresso",
        )
        self.assertEqual(result["final_qa"]["final_checkpoint_id"], "cp2")
        self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
        self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
