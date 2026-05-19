#!/usr/bin/env python3
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.MemoryOS import tce_adapter as memoryos_tce


class _FakeLLMClient:
    def __init__(self, *args, **kwargs):
        return None

    def ask(self, prompt: str, response_type: str = "json"):
        target_key = (
            "profile_state:backup_coffee"
            if "profile_state:backup_coffee" in prompt
            else "profile_state:favorite_coffee"
        )
        return {
            "snapshot_state": {target_key: "latte"},
            "evidence": {
                target_key: [
                    {
                        "app_log_id": "log_0001",
                        "evidence_content": "favorite_coffee latte",
                    }
                ]
            },
        }

    def ask_structured(self, prompt: str, text_format):
        return self.ask(prompt, response_type="json")

    def supports_structured_response(self):
        return True

    def usage_summary(self):
        return {
            "turn_count": 1,
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": 18,
        }

    def close(self):
        return None


class _FakeJsonStore:
    def __init__(self, file_path: Path):
        self.file_path = str(file_path)
        self.memory = []
        self.sessions = {}
        self.access_frequency = {}
        self.user_profiles = {}
        self.knowledge_base = []
        self.assistant_knowledge = []

    def load(self):
        payload = json.loads(Path(self.file_path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            self.memory = payload
            return
        if isinstance(payload, dict):
            self.sessions = payload.get("sessions", {})
            self.access_frequency = payload.get("access_frequency", {})
            self.user_profiles = payload.get("user_profiles", {})
            self.knowledge_base = payload.get("knowledge_base", [])
            self.assistant_knowledge = payload.get("assistant_knowledge", [])

    def get_all(self):
        return list(self.memory)

    def get_raw_user_profile(self, user_id: str):
        return (self.user_profiles.get(user_id) or {}).get("data", "None")


class _FakeMemoryOSRetriever:
    def __init__(self, owner):
        self.owner = owner

    def retrieve_context(self, user_query: str, user_id: str):
        return {"retrieved_pages": list(self.owner.short_term_memory.memory)}


class _FakeMemoryOSClient:
    def __init__(self, owner):
        self.owner = owner

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        del model, temperature, max_tokens
        prompt = messages[-1]["content"]
        payload = self.owner._response_payload(prompt)
        return json.dumps(payload, ensure_ascii=False)


class _FakeMemoryOS:
    def __init__(self, data_storage_root: Path, memory_user_id: str):
        root = Path(data_storage_root)
        root.mkdir(parents=True, exist_ok=True)
        self.user_id = memory_user_id
        self.llm_model = "gpt-5-mini"
        self.short_term_memory = _FakeJsonStore(root / "short_term.json")
        self.mid_term_memory = _FakeJsonStore(root / "mid_term.json")
        self.user_long_term_memory = _FakeJsonStore(root / "long_term_user.json")
        self.assistant_long_term_memory = _FakeJsonStore(root / "long_term_assistant.json")
        self.retriever = _FakeMemoryOSRetriever(self)
        self.client = _FakeMemoryOSClient(self)

    def _favorite_coffee(self) -> str:
        for item in reversed(self.short_term_memory.memory):
            user_input = str(item.get("user_input", "")).strip()
            if user_input.startswith("[APP_LOG] "):
                user_input = user_input.split("[APP_LOG] ", 1)[1].strip()
            try:
                payload = json.loads(user_input)
            except Exception:
                continue
            response = payload.get("response") if isinstance(payload, dict) else None
            if isinstance(response, dict) and response.get("favorite_coffee"):
                return str(response["favorite_coffee"])
        return "latte"

    def _last_app_log_id(self) -> str:
        for item in reversed(self.short_term_memory.memory):
            user_input = str(item.get("user_input", "")).strip()
            if user_input.startswith("[APP_LOG] "):
                user_input = user_input.split("[APP_LOG] ", 1)[1].strip()
            try:
                payload = json.loads(user_input)
            except Exception:
                continue
            app_log_id = payload.get("app_log_id") if isinstance(payload, dict) else None
            if app_log_id:
                return str(app_log_id)
        return "log_0001"

    def _response_payload(self, prompt: str):
        coffee = self._favorite_coffee()
        app_log_id = self._last_app_log_id()
        target_key = "profile_state:backup_coffee" if "profile_state:backup_coffee" in prompt else "profile_state:favorite_coffee"
        if '"change_analysis"' in prompt:
            return {"change_analysis": {}}
        if '"user_state"' in prompt or '"snapshot_state"' in prompt:
            return {
                "snapshot_state": {target_key: coffee},
                "evidence": {
                    target_key: [
                        {"app_log_id": app_log_id, "evidence_content": f"favorite_coffee {coffee}"}
                    ]
                },
            }
        return {
            "answer": f"Recommend {coffee}",
            "evidence": [{"app_log_id": app_log_id, "evidence_content": f"favorite_coffee {coffee}"}],
        }

    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, meta_data: dict = None):
        del meta_data
        self.short_term_memory.memory.append(
            {
                "user_input": user_input,
                "agent_response": agent_response,
                "timestamp": timestamp or "",
            }
        )


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class MemoryosTceConfigTest(unittest.TestCase):
    def test_memoryos_usage_summary_reads_reasoning_tokens_from_completion_details(self) -> None:
        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.SentenceTransformer = object
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModel = object
        fake_transformers.AutoTokenizer = object
        with mock.patch.dict(
            sys.modules,
            {
                "sentence_transformers": fake_sentence_transformers,
                "transformers": fake_transformers,
            },
        ):
            utils_mod = memoryos_tce._load_memoryos_utils_module()
            utils_mod.reset_usage_tracker()
            try:
                utils_mod._record_usage(
                    "chat_completion",
                    "gpt-5-mini",
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 6,
                        "completion_tokens_details": {"reasoning_tokens": 4},
                        "total_tokens": 16,
                    },
                )
                summary = utils_mod.get_usage_summary()
            finally:
                utils_mod.reset_usage_tracker()

        self.assertEqual(summary["prompt_tokens"], 10)
        self.assertEqual(summary["completion_tokens"], 6)
        self.assertEqual(summary["reasoning_tokens"], 4)
        self.assertEqual(summary["total_tokens"], 16)

    def test_query_time_memoryos_instance_uses_shared_retriever_provider(self) -> None:
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": "2025-01-01 08:00:00",
                        "log_index": 0,
                        "app_log_id": "log_0001",
                    },
                    "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp_1",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer the user's current favorite coffee.",
                                "answer_template": "<fill the blank>",
                                "retrieval_query": "Infer the user's current favorite coffee.",
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "version": "v1",
                        "previous_checkpoint_id": "",
                        "previous_cutoff_ts": "",
                        "keys": {},
                    },
                }
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
            output_path = root / "prediction.json"
            snapshot_root = root / "memoryos_snapshots"
            snapshot_dir = snapshot_root / "snap1"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_json(
                snapshot_root / "manifest.json",
                {
                    "snapshots": [
                        {
                            "snapshot_id": "snap1",
                            "created_at": "2025-01-01T09:00:00",
                            "last_event_idx": 0,
                            "checkpoint_id": "cp_0001",
                            "checkpoint_app_log_id": "log_0001",
                            "short_term_path": "snap1/short_term.json",
                            "mid_term_path": "snap1/mid_term.json",
                            "long_term_path": "snap1/long_term_user.json",
                            "assistant_long_term_path": "snap1/long_term_assistant.json",
                        }
                    ]
                },
            )
            _write_json(
                snapshot_dir / "short_term.json",
                [{"user_input": "[APP_LOG] {}".format(json.dumps(app_logs[0], ensure_ascii=False))}],
            )
            _write_json(snapshot_dir / "mid_term.json", {"sessions": {}, "access_frequency": {}})
            _write_json(
                snapshot_dir / "long_term_user.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )
            _write_json(
                snapshot_dir / "long_term_assistant.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )

            seen_build_kwargs = []

            def _fake_memoryos_instance(**kwargs):
                seen_build_kwargs.append(dict(kwargs))
                return _FakeMemoryOS(
                    data_storage_root=Path(kwargs["data_storage_root"]),
                    memory_user_id=str(kwargs["memory_user_id"]),
                )

            build_usage = {
                "request_count": 3,
                "chat_request_count": 1,
                "embedding_request_count": 2,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "reasoning_tokens": 12,
                "total_tokens": 150,
                "by_model": [],
            }
            retrieval_usage = {
                "request_count": 1,
                "chat_request_count": 0,
                "embedding_request_count": 1,
                "prompt_tokens": 25,
                "completion_tokens": 0,
                "total_tokens": 25,
                "by_model": [],
            }
            usage_sequence = iter([build_usage, build_usage, retrieval_usage])

            def _fake_usage_summary():
                try:
                    return next(usage_sequence)
                except StopIteration:
                    return retrieval_usage

            with mock.patch("generation.MemoryOS.tce_adapter.LLMClient", _FakeLLMClient), mock.patch(
                "generation.MemoryOS.tce_adapter._ensure_snapshots_for_benchmark",
                lambda **kwargs: snapshot_root,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._build_memoryos_instance",
                _fake_memoryos_instance,
            ):
                result = memoryos_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    data_storage_path=None,
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_model_name="text-embedding-3-large",
                    llm_controller_model="gpt-5-mini",
                    assistant_id="assistant",
                    retriever_provider="azure",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

        self.assertEqual(len(seen_build_kwargs), 1)
        self.assertEqual(seen_build_kwargs[0]["retriever_provider"], "azure")
        self.assertEqual(seen_build_kwargs[0]["retrieval_queue_capacity"], 5)
        retrieval_meta = result["predictions"][0]["metadata"]["per_key_retrieval"][0]["retrieval_metadata"]
        self.assertEqual(retrieval_meta["retriever_provider"], "azure")
        self.assertEqual(retrieval_meta["retrieval_mode"], "memoryos_snapshot")
        self.assertEqual(retrieval_meta["retrieval_queue_capacity"], 5)
        self.assertIn("[Retrieved Historical Memory]", result["predictions"][0]["metadata"]["prompt"][0])
        self.assertIn("memoryos_get_response_text", retrieval_meta["inline_memory_format"])
        self.assertNotIn("Use your memory about the user", result["predictions"][0]["metadata"]["prompt"][0])

    def test_run_generation_writes_usage_sidecar_and_build_timing(self) -> None:
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": "2025-01-01 08:00:00",
                        "log_index": 0,
                        "app_log_id": "log_0001",
                    },
                    "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp_1",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer the user's current favorite coffee.",
                                "answer_template": "<fill the blank>",
                                "retrieval_query": "Infer the user's current favorite coffee.",
                            }
                        },
                    },
                    "change_tracking_pack": {
                        "version": "v1",
                        "previous_checkpoint_id": "",
                        "previous_cutoff_ts": "",
                        "keys": {},
                    },
                }
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
            output_path = root / "prediction.json"
            snapshot_root = root / "memoryos_snapshots"
            snapshot_dir = snapshot_root / "snap1"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_json(
                snapshot_root / "manifest.json",
                {
                    "snapshots": [
                        {
                            "snapshot_id": "snap1",
                            "created_at": "2025-01-01T09:00:00",
                            "last_event_idx": 0,
                            "checkpoint_id": "cp_0001",
                            "checkpoint_app_log_id": "log_0001",
                            "short_term_path": "snap1/short_term.json",
                            "mid_term_path": "snap1/mid_term.json",
                            "long_term_path": "snap1/long_term_user.json",
                            "assistant_long_term_path": "snap1/long_term_assistant.json",
                        }
                    ]
                },
            )
            _write_json(
                snapshot_dir / "short_term.json",
                [{"user_input": json.dumps(app_logs[0], ensure_ascii=False)}],
            )
            _write_json(snapshot_dir / "mid_term.json", {"sessions": {}, "access_frequency": {}})
            _write_json(
                snapshot_dir / "long_term_user.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )
            _write_json(
                snapshot_dir / "long_term_assistant.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )

            build_usage = {
                "request_count": 3,
                "chat_request_count": 1,
                "embedding_request_count": 2,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "reasoning_tokens": 12,
                "total_tokens": 150,
                "by_model": [],
            }
            retrieval_usage = {
                "request_count": 1,
                "chat_request_count": 0,
                "embedding_request_count": 1,
                "prompt_tokens": 25,
                "completion_tokens": 0,
                "total_tokens": 25,
                "by_model": [],
            }
            usage_sequence = iter([build_usage, build_usage, retrieval_usage])

            def _fake_usage_summary():
                try:
                    return next(usage_sequence)
                except StopIteration:
                    return retrieval_usage

            with mock.patch("generation.MemoryOS.tce_adapter.LLMClient", _FakeLLMClient), mock.patch(
                "generation.MemoryOS.tce_adapter._ensure_snapshots_for_benchmark",
                lambda **kwargs: snapshot_root,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._build_memoryos_instance",
                lambda **kwargs: _FakeMemoryOS(
                    data_storage_root=Path(kwargs["data_storage_root"]),
                    memory_user_id=str(kwargs["memory_user_id"]),
                ),
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._reset_memoryos_usage_tracker",
                lambda: None,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._memoryos_usage_summary",
                side_effect=_fake_usage_summary,
            ):
                memoryos_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    data_storage_path=None,
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_model_name="text-embedding-3-large",
                    llm_controller_model="gpt-5-mini",
                    assistant_id="assistant",
                    retriever_provider="azure",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

            sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))

        self.assertEqual(sidecar["llm_provider"], "azure")
        self.assertEqual(sidecar["llm_model"], "gpt-5-mini")
        self.assertEqual(sidecar["retriever_provider"], "azure")
        self.assertEqual(sidecar["embedding_model"], "text-embedding-3-large")
        self.assertGreaterEqual(float(sidecar["timing"]["build_memory_duration_s"]), 0.0)
        self.assertEqual(sidecar["usage"]["build_memory"]["request_count"], 3)
        self.assertEqual(sidecar["usage"]["build_memory"]["reasoning_tokens"], 12)
        self.assertEqual(sidecar["usage"]["retrieval"]["request_count"], 1)
        self.assertEqual(sidecar["usage"]["answer_llm"]["prompt_tokens"], 11)
        self.assertEqual(sidecar["usage"]["answer_llm"]["completion_tokens"], 7)

    def test_run_generation_build_only_stops_after_build_and_writes_sidecar(self) -> None:
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": "2025-01-01 08:00:00",
                        "log_index": 0,
                        "app_log_id": "log_0001",
                    },
                }
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
            output_path = root / "prediction.json"
            snapshot_root = root / "memoryos_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            build_usage = {
                "request_count": 3,
                "chat_request_count": 1,
                "embedding_request_count": 2,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "reasoning_tokens": 12,
                "total_tokens": 150,
                "by_model": [],
            }
            with mock.patch(
                "generation.MemoryOS.tce_adapter._ensure_snapshots_for_benchmark",
                lambda **kwargs: snapshot_root,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter.LLMClient",
                side_effect=AssertionError("LLMClient should not be constructed in build_only mode"),
            ), mock.patch(
                "generation.MemoryOS.tce_adapter.run_pipeline",
                side_effect=AssertionError("run_pipeline should not be called in build_only mode"),
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._reset_memoryos_usage_tracker",
                lambda: None,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._memoryos_usage_summary",
                return_value=build_usage,
            ):
                result = memoryos_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    data_storage_path=None,
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_model_name="text-embedding-3-large",
                    llm_controller_model="gpt-5-mini",
                    assistant_id="assistant",
                    retriever_provider="azure",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    build_only=True,
                )

            sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"], [])
        self.assertTrue(result["build_only"]["enabled"])
        self.assertEqual(result["build_only"]["requested_checkpoint_ids"], ["cp_0001"])
        self.assertEqual(result["build_only"]["snapshot_root"], str(snapshot_root))
        self.assertFalse(output_path.exists())
        self.assertEqual(sidecar["usage"]["build_memory"]["request_count"], 3)
        self.assertEqual(sidecar["usage"]["build_memory"]["reasoning_tokens"], 12)
        self.assertEqual(sidecar["usage"]["retrieval"]["request_count"], 0)
        self.assertEqual(sidecar["usage"]["answer_llm"]["request_count"], 0)
        self.assertGreaterEqual(float(sidecar["timing"]["build_memory_duration_s"]), 0.0)
        self.assertEqual(float(sidecar["timing"]["generation_duration_s"]), 0.0)

    def test_query_time_uses_fresh_isolated_snapshot_per_key(self) -> None:
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": "2025-01-01 08:00:00",
                        "log_index": 0,
                        "app_log_id": "log_0001",
                    },
                    "validated_snapshot_state": {
                        "profile_state": {
                            "favorite_coffee": "latte",
                            "backup_coffee": "latte",
                        }
                    },
                    "state_completion_pack": {
                        "version": "v1",
                        "keys": {
                            "profile_state:favorite_coffee": {
                                "item_id": "scp_1",
                                "state_key": "profile_state:favorite_coffee",
                                "question_text": "Infer the user's current favorite coffee.",
                                "answer_template": "<fill the blank>",
                                "retrieval_query": "Infer the user's current favorite coffee.",
                            },
                            "profile_state:backup_coffee": {
                                "item_id": "scp_2",
                                "state_key": "profile_state:backup_coffee",
                                "question_text": "Infer the user's backup coffee.",
                                "answer_template": "<fill the blank>",
                                "retrieval_query": "Infer the user's backup coffee.",
                            },
                        },
                    },
                    "change_tracking_pack": {"version": "v1", "previous_checkpoint_id": "", "previous_cutoff_ts": "", "keys": {}},
                }
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
            output_path = root / "prediction.json"
            snapshot_root = root / "memoryos_snapshots"
            snapshot_dir = snapshot_root / "snap1"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            original_short_term = [{"user_input": "[APP_LOG] {}".format(json.dumps(app_logs[0], ensure_ascii=False))}]
            _write_json(
                snapshot_root / "manifest.json",
                {
                    "snapshots": [
                        {
                            "snapshot_id": "snap1",
                            "created_at": "2025-01-01T09:00:00",
                            "last_event_idx": 0,
                            "checkpoint_id": "cp_0001",
                            "checkpoint_app_log_id": "log_0001",
                            "short_term_path": "snap1/short_term.json",
                            "mid_term_path": "snap1/mid_term.json",
                            "long_term_path": "snap1/long_term_user.json",
                            "assistant_long_term_path": "snap1/long_term_assistant.json",
                        }
                    ]
                },
            )
            _write_json(snapshot_dir / "short_term.json", original_short_term)
            _write_json(snapshot_dir / "mid_term.json", {"sessions": {}, "access_frequency": {}})
            _write_json(snapshot_dir / "long_term_user.json", {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []})
            _write_json(snapshot_dir / "long_term_assistant.json", {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []})

            seen_build_kwargs = []

            def _fake_memoryos_instance(**kwargs):
                seen_build_kwargs.append(dict(kwargs))
                return _FakeMemoryOS(
                    data_storage_root=Path(kwargs["data_storage_root"]),
                    memory_user_id=str(kwargs["memory_user_id"]),
                )

            with mock.patch("generation.MemoryOS.tce_adapter.LLMClient", _FakeLLMClient), mock.patch(
                "generation.MemoryOS.tce_adapter._ensure_snapshots_for_benchmark",
                lambda **kwargs: snapshot_root,
            ), mock.patch(
                "generation.MemoryOS.tce_adapter._build_memoryos_instance",
                _fake_memoryos_instance,
            ):
                result = memoryos_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    data_storage_path=None,
                    retrieval_top_k=20,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_model_name="text-embedding-3-large",
                    llm_controller_model="gpt-5-mini",
                    assistant_id="assistant",
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

            short_term_after = json.loads((snapshot_dir / "short_term.json").read_text(encoding="utf-8"))

        prediction = result["predictions"][0]
        self.assertEqual(prediction["snapshot_state"]["profile_state:favorite_coffee"], "latte")
        self.assertEqual(prediction["snapshot_state"]["profile_state:backup_coffee"], "latte")
        self.assertEqual(short_term_after, original_short_term)
        self.assertEqual(len(seen_build_kwargs), 2)
        self.assertEqual(seen_build_kwargs[0]["retrieval_queue_capacity"], 20)
        self.assertEqual(seen_build_kwargs[1]["retrieval_queue_capacity"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
