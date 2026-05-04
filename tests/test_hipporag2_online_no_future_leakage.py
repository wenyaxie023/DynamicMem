#!/usr/bin/env python3
import importlib
import json
import numpy as np
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.adapters import hipporag2 as hipporag2_adapter
from generation.adapters.base import TceAdapterArgs


def _install_import_stubs():
    openai_mod = types.ModuleType("openai")

    class _DummyOpenAIClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    openai_mod.OpenAI = _DummyOpenAIClient
    openai_mod.AzureOpenAI = _DummyOpenAIClient

    transformers_mod = types.ModuleType("transformers")
    transformers_utils_mod = types.ModuleType("transformers.utils")
    transformers_import_utils_mod = types.ModuleType("transformers.utils.import_utils")
    transformers_modeling_utils_mod = types.ModuleType("transformers.modeling_utils")
    transformers_import_utils_mod.check_torch_load_is_safe = lambda *args, **kwargs: True
    transformers_modeling_utils_mod.check_torch_load_is_safe = lambda *args, **kwargs: True
    transformers_utils_mod.import_utils = transformers_import_utils_mod
    transformers_mod.utils = transformers_utils_mod
    transformers_mod.modeling_utils = transformers_modeling_utils_mod

    hipporag_mod = types.ModuleType("hipporag")
    hipporag_mod.HippoRAG = object
    hipporag_utils_mod = types.ModuleType("hipporag.utils")
    hipporag_config_utils_mod = types.ModuleType("hipporag.utils.config_utils")
    hipporag_embedding_store_mod = types.ModuleType("hipporag.embedding_store")

    class _FakeBaseConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FakeEmbeddingStore:
        def __init__(self, embedding_model, db_filename, batch_size, namespace):
            del embedding_model, batch_size
            self.root = Path(db_filename)
            self.root.mkdir(parents=True, exist_ok=True)
            self.file = self.root / "rows.json"
            self.namespace = namespace
            self._load()

        def _load(self):
            if self.file.exists():
                payload = json.loads(self.file.read_text(encoding="utf-8"))
            else:
                payload = {"rows": []}
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            self.rows = {str(row["hash_id"]): dict(row) for row in rows if isinstance(row, dict) and row.get("hash_id")}
            self.text_to_hash = {str(row.get("content") or ""): str(row["hash_id"]) for row in self.rows.values()}

        def _save(self):
            payload = {"rows": list(self.rows.values())}
            self.file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        def upsert_rows(self, rows):
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                hash_id = str(row.get("hash_id") or "")
                if not hash_id or hash_id in self.rows:
                    continue
                self.rows[hash_id] = {
                    "hash_id": hash_id,
                    "content": str(row.get("content") or ""),
                    "embedding": list(row.get("embedding") or []),
                }
                self.text_to_hash[str(row.get("content") or "")] = hash_id
            self._save()

        def get_embedding(self, hash_id, dtype=np.float32):
            return np.asarray(self.rows[str(hash_id)]["embedding"], dtype=dtype)

        def get_hash_id(self, text):
            return self.text_to_hash[str(text)]

    hipporag_config_utils_mod.BaseConfig = _FakeBaseConfig
    hipporag_embedding_store_mod.EmbeddingStore = _FakeEmbeddingStore
    hipporag_utils_mod.config_utils = hipporag_config_utils_mod

    return mock.patch.dict(
        sys.modules,
        {
            "openai": openai_mod,
            "transformers": transformers_mod,
            "transformers.utils": transformers_utils_mod,
            "transformers.utils.import_utils": transformers_import_utils_mod,
            "transformers.modeling_utils": transformers_modeling_utils_mod,
            "hipporag": hipporag_mod,
            "hipporag.embedding_store": hipporag_embedding_store_mod,
            "hipporag.utils": hipporag_utils_mod,
            "hipporag.utils.config_utils": hipporag_config_utils_mod,
        },
    )


def _reload_online_tce():
    module_name = "generation.HippoRAG2.generation_tce.online_tce"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


class _FakeHippoRAG:
    created = []
    global_index_invocations = 0
    global_preprocess_invocations = 0
    raise_on_index_call_numbers = set()

    @classmethod
    def reset(cls):
        cls.created = []
        cls.global_index_invocations = 0
        cls.global_preprocess_invocations = 0
        cls.raise_on_index_call_numbers = set()

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        raw_save_dir = kwargs.get("save_dir")
        if raw_save_dir is None:
            global_config = kwargs.get("global_config")
            raw_save_dir = getattr(global_config, "save_dir", None)
        self.save_dir = Path(raw_save_dir or tempfile.mkdtemp(prefix="fake_hipporag_"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.save_dir / "fake_docs.json"
        self.index_calls = []
        self.retrieve_calls = []
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                payload = []
            self.indexed_docs = list(payload) if isinstance(payload, list) else []
        else:
            self.indexed_docs = []
        llm_model_name = str(kwargs.get("llm_model_name") or "")
        embedding_model_name = str(kwargs.get("embedding_model_name") or "")
        if llm_model_name and embedding_model_name:
            (self.save_dir / "{}_{}".format(llm_model_name, embedding_model_name)).mkdir(
                parents=True,
                exist_ok=True,
            )
        self.ready_to_retrieve = True
        self.preprocess_calls = []
        self.__class__.created.append(self)

    def _embedding_dir(self, namespace: str) -> Path:
        return self.save_dir / "{}_embeddings".format(namespace)

    def _rows_path(self, namespace: str) -> Path:
        return self._embedding_dir(namespace) / "rows.json"

    def _load_rows(self, namespace: str):
        path = self._rows_path(namespace)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return {str(row["hash_id"]): dict(row) for row in rows if isinstance(row, dict) and row.get("hash_id")}

    def _upsert_rows(self, namespace: str, rows):
        current = self._load_rows(namespace)
        for row in rows:
            hash_id = str(row.get("hash_id") or "")
            if not hash_id or hash_id in current:
                continue
            current[hash_id] = {
                "hash_id": hash_id,
                "content": str(row.get("content") or ""),
                "embedding": list(row.get("embedding") or []),
            }
        target = self._embedding_dir(namespace)
        target.mkdir(parents=True, exist_ok=True)
        self._rows_path(namespace).write_text(
            json.dumps({"rows": list(current.values())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def preprocess_docs(self, docs):
        self.__class__.global_preprocess_invocations += 1
        self.preprocess_calls.append(list(docs))

        docs_out = []
        chunk_rows = []
        entity_rows = []
        fact_rows = []
        seen_entities = set()
        seen_facts = set()

        for doc in docs:
            try:
                payload = json.loads(doc)
            except Exception:
                payload = {"app_log_id": "unknown", "raw": doc}
            app_log_id = str(payload.get("app_log_id") or "unknown")
            chunk_id = "chunk-{}".format(app_log_id)
            entity = "entity-{}".format(app_log_id)
            fact_text = "('{}', 'mentions', '{}')".format(app_log_id, entity)
            docs_out.append(
                types.SimpleNamespace(
                    chunk_id=chunk_id,
                    passage=doc,
                    extracted_entities=[entity],
                    extracted_triples=[[app_log_id, "mentions", entity]],
                    chunk_triples=[[app_log_id, "mentions", entity]],
                    entity_nodes=[entity],
                    fact_texts=[fact_text],
                )
            )
            chunk_rows.append({"hash_id": chunk_id, "content": doc, "embedding": [float(len(doc)), 0.0, 1.0]})
            if entity not in seen_entities:
                seen_entities.add(entity)
                entity_rows.append({"hash_id": "entity-{}".format(entity), "content": entity, "embedding": [1.0, 0.0, 0.0]})
            if fact_text not in seen_facts:
                seen_facts.add(fact_text)
                fact_rows.append({"hash_id": "fact-{}".format(app_log_id), "content": fact_text, "embedding": [0.0, 1.0, 0.0]})

        self._upsert_rows("chunk", chunk_rows)
        self._upsert_rows("entity", entity_rows)
        self._upsert_rows("fact", fact_rows)
        return types.SimpleNamespace(docs=docs_out, chunk_rows=chunk_rows, entity_rows=entity_rows, fact_rows=fact_rows)

    def index_preprocessed(self, batch):
        self.__class__.global_index_invocations += 1
        if self.__class__.global_index_invocations in self.__class__.raise_on_index_call_numbers:
            raise RuntimeError("synthetic HippoRAG build failure")
        docs = [str(getattr(doc, "passage", "")) for doc in list(getattr(batch, "docs", []) or [])]
        self.index_calls.append(docs)
        self.indexed_docs.extend(docs)
        self.state_path.write_text(json.dumps(self.indexed_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    def index(self, docs):
        batch = self.preprocess_docs(docs)
        self.index_preprocessed(batch)

    def retrieve(self, queries):
        self.retrieve_calls.append(list(queries))
        return [types.SimpleNamespace(docs=list(self.indexed_docs))]


class _FakeLLMClient:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self._request_count = 0
        self._lock = threading.Lock()

    def ask(self, prompt: str, response_type: str = "json"):
        del response_type
        with self._lock:
            self._request_count += 1
        if '"change_analysis"' in prompt:
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
        if '"answer"' in prompt and '"snapshot_state"' not in prompt and '"user_state"' not in prompt:
            return {
                "answer": "Recommend espresso",
                "evidence": [
                    {
                        "app_log_id": "log_0002",
                        "evidence_content": "favorite_coffee espresso",
                    }
                ],
            }
        return {
            "snapshot_state": {"profile_state:favorite_coffee": "espresso"},
            "evidence": {
                "profile_state:favorite_coffee": [
                    {
                        "app_log_id": "log_0002",
                        "evidence_content": "favorite_coffee espresso",
                    }
                ]
            },
        }

    def ask_structured(self, prompt: str, text_format):
        del text_format
        return self.ask(prompt, response_type="json")

    def supports_structured_response(self):
        return False

    def close(self):
        return None

    def usage_summary(self):
        with self._lock:
            request_count = self._request_count
        prompt_tokens = request_count * 100
        completion_tokens = request_count * 20
        total_tokens = prompt_tokens + completion_tokens
        return {
            "request_count": request_count,
            "turn_count": request_count,
            "chat_request_count": request_count,
            "embedding_request_count": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": total_tokens,
            "by_model": [
                {
                    "request_kind": "response",
                    "model": str(self.kwargs.get("model_name") or ""),
                    "request_count": request_count,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            ],
        }


def _make_benchmark():
    return {
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
                "expected_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                "state_completion_pack": {
                    "version": "v1",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "scp_1",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer the user's current favorite coffee.",
                            "answer_template": "<fill the blank>",
                            "retrieval_query": "Retrieve current favorite coffee evidence.",
                        }
                    },
                },
                "change_tracking_pack": {
                    "version": "v1",
                    "previous_checkpoint_id": "",
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
                                    "service_category": "recommendation",
                                    "apply_scenario": "The assistant should recommend a coffee order.",
                                    "question": "What coffee should the assistant recommend?",
                                    "reference_answer": "Recommend latte",
                                    "retrieval_query": "Service scenario: recommend a coffee order.",
                                }
                            ]
                        }
                    },
                },
            },
            {
                "checkpoint_id": "cp_0002",
                "as_of": {
                    "timestamp": "2025-01-02 08:00:00",
                    "log_index": 1,
                    "app_log_id": "log_0002",
                },
                "expected_snapshot_state": {"profile_state": {"favorite_coffee": "espresso"}},
                "validated_snapshot_state": {"profile_state": {"favorite_coffee": "espresso"}},
                "state_completion_pack": {
                    "version": "v1",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "scp_2",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer the user's current favorite coffee.",
                            "answer_template": "<fill the blank>",
                            "retrieval_query": "Retrieve current favorite coffee evidence.",
                        }
                    },
                },
                "change_tracking_pack": {
                    "version": "v1",
                    "previous_checkpoint_id": "cp_0001",
                    "previous_cutoff_ts": "2025-01-01 08:00:00",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "ctp_1",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer how the favorite coffee changed.",
                            "before_template": "<fill the blank>",
                            "after_template": "<fill the blank>",
                            "retrieval_query": "Retrieve evidence for how the favorite coffee changed.",
                        }
                    },
                },
                "rq3_apply_service_qa": {
                    "version": "v1",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "items": [
                                {
                                    "qa_id": "q2",
                                    "service_category": "recommendation",
                                    "apply_scenario": "The assistant should recommend a coffee order.",
                                    "question": "What coffee should the assistant recommend now?",
                                    "reference_answer": "Recommend espresso",
                                    "retrieval_query": "Service scenario: recommend the current favorite coffee.",
                                }
                            ]
                        }
                    },
                },
            },
        ],
    }


def _make_app_logs():
    return [
        {
            "app_log_id": "log_0001",
            "timestamp": "2025-01-01 08:00:00",
            "app_name": "coffee_app",
            "api_name": "save_preference",
            "request": {},
            "response": {"favorite_coffee": "latte"},
        },
        {
            "app_log_id": "log_0002",
            "timestamp": "2025-01-02 08:00:00",
            "app_name": "coffee_app",
            "api_name": "save_preference",
            "request": {},
            "response": {"favorite_coffee": "espresso"},
        },
    ]


class HippoRAG2ProtocolTests(unittest.TestCase):
    def test_materializer_writes_snapshots_and_prepare_checkpoint_state_is_load_only(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        all_logs = [
            {"app_log_id": "log_00001", "timestamp": "2023-10-01 00:00:00", "text": "past-a"},
            {"app_log_id": "log_00002", "timestamp": "2023-10-01 01:00:00", "text": "past-b"},
            {"app_log_id": "log_00003", "timestamp": "2023-10-01 02:00:00", "text": "future-a"},
            {"app_log_id": "log_00004", "timestamp": "2023-10-01 03:00:00", "text": "future-b"},
        ]
        cp1 = {
            "checkpoint_id": "cp1",
            "as_of": {"timestamp": "2023-10-01 01:00:00", "log_index": 1, "app_log_id": "log_00002"},
        }
        cp2 = {
            "checkpoint_id": "cp2",
            "as_of": {"timestamp": "2023-10-01 03:00:00", "log_index": 3, "app_log_id": "log_00004"},
        }
        benchmark = {"checkpoints": [cp1, cp2]}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            _FakeHippoRAG.reset()
            runner = mod.HippoRAG2Runner(
                data_storage_root=data_storage_root,
                snapshot_root=snapshot_root,
                all_logs=all_logs,
                llm_model="gpt-5-mini",
                llm_base_url="https://example.com/v1",
                embedding_model="text-embedding-3-large",
                embedding_base_url="https://example.com/v1",
                batch_size=8,
                retrieval_top_k=5,
                openie_mode="online",
            )
            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=False,
                    max_checkpoints=2,
                    builder_save_every_logs=2,
                )

                progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))
                manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["confirmed_last_log_index"], 3)
                self.assertEqual(progress["status"], "complete")
                self.assertEqual(progress["artifact_version"], 2)
                self.assertEqual(progress["preprocess_fingerprint"], "")
                self.assertEqual(len(manifest["checkpoints"]), 2)
                self.assertFalse((data_storage_root / "preprocess").exists())
                self.assertTrue((snapshot_root / "periodic" / "periodic_00000002").exists())
                self.assertTrue((snapshot_root / "periodic" / "final_00000004").exists())
                self.assertTrue((snapshot_root / "checkpoints" / "cp1").exists())
                self.assertTrue((snapshot_root / "checkpoints" / "cp2").exists())

                _FakeHippoRAG.created = []

                handle1 = runner.prepare_checkpoint_state(cp1, all_logs[:2])
                self.assertEqual(len(_FakeHippoRAG.created), 1)
                self.assertEqual(_FakeHippoRAG.created[0].index_calls, [])

                query_spec = types.SimpleNamespace(
                    task_name="Task A",
                    item_key="finance:balance",
                    target_keys=["finance:balance"],
                    checkpoint_timestamp="2023-10-01 01:00:00",
                    task_query_text="ignored",
                    retrieval_query_text="shared retrieval query",
                    answer_query_text="answer question",
                    task_payload={},
                )
                retrieval_result = runner.retrieve_context_for_query(
                    handle1,
                    query_spec,
                    types.SimpleNamespace(common={"top_k": 1}, backend={}),
                    all_logs[:2],
                )

                self.assertEqual(_FakeHippoRAG.created[0].retrieve_calls, [["shared retrieval query"]])
                self.assertEqual(retrieval_result.debug_metadata["retrieved_app_log_ids"], ["log_00001"])
                self.assertEqual(retrieval_result.debug_metadata["num_input_context_logs"], 1)
                self.assertEqual(len(retrieval_result.inline_memory_blocks), 1)
                self.assertIn("log_00001", retrieval_result.inline_memory_blocks[0])

                handle2 = runner.prepare_checkpoint_state(cp2, all_logs[:4])
                self.assertEqual(len(_FakeHippoRAG.created), 2)
                self.assertEqual(_FakeHippoRAG.created[1].index_calls, [])
                self.assertEqual(handle2.metadata["indexed_log_count"], 4)

    def test_materializer_resume_uses_confirmed_progress_and_snapshot_fallback(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = {
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0, "app_log_id": "log_0001"},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-01-03 08:00:00", "log_index": 2, "app_log_id": "log_0003"},
                },
            ]
        }
        app_logs = _make_app_logs() + [
            {
                "app_log_id": "log_0003",
                "timestamp": "2025-01-03 08:00:00",
                "app_name": "coffee_app",
                "api_name": "save_preference",
                "request": {},
                "response": {"favorite_coffee": "cappuccino"},
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                _FakeHippoRAG.raise_on_index_call_numbers = {2}
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                with self.assertRaises(RuntimeError):
                    runner.materialize_checkpoint_snapshots(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        resume=False,
                        max_checkpoints=2,
                        builder_save_every_logs=1,
                    )

                progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["confirmed_last_log_index"], 0)
                self.assertEqual(progress["status"], "interrupted")
                shutil.rmtree(data_storage_root / "builder" / "workspace")

                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )

                progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))
                manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["confirmed_last_log_index"], 2)
                self.assertEqual(progress["status"], "complete")
                self.assertEqual({item["checkpoint_id"] for item in manifest["checkpoints"]}, {"cp_0001", "cp_0002"})
                self.assertEqual(len(_FakeHippoRAG.created[0].index_calls), 2)
                self.assertIn("log_0002", _FakeHippoRAG.created[0].index_calls[0][0])
                self.assertTrue((snapshot_root / "checkpoints" / "cp_0002").exists())

    def test_materializer_config_mismatch_forces_clean_rebuild(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = {
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0, "app_log_id": "log_0001"},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1, "app_log_id": "log_0002"},
                },
            ]
        }
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=False,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                first_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))

                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-small",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                second_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))

                self.assertNotEqual(first_progress["config_fingerprint"], second_progress["config_fingerprint"])
                self.assertEqual(second_progress["status"], "complete")
                replayed_docs = [
                    doc
                    for instance in _FakeHippoRAG.created
                    for call in instance.index_calls
                    for doc in call
                ]
                self.assertTrue(any("log_0001" in doc for doc in replayed_docs))

    def test_materializer_reuses_snapshots_when_eval_pack_path_changes(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        build_benchmark = _make_benchmark()
        eval_benchmark = _make_benchmark()
        eval_benchmark["checkpoints"][0]["rq3_apply_service_qa"]["keys"][
            "profile_state:favorite_coffee"
        ]["items"][0]["question"] = "What coffee should be recommended in the updated eval pack?"
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_benchmark_path = root / "benchmark_build.json"
            eval_benchmark_path = root / "benchmark_eval.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            build_benchmark_path.write_text(json.dumps(build_benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            eval_benchmark_path.write_text(json.dumps(eval_benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=build_benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=False,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                first_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))

                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=False,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=eval_benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                second_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))

                self.assertEqual(first_progress["config_fingerprint"], second_progress["config_fingerprint"])
                self.assertEqual(_FakeHippoRAG.global_index_invocations, 0)

    def test_run_generation_emits_pack_first_tce_predictions(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        _FakeHippoRAG.reset()
        benchmark = _make_benchmark()
        app_logs = _make_app_logs()
        qa_list = [
            {
                "id": "qa_1",
                "query": "What coffee should the assistant recommend now?",
                "reference": "Recommend espresso",
                "reference_app_logs": [app_logs[1]],
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            qa_path = root / "qa.json"
            output_path = root / "prediction.json"
            data_storage_path = root / "runtime_state"
            snapshot_dir = root / "snapshots"

            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG), mock.patch.object(
                mod,
                "LLMClient",
                _FakeLLMClient,
            ), mock.patch.object(
                mod,
                "_configure_hipporag_openai_env",
                return_value={
                    "llm_api_key": "dummy",
                    "llm_base_url": "https://example.com/v1",
                    "retriever_api_key": "dummy",
                    "retriever_base_url": "https://example.com/v1",
                },
            ):
                result = mod.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=snapshot_dir,
                    data_storage_path=data_storage_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=2,
                    resume=False,
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    retriever_provider="azure",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    retrieval_top_k=2,
                    builder_save_every_logs=1,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_save_prompt_and_raw=True,
                    rq3_apply_retrieval_top_k=2,
                    checkpoint_workers=2,
                    within_checkpoint_workers=2,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=str(qa_path),
                    final_qa_retrieval_top_k=2,
                    final_qa_save_prompt_and_raw=True,
                )

            self.assertEqual(len(result["predictions"]), 2)
            cp2 = result["predictions"][-1]
            self.assertEqual(cp2["snapshot_state"]["profile_state:favorite_coffee"], "espresso")
            self.assertIn("profile_state:favorite_coffee", cp2["change_analysis"])
            self.assertIn("profile_state:favorite_coffee", cp2["rq3_apply_answers"])
            self.assertEqual(cp2["metadata"]["baseline"], "hipporag2")
            self.assertEqual(
                cp2["metadata"]["concurrency_policy"],
                {"checkpoint_parallelism": "allowed", "within_checkpoint_parallelism": "allowed"},
            )
            self.assertEqual(cp2["metadata"]["effective_checkpoint_workers"], 2)
            self.assertEqual(cp2["metadata"]["effective_within_checkpoint_workers"], 2)
            self.assertEqual(cp2["metadata"]["per_key_retrieval"][0]["retrieval_query"], "Retrieve current favorite coffee evidence.")
            self.assertEqual(cp2["metadata"]["per_key_retrieval"][0]["context_log_ids"], ["log_0001", "log_0002"])
            self.assertFalse((data_storage_path / "preprocess").exists())
            self.assertTrue((data_storage_path / "builder" / "progress.json").exists())
            self.assertTrue((snapshot_dir / "periodic" / "periodic_00000001").exists())
            self.assertTrue((snapshot_dir / "manifest.json").exists())

            final_qa_results = json.loads(output_path.with_name("prediction_final_qa.json").read_text(encoding="utf-8"))
            self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
            self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp_0002")

            usage_sidecar = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))
            live_usage_sidecar = json.loads((output_path.parent / "usage_cost_live.json").read_text(encoding="utf-8"))
            self.assertEqual(usage_sidecar["llm_provider"], "azure")
            self.assertEqual(usage_sidecar["llm_model"], "gpt-5-mini")
            self.assertEqual(usage_sidecar["retriever_provider"], "azure")
            self.assertEqual(usage_sidecar["embedding_model"], "text-embedding-3-large")
            self.assertEqual(usage_sidecar["usage"]["answer_llm"]["request_count"], 6)
            self.assertEqual(usage_sidecar["usage"]["answer_llm"]["prompt_tokens"], 600)
            self.assertEqual(usage_sidecar["usage"]["answer_llm"]["completion_tokens"], 120)
            self.assertEqual(live_usage_sidecar["usage"]["answer_llm"]["request_count"], 6)
            self.assertTrue(live_usage_sidecar["is_live"])

    def test_materializer_builder_save_cadence_change_does_not_force_rebuild(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0, "app_log_id": "log_0001"},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1, "app_log_id": "log_0002"},
                },
            ],
        }
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=False,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                progress_path = data_storage_root / "builder" / "progress.json"

                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=2,
                    builder_save_every_logs=64,
                )
                second_progress = json.loads(progress_path.read_text(encoding="utf-8"))

                expected_fingerprint = mod._build_builder_fingerprint(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    checkpoint_specs=[
                        {
                            "checkpoint_id": "cp_0001",
                            "checkpoint_app_log_id": "log_0001",
                            "cut_index": 0,
                        },
                        {
                            "checkpoint_id": "cp_0002",
                            "checkpoint_app_log_id": "log_0002",
                            "cut_index": 1,
                        },
                    ],
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    openie_mode="online",
                )
                self.assertEqual(second_progress["config_fingerprint"], expected_fingerprint)
                self.assertEqual(second_progress["preprocess_fingerprint"], "")
                self.assertEqual(second_progress["confirmed_last_log_index"], 1)
                self.assertEqual(_FakeHippoRAG.global_preprocess_invocations, 0)
                self.assertEqual(_FakeHippoRAG.global_index_invocations, 0)
                self.assertFalse((data_storage_root / "preprocess").exists())

    def test_materializer_resume_uses_snapshot_dir_without_local_builder_progress(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0, "app_log_id": "log_0001"},
                },
                {
                    "checkpoint_id": "cp_0002",
                    "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 1, "app_log_id": "log_0002"},
                },
            ],
        }
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=False,
                    max_checkpoints=2,
                    builder_save_every_logs=1,
                )
                self.assertTrue((data_storage_root / "builder" / "progress.json").exists())
                self.assertTrue((snapshot_root / "manifest.json").exists())

                (data_storage_root / "builder" / "progress.json").unlink()
                shutil.rmtree(data_storage_root / "builder" / "workspace", ignore_errors=True)

                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=2,
                    builder_save_every_logs=64,
                )

                resumed_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))
                self.assertIn(resumed_progress["status"], {"restored_from_snapshot", "complete"})
                self.assertEqual(resumed_progress["confirmed_last_log_index"], 1)
                self.assertEqual(_FakeHippoRAG.global_index_invocations, 0)

    def test_run_generation_interleaves_build_and_test_when_enabled(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = _make_benchmark()
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            data_storage_path = root / "runtime_state"
            snapshot_dir = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            materialize_calls = []
            pipeline_calls = []

            def _fake_materialize(self, **kwargs):
                materialize_calls.append(
                    {
                        "resume": kwargs.get("resume"),
                        "max_checkpoints": kwargs.get("max_checkpoints"),
                        "target_log_index": kwargs.get("target_log_index"),
                    }
                )
                return self.save_root

            def _fake_run_pipeline(**kwargs):
                pipeline_calls.append(
                    {
                        "resume": kwargs.get("resume"),
                        "max_checkpoints": kwargs.get("max_checkpoints"),
                        "enable_final_qa": kwargs.get("enable_final_qa"),
                    }
                )
                max_cp = int(kwargs.get("max_checkpoints") or 0)
                return {"predictions": [{"checkpoint_id": "cp_{:04d}".format(max_cp)}]}

            with mock.patch.object(mod.HippoRAG2Runner, "materialize_checkpoint_snapshots", _fake_materialize), mock.patch.object(
                mod,
                "run_pipeline",
                side_effect=_fake_run_pipeline,
            ), mock.patch.object(
                mod,
                "LLMClient",
                _FakeLLMClient,
            ), mock.patch.object(
                mod,
                "_configure_hipporag_openai_env",
                return_value={
                    "llm_api_key": "dummy",
                    "llm_base_url": "https://example.com/v1",
                    "retriever_api_key": "dummy",
                    "retriever_base_url": "https://example.com/v1",
                },
            ):
                result = mod.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=snapshot_dir,
                    data_storage_path=data_storage_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=2,
                    resume=False,
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    retriever_provider="azure",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    retrieval_top_k=2,
                    builder_save_every_logs=1,
                    interleave_build_and_test=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_save_prompt_and_raw=True,
                    rq3_apply_retrieval_top_k=2,
                    checkpoint_workers=2,
                    within_checkpoint_workers=2,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=None,
                    final_qa_output_path=None,
                    final_qa_retrieval_top_k=2,
                    final_qa_save_prompt_and_raw=True,
                )

            self.assertEqual(
                materialize_calls,
                [
                    {"resume": False, "max_checkpoints": 1, "target_log_index": 0},
                    {"resume": True, "max_checkpoints": 2, "target_log_index": 1},
                ],
            )
            self.assertEqual(
                pipeline_calls,
                [
                    {"resume": False, "max_checkpoints": 1, "enable_final_qa": False},
                    {"resume": True, "max_checkpoints": 2, "enable_final_qa": True},
                ],
            )
            self.assertEqual(result, {"predictions": [{"checkpoint_id": "cp_0002"}]})
            self.assertTrue((output_path.parent / "usage_cost.json").exists())

    def test_run_generation_can_stop_after_first_checkpoint_build_only(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = _make_benchmark()
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            data_storage_path = root / "runtime_state"
            snapshot_dir = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            materialize_calls = []

            def _fake_materialize(self, **kwargs):
                materialize_calls.append(
                    {
                        "resume": kwargs.get("resume"),
                        "max_checkpoints": kwargs.get("max_checkpoints"),
                        "target_log_index": kwargs.get("target_log_index"),
                    }
                )
                return self.save_root

            with mock.patch.object(mod.HippoRAG2Runner, "materialize_checkpoint_snapshots", _fake_materialize), mock.patch.object(
                mod,
                "run_pipeline",
                side_effect=AssertionError("run_pipeline should not be called in build_only mode"),
            ), mock.patch.object(
                mod,
                "LLMClient",
                _FakeLLMClient,
            ), mock.patch.object(
                mod,
                "_configure_hipporag_openai_env",
                return_value={
                    "llm_api_key": "dummy",
                    "llm_base_url": "https://example.com/v1",
                    "retriever_api_key": "dummy",
                    "retriever_base_url": "https://example.com/v1",
                },
            ):
                result = mod.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=snapshot_dir,
                    data_storage_path=data_storage_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=2,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    retriever_provider="azure",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    retrieval_top_k=2,
                    builder_save_every_logs=1,
                    interleave_build_and_test=False,
                    build_only=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_save_prompt_and_raw=True,
                    rq3_apply_retrieval_top_k=2,
                    checkpoint_workers=2,
                    within_checkpoint_workers=2,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=None,
                    final_qa_output_path=None,
                    final_qa_retrieval_top_k=2,
                    final_qa_save_prompt_and_raw=True,
                )

            self.assertEqual(
                materialize_calls,
                [{"resume": False, "max_checkpoints": 1, "target_log_index": 0}],
            )
            self.assertEqual(result["predictions"], [])
            self.assertTrue(result["build_only"]["enabled"])
            self.assertEqual(result["build_only"]["snapshot_dir"], str(snapshot_dir))
            self.assertEqual(result["build_only"]["data_storage_path"], str(data_storage_path))
            self.assertEqual(result["build_only"]["target_log_index"], 0)
            self.assertEqual(result["build_only"]["requested_checkpoint_ids"], ["cp_0001"])
            self.assertTrue((output_path.parent / "usage_cost.json").exists())

    def test_run_generation_predict_from_prebuilt_skips_materialization(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = _make_benchmark()
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            data_storage_path = root / "runtime_state"
            snapshot_dir = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            pipeline_calls = []

            def _fake_run_pipeline(**kwargs):
                pipeline_calls.append(
                    {
                        "resume": kwargs.get("resume"),
                        "max_checkpoints": kwargs.get("max_checkpoints"),
                        "backend": kwargs.get("retrieval_options_backend"),
                    }
                )
                return {"predictions": [{"checkpoint_id": "cp_0001"}]}

            with mock.patch.object(
                mod.HippoRAG2Runner,
                "materialize_checkpoint_snapshots",
                side_effect=AssertionError("predict_from_prebuilt must not materialize snapshots"),
            ), mock.patch.object(
                mod,
                "run_pipeline",
                side_effect=_fake_run_pipeline,
            ), mock.patch.object(
                mod,
                "LLMClient",
                _FakeLLMClient,
            ), mock.patch.object(
                mod,
                "_configure_hipporag_openai_env",
                return_value={
                    "llm_api_key": "dummy",
                    "llm_base_url": "https://example.com/v1",
                    "retriever_api_key": "dummy",
                    "retriever_base_url": "https://example.com/v1",
                },
            ):
                result = mod.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=snapshot_dir,
                    data_storage_path=data_storage_path,
                    max_visible_logs=None,
                    llm_provider="azure",
                    llm_model="gpt-5-mini",
                    llm_max_workers=2,
                    resume=True,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    retriever_provider="azure",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    retrieval_top_k=2,
                    builder_save_every_logs=1,
                    interleave_build_and_test=False,
                    predict_from_prebuilt=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_save_prompt_and_raw=True,
                    rq3_apply_retrieval_top_k=2,
                    checkpoint_workers=2,
                    within_checkpoint_workers=2,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=None,
                    final_qa_output_path=None,
                    final_qa_retrieval_top_k=2,
                    final_qa_save_prompt_and_raw=True,
                )

            self.assertEqual(result, {"predictions": [{"checkpoint_id": "cp_0001"}]})
            self.assertEqual(len(pipeline_calls), 1)
            self.assertTrue(pipeline_calls[0]["resume"])
            self.assertEqual(pipeline_calls[0]["max_checkpoints"], 1)
            self.assertTrue(pipeline_calls[0]["backend"]["predict_from_prebuilt"])
            self.assertFalse(pipeline_calls[0]["backend"]["build_only"])
            self.assertTrue((output_path.parent / "usage_cost.json").exists())

    def test_non_v2_builder_root_triggers_clean_rebuild(self):
        with _install_import_stubs():
            mod = _reload_online_tce()

        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 0, "app_log_id": "log_0001"},
                }
            ],
        }
        app_logs = _make_app_logs()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            data_storage_root = root / "hipporag_runtime"
            snapshot_root = root / "hipporag_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            legacy_preprocess_root = data_storage_root / "preprocess"
            legacy_preprocess_root.mkdir(parents=True, exist_ok=True)
            (legacy_preprocess_root / "progress.json").write_text(
                json.dumps(
                    {
                        "artifact_version": 1,
                        "preprocess_fingerprint": "legacy",
                        "confirmed_last_log_index": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (data_storage_root / "legacy_marker.txt").write_text("stale", encoding="utf-8")

            with mock.patch.object(mod, "HippoRAG", _FakeHippoRAG):
                _FakeHippoRAG.reset()
                runner = mod.HippoRAG2Runner(
                    data_storage_root=data_storage_root,
                    snapshot_root=snapshot_root,
                    all_logs=app_logs,
                    llm_model="gpt-5-mini",
                    llm_base_url="https://example.com/v1",
                    embedding_model="text-embedding-3-large",
                    embedding_base_url="https://example.com/v1",
                    batch_size=1,
                    retrieval_top_k=5,
                    openie_mode="online",
                    allow_destructive_rebuild=True,
                )
                runner.materialize_checkpoint_snapshots(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    resume=True,
                    max_checkpoints=1,
                    builder_save_every_logs=1,
                )

                builder_progress = json.loads((data_storage_root / "builder" / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(builder_progress["artifact_version"], 2)
                self.assertEqual(builder_progress["status"], "complete")
                self.assertFalse((data_storage_root / "legacy_marker.txt").exists())
                self.assertFalse((data_storage_root / "preprocess").exists())
                self.assertTrue((snapshot_root / "checkpoints" / "cp_0001").exists())

    def test_reset_save_root_refuses_existing_artifacts_without_opt_in(self):
        with _install_import_stubs():
            mod = _reload_online_tce()
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                data_storage_root = root / "runtime"
                snapshot_root = root / "snapshots"
                (data_storage_root / "builder").mkdir(parents=True, exist_ok=True)
                (data_storage_root / "builder" / "stale.txt").write_text("stale", encoding="utf-8")
                (snapshot_root / "checkpoints").mkdir(parents=True, exist_ok=True)
                (snapshot_root / "checkpoints" / "stale.txt").write_text("stale", encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "allow_destructive_rebuild"):
                    mod._reset_save_root(
                        data_storage_root,
                        snapshot_root,
                        allow_destructive_rebuild=False,
                    )

                self.assertTrue((data_storage_root / "builder" / "stale.txt").exists())
                self.assertTrue((snapshot_root / "checkpoints" / "stale.txt").exists())

    def test_adapter_routes_to_canonical_online_runtime(self):
        args = TceAdapterArgs(
            baseline="hipporag2",
            user_id="001_user_001",
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/prediction.json"),
            llm_provider="azure",
            llm_model="gpt-5-mini",
            llm_max_workers=4,
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_top_k=None,
            resume=False,
            max_checkpoints=3,
            debug=False,
            debug_dir=None,
            save_prompt_and_raw=True,
            enable_change_reasoning=True,
            enable_rq3_apply_service_qa=True,
            rq3_apply_save_prompt_and_raw=True,
            checkpoint_workers=4,
            within_checkpoint_workers=4,
            save_every_generation_keys=5,
            retriever_provider="azure",
            retriever_model="text-embedding-3-large",
            retriever_batch_size=64,
            retrieval_top_k=20,
            rq3_apply_retrieval_top_k=20,
            enable_final_qa=True,
            final_qa_path="/tmp/qa.json",
            final_qa_output_path="/tmp/qa_out.json",
            final_qa_retrieval_top_k=5,
            final_qa_save_prompt_and_raw=True,
            extras={
                "snapshot_dir": "/tmp/hipporag2_snapshots",
                "data_storage_path": "/tmp/hipporag2_runtime",
                "builder_save_every_logs": "7",
                "interleave_build_and_test": "true",
                "build_only": "true",
                "__task_selection__": "task_c_only",
            },
        )

        with _install_import_stubs():
            with mock.patch(
                "generation.HippoRAG2.generation_tce.online_tce.run_generation",
                return_value={"predictions": []},
            ) as mocked_run:
                result = hipporag2_adapter.run(args)

        self.assertEqual(result, {"predictions": []})
        mocked_run.assert_called_once()
        kwargs = mocked_run.call_args[1]
        self.assertEqual(kwargs["snapshot_dir"], Path("/tmp/hipporag2_snapshots"))
        self.assertEqual(kwargs["data_storage_path"], Path("/tmp/hipporag2_runtime"))
        self.assertEqual(kwargs["retriever_batch_size"], 64)
        self.assertEqual(kwargs["retrieval_top_k"], 20)
        self.assertEqual(kwargs["builder_save_every_logs"], 7)
        self.assertTrue(kwargs["interleave_build_and_test"])
        self.assertTrue(kwargs["build_only"])
        self.assertFalse(kwargs["predict_from_prebuilt"])
        self.assertTrue(kwargs["enable_change_reasoning"])
        self.assertTrue(kwargs["enable_rq3_apply_service_qa"])
        self.assertEqual(kwargs["task_selection"], "task_c_only")
        self.assertTrue(kwargs["enable_final_qa"])

    def test_adapter_routes_predict_from_prebuilt(self):
        args = TceAdapterArgs(
            baseline="hipporag2",
            user_id="001_user_001",
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/prediction.json"),
            llm_provider="azure",
            llm_model="gpt-5-mini",
            llm_max_workers=4,
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_top_k=None,
            resume=True,
            max_checkpoints=3,
            debug=False,
            debug_dir=None,
            save_prompt_and_raw=True,
            enable_change_reasoning=False,
            enable_rq3_apply_service_qa=True,
            rq3_apply_save_prompt_and_raw=True,
            checkpoint_workers=4,
            within_checkpoint_workers=4,
            save_every_generation_keys=5,
            retriever_provider="azure",
            retriever_model="text-embedding-3-large",
            retriever_batch_size=64,
            retrieval_top_k=20,
            rq3_apply_retrieval_top_k=20,
            enable_final_qa=False,
            final_qa_path=None,
            final_qa_output_path=None,
            final_qa_retrieval_top_k=5,
            final_qa_save_prompt_and_raw=False,
            extras={
                "memory_action": "predict_from_prebuilt",
                "snapshot_dir": "/tmp/hipporag2_snapshots",
                "data_storage_path": "/tmp/hipporag2_runtime",
            },
        )

        with _install_import_stubs():
            with mock.patch(
                "generation.HippoRAG2.generation_tce.online_tce.run_generation",
                return_value={"predictions": []},
            ) as mocked_run:
                hipporag2_adapter.run(args)

        kwargs = mocked_run.call_args[1]
        self.assertFalse(kwargs["build_only"])
        self.assertTrue(kwargs["predict_from_prebuilt"])


if __name__ == "__main__":
    unittest.main()
