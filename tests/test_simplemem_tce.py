#!/usr/bin/env python3
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import types

fake_openai = types.ModuleType("openai")


class _DummyOpenAI(object):
    def __init__(self, *args, **kwargs):
        del args, kwargs


fake_openai.OpenAI = _DummyOpenAI
sys.modules["openai"] = fake_openai

fake_google = types.ModuleType("google")
fake_genai = types.ModuleType("google.genai")
fake_google.genai = fake_genai
sys.modules["google"] = fake_google
sys.modules["google.genai"] = fake_genai
sys.modules["lancedb"] = types.ModuleType("lancedb")
sys.modules["pyarrow"] = types.ModuleType("pyarrow")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.adapters import simplemem as simplemem_adapter
from baseline_prediction.adapters.base import TceAdapterArgs
from baseline_prediction.simplemem.generation_tce import tce as simplemem_tce
from baseline_prediction.simplemem.upstream_vendor.models.memory_entry import Dialogue, MemoryEntry
from tce_core.orchestrator_protocol import RetrievalOptions
from tce_core.pipeline import build_state_completion_prompt


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _benchmark_payload():
    return {
        "user_id": "001_user_001",
        "sampling_strategy": {"stage": "benchmark_build"},
        "checkpoints": [
            {
                "checkpoint_id": "cp1",
                "as_of": {"timestamp": "2025-01-01 08:00:00", "log_index": 1, "app_log_id": "log_0002"},
                "validated_snapshot_state": {"profile_state": {"favorite_coffee": "latte"}},
                "state_completion_pack": {
                    "version": "v1",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "scp1",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer the user's current favorite coffee.",
                            "answer_template": "<fill the blank>",
                            "retrieval_query": "Retrieve current favorite coffee evidence.",
                        }
                    },
                },
                "change_tracking_pack": {"version": "v1", "previous_cutoff_ts": "", "keys": {}},
                "rq3_apply_service_qa": {"version": "v1", "keys": {}},
            },
            {
                "checkpoint_id": "cp2",
                "as_of": {"timestamp": "2025-01-02 08:00:00", "log_index": 2, "app_log_id": "log_0003"},
                "validated_snapshot_state": {"profile_state": {"favorite_coffee": "espresso"}},
                "state_completion_pack": {
                    "version": "v1",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "scp2",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer the user's current favorite coffee.",
                            "answer_template": "<fill the blank>",
                            "retrieval_query": "Retrieve current favorite coffee evidence.",
                        }
                    },
                },
                "change_tracking_pack": {
                    "version": "v1",
                    "previous_cutoff_ts": "2025-01-01 08:00:00",
                    "keys": {
                        "profile_state:favorite_coffee": {
                            "item_id": "ctp1",
                            "state_key": "profile_state:favorite_coffee",
                            "question_text": "Infer how the favorite coffee changed.",
                            "before_template": "<fill the blank>",
                            "after_template": "<fill the blank>",
                            "retrieval_query": "Retrieve evidence for how the favorite coffee changed.",
                        }
                    },
                },
                "rq3_apply_service_qa": {"version": "v1", "keys": {}},
            },
        ],
    }


def _benchmark_payload_with_updated_eval_pack():
    payload = _benchmark_payload()
    for checkpoint in payload.get("checkpoints") or []:
        if not isinstance(checkpoint, dict):
            continue
        state_pack = checkpoint.get("state_completion_pack") if isinstance(checkpoint.get("state_completion_pack"), dict) else {}
        keys = state_pack.get("keys") if isinstance(state_pack.get("keys"), dict) else {}
        item = keys.get("profile_state:favorite_coffee") if isinstance(keys.get("profile_state:favorite_coffee"), dict) else None
        if item is None:
            continue
        item["question_text"] = "UPDATED PACK question for {}".format(checkpoint.get("checkpoint_id"))
        item["retrieval_query"] = "UPDATED PACK retrieval for {}".format(checkpoint.get("checkpoint_id"))
    return payload


def _app_logs_payload():
    return [
        {
            "app_log_id": "log_0001",
            "timestamp": "2025-01-01 06:00:00",
            "app_name": "CoffeeApp",
            "api_name": "RecordPreference",
            "request": {"favorite_coffee": "latte"},
            "response": {"status": "ok"},
        },
        {
            "app_log_id": "log_0002",
            "timestamp": "2025-01-01 07:00:00",
            "app_name": "CoffeeApp",
            "api_name": "OpenCafe",
            "request": {"store": "Downtown"},
            "response": {"status": "ok"},
        },
        {
            "app_log_id": "log_0003",
            "timestamp": "2025-01-02 07:00:00",
            "app_name": "CoffeeApp",
            "api_name": "RecordPreference",
            "request": {"favorite_coffee": "espresso"},
            "response": {"status": "ok"},
        },
    ]


class _FakeLLMClient:
    build_calls = 0

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def ask(self, prompt: str, response_type: str = "json"):
        self.calls.append(prompt)
        if "[Current Window Raw App Logs]" in prompt:
            self.__class__.build_calls += 1
            marker = "[Current Window Raw App Logs]\n"
            end_marker = "\n\n[Output Schema]"
            start = prompt.index(marker) + len(marker)
            end = prompt.index(end_marker, start)
            payload = json.loads(prompt[start:end])
            entries = []
            for item in payload:
                source_log_id = str(item.get("source_log_id") or "")
                entries.append(
                    {
                        "lossless_restatement": "Memory for {}".format(source_log_id),
                        "keywords": [source_log_id],
                        "timestamp": item.get("timestamp"),
                        "location": None,
                        "persons": [],
                        "entities": [source_log_id],
                        "topic": "coffee",
                    }
                )
            return entries
        del response_type
        log_ids = re.findall(r'"app_log_id"\s*:\s*"([^"]+)"', prompt)
        app_log_id = log_ids[-1] if log_ids else ""
        return {
            "snapshot_state": {"profile_state:favorite_coffee": "latte"},
            "evidence": {
                "profile_state:favorite_coffee": [
                    {
                        "app_log_id": app_log_id,
                        "evidence_content": "support from {}".format(app_log_id),
                    }
                ]
            },
        }

    def ask_structured(self, prompt: str, *, text_format):
        del text_format
        return self.ask(prompt)

    def supports_structured_response(self):
        return False

    def usage_summary(self):
        count = len(self.calls)
        return {
            "request_count": count,
            "turn_count": count,
            "prompt_tokens": count * 10,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "context_tokens": 0,
            "total_tokens": count * 10,
            "by_model": [
                {
                    "request_kind": "response",
                    "model": str(self.kwargs.get("model_name") or "fake"),
                    "request_count": count,
                    "prompt_tokens": count * 10,
                    "completion_tokens": 0,
                    "total_tokens": count * 10,
                }
            ],
        }

    def close(self):
        return None


class _FakeEmbeddingClient:
    def __init__(self, provider: str, model: str, batch_size: int = 64, tracker=None):
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.tracker = tracker

    def encode_documents(self, texts):
        if self.tracker is not None and texts:
            self.tracker.record({"prompt_tokens": len(texts), "total_tokens": len(texts)})
        return [[float(len(text))] for text in texts]

    def encode_single(self, text: str):
        if self.tracker is not None:
            self.tracker.record({"prompt_tokens": 1, "total_tokens": 1})
        return [float(len(text))]


class _FakeVectorStore:
    def __init__(self, config, embedding_client):
        self.config = config
        self.embedding_client = embedding_client
        self.entries = []

    def _path(self) -> Path:
        return Path(self.config.db_path) / "entries.json"

    def _save(self):
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def initialize(self, clear: bool = False):
        if clear and Path(self.config.db_path).exists():
            import shutil

            shutil.rmtree(self.config.db_path, ignore_errors=True)
        Path(self.config.db_path).mkdir(parents=True, exist_ok=True)
        self.entries = []
        self._save()

    def open_existing(self) -> bool:
        path = self._path()
        if not path.exists():
            return False
        self.entries = json.loads(path.read_text(encoding="utf-8"))
        return True

    def add_entries(self, entries):
        if not entries:
            return
        self.embedding_client.encode_documents([entry.lossless_restatement for entry in entries])
        self.entries.extend([entry.to_dict() for entry in entries])
        self._save()

    def count_rows(self) -> int:
        return len(self.entries)

    def get_all_entries(self):
        return [MemoryEntry.from_dict(item) for item in self.entries]

    def semantic_search(self, query: str, top_k: int):
        del query
        limit = len(self.entries) if top_k <= 0 else min(top_k, len(self.entries))
        return [MemoryEntry.from_dict(item) for item in self.entries[:limit]]

    def keyword_search(self, keywords, top_k: int):
        del keywords
        return self.semantic_search("", top_k)

    def structured_search(self, **kwargs):
        del kwargs
        return []


def _usage_summary(request_count: int, model: str = "fake-model") -> dict:
    return {
        "request_count": int(request_count),
        "turn_count": int(request_count),
        "prompt_tokens": int(request_count) * 10,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "context_tokens": 0,
        "total_tokens": int(request_count) * 10,
        "by_model": [
            {
                "request_kind": "response",
                "model": model,
                "request_count": int(request_count),
                "prompt_tokens": int(request_count) * 10,
                "completion_tokens": 0,
                "total_tokens": int(request_count) * 10,
            }
        ],
    }


class _FakeSimpleMemVectorStore:
    def __init__(self, db_path: str, tracker=None):
        self.db_path = Path(db_path)
        self.tracker = tracker
        self.entries = []
        self._load()

    def _path(self) -> Path:
        return self.db_path / "entries.json"

    def _load(self) -> None:
        path = self._path()
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.entries = []

    def _save(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._path().write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self.db_path.exists():
            import shutil

            shutil.rmtree(self.db_path, ignore_errors=True)
        self.entries = []
        self._save()

    def count_rows(self) -> int:
        self._load()
        return len(self.entries)

    def add_entries(self, entries) -> None:
        if not entries:
            return
        if self.tracker is not None:
            self.tracker.record({"prompt_tokens": len(entries), "total_tokens": len(entries)})
        self._load()
        self.entries.extend([entry.to_dict() for entry in entries])
        self._save()

    def get_all_entries(self):
        self._load()
        return [MemoryEntry.from_dict(item) for item in self.entries]


class _FakeSimpleMemBuilder:
    def __init__(self, system, window_size: int, overlap_size: int):
        self.system = system
        self.window_size = max(1, int(window_size))
        self.overlap_size = max(0, int(overlap_size))
        self.step_size = max(1, self.window_size - self.overlap_size)
        self.dialogue_buffer = []
        self.previous_entries = []
        self.processed_count = 0

    def export_state(self):
        return {
            "processed_count": self.processed_count,
            "dialogue_buffer": [item.to_dict() for item in self.dialogue_buffer],
            "previous_entries": [item.to_dict() for item in self.previous_entries],
        }

    def import_state(self, payload):
        raw = payload if isinstance(payload, dict) else {}
        self.processed_count = int(raw.get("processed_count") or 0)
        self.dialogue_buffer = [
            Dialogue.from_dict(item)
            for item in (raw.get("dialogue_buffer") or [])
            if isinstance(item, dict)
        ]
        self.previous_entries = [
            MemoryEntry.from_dict(item)
            for item in (raw.get("previous_entries") or [])
            if isinstance(item, dict)
        ]

    def add_dialogues(self, dialogues, auto_process=True):
        self.dialogue_buffer.extend(list(dialogues))
        if not auto_process:
            return
        while len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def _make_entries(self, dialogues):
        user_turns = [
            dialogue
            for dialogue in dialogues
            if str(getattr(dialogue, "speaker", "")).lower() == "user" and getattr(dialogue, "source_log_id", None)
        ]
        if not user_turns:
            return []
        self.system.__class__.build_calls += 1
        self.system.phase_counts["build_memory"] += 1
        entries = []
        for dialogue in user_turns:
            source_log_id = str(dialogue.source_log_id or "")
            entries.append(
                MemoryEntry(
                    lossless_restatement="Memory for {}".format(source_log_id),
                    keywords=[source_log_id],
                    timestamp=dialogue.timestamp,
                    location=None,
                    persons=[],
                    entities=[source_log_id],
                    topic="coffee",
                )
            )
        return entries

    def process_window(self):
        if not self.dialogue_buffer:
            return []
        window = self.dialogue_buffer[: self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size :]
        entries = self._make_entries(window)
        if entries:
            self.system.vector_store.add_entries(entries)
            self.previous_entries = entries[-10:]
        self.processed_count += len(window)
        return entries

    def process_remaining(self):
        if not self.dialogue_buffer:
            return []
        window = list(self.dialogue_buffer)
        self.dialogue_buffer = []
        entries = self._make_entries(window)
        if entries:
            self.system.vector_store.add_entries(entries)
            self.previous_entries = entries[-10:]
        self.processed_count += len(window)
        return entries


class _FakeSimpleMemSystem:
    build_calls = 0

    def __init__(
        self,
        *,
        db_path,
        embedding_tracker=None,
        clear_db=False,
        window_size=5,
        overlap_size=1,
        **kwargs,
    ):
        del kwargs
        self.db_path = Path(db_path)
        self.embedding_tracker = embedding_tracker
        self.phase_counts = {"build_memory": 0, "retrieval": 0, "answer_llm": 0}
        self.vector_store = _FakeSimpleMemVectorStore(str(self.db_path), tracker=self.embedding_tracker)
        if clear_db:
            self.vector_store.clear()
        self.memory_builder = _FakeSimpleMemBuilder(self, window_size=window_size, overlap_size=overlap_size)

    def export_builder_state(self):
        return self.memory_builder.export_state()

    def import_builder_state(self, payload):
        self.memory_builder.import_state(payload)

    def finalize(self):
        return self.memory_builder.process_remaining()

    def retrieve(self, question, enable_reflection=None):
        del question, enable_reflection
        self.phase_counts["retrieval"] += 1
        if self.embedding_tracker is not None:
            self.embedding_tracker.record({"prompt_tokens": 1, "total_tokens": 1})
        return self.vector_store.get_all_entries()

    def ask(
        self,
        question,
        *,
        answer_prompt_override=None,
        enable_reflection=None,
        precomputed_contexts=None,
        return_debug=False,
    ):
        del question, enable_reflection
        self.phase_counts["answer_llm"] += 1
        contexts = list(precomputed_contexts or [])
        matched_log_id = ""
        prompt_text = str(answer_prompt_override or "")
        match = re.search(r"app_log_ids=([^\n]+)", prompt_text)
        if match:
            first_value = str(match.group(1) or "").split(",")[0].strip()
            if first_value and first_value != "<none>":
                matched_log_id = first_value
        raw_output = {
            "snapshot_state": {"profile_state:favorite_coffee": "latte"},
            "evidence": {
                "profile_state:favorite_coffee": [
                    {
                        "app_log_id": matched_log_id,
                        "evidence_content": contexts[0].lossless_restatement if contexts else "",
                    }
                ]
            },
        }
        if not return_debug:
            return raw_output
        return {
            "raw_output": raw_output,
            "prompt": "{}\n\n[Retrieved Agent Memory]\n{}".format(
                answer_prompt_override or "",
                "\n".join(entry.lossless_restatement for entry in contexts),
            ).strip(),
            "contexts": [entry.to_dict() for entry in contexts],
        }

    def llm_usage_summary(self, phase=None):
        if phase is None:
            return _usage_summary(sum(self.phase_counts.values()))
        return _usage_summary(self.phase_counts.get(str(phase), 0))

    def close(self):
        return None


def _fake_run_pipeline(**kwargs):
    benchmark = json.loads(Path(kwargs["benchmark_path"]).read_text(encoding="utf-8"))
    app_logs = json.loads(Path(kwargs["app_logs_path"]).read_text(encoding="utf-8"))
    rows = []
    prepare_checkpoint_state = kwargs["prepare_checkpoint_state"]
    retrieve_context_for_query = kwargs["retrieve_context_for_query"]
    answer_query = kwargs.get("answer_query")
    ask_json = kwargs["ask_json"]
    close = kwargs["close"]
    try:
        for checkpoint in benchmark.get("checkpoints") or []:
            as_of = checkpoint.get("as_of") or {}
            log_index = int(as_of.get("log_index") or -1)
            memory_pool = app_logs[: log_index + 1] if log_index >= 0 else []
            handle = prepare_checkpoint_state(checkpoint, memory_pool)
            key = "profile_state:favorite_coffee"
            retrieval = retrieve_context_for_query(
                handle,
                SimpleNamespace(
                    task_name="Task A",
                    retrieval_query_text="Retrieve evidence for {}".format(checkpoint["checkpoint_id"]),
                    answer_query_text="Infer the user's current favorite coffee.",
                    checkpoint_timestamp=str(as_of.get("timestamp") or ""),
                    target_keys=[key],
                    task_payload={"target_value_templates": {key: "<fill the blank>"}},
                ),
                RetrievalOptions(common={"top_k": 10}, backend={}),
                memory_pool,
            )
            query_spec = SimpleNamespace(
                task_name="Task A",
                retrieval_query_text="Retrieve evidence for {}".format(checkpoint["checkpoint_id"]),
                answer_query_text="Infer the user's current favorite coffee.",
                checkpoint_timestamp=str(as_of.get("timestamp") or ""),
                target_keys=[key],
                task_payload={"target_value_templates": {key: "<fill the blank>"}},
            )
            if answer_query is not None:
                answer = answer_query(handle, query_spec, retrieval)
            else:
                prompt = build_state_completion_prompt(
                    checkpoint={"as_of": {"timestamp": str(as_of.get("timestamp") or "")}},
                    context_logs=None,
                    target_keys=[key],
                    target_value_templates={key: "<fill the blank>"},
                    memory_prompt_mode=kwargs.get("memory_prompt_mode", "inline_memory"),
                    inline_memory_blocks=list(retrieval.inline_memory_blocks),
                    task_text_override=query_spec.answer_query_text,
                )
                answer = SimpleNamespace(
                    raw_output=ask_json(prompt),
                    prompt=prompt,
                    debug_metadata={"retrieval_metadata": dict(retrieval.debug_metadata or {})},
                )
            rows.append(
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "retrieval_query": "Retrieve evidence for {}".format(checkpoint["checkpoint_id"]),
                    "retrieval_mode": retrieval.mode,
                    "debug_metadata": dict(retrieval.debug_metadata),
                    "prompt": answer.prompt,
                    "raw_output": answer.raw_output,
                }
            )
        Path(kwargs["output_path"]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"predictions": rows}
    finally:
        close()


class SimpleMemTceTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSimpleMemSystem.build_calls = 0

    def test_adapter_passes_simplemem_specific_extras(self) -> None:
        captured = {}

        def _fake_run_generation(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        args = TceAdapterArgs(
            baseline="simplemem",
            user_id="001_user_001",
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retriever_batch_size=16,
            retrieval_top_k=12,
            extras={
                "snapshot_dir": "/tmp/simplemem_snapshots",
                "data_storage_path": "/tmp/simplemem_runtime",
                "build_only": "true",
                "builder_llm_provider": "azure",
                "builder_llm_model": "gpt-5-mini",
                "builder_llm_max_workers": "2",
                "builder_llm_temperature": "0.2",
                "window_size": "7",
                "overlap_size": "2",
                "save_every_logs": "9",
                "semantic_top_k": "11",
                "keyword_top_k": "13",
                "structured_top_k": "15",
                "enable_parallel_processing": "true",
                "max_parallel_workers": "4",
                "enable_planning": "true",
                "enable_reflection": "true",
                "max_reflection_rounds": "3",
                "enable_parallel_retrieval": "true",
                "max_retrieval_workers": "5",
            },
        )
        with mock.patch("generation.simplemem.generation_tce.tce.run_generation", side_effect=_fake_run_generation):
            simplemem_adapter.run(args)

        self.assertEqual(captured["snapshot_dir"], "/tmp/simplemem_snapshots")
        self.assertEqual(captured["data_storage_path"], "/tmp/simplemem_runtime")
        self.assertTrue(captured["build_only"])
        self.assertFalse(captured["predict_from_prebuilt"])
        self.assertEqual(captured["builder_llm_provider"], "azure")
        self.assertEqual(captured["window_size"], 7)
        self.assertEqual(captured["overlap_size"], 2)
        self.assertEqual(captured["save_every_logs"], 9)
        self.assertEqual(captured["semantic_top_k"], 11)
        self.assertEqual(captured["keyword_top_k"], 13)
        self.assertEqual(captured["structured_top_k"], 15)
        self.assertTrue(captured["enable_parallel_processing"])
        self.assertEqual(captured["max_parallel_workers"], 4)
        self.assertTrue(captured["enable_planning"])
        self.assertTrue(captured["enable_reflection"])
        self.assertEqual(captured["max_reflection_rounds"], 3)
        self.assertTrue(captured["enable_parallel_retrieval"])
        self.assertEqual(captured["max_retrieval_workers"], 5)

    def test_adapter_routes_predict_from_prebuilt(self) -> None:
        captured = {}

        def _fake_run_generation(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        args = TceAdapterArgs(
            baseline="simplemem",
            user_id="001_user_001",
            benchmark=Path("/tmp/benchmark.json"),
            app_logs_path=Path("/tmp/app_logs.json"),
            output=Path("/tmp/out.json"),
            llm_provider="openai",
            llm_model="gpt-5-mini",
            llm_max_workers=1,
            retriever_provider="openai",
            retriever_model="text-embedding-3-large",
            retriever_batch_size=16,
            retrieval_top_k=12,
            extras={
                "snapshot_dir": "/tmp/simplemem_snapshots",
                "data_storage_path": "/tmp/simplemem_runtime",
                "memory_action": "predict_from_prebuilt",
            },
        )
        with mock.patch("generation.simplemem.generation_tce.tce.run_generation", side_effect=_fake_run_generation):
            simplemem_adapter.run(args)

        self.assertFalse(captured["build_only"])
        self.assertTrue(captured["predict_from_prebuilt"])

    def test_run_generation_builds_snapshots_and_retrieves_raw_logs_from_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_pred.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(
                simplemem_tce, "run_pipeline", side_effect=_fake_run_pipeline
            ):
                result = simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )

            progress = json.loads((data_storage_path / "builder_progress.json").read_text(encoding="utf-8"))
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "build_completed")
            self.assertEqual(progress["resume_source"], "snapshot_bundle")
            self.assertEqual(progress["completed_snapshot_ids"], ["cp1", "cp2"])
            self.assertEqual(progress["confirmed_last_log_idx"], 2)
            self.assertEqual(len(manifest["snapshots"]), 2)

            cp1_entry = next(entry for entry in manifest["snapshots"] if entry["checkpoint_id"] == "cp1")
            cp2_entry = next(entry for entry in manifest["snapshots"] if entry["checkpoint_id"] == "cp2")
            latest_checkpoint_payload = json.loads((snapshot_dir / cp2_entry["checkpoint_path"]).read_text(encoding="utf-8"))
            self.assertIn("builder_state", latest_checkpoint_payload)
            cp1_lineage = json.loads(
                (snapshot_dir / cp1_entry["db_dir"]).joinpath("entry_lineage.json").read_text(
                    encoding="utf-8"
                )
            )
            cp2_lineage = json.loads(
                (snapshot_dir / cp2_entry["db_dir"]).joinpath("entry_lineage.json").read_text(
                    encoding="utf-8"
                )
            )
            cp1_source_ids = sorted(
                {
                    source
                    for source_ids in (cp1_lineage.get("entry_lineage") or {}).values()
                    for source in (source_ids or [])
                }
            )
            cp2_source_ids = sorted(
                {
                    source
                    for source_ids in (cp2_lineage.get("entry_lineage") or {}).values()
                    for source in (source_ids or [])
                }
            )
            self.assertEqual(cp1_source_ids, ["log_0001", "log_0002"])
            self.assertEqual(cp2_source_ids, ["log_0001", "log_0002", "log_0003"])

            predictions = result["predictions"]
            self.assertEqual(predictions[0]["checkpoint_id"], "cp1")
            self.assertEqual(predictions[0]["retrieval_query"], "Retrieve evidence for cp1")
            self.assertEqual(predictions[0]["retrieval_mode"], "inline_memory")
            self.assertIn("[Memory]", predictions[0]["prompt"])
            self.assertIn('"app_log_id": "log_0001"', predictions[0]["prompt"])
            self.assertEqual(predictions[1]["debug_metadata"]["retrieved_app_log_ids"][-1], "log_0003")
            self.assertTrue(predictions[1]["debug_metadata"]["retrieved_memory_entry_ids"])

    def test_resume_restores_from_snapshot_without_builder_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_pred.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(
                simplemem_tce, "run_pipeline", side_effect=_fake_run_pipeline
            ):
                simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )
                initial_build_calls = _FakeSimpleMemSystem.build_calls
                progress_path = data_storage_path / "builder_progress.json"
                self.assertTrue(progress_path.exists())
                if progress_path.exists():
                    progress_path.unlink()
                if (data_storage_path / "simplemem_live_db").exists():
                    import shutil

                    shutil.rmtree(data_storage_path / "simplemem_live_db", ignore_errors=True)
                simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=True,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )

            self.assertEqual(_FakeSimpleMemSystem.build_calls, initial_build_calls)
            live_usage = json.loads((output_path.parent / "usage_cost_live.json").read_text(encoding="utf-8"))
            final_usage = json.loads((output_path.parent / "usage_cost.json").read_text(encoding="utf-8"))
            self.assertIn("build_memory", live_usage["usage"])
            self.assertIn("retrieval", live_usage["usage"])
            self.assertIn("answer_llm", live_usage["usage"])
            self.assertIn("build_memory_duration_s", final_usage["timing"])
            self.assertGreaterEqual(final_usage["usage"]["build_memory"]["request_count"], 1)

    def test_build_only_skips_generation_and_writes_builder_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_build_only.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(
                simplemem_tce, "run_pipeline", side_effect=AssertionError("run_pipeline should not execute in build_only mode")
            ):
                result = simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    build_only=True,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )

            self.assertEqual(result["predictions"], [])
            self.assertEqual(result["build_only"]["snapshot_dir"], str(snapshot_dir))
            self.assertEqual(result["build_only"]["data_storage_path"], str(data_storage_path))
            self.assertTrue((snapshot_dir / "manifest.json").exists())
            self.assertTrue((data_storage_path / "builder_progress.json").exists())

    def test_build_only_resume_continues_from_checkpoint_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_build_only.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(
                simplemem_tce, "run_pipeline", side_effect=AssertionError("run_pipeline should not execute in build_only mode")
            ):
                simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    build_only=True,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )
                first_manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual([entry["checkpoint_id"] for entry in first_manifest["snapshots"]], ["cp1"])
                cp1_checkpoint_path = snapshot_dir / first_manifest["snapshots"][0]["checkpoint_path"]
                cp1_checkpoint = json.loads(cp1_checkpoint_path.read_text(encoding="utf-8"))
                cp1_checkpoint["source_data_fingerprint"] = None
                cp1_checkpoint["app_logs_path"] = "app_logs.json"
                _write_json(cp1_checkpoint_path, cp1_checkpoint)

                with mock.patch.object(simplemem_tce, "REPO_ROOT_DIR", root):
                    result = simplemem_tce.run_generation(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        output_path=output_path,
                        snapshot_dir=str(snapshot_dir),
                        data_storage_path=str(data_storage_path),
                        max_visible_logs=None,
                        llm_provider="openai",
                        llm_model="gpt-5-mini",
                        llm_max_workers=1,
                        retriever_provider="openai",
                        retriever_model="text-embedding-3-large",
                        retriever_batch_size=8,
                        resume=True,
                        max_checkpoints=None,
                        debug=False,
                        debug_dir=None,
                        save_prompt_and_raw=False,
                        retrieval_top_k=10,
                        build_only=True,
                        allow_destructive_rebuild=False,
                        window_size=2,
                        overlap_size=0,
                        save_every_logs=1,
                    )

            self.assertEqual(result["predictions"], [])
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            progress = json.loads((data_storage_path / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["checkpoint_id"] for entry in manifest["snapshots"]], ["cp1", "cp2"])
            self.assertEqual(progress["completed_snapshot_ids"], ["cp1", "cp2"])
            self.assertEqual(progress["confirmed_last_log_idx"], 2)

    def test_resume_reuses_build_only_memory_when_only_eval_pack_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_benchmark_path = root / "benchmark_build.json"
            eval_benchmark_path = root / "benchmark_eval.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_pred.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(build_benchmark_path, _benchmark_payload())
            _write_json(eval_benchmark_path, _benchmark_payload_with_updated_eval_pack())
            _write_json(app_logs_path, _app_logs_payload())

            def _pack_echo_run_pipeline(**kwargs):
                benchmark = json.loads(Path(kwargs["benchmark_path"]).read_text(encoding="utf-8"))
                prepare_checkpoint_state = kwargs["prepare_checkpoint_state"]
                close = kwargs["close"]
                rows = []
                try:
                    for checkpoint in benchmark.get("checkpoints") or []:
                        handle = prepare_checkpoint_state(checkpoint, [])
                        item = (
                            (((checkpoint.get("state_completion_pack") or {}).get("keys") or {}).get("profile_state:favorite_coffee"))
                            or {}
                        )
                        rows.append(
                            {
                                "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                                "question_text": str(item.get("question_text") or ""),
                                "snapshot_id": str((handle.metadata or {}).get("snapshot_id") or ""),
                            }
                        )
                    Path(kwargs["output_path"]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    return {"predictions": rows}
                finally:
                    close()

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(simplemem_tce, "run_pipeline", side_effect=_pack_echo_run_pipeline):
                simplemem_tce.run_generation(
                    benchmark_path=build_benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    build_only=True,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )
                initial_build_calls = _FakeSimpleMemSystem.build_calls
                progress_path = data_storage_path / "builder_progress.json"
                if progress_path.exists():
                    progress_path.unlink()
                if (data_storage_path / "simplemem_live_db").exists():
                    import shutil

                    shutil.rmtree(data_storage_path / "simplemem_live_db", ignore_errors=True)
                result = simplemem_tce.run_generation(
                    benchmark_path=eval_benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=True,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    predict_from_prebuilt=True,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )

            self.assertEqual(_FakeSimpleMemSystem.build_calls, initial_build_calls)
            self.assertEqual(
                [row["question_text"] for row in result["predictions"]],
                [
                    "UPDATED PACK question for cp1",
                    "UPDATED PACK question for cp2",
                ],
            )
            progress = json.loads((data_storage_path / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "build_completed")
            self.assertEqual(progress["benchmark_path"], str(eval_benchmark_path.resolve()))
            self.assertTrue(progress.get("source_data_fingerprint"))
            self.assertTrue(progress.get("evaluation_pack_fingerprint"))

    def test_predict_from_prebuilt_fails_closed_when_snapshots_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction" / "simplemem_pred.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(simplemem_tce, "run_pipeline", side_effect=AssertionError("run_pipeline should not execute")):
                with self.assertRaisesRegex(FileNotFoundError, "predict_from_prebuilt requires prebuilt checkpoint snapshots"):
                    simplemem_tce.run_generation(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        output_path=output_path,
                        snapshot_dir=str(snapshot_dir),
                        data_storage_path=str(data_storage_path),
                        max_visible_logs=None,
                        llm_provider="openai",
                        llm_model="gpt-5-mini",
                        llm_max_workers=1,
                        retriever_provider="openai",
                        retriever_model="text-embedding-3-large",
                        retriever_batch_size=8,
                        resume=True,
                        max_checkpoints=None,
                        debug=False,
                        debug_dir=None,
                        save_prompt_and_raw=False,
                        retrieval_top_k=10,
                        predict_from_prebuilt=True,
                        window_size=2,
                        overlap_size=0,
                        save_every_logs=1,
                    )

            self.assertEqual(_FakeSimpleMemSystem.build_calls, 0)

    def test_resume_refuses_destructive_rebuild_without_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            changed_app_logs_path = root / "app_logs_changed.json"
            output_path = root / "prediction" / "simplemem_pred.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, _benchmark_payload())
            _write_json(app_logs_path, _app_logs_payload())
            changed_logs = _app_logs_payload()
            changed_logs[0]["request"]["favorite_coffee"] = "mocha"
            _write_json(changed_app_logs_path, changed_logs)

            with mock.patch.object(simplemem_tce, "SimpleMemSystem", _FakeSimpleMemSystem), mock.patch.object(
                simplemem_tce, "SharedLLMClient", _FakeLLMClient
            ), mock.patch.object(
                simplemem_tce, "run_pipeline", side_effect=_fake_run_pipeline
            ):
                simplemem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    retriever_batch_size=8,
                    resume=False,
                    max_checkpoints=None,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=False,
                    retrieval_top_k=10,
                    build_only=True,
                    allow_destructive_rebuild=True,
                    window_size=2,
                    overlap_size=0,
                    save_every_logs=1,
                )

                with self.assertRaisesRegex(RuntimeError, "refused to delete existing memory artifacts"):
                    simplemem_tce.run_generation(
                        benchmark_path=benchmark_path,
                        app_logs_path=changed_app_logs_path,
                        output_path=output_path,
                        snapshot_dir=str(snapshot_dir),
                        data_storage_path=str(data_storage_path),
                        max_visible_logs=None,
                        llm_provider="openai",
                        llm_model="gpt-5-mini",
                        llm_max_workers=1,
                        retriever_provider="openai",
                        retriever_model="text-embedding-3-large",
                        retriever_batch_size=8,
                        resume=True,
                        max_checkpoints=None,
                        debug=False,
                        debug_dir=None,
                        save_prompt_and_raw=False,
                        retrieval_top_k=10,
                        build_only=False,
                        allow_destructive_rebuild=False,
                        window_size=2,
                        overlap_size=0,
                        save_every_logs=1,
                    )

            self.assertTrue((snapshot_dir / "manifest.json").exists())
            progress = json.loads((data_storage_path / "builder_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "build_completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
