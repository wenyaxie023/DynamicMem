#!/usr/bin/env python3
import json
import pickle
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.Amem import amem as amem_builder
from generation.Amem import tce as amem_tce
from generation.MemoryOS import tce_adapter as memoryos_tce
from generation.mem0 import tce as mem0_tce
from generation.rag import rag_tce


class _FakeLLMClient:
    _LOG_RE = re.compile(
        r'"app_log_id":\s*"([^"]+)".*?"response":\s*\{"favorite_coffee":\s*"([^"]+)"\}',
        re.DOTALL,
    )

    def __init__(self, *args, **kwargs):
        return None

    def _extract_entries(self, prompt: str):
        entries = list(self._LOG_RE.findall(prompt))
        if entries:
            return entries
        normalized = prompt.replace('\\"', '"')
        return list(self._LOG_RE.findall(normalized))

    def _select_current_and_previous(self, prompt: str):
        entries = self._extract_entries(prompt)
        is_amem_native_context = "talk start time:" in prompt and "memory content:" in prompt
        if is_amem_native_context:
            current = entries[0] if entries else ("", None)
            previous = entries[1] if len(entries) >= 2 else ("", None)
        else:
            current = entries[-1] if entries else ("", None)
            previous = entries[-2] if len(entries) >= 2 else ("", None)
        return entries, current, previous

    def ask(self, prompt: str, response_type: str = "json"):
        if "generate several keywords" in prompt.lower() and '"keywords"' in prompt.lower():
            if "espresso" in prompt:
                payload = {"keywords": "espresso, coffee"}
            elif "latte" in prompt:
                payload = {"keywords": "latte, coffee"}
            else:
                payload = {"keywords": "coffee, preference"}
            if response_type == "text":
                return payload["keywords"]
            return payload
        if response_type == "text":
            return prompt
        entries, current_entry, previous_entry = self._select_current_and_previous(prompt)
        current_log_id = current_entry[0] if entries else ""
        current_value = current_entry[1] if entries else None
        previous_value = previous_entry[1] if len(entries) >= 2 else None
        if '"change_analysis"' in prompt:
            before_value = previous_value
            after_value = current_value
            evidence = []
            if current_log_id and after_value is not None:
                evidence = [{"app_log_id": current_log_id, "evidence_content": "favorite_coffee {}".format(after_value)}]
            return {
                "change_analysis": {
                    "profile_state:favorite_coffee": {
                        "before": before_value,
                        "after": after_value,
                        "change_reason": "updated coffee preference" if after_value is not None else "",
                        "evidence": evidence,
                    }
                }
            }
        if '"answer"' in prompt and '"snapshot_state"' not in prompt:
            evidence = []
            if current_log_id and current_value is not None:
                evidence = [{"app_log_id": current_log_id, "evidence_content": "favorite_coffee {}".format(current_value)}]
            return {
                "answer": "Recommend {}".format(current_value) if current_value is not None else "",
                "evidence": evidence,
            }
        evidence = []
        if current_log_id and current_value is not None:
            evidence = [{"app_log_id": current_log_id, "evidence_content": "favorite_coffee {}".format(current_value)}]
        return {
            "snapshot_state": {"profile_state:favorite_coffee": current_value},
            "evidence": {"profile_state:favorite_coffee": evidence},
        }

    def ask_structured(self, prompt: str, text_format):
        return self.ask(prompt, response_type="json")

    def supports_structured_response(self):
        return True

    def close(self):
        return None


class _FakeRagLLMClient(_FakeLLMClient):
    def ask(self, prompt: str, response_type: str = "json"):
        entries = self._extract_entries(prompt)
        first_log_id = entries[0][0] if entries else ""
        first_value = entries[0][1] if entries else None
        if '"change_analysis"' in prompt:
            before_value = entries[1][1] if len(entries) >= 2 else None
            after_value = first_value
            evidence = []
            if first_log_id and after_value is not None:
                evidence = [{"app_log_id": first_log_id, "evidence_content": "favorite_coffee {}".format(after_value)}]
            return {
                "change_analysis": {
                    "profile_state:favorite_coffee": {
                        "before": before_value,
                        "after": after_value,
                        "change_reason": "updated coffee preference" if after_value is not None else "",
                        "evidence": evidence,
                    }
                }
            }
        if '"answer"' in prompt and '"snapshot_state"' not in prompt:
            evidence = []
            if first_log_id and first_value is not None:
                evidence = [{"app_log_id": first_log_id, "evidence_content": "favorite_coffee {}".format(first_value)}]
            return {
                "answer": "Recommend {}".format(first_value) if first_value is not None else "",
                "evidence": evidence,
            }
        evidence = []
        if first_log_id and first_value is not None:
            evidence = [{"app_log_id": first_log_id, "evidence_content": "favorite_coffee {}".format(first_value)}]
        return {
            "snapshot_state": {"profile_state:favorite_coffee": first_value},
            "evidence": {"profile_state:favorite_coffee": evidence},
        }


def _fake_rag_embed_texts(client, model: str, texts, batch_size: int = 64):
    rows = []
    for text in texts:
        text = str(text)
        if "espresso" in text:
            rows.append([2.0])
        elif "latte" in text:
            rows.append([1.0])
        elif "What drink should the assistant recommend?" in text:
            rows.append([2.0])
        else:
            rows.append([1.0])
    return np.array(rows, dtype="float32")


class _FakeAmemRetriever:
    def __init__(self, *args, **kwargs):
        self.directory = Path(str(kwargs.get("directory", "")))
        return None

    def search(self, query: str, k: int = 1):
        del query
        docs_path = self.directory / "documents.json"
        if not docs_path.exists():
            return []
        documents = json.loads(docs_path.read_text(encoding="utf-8"))
        count = max(1, min(k, len(documents)))
        start_index = len(documents) - count
        return list(range(len(documents) - 1, start_index - 1, -1))


class _FakeAmemLiveRetriever:
    def __init__(self, owner):
        self.owner = owner

    def clone_collection_to_directory(
        self,
        dest_directory,
        dest_collection_name=None,
        overwrite=False,
        batch_size=100,
    ):
        dest = Path(dest_directory)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "documents.json").write_text(
            json.dumps(list(self.owner.documents), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "dest_directory": str(dest),
            "collection_name": dest_collection_name or self.owner.collection_name,
            "count": len(self.owner.documents),
        }


class _FakeAmemMemorySystem:
    fail_after_calls = None
    added_app_log_ids = []

    def __init__(self, *args, **kwargs):
        self.collection_name = kwargs.get("collection_name", "memories")
        self.documents = []
        self.memories = {}
        self.retriever = _FakeAmemLiveRetriever(self)

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"documents": list(self.documents)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path):
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        self.documents = list(payload.get("documents", []))
        self.memories = {
            f"note_{idx}": types.SimpleNamespace(content=content, id=f"note_{idx}")
            for idx, content in enumerate(self.documents, start=1)
        }
        self.retriever = _FakeAmemLiveRetriever(self)

    def rebuild_retriever(self):
        self.retriever = _FakeAmemLiveRetriever(self)

    def add_note(self, content: str, time: str = None, **kwargs):
        del time, kwargs
        try:
            payload = json.loads(content)
        except Exception:
            lines = [line.strip() for line in str(content).splitlines() if line.strip()]
            if not lines or not lines[0].startswith("User:"):
                raise
            payload = json.loads(lines[0].split("User:", 1)[1].strip())
        app_log_id = str(payload.get("app_log_id") or "")
        _FakeAmemMemorySystem.added_app_log_ids.append(app_log_id)
        if (
            isinstance(_FakeAmemMemorySystem.fail_after_calls, int)
            and len(_FakeAmemMemorySystem.added_app_log_ids) > _FakeAmemMemorySystem.fail_after_calls
        ):
            raise RuntimeError("synthetic ingest failure")
        note_id = f"note_{len(self.documents) + 1}"
        self.documents.append(content)
        self.memories[note_id] = types.SimpleNamespace(content=content, id=note_id)
        return note_id


def _run_fake_amem_builder(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    checkpoint_dir: Path,
    snapshot_root: Path,
    fail_after_calls=None,
    resume: bool = False,
):
    resolved_snapshot_root = snapshot_root / "001_user_001" / "large"
    canonical_user_dir = app_logs_path.parent / "001_user_001"
    canonical_user_dir.mkdir(parents=True, exist_ok=True)
    canonical_app_logs_path = canonical_user_dir / "app_log_large.json"
    canonical_app_logs_path.write_text(app_logs_path.read_text(encoding="utf-8"), encoding="utf-8")

    _FakeAmemMemorySystem.fail_after_calls = fail_after_calls
    _FakeAmemMemorySystem.added_app_log_ids = []
    with mock.patch("generation.Amem.amem.AgenticMemorySystem", _FakeAmemMemorySystem), mock.patch(
        "generation.Amem.amem._ensure_nltk", lambda: None
    ):
        summary = amem_builder.evaluate_membench(
            user_id="001_user_001",
            app_log_path=str(canonical_app_logs_path),
            benchmark_path=str(benchmark_path),
            size="large",
            embedding_model_name="text-embedding-3-large",
            embedding_backend="openai",
            llm_controller_backend="openai",
            llm_controller_model_name="gpt-5-mini",
            resume=resume,
            checkpoint_dir=str(checkpoint_dir),
            save_every=5,
            snapshot_dir=str(snapshot_root),
        )
    return summary, resolved_snapshot_root


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
        if '"snapshot_state"' in prompt:
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


class _FakeMem0Memory:
    store_by_collection = {}
    search_events = []

    def __init__(self, config):
        self.config = config
        self.collection_name = config["vector_store"]["config"]["collection_name"]
        _FakeMem0Memory.store_by_collection.setdefault(self.collection_name, [])

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def add(self, messages, metadata=None, user_id=None):
        _FakeMem0Memory.store_by_collection.setdefault(self.collection_name, []).append(
            {
                "memory": messages[0]["content"],
                "metadata": dict(metadata or {}),
                "user_id": user_id,
            }
        )

    def search(self, query, user_id=None):
        before = len(_FakeMem0Memory.store_by_collection.get(self.collection_name, []))
        results = list(_FakeMem0Memory.store_by_collection.get(self.collection_name, []))
        after = len(_FakeMem0Memory.store_by_collection.get(self.collection_name, []))
        _FakeMem0Memory.search_events.append(
            {
                "collection": self.collection_name,
                "query": query,
                "before": before,
                "after": after,
            }
        )
        return {"results": results}

    def delete_all_memories(self):
        _FakeMem0Memory.store_by_collection[self.collection_name] = []


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_app_logs(values):
    logs = []
    for idx, value in enumerate(values, start=1):
        logs.append(
            {
                "app_log_id": "log_{:04d}".format(idx),
                "timestamp": "2025-01-{:02d} 08:00:00".format(idx),
                "app_name": "coffee_app",
                "api_name": "save_preference",
                "request": {},
                "response": {"favorite_coffee": value},
            }
        )
    return logs


def _make_checkpoint(idx: int, value: str, previous_value=None):
    checkpoint_id = "cp_{:04d}".format(idx)
    app_log_id = "log_{:04d}".format(idx)
    state_key = "profile_state:favorite_coffee"
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "as_of": {
            "timestamp": "2025-01-{:02d} 08:00:00".format(idx),
            "log_index": idx - 1,
            "app_log_id": app_log_id,
        },
        "validated_snapshot_state": {"profile_state": {"favorite_coffee": value}},
        "state_completion_pack": {
            "version": "v1",
            "pack_authoring": "deterministic",
            "keys": {
                state_key: {
                    "item_id": "scp_{}".format(idx),
                    "state_key": state_key,
                    "question_text": "Infer the user's current favorite coffee.",
                    "answer_template": "<fill the blank>",
                    "retrieval_query": "Infer the user's current favorite coffee.",
                }
            },
        },
        "rq3_apply_service_qa": {
            "version": "v1",
            "keys": {
                state_key: {
                    "items": [
                        {
                            "qa_id": "rq3_{}".format(idx),
                            "service_category": "beverage_recommendation",
                            "apply_scenario": "A cafe reminder should suggest one drink.",
                            "apply_question": "What drink should the assistant recommend?",
                            "apply_reference_answer": "Recommend {}".format(value),
                            "retrieval_query": "What drink should the assistant recommend?",
                        }
                    ]
                }
            },
        },
    }
    if previous_value is None:
        checkpoint["change_tracking_pack"] = {
            "version": "v1",
            "pack_authoring": "deterministic",
            "previous_checkpoint_id": "",
            "previous_cutoff_ts": "",
            "keys": {},
        }
    else:
        checkpoint["change_tracking_pack"] = {
            "version": "v1",
            "pack_authoring": "deterministic",
            "previous_checkpoint_id": "cp_{:04d}".format(idx - 1),
            "previous_cutoff_ts": "2025-01-{:02d} 08:00:00".format(idx - 1),
            "keys": {
                state_key: {
                    "item_id": "ctp_{}".format(idx),
                    "question_text": "Explain why the favorite coffee changed.",
                    "retrieval_query": "Explain why the favorite coffee changed.",
                    "before_template": "<fill the blank>",
                    "after_template": "<fill the blank>",
                }
            },
        }
    return checkpoint


def _make_benchmark(values):
    checkpoints = []
    previous_value = None
    for idx, value in enumerate(values, start=1):
        checkpoints.append(_make_checkpoint(idx, value, previous_value=previous_value))
        previous_value = value
    return {"user_id": "001_user_001", "sampling_strategy": {"stage": "benchmark_build"}, "checkpoints": checkpoints}


def _make_qa_list(value: str, app_log_id: str):
    return [
        {
            "id": 1,
            "query": "What drink should the assistant recommend?",
            "reference": "Recommend {}".format(value),
            "metadata": {"app_log_ids": [app_log_id]},
        }
    ]


class SnapshotBaselineAcceptance(unittest.TestCase):
    def test_amem_minimal_pack_first_generation(self):
        benchmark = _make_benchmark(["latte"])
        app_logs = _make_app_logs(["latte"])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            checkpoint_dir = root / "ckpts"
            snapshot_root = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            fake_retrievers_module = types.ModuleType("generation.Amem.agentic_memory.retrievers")
            fake_retrievers_module.SimpleEmbeddingRetriever = _FakeAmemRetriever
            _, resolved_snapshot_root = _run_fake_amem_builder(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                checkpoint_dir=checkpoint_dir,
                snapshot_root=snapshot_root,
                fail_after_calls=None,
                resume=False,
            )
            with mock.patch("generation.Amem.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.Amem.tce._ensure_nltk", lambda: None
            ), mock.patch.dict(
                sys.modules,
                {"generation.Amem.agentic_memory.retrievers": fake_retrievers_module},
            ):
                result = amem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    checkpoint_dir=str(checkpoint_dir),
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_backend="openai",
                    embedding_model_name="text-embedding-3-large",
                    embedding_api_key=None,
                    embedding_api_base_url=None,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    snapshot_root=str(resolved_snapshot_root),
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )
            manifest_exists = (resolved_snapshot_root / "manifest.json").exists()
            manifest_payload = json.loads((resolved_snapshot_root / "manifest.json").read_text(encoding="utf-8"))
            snapshot_entry = manifest_payload["snapshots"][0]
            docs_path = resolved_snapshot_root / snapshot_entry["chroma_dir"] / "documents.json"
            snapshot_documents = json.loads(docs_path.read_text(encoding="utf-8"))

        prediction = result["predictions"][0]
        retrieval_record = prediction["metadata"]["per_key_retrieval"][0]
        retrieval_meta = retrieval_record["retrieval_metadata"]
        self.assertEqual(prediction["snapshot_state"]["profile_state:favorite_coffee"], "latte")
        self.assertEqual(prediction["metadata"]["baseline"], "amem")
        self.assertEqual(prediction["metadata"]["requested_checkpoint_workers"], 1)
        self.assertEqual(prediction["metadata"]["effective_checkpoint_workers"], 1)
        self.assertTrue(manifest_exists)
        self.assertEqual(retrieval_record["retrieval_query"], "Infer the user's current favorite coffee.")
        self.assertEqual(retrieval_meta["retrieval_mode"], "amem_snapshot_native")
        self.assertEqual(retrieval_meta["generated_keywords"], "coffee, preference")
        self.assertEqual(retrieval_meta["retrieved_app_log_ids"], ["log_0001"])
        self.assertIn('"app_log_id": "log_0001"', retrieval_meta["native_raw_context"])
        self.assertTrue(snapshot_documents[0].startswith('{"app_log_id": "log_0001"'))
        self.assertFalse(snapshot_documents[0].startswith("User:"))
        self.assertTrue(prediction["metadata"]["rq3_apply"]["enabled"])
        self.assertEqual(
            prediction["rq3_apply_answers"]["profile_state:favorite_coffee"]["items"][0]["answer"],
            "Recommend latte",
        )

    def test_memoryos_minimal_pack_first_generation(self):
        benchmark = _make_benchmark(["latte"])
        app_logs = _make_app_logs(["latte"])

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

            def _fake_memoryos_instance(**kwargs):
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
                    retrieval_top_k=5,
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
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

        prediction = result["predictions"][0]
        self.assertEqual(prediction["snapshot_state"]["profile_state:favorite_coffee"], "latte")
        self.assertEqual(prediction["metadata"]["baseline"], "memoryos")
        self.assertEqual(prediction["metadata"]["requested_checkpoint_workers"], 1)
        self.assertEqual(prediction["metadata"]["effective_checkpoint_workers"], 1)
        self.assertEqual(
            prediction["metadata"]["per_key_retrieval"][0]["retrieval_metadata"]["retrieval_mode"],
            "memoryos_snapshot",
        )
        self.assertTrue(prediction["metadata"]["rq3_apply"]["enabled"])
        self.assertEqual(
            prediction["rq3_apply_answers"]["profile_state:favorite_coffee"]["items"][0]["answer"],
            "Recommend latte",
        )

    def test_amem_final_checkpoint_qa_uses_final_snapshot_memory(self):
        benchmark = _make_benchmark(["latte", "espresso"])
        app_logs = _make_app_logs(["latte", "espresso"])
        qa_list = _make_qa_list("espresso", "log_0002")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            qa_path = root / "qa.json"
            output_path = root / "prediction.json"
            checkpoint_dir = root / "ckpts"
            snapshot_root = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

            fake_retrievers_module = types.ModuleType("generation.Amem.agentic_memory.retrievers")
            fake_retrievers_module.SimpleEmbeddingRetriever = _FakeAmemRetriever
            _, resolved_snapshot_root = _run_fake_amem_builder(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                checkpoint_dir=checkpoint_dir,
                snapshot_root=snapshot_root,
                fail_after_calls=None,
                resume=False,
            )
            with mock.patch("generation.Amem.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.Amem.tce._ensure_nltk", lambda: None
            ), mock.patch.dict(
                sys.modules,
                {"generation.Amem.agentic_memory.retrievers": fake_retrievers_module},
            ):
                result = amem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    checkpoint_dir=str(checkpoint_dir),
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_backend="openai",
                    embedding_model_name="text-embedding-3-large",
                    embedding_api_key=None,
                    embedding_api_base_url=None,
                    resume=False,
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    snapshot_root=str(resolved_snapshot_root),
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=str(qa_path),
                    final_qa_retrieval_top_k=5,
                )

            final_qa_path = output_path.with_name("prediction_final_qa.json")
            final_qa_results = json.loads(final_qa_path.read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"][-1]["snapshot_state"]["profile_state:favorite_coffee"], "espresso")
        self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
        self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp_0002")

    def test_amem_builder_resume_uses_builder_progress_files(self):
        benchmark = {
            "user_id": "001_user_001",
            "sampling_strategy": {"stage": "benchmark_build"},
            "checkpoints": [_make_checkpoint(10, "latte", previous_value="latte")],
        }
        app_logs = _make_app_logs(["latte"] * 10)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            checkpoint_dir = root / "ckpts"
            snapshot_root = root / "snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            fake_retrievers_module = types.ModuleType("generation.Amem.agentic_memory.retrievers")
            fake_retrievers_module.SimpleEmbeddingRetriever = _FakeAmemRetriever

            with self.assertRaises(RuntimeError):
                _run_fake_amem_builder(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    checkpoint_dir=checkpoint_dir,
                    snapshot_root=snapshot_root,
                    fail_after_calls=7,
                    resume=False,
                )

            progress_path = checkpoint_dir / "membench_amem_001_user_001_large.json"
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["last_event_idx"], 6)
            self.assertEqual(progress_payload["events_processed"], 7)

            _, resolved_snapshot_root = _run_fake_amem_builder(
                benchmark_path=benchmark_path,
                app_logs_path=app_logs_path,
                checkpoint_dir=checkpoint_dir,
                snapshot_root=snapshot_root,
                fail_after_calls=None,
                resume=True,
            )
            self.assertEqual(_FakeAmemMemorySystem.added_app_log_ids, ["log_0008", "log_0009", "log_0010"])

            with mock.patch("generation.Amem.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.Amem.tce._ensure_nltk", lambda: None
            ), mock.patch.dict(
                sys.modules,
                {"generation.Amem.agentic_memory.retrievers": fake_retrievers_module},
            ):
                result = amem_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=None,
                    checkpoint_dir=str(checkpoint_dir),
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_backend="openai",
                    embedding_model_name="text-embedding-3-large",
                    embedding_api_key=None,
                    embedding_api_base_url=None,
                    resume=True,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    snapshot_root=str(resolved_snapshot_root),
                )

        self.assertEqual(
            result["predictions"][0]["snapshot_state"]["profile_state:favorite_coffee"],
            "latte",
        )

    def test_memoryos_final_checkpoint_qa_uses_final_snapshot_memory(self):
        benchmark = _make_benchmark(["latte", "espresso"])
        app_logs = _make_app_logs(["latte", "espresso"])
        qa_list = _make_qa_list("espresso", "log_0002")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            qa_path = root / "qa.json"
            output_path = root / "prediction.json"
            snapshot_root = root / "memoryos_snapshots"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")
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
                        },
                        {
                            "snapshot_id": "snap2",
                            "created_at": "2025-01-02T09:00:00",
                            "last_event_idx": 1,
                            "checkpoint_id": "cp_0002",
                            "checkpoint_app_log_id": "log_0002",
                            "short_term_path": "snap2/short_term.json",
                            "mid_term_path": "snap2/mid_term.json",
                            "long_term_path": "snap2/long_term_user.json",
                            "assistant_long_term_path": "snap2/long_term_assistant.json",
                        },
                    ]
                },
            )
            _write_json(
                snapshot_root / "snap1/short_term.json",
                [{"user_input": "[APP_LOG] {}".format(json.dumps(app_logs[0], ensure_ascii=False))}],
            )
            _write_json(snapshot_root / "snap1/mid_term.json", {"sessions": {}, "access_frequency": {}})
            _write_json(
                snapshot_root / "snap1/long_term_user.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )
            _write_json(
                snapshot_root / "snap1/long_term_assistant.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )
            _write_json(
                snapshot_root / "snap2/short_term.json",
                [
                    {"user_input": "[APP_LOG] {}".format(json.dumps(app_logs[0], ensure_ascii=False))},
                    {"user_input": "[APP_LOG] {}".format(json.dumps(app_logs[1], ensure_ascii=False))},
                ],
            )
            _write_json(snapshot_root / "snap2/mid_term.json", {"sessions": {}, "access_frequency": {}})
            _write_json(
                snapshot_root / "snap2/long_term_user.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )
            _write_json(
                snapshot_root / "snap2/long_term_assistant.json",
                {"user_profiles": {}, "knowledge_base": [], "assistant_knowledge": []},
            )

            def _fake_memoryos_instance(**kwargs):
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
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    embedding_model_name="text-embedding-3-large",
                    llm_controller_model="gpt-5-mini",
                    assistant_id="assistant",
                    resume=False,
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=str(qa_path),
                    final_qa_retrieval_top_k=5,
                )

            final_qa_path = output_path.with_name("prediction_final_qa.json")
            final_qa_results = json.loads(final_qa_path.read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"][-1]["snapshot_state"]["profile_state:favorite_coffee"], "espresso")
        self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
        self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp_0002")

    def test_rag_final_checkpoint_qa_uses_final_checkpoint_memory(self):
        benchmark = _make_benchmark(["latte", "espresso"])
        app_logs = _make_app_logs(["latte", "espresso"])
        qa_list = _make_qa_list("espresso", "log_0002")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            qa_path = root / "qa.json"
            output_path = root / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch("generation.rag.rag_tce.LLMClient", _FakeRagLLMClient), mock.patch(
                "generation.rag.rag_tce._build_openai_client",
                lambda provider: object(),
            ), mock.patch(
                "generation.rag.rag_tce._embed_texts",
                _fake_rag_embed_texts,
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
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    predict_per_key=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=str(qa_path),
                    final_qa_retrieval_top_k=5,
                )

            final_qa_path = output_path.with_name("prediction_final_qa.json")
            final_qa_results = json.loads(final_qa_path.read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"][-1]["snapshot_state"]["profile_state:favorite_coffee"], "espresso")
        self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
        self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp_0002")

    def test_mem0_final_checkpoint_qa_uses_final_checkpoint_memory(self):
        _FakeMem0Memory.store_by_collection = {}
        _FakeMem0Memory.search_events = []
        benchmark = _make_benchmark(["latte", "espresso"])
        app_logs = _make_app_logs(["latte", "espresso"])
        qa_list = _make_qa_list("espresso", "log_0002")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            qa_path = root / "qa.json"
            output_path = root / "prediction.json"
            checkpoint_root = root / "checkpoints"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch("generation.mem0.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.mem0.tce._load_mem0_class",
                lambda: _FakeMem0Memory,
            ):
                result = mem0_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    collection_name="membench_mem0_user1_finalqa",
                    qdrant_host="localhost",
                    qdrant_port=6333,
                    config_path=None,
                    checkpoint_dir=str(checkpoint_root),
                    embedder_api_key=None,
                    embedder_api_base=None,
                    llm_api_key=None,
                    llm_api_base=None,
                    embedder_model=None,
                    answer_llm_model="gpt-5-mini",
                    reset_collections=True,
                    resume=False,
                    max_checkpoints=2,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                    enable_final_qa=True,
                    final_qa_path=str(qa_path),
                    final_qa_retrieval_top_k=5,
                )

            final_qa_results = json.loads(output_path.with_name("prediction_final_qa.json").read_text(encoding="utf-8"))

        self.assertEqual(result["predictions"][-1]["snapshot_state"]["profile_state:favorite_coffee"], "espresso")
        self.assertEqual(final_qa_results[0]["prediction"], "Recommend espresso")
        self.assertEqual(final_qa_results[0]["metadata"]["tce_final_checkpoint_id"], "cp_0002")

    def test_mem0_minimal_pack_first_generation(self):
        _FakeMem0Memory.store_by_collection = {}
        _FakeMem0Memory.search_events = []
        benchmark = _make_benchmark(["latte"])
        app_logs = _make_app_logs(["latte"])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch("generation.mem0.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.mem0.tce._load_mem0_class",
                lambda: _FakeMem0Memory,
            ):
                result = mem0_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    retrieval_top_k=5,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    collection_name="membench_mem0_user1",
                    qdrant_host="localhost",
                    qdrant_port=6333,
                    config_path=None,
                    checkpoint_dir=str(root / "checkpoints"),
                    embedder_api_key=None,
                    embedder_api_base=None,
                    llm_api_key=None,
                    llm_api_base=None,
                    embedder_model=None,
                    answer_llm_model="gpt-5-mini",
                    reset_collections=True,
                    resume=False,
                    max_checkpoints=1,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=5,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

            manifest = json.loads(
                (root / "checkpoints" / "001_user_001" / "membench_mem0_user1" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        prediction = result["predictions"][0]
        self.assertEqual(prediction["snapshot_state"]["profile_state:favorite_coffee"], "latte")
        self.assertEqual(prediction["metadata"]["baseline"], "mem0")
        self.assertEqual(prediction["metadata"]["requested_checkpoint_workers"], 1)
        self.assertEqual(prediction["metadata"]["effective_checkpoint_workers"], 1)
        self.assertTrue(prediction["metadata"]["rq3_apply"]["enabled"])
        self.assertEqual(
            prediction["rq3_apply_answers"]["profile_state:favorite_coffee"]["items"][0]["answer"],
            "Recommend latte",
        )
        self.assertEqual(len(manifest["checkpoints"]), 1)
        self.assertEqual(manifest["checkpoints"][0]["builder_collection_name"], "membench_mem0_user1")
        self.assertEqual(
            manifest["checkpoints"][0]["snapshot_collection_name"],
            "membench_mem0_user1__snapshot__cp_0001",
        )
        self.assertEqual(len(_FakeMem0Memory.store_by_collection["membench_mem0_user1"]), 1)
        snapshot_collections = {
            name: rows
            for name, rows in _FakeMem0Memory.store_by_collection.items()
            if name.startswith("membench_mem0_user1__snapshot__")
        }
        self.assertTrue(snapshot_collections)
        self.assertTrue(all(len(rows) > 0 for rows in snapshot_collections.values()))

    def test_mem0_stateful_checkpoint_semantics_and_non_polluting_queries(self):
        _FakeMem0Memory.store_by_collection = {}
        _FakeMem0Memory.search_events = []
        values = ["latte", "espresso", "mocha", "americano", "cappuccino"]
        benchmark = _make_benchmark(values)
        app_logs = _make_app_logs(values)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            checkpoint_root = root / "checkpoints"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch("generation.mem0.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.mem0.tce._load_mem0_class",
                lambda: _FakeMem0Memory,
            ):
                result = mem0_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    retrieval_top_k=20,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    collection_name="membench_mem0_user1_v14",
                    qdrant_host="localhost",
                    qdrant_port=6333,
                    config_path=None,
                    checkpoint_dir=str(checkpoint_root),
                    embedder_api_key=None,
                    embedder_api_base=None,
                    llm_api_key=None,
                    llm_api_base=None,
                    embedder_model=None,
                    answer_llm_model="gpt-5-mini",
                    reset_collections=True,
                    resume=False,
                    max_checkpoints=5,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=20,
                    checkpoint_workers=2,
                    within_checkpoint_workers=2,
                    save_every_generation_keys=1,
                )

            manifest = json.loads(
                (checkpoint_root / "001_user_001" / "membench_mem0_user1_v14" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest_root = checkpoint_root / "001_user_001" / "membench_mem0_user1_v14"
            snapshot_payloads = []
            for entry in manifest["checkpoints"]:
                snapshot_path = manifest_root / str(entry["snapshot_path"])
                snapshot_payloads.append(json.loads(snapshot_path.read_text(encoding="utf-8")))

        self.assertEqual(len(result["predictions"]), 5)
        self.assertEqual(len(manifest["checkpoints"]), 5)
        for idx, prediction in enumerate(result["predictions"], start=1):
            expected_value = values[idx - 1]
            self.assertEqual(prediction["snapshot_state"]["profile_state:favorite_coffee"], expected_value)
            self.assertEqual(
                prediction["rq3_apply_answers"]["profile_state:favorite_coffee"]["items"][0]["answer"],
                "Recommend {}".format(expected_value),
            )
            self.assertEqual(prediction["metadata"]["effective_checkpoint_workers"], 2)
            self.assertEqual(prediction["metadata"]["effective_within_checkpoint_workers"], 2)
            if idx == 1:
                self.assertEqual(prediction["change_analysis"], {})
            else:
                self.assertEqual(
                    prediction["change_analysis"]["profile_state:favorite_coffee"]["before"],
                    values[idx - 2],
                )
                self.assertEqual(
                    prediction["change_analysis"]["profile_state:favorite_coffee"]["after"],
                    expected_value,
                )

        self.assertEqual(len(_FakeMem0Memory.store_by_collection["membench_mem0_user1_v14"]), len(values))
        snapshot_collections = {
            name: rows
            for name, rows in _FakeMem0Memory.store_by_collection.items()
            if name.startswith("membench_mem0_user1_v14__snapshot__")
        }
        self.assertEqual(len(snapshot_collections), 5)
        self.assertEqual(sorted(len(rows) for rows in snapshot_collections.values()), [1, 2, 3, 4, 5])
        self.assertEqual(
            {int(event["before"]) for event in _FakeMem0Memory.search_events},
            {1, 2, 3, 4, 5},
        )
        self.assertTrue(
            all(
                str(event["collection"]).startswith("membench_mem0_user1_v14__snapshot__")
                for event in _FakeMem0Memory.search_events
            )
        )

        for idx, (entry, snapshot_payload) in enumerate(zip(manifest["checkpoints"], snapshot_payloads), start=1):
            self.assertEqual(entry["builder_collection_name"], "membench_mem0_user1_v14")
            self.assertEqual(
                entry["snapshot_collection_name"],
                "membench_mem0_user1_v14__snapshot__cp_{:04d}".format(idx),
            )
            self.assertEqual(snapshot_payload["snapshot_id"], entry["snapshot_id"])
            self.assertEqual(snapshot_payload["checkpoint_id"], entry["checkpoint_id"])
            self.assertEqual(snapshot_payload["snapshot_collection_name"], entry["snapshot_collection_name"])
            self.assertEqual(snapshot_payload["last_event_idx"], idx - 1)
            self.assertEqual(len(snapshot_payload["app_logs"]), idx)

        self.assertTrue(_FakeMem0Memory.search_events)
        self.assertTrue(all(event["before"] == event["after"] for event in _FakeMem0Memory.search_events))

    def test_mem0_resume_uses_confirmed_builder_progress(self):
        class _InterruptingMem0Memory(_FakeMem0Memory):
            fail_enabled = True
            add_attempts = 0
            fail_after_successes = 2

            def add(self, messages, metadata=None, user_id=None):
                if (
                    self.collection_name == "membench_mem0_user1_resume"
                    and self.fail_enabled
                    and _InterruptingMem0Memory.add_attempts >= _InterruptingMem0Memory.fail_after_successes
                ):
                    raise RuntimeError("simulated mem0 builder interruption")
                if self.collection_name == "membench_mem0_user1_resume" and self.fail_enabled:
                    _InterruptingMem0Memory.add_attempts += 1
                return super().add(messages, metadata=metadata, user_id=user_id)

        _FakeMem0Memory.store_by_collection = {}
        _FakeMem0Memory.search_events = []
        benchmark = _make_benchmark(["latte", "espresso", "mocha"])
        app_logs = _make_app_logs(["latte", "espresso", "mocha"])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            output_path = root / "prediction.json"
            checkpoint_root = root / "checkpoints"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
            app_logs_path.write_text(json.dumps(app_logs, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch("generation.mem0.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.mem0.tce._load_mem0_class",
                lambda: _InterruptingMem0Memory,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated mem0 builder interruption"):
                    mem0_tce.run_generation(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        output_path=output_path,
                        user_id="001_user_001",
                        retrieval_top_k=20,
                        max_visible_logs=None,
                        llm_provider="openai",
                        llm_model="gpt-5-mini",
                        llm_max_workers=1,
                        retriever_provider="openai",
                        retriever_model="text-embedding-3-large",
                        collection_name="membench_mem0_user1_resume",
                        qdrant_host="localhost",
                        qdrant_port=6333,
                        config_path=None,
                        checkpoint_dir=str(checkpoint_root),
                        embedder_api_key=None,
                        embedder_api_base=None,
                        llm_api_key=None,
                        llm_api_base=None,
                        embedder_model=None,
                        answer_llm_model="gpt-5-mini",
                        reset_collections=True,
                        resume=False,
                        max_checkpoints=3,
                        debug=False,
                        debug_dir=None,
                        save_prompt_and_raw=True,
                        enable_change_reasoning=True,
                        enable_rq3_apply_service_qa=True,
                        rq3_apply_retrieval_top_k=20,
                        checkpoint_workers=1,
                        within_checkpoint_workers=1,
                        save_every_generation_keys=1,
                    )

            progress_path = (
                checkpoint_root / "001_user_001" / "membench_mem0_user1_resume" / "builder_progress.json"
            )
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["confirmed_last_event_idx"], 1)
            self.assertEqual(progress_payload["confirmed_app_log_id"], "log_0002")

            _InterruptingMem0Memory.fail_enabled = False
            with mock.patch("generation.mem0.tce.LLMClient", _FakeLLMClient), mock.patch(
                "generation.mem0.tce._load_mem0_class",
                lambda: _InterruptingMem0Memory,
            ):
                result = mem0_tce.run_generation(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    output_path=output_path,
                    user_id="001_user_001",
                    retrieval_top_k=20,
                    max_visible_logs=None,
                    llm_provider="openai",
                    llm_model="gpt-5-mini",
                    llm_max_workers=1,
                    retriever_provider="openai",
                    retriever_model="text-embedding-3-large",
                    collection_name="membench_mem0_user1_resume",
                    qdrant_host="localhost",
                    qdrant_port=6333,
                    config_path=None,
                    checkpoint_dir=str(checkpoint_root),
                    embedder_api_key=None,
                    embedder_api_base=None,
                    llm_api_key=None,
                    llm_api_base=None,
                    embedder_model=None,
                    answer_llm_model="gpt-5-mini",
                    reset_collections=False,
                    resume=True,
                    max_checkpoints=3,
                    debug=False,
                    debug_dir=None,
                    save_prompt_and_raw=True,
                    enable_change_reasoning=True,
                    enable_rq3_apply_service_qa=True,
                    rq3_apply_retrieval_top_k=20,
                    checkpoint_workers=1,
                    within_checkpoint_workers=1,
                    save_every_generation_keys=1,
                )

            manifest = json.loads(
                (
                    checkpoint_root / "001_user_001" / "membench_mem0_user1_resume" / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(result["predictions"][-1]["snapshot_state"]["profile_state:favorite_coffee"], "mocha")
        self.assertEqual(len(manifest["checkpoints"]), 3)
        self.assertEqual(len(_FakeMem0Memory.store_by_collection["membench_mem0_user1_resume"]), 3)
        self.assertEqual(progress_payload["confirmed_last_event_idx"], 2)
        self.assertEqual(progress_payload["confirmed_app_log_id"], "log_0003")
        self.assertEqual(progress_payload["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
