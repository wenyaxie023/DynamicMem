#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_prediction.MemoryOS import tce_adapter as memoryos_tce


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


class _FakeUpdater:
    def __init__(self):
        self.last_evicted_page_for_continuity = None


class _FakeMemoryOS:
    fail_before_add_count = None
    factory_instances = []
    resumed_continuity_values = []

    def __init__(self, data_storage_root: Path, memory_user_id: str):
        root = Path(data_storage_root)
        (root / "users" / memory_user_id).mkdir(parents=True, exist_ok=True)
        (root / "assistants" / "assistant").mkdir(parents=True, exist_ok=True)
        self.user_id = memory_user_id
        self.short_term_memory = _FakeJsonStore(root / "users" / memory_user_id / "short_term.json")
        self.mid_term_memory = _FakeJsonStore(root / "users" / memory_user_id / "mid_term.json")
        self.user_long_term_memory = _FakeJsonStore(root / "users" / memory_user_id / "long_term_user.json")
        self.assistant_long_term_memory = _FakeJsonStore(root / "assistants" / "assistant" / "long_term_assistant.json")
        self.updater = _FakeUpdater()
        self.add_calls = []
        self.instance_index = len(_FakeMemoryOS.factory_instances)
        _FakeMemoryOS.factory_instances.append(self)

    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, meta_data: dict = None):
        next_count = len(self.add_calls) + 1
        if self.instance_index > 0 and not self.add_calls:
            _FakeMemoryOS.resumed_continuity_values.append(self.updater.last_evicted_page_for_continuity)
        fail_before = _FakeMemoryOS.fail_before_add_count
        if self.instance_index == 0 and fail_before is not None and next_count > int(fail_before):
            raise RuntimeError("simulated builder interrupt")
        self.add_calls.append(user_input)
        self.short_term_memory.memory.append(
            {
                "user_input": user_input,
                "agent_response": agent_response,
                "timestamp": timestamp or "",
            }
        )
        self.updater.last_evicted_page_for_continuity = {
            "page_id": "page_{:04d}".format(next_count),
            "user_input": user_input,
        }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_app_logs(n: int):
    return [
        {
            "app_log_id": "log_{:04d}".format(idx),
            "timestamp": "2025-01-{:02d} 08:00:00".format(idx),
            "app_name": "app",
            "api_name": "save",
            "request": {"i": idx},
            "response": {"value": idx},
        }
        for idx in range(1, n + 1)
    ]


class MemoryosBuilderResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeMemoryOS.fail_before_add_count = None
        _FakeMemoryOS.factory_instances = []
        _FakeMemoryOS.resumed_continuity_values = []

    def _fake_factory(self, **kwargs):
        return _FakeMemoryOS(
            data_storage_root=Path(kwargs["data_storage_root"]),
            memory_user_id=str(kwargs["memory_user_id"]),
        )

    def test_builder_resume_continues_from_latest_durable_snapshot(self) -> None:
        app_logs = _make_app_logs(12)
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0012",
                    "as_of": {
                        "timestamp": app_logs[-1]["timestamp"],
                        "log_index": 11,
                        "app_log_id": app_logs[-1]["app_log_id"],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, benchmark)
            _write_json(app_logs_path, app_logs)

            _FakeMemoryOS.fail_before_add_count = 5
            with mock.patch("generation.MemoryOS.tce_adapter._build_memoryos_instance", self._fake_factory):
                with self.assertRaisesRegex(RuntimeError, "simulated builder interrupt"):
                    memoryos_tce._ensure_snapshots_for_benchmark(
                        benchmark_path=benchmark_path,
                        app_logs_path=app_logs_path,
                        user_id="001_user_001",
                        size="large",
                        snapshot_dir=str(snapshot_dir),
                        data_storage_path=str(data_storage_path),
                        llm_controller_model="gpt-5-mini",
                        embedding_model_name="text-embedding-3-large",
                        assistant_id="assistant",
                        max_checkpoints=1,
                        retriever_provider="openai",
                        resume=True,
                    )

            latest_entry = memoryos_tce._latest_snapshot_entry(snapshot_dir / "001_user_001_large")
            self.assertIsNotNone(latest_entry)
            self.assertEqual(int(latest_entry["last_event_idx"]), 4)

            _FakeMemoryOS.fail_before_add_count = None
            with mock.patch("generation.MemoryOS.tce_adapter._build_memoryos_instance", self._fake_factory):
                snapshot_root = memoryos_tce._ensure_snapshots_for_benchmark(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    llm_controller_model="gpt-5-mini",
                    embedding_model_name="text-embedding-3-large",
                    assistant_id="assistant",
                    max_checkpoints=1,
                    retriever_provider="openai",
                    resume=True,
                )

            self.assertEqual(snapshot_root, snapshot_dir / "001_user_001_large")
            self.assertIn(
                "cp_0012",
                memoryos_tce._existing_checkpoint_ids(snapshot_dir / "001_user_001_large"),
            )

        self.assertEqual(len(_FakeMemoryOS.factory_instances), 2)
        self.assertEqual(len(_FakeMemoryOS.factory_instances[0].add_calls), 5)
        self.assertEqual(len(_FakeMemoryOS.factory_instances[1].add_calls), 7)
        self.assertEqual(
            _FakeMemoryOS.resumed_continuity_values[0]["page_id"],
            "page_0005",
        )

    def test_builder_snapshots_store_raw_app_log_payload(self) -> None:
        app_logs = _make_app_logs(1)
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": app_logs[0]["timestamp"],
                        "log_index": 0,
                        "app_log_id": app_logs[0]["app_log_id"],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, benchmark)
            _write_json(app_logs_path, app_logs)

            with mock.patch("generation.MemoryOS.tce_adapter._build_memoryos_instance", self._fake_factory):
                snapshot_root = memoryos_tce._ensure_snapshots_for_benchmark(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    llm_controller_model="gpt-5-mini",
                    embedding_model_name="text-embedding-3-large",
                    assistant_id="assistant",
                    max_checkpoints=1,
                    retriever_provider="openai",
                    resume=False,
                )

            bundle = memoryos_tce._load_snapshot_bundle(
                snapshot_root,
                benchmark["checkpoints"][0],
            )
            short_term_payload = json.loads(bundle["short_term_path"].read_text(encoding="utf-8"))

        self.assertEqual(len(short_term_payload), 1)
        self.assertEqual(short_term_payload[0]["user_input"], json.dumps(app_logs[0], ensure_ascii=False))

    def test_fresh_build_refuses_clearing_existing_artifacts_without_opt_in(self) -> None:
        app_logs = _make_app_logs(1)
        benchmark = {
            "user_id": "001_user_001",
            "checkpoints": [
                {
                    "checkpoint_id": "cp_0001",
                    "as_of": {
                        "timestamp": app_logs[0]["timestamp"],
                        "log_index": 0,
                        "app_log_id": app_logs[0]["app_log_id"],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark_path = root / "benchmark.json"
            app_logs_path = root / "app_logs.json"
            snapshot_dir = root / "snapshots"
            data_storage_path = root / "runtime_data"
            _write_json(benchmark_path, benchmark)
            _write_json(app_logs_path, app_logs)
            stale_runtime = data_storage_path / "001_user_001_large" / "builder" / "stale.txt"
            stale_snapshot = snapshot_dir / "001_user_001_large" / "manifest.json"
            stale_runtime.parent.mkdir(parents=True, exist_ok=True)
            stale_snapshot.parent.mkdir(parents=True, exist_ok=True)
            stale_runtime.write_text("stale", encoding="utf-8")
            stale_snapshot.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "allow_destructive_rebuild"):
                memoryos_tce._ensure_snapshots_for_benchmark(
                    benchmark_path=benchmark_path,
                    app_logs_path=app_logs_path,
                    user_id="001_user_001",
                    size="large",
                    snapshot_dir=str(snapshot_dir),
                    data_storage_path=str(data_storage_path),
                    llm_controller_model="gpt-5-mini",
                    embedding_model_name="text-embedding-3-large",
                    assistant_id="assistant",
                    max_checkpoints=1,
                    retriever_provider="openai",
                    resume=False,
                    allow_destructive_rebuild=False,
                )

            self.assertEqual(stale_runtime.read_text(encoding="utf-8"), "stale")
            self.assertEqual(stale_snapshot.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
