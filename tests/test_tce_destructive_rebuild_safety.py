#!/usr/bin/env python3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
fake_openai.AzureOpenAI = object
sys.modules.setdefault("openai", fake_openai)
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.genai", types.ModuleType("google.genai"))

from generation.mem0 import tce as mem0_tce
from generation.tce_config import build_from_config
from generation.tce_safety import ensure_destructive_rebuild_allowed


class TceDestructiveRebuildSafetyTest(unittest.TestCase):
    def test_runtime_allow_destructive_rebuild_parses_from_shared_runtime(self):
        args = build_from_config(
            {
                "runtime": {
                    "baseline": "simplemem",
                    "user_id": "001_user_001",
                    "resume": True,
                    "allow_destructive_rebuild": True,
                },
                "data": {
                    "benchmark": "data/benchmark.json",
                    "app_logs_path": "data/app_logs.json",
                },
                "output": {"prediction_path": "results/prediction.json"},
            }
        )
        self.assertTrue(args.allow_destructive_rebuild)

    def test_shared_safety_helper_refuses_existing_artifacts_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "memory"
            target.mkdir(parents=True, exist_ok=True)
            (target / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "allow_destructive_rebuild"):
                ensure_destructive_rebuild_allowed(
                    allow_destructive_rebuild=False,
                    operation="clear existing artifacts",
                    paths=[target],
                )

    def test_mem0_clone_refuses_replacing_existing_snapshot_collection_without_opt_in(self):
        def _fake_qdrant_request(*, host, port, method, path, payload=None):
            del host, port, payload
            if method == "GET" and path == "/collections/source_collection":
                return {
                    "result": {
                        "config": {
                            "params": {"vectors": {"size": 3, "distance": "Cosine"}},
                            "optimizer_config": {},
                        }
                    }
                }
            if method == "GET" and path == "/collections":
                return {"result": {"collections": [{"name": "target_collection"}]}}
            raise AssertionError("unexpected qdrant request {} {}".format(method, path))

        with mock.patch.object(mem0_tce, "_qdrant_request", side_effect=_fake_qdrant_request), mock.patch.object(
            mem0_tce,
            "_load_mem0_class",
            return_value=types.SimpleNamespace(store_by_collection=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "allow_destructive_rebuild"):
                mem0_tce._clone_qdrant_collection(
                    host="localhost",
                    port=6333,
                    source_collection_name="source_collection",
                    target_collection_name="target_collection",
                    allow_destructive_rebuild=False,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
