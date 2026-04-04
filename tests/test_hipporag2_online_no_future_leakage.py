import importlib
import json
import sys
import types


def _install_import_stubs(monkeypatch):
    # Stub heavy optional deps used at module import time.
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
    hipporag_misc_utils_mod = types.ModuleType("hipporag.utils.misc_utils")
    hipporag_config_utils_mod.BaseConfig = object
    hipporag_misc_utils_mod.compute_mdhash_id = lambda content, prefix="": f"{prefix}{hash(content)}"
    hipporag_utils_mod.config_utils = hipporag_config_utils_mod
    hipporag_utils_mod.misc_utils = hipporag_misc_utils_mod

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", transformers_import_utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", transformers_modeling_utils_mod)
    monkeypatch.setitem(sys.modules, "hipporag", hipporag_mod)
    monkeypatch.setitem(sys.modules, "hipporag.utils", hipporag_utils_mod)
    monkeypatch.setitem(sys.modules, "hipporag.utils.config_utils", hipporag_config_utils_mod)
    monkeypatch.setitem(sys.modules, "hipporag.utils.misc_utils", hipporag_misc_utils_mod)


def test_online_retrieve_context_filters_future_logs(monkeypatch):
    """
    Regression test for HippoRAG2 online leakage fix.

    Given checkpoints advance from log_index=1 to log_index=3,
    retrieve_context should only index newly observed logs each time
    (strict incremental indexing, no global prefetch of future logs).
    """
    _install_import_stubs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    module_name = "generation.HippoRAG2.generation_tce.online_tce"
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)

    all_logs = [
        {"app_log_id": "log_00001", "timestamp": "2023-10-01 00:00:00", "text": "past-a"},
        {"app_log_id": "log_00002", "timestamp": "2023-10-01 01:00:00", "text": "past-b"},
        {"app_log_id": "log_00003", "timestamp": "2023-10-01 02:00:00", "text": "future-a"},
        {"app_log_id": "log_00004", "timestamp": "2023-10-01 03:00:00", "text": "future-b"},
    ]

    class FakeHippoRAG:
        def __init__(self):
            self.ready_to_retrieve = True
            self.index_calls = []
            self.indexed_docs = []

        def index(self, docs):
            self.index_calls.append(list(docs))
            self.indexed_docs.extend(docs)

        def retrieve(self, queries):
            # Return only docs that have actually been indexed so far.
            return [types.SimpleNamespace(docs=list(self.indexed_docs))]

    fake = FakeHippoRAG()
    runner = mod.OnlineDSPRunner(
        hipporag=fake,
        all_logs=all_logs,
        batch_size=8,
    )
    runner.last_indexed_idx = -1

    cp1 = {"checkpoint_id": "cp1", "as_of": {"log_index": 1}}
    result1 = runner.retrieve_context(
        cp=cp1,
        memory_pool=all_logs[:2],
        target_keys=["finance:balance"],
    )
    assert len(fake.index_calls) == 1
    assert len(fake.index_calls[0]) == 2  # log_00001..log_00002
    assert runner.last_indexed_idx == 1

    returned1 = result1["context_logs"]
    assert returned1, "Expected retrieved logs for cp1."
    ids1 = {str(x.get("app_log_id")) for x in returned1}
    assert ids1.issubset({"log_00001", "log_00002"})

    cp2 = {"checkpoint_id": "cp2", "as_of": {"log_index": 3}}
    result2 = runner.retrieve_context(
        cp=cp2,
        memory_pool=all_logs[:4],
        target_keys=["finance:balance"],
    )
    assert len(fake.index_calls) == 2
    assert len(fake.index_calls[1]) == 2  # only new logs: log_00003..log_00004
    assert runner.last_indexed_idx == 3

    returned2 = result2["context_logs"]
    assert returned2, "Expected retrieved logs for cp2."
    ids2 = {str(x.get("app_log_id")) for x in returned2}
    assert ids2.issubset({"log_00001", "log_00002", "log_00003", "log_00004"})
