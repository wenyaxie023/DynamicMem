#!/usr/bin/env python3
import sys
import types
import unittest
from unittest import mock

if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.SimpleNamespace(
        ndarray=object,
        asarray=lambda *args, **kwargs: [],
        zeros=lambda *args, **kwargs: [],
        argsort=lambda *args, **kwargs: [],
        vstack=lambda values: values,
        save=lambda *args, **kwargs: None,
        load=lambda *args, **kwargs: [],
        linalg=types.SimpleNamespace(norm=lambda *args, **kwargs: 1.0),
    )

from generation.Amem.agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


class _FakeRetriever:
    instances = []

    def __init__(self, *args, **kwargs):
        self.search_result = []
        self.add_calls = []
        self.deleted_ids = []
        _FakeRetriever.instances.append(self)

    def add_document(self, document, metadata, doc_id):
        self.add_calls.append({"document": document, "metadata": dict(metadata), "doc_id": doc_id})

    def delete_document(self, doc_id):
        self.deleted_ids.append(doc_id)

    def search(self, query, k=5):
        del query, k
        return self.search_result


class _FakeLLMController:
    def __init__(self, *args, **kwargs):
        self.llm = types.SimpleNamespace(get_completion=self._get_completion)
        self.responses = []

    def _get_completion(self, prompt, response_format=None, temperature=None):
        del prompt, response_format, temperature
        if not self.responses:
            raise AssertionError("No fake LLM response configured")
        return self.responses.pop(0)


class _FakeOpenAIChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": true}'))],
            usage=None,
        )


class AmemFaithfulTest(unittest.TestCase):
    def test_openai_controller_omits_temperature_for_gpt5_models(self):
        from generation.Amem.agentic_memory.llm_controller import OpenAIController

        completions = _FakeOpenAIChatCompletions()
        controller = object.__new__(OpenAIController)
        controller.model = "gpt-5-mini"
        controller.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions),
        )

        controller.get_completion("Return JSON", temperature=0.7)

        self.assertNotIn("temperature", completions.calls[0])

    def test_openai_controller_keeps_temperature_for_non_gpt5_models(self):
        from generation.Amem.agentic_memory.llm_controller import OpenAIController

        completions = _FakeOpenAIChatCompletions()
        controller = object.__new__(OpenAIController)
        controller.model = "gpt-4o-mini"
        controller.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions),
        )

        controller.get_completion("Return JSON", temperature=0.7)

        self.assertEqual(completions.calls[0]["temperature"], 0.7)

    def test_add_note_auto_generates_missing_metadata(self):
        with mock.patch(
            "generation.Amem.agentic_memory.memory_system.SimpleEmbeddingRetriever",
            _FakeRetriever,
        ), mock.patch(
            "generation.Amem.agentic_memory.memory_system.LLMController",
            _FakeLLMController,
        ):
            memory_system = AgenticMemorySystem(reset_collection=False)
            memory_system.analyze_content = mock.Mock(
                return_value={
                    "keywords": ["latte", "coffee"],
                    "context": "coffee preference note",
                    "tags": ["coffee", "profile"],
                }
            )
            memory_system.process_memory = mock.Mock(side_effect=lambda note: (False, note))

            note_id = memory_system.add_note('{"app_log_id":"log_0001","response":{"favorite_coffee":"latte"}}')

        note = memory_system.memories[note_id]
        self.assertEqual(note.keywords, ["latte", "coffee"])
        self.assertEqual(note.context, "coffee preference note")
        self.assertEqual(note.tags, ["coffee", "profile"])
        self.assertEqual(memory_system.retriever.add_calls[0]["metadata"]["keywords"], ["latte", "coffee"])

    def test_find_related_memories_returns_insertion_order_indices(self):
        with mock.patch(
            "generation.Amem.agentic_memory.memory_system.SimpleEmbeddingRetriever",
            _FakeRetriever,
        ), mock.patch(
            "generation.Amem.agentic_memory.memory_system.LLMController",
            _FakeLLMController,
        ):
            memory_system = AgenticMemorySystem(reset_collection=False)

        memory_system.memories = {
            "note_1": MemoryNote("content 1", id="note_1", timestamp="1", context="c1", keywords=["k1"], tags=["t1"]),
            "note_2": MemoryNote("content 2", id="note_2", timestamp="2", context="c2", keywords=["k2"], tags=["t2"]),
            "note_3": MemoryNote("content 3", id="note_3", timestamp="3", context="c3", keywords=["k3"], tags=["t3"]),
        }
        memory_system.retriever.search_result = [2, 0]

        _, indices = memory_system.find_related_memories("question", k=2)

        self.assertEqual(indices, [2, 0])

    def test_process_memory_updates_index_based_links_and_neighbors(self):
        with mock.patch(
            "generation.Amem.agentic_memory.memory_system.SimpleEmbeddingRetriever",
            _FakeRetriever,
        ), mock.patch(
            "generation.Amem.agentic_memory.memory_system.LLMController",
            _FakeLLMController,
        ):
            memory_system = AgenticMemorySystem(reset_collection=False)

        memory_system.memories = {
            "note_1": MemoryNote("content 1", id="note_1", timestamp="1", context="ctx1", keywords=["k1"], tags=["t1"]),
            "note_2": MemoryNote("content 2", id="note_2", timestamp="2", context="ctx2", keywords=["k2"], tags=["t2"]),
            "note_3": MemoryNote("content 3", id="note_3", timestamp="3", context="ctx3", keywords=["k3"], tags=["t3"]),
        }
        memory_system.find_related_memories = mock.Mock(return_value=("neighbors", [1, 2]))
        memory_system.llm_controller.responses = [
            "DECISION: STRENGTHEN_AND_UPDATE\nREASON: related preference cluster",
            "CONNECTIONS: 1\nTAGS: new_tag",
            "NEIGHBOR 0:\nCONTEXT: ctx2_updated\nTAGS: tag2_updated\n\nNEIGHBOR 1:\nCONTEXT: ctx3_updated\nTAGS: tag3_updated",
        ]

        should_evolve, note = memory_system.process_memory(
            MemoryNote("new content", id="note_new", context="ctx_new", keywords=["kw"], tags=["tag"])
        )

        self.assertTrue(should_evolve)
        self.assertEqual(note.links, [1])
        self.assertEqual(note.tags, ["new_tag"])
        self.assertEqual(memory_system.memories["note_2"].context, "ctx2_updated")
        self.assertEqual(memory_system.memories["note_2"].tags, ["tag2_updated"])
        self.assertEqual(memory_system.memories["note_3"].context, "ctx3_updated")
        self.assertEqual(memory_system.memories["note_3"].tags, ["tag3_updated"])

    def test_analyze_content_accepts_plain_text_robust_format(self):
        with mock.patch(
            "generation.Amem.agentic_memory.memory_system.SimpleEmbeddingRetriever",
            _FakeRetriever,
        ), mock.patch(
            "generation.Amem.agentic_memory.memory_system.LLMController",
            _FakeLLMController,
        ):
            memory_system = AgenticMemorySystem(reset_collection=False)

        memory_system.llm_controller.responses = [
            "KEYWORDS: latte, coffee, preference\nCONTEXT: A coffee preference note.\nTAGS: coffee, profile, preference"
        ]

        analysis = memory_system.analyze_content("User likes latte.")

        self.assertEqual(analysis["keywords"], ["latte", "coffee", "preference"])
        self.assertEqual(analysis["context"], "A coffee preference note.")
        self.assertEqual(analysis["tags"], ["coffee", "profile", "preference"])

    def test_find_related_memories_raw_expands_linked_neighbors(self):
        with mock.patch(
            "generation.Amem.agentic_memory.memory_system.SimpleEmbeddingRetriever",
            _FakeRetriever,
        ), mock.patch(
            "generation.Amem.agentic_memory.memory_system.LLMController",
            _FakeLLMController,
        ):
            memory_system = AgenticMemorySystem(reset_collection=False)

        memory_system.memories = {
            "note_1": MemoryNote("content 1", id="note_1", timestamp="1", context="ctx1", keywords=["k1"], tags=["t1"]),
            "note_2": MemoryNote(
                "content 2",
                id="note_2",
                timestamp="2",
                context="ctx2",
                keywords=["k2"],
                tags=["t2"],
                links=[0],
            ),
        }
        memory_system.find_related_memories = mock.Mock(return_value=("neighbors", [1]))

        rendered = memory_system.find_related_memories_raw("question", k=2)

        self.assertIn("content 2", rendered)
        self.assertIn("content 1", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
