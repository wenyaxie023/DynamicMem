"""Vendored SimpleMem system adapted to TCE runtime controls."""

from typing import Any, Dict, List, Optional, Sequence

from .models.memory_entry import Dialogue, MemoryEntry
from .utils.llm_client import LLMClient
from .utils.embedding import EmbeddingModel
from .database.vector_store import VectorStore
from .core.memory_builder import MemoryBuilder
from .core.hybrid_retriever import HybridRetriever
from .core.answer_generator import AnswerGenerator
from .utils.usage import UsageTracker
from . import config


class SimpleMemSystem:
    """
    SimpleMem Main System

    Three-stage pipeline:
    1. Semantic Structured Compression: add_dialogue() -> MemoryBuilder -> VectorStore
    2. Online Semantic Synthesis: intra-session consolidation during write
    3. Intent-Aware Retrieval Planning: ask() -> HybridRetriever -> AnswerGenerator
    """

    def __init__(
        self,
        *,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        llm_max_workers: int = 1,
        llm_temperature: Optional[float] = None,
        llm_top_p: Optional[float] = None,
        llm_top_k: Optional[int] = None,
        embedding_provider: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        embedding_tracker: Optional[UsageTracker] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        structured_top_k: Optional[int] = None,
        window_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
    ):
        config.apply_runtime_config(
            {
                "LLM_PROVIDER": str(llm_provider or config.LLM_PROVIDER),
                "LLM_MODEL": str(model or config.LLM_MODEL),
                "LLM_MAX_WORKERS": int(llm_max_workers or config.LLM_MAX_WORKERS),
                "LLM_TEMPERATURE": llm_temperature if llm_temperature is not None else config.LLM_TEMPERATURE,
                "LLM_TOP_P": llm_top_p if llm_top_p is not None else config.LLM_TOP_P,
                "LLM_TOP_K": llm_top_k if llm_top_k is not None else config.LLM_TOP_K,
                "EMBEDDING_PROVIDER": str(embedding_provider or config.EMBEDDING_PROVIDER),
                "EMBEDDING_MODEL": str(embedding_model_name or config.EMBEDDING_MODEL),
                "EMBEDDING_DIM": int(
                    embedding_dim
                    or config.EMBEDDING_DIM
                    or (3072 if "large" in str(embedding_model_name or config.EMBEDDING_MODEL).lower() else 1536)
                ),
                "RETRIEVER_BATCH_SIZE": int(embedding_batch_size or config.RETRIEVER_BATCH_SIZE or 64),
                "LANCEDB_PATH": str(db_path or config.LANCEDB_PATH),
                "MEMORY_TABLE_NAME": str(table_name or config.MEMORY_TABLE_NAME),
                "WINDOW_SIZE": int(window_size or config.WINDOW_SIZE),
                "OVERLAP_SIZE": int(overlap_size if overlap_size is not None else config.OVERLAP_SIZE),
                "ENABLE_PARALLEL_PROCESSING": bool(
                    config.ENABLE_PARALLEL_PROCESSING if enable_parallel_processing is None else enable_parallel_processing
                ),
                "MAX_PARALLEL_WORKERS": int(max_parallel_workers or config.MAX_PARALLEL_WORKERS or 1),
                "SEMANTIC_TOP_K": int(semantic_top_k or config.SEMANTIC_TOP_K),
                "KEYWORD_TOP_K": int(keyword_top_k or config.KEYWORD_TOP_K),
                "STRUCTURED_TOP_K": int(structured_top_k or config.STRUCTURED_TOP_K),
                "ENABLE_PLANNING": config.ENABLE_PLANNING if enable_planning is None else bool(enable_planning),
                "ENABLE_REFLECTION": config.ENABLE_REFLECTION if enable_reflection is None else bool(enable_reflection),
                "MAX_REFLECTION_ROUNDS": int(max_reflection_rounds or config.MAX_REFLECTION_ROUNDS or 2),
                "ENABLE_PARALLEL_RETRIEVAL": (
                    config.ENABLE_PARALLEL_RETRIEVAL
                    if enable_parallel_retrieval is None
                    else bool(enable_parallel_retrieval)
                ),
                "MAX_RETRIEVAL_WORKERS": int(max_retrieval_workers or config.MAX_RETRIEVAL_WORKERS or 1),
                "USE_JSON_FORMAT": bool(getattr(config, "USE_JSON_FORMAT", False)),
            }
        )

        print("=" * 60)
        print("Initializing SimpleMem System")
        print("=" * 60)

        self.llm_client = LLMClient(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            max_workers=config.LLM_MAX_WORKERS,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
            top_k=config.LLM_TOP_K,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
        )
        self.embedding_tracker = embedding_tracker
        self.embedding_model = EmbeddingModel(
            provider=config.EMBEDDING_PROVIDER,
            model_name=config.EMBEDDING_MODEL,
            batch_size=config.RETRIEVER_BATCH_SIZE,
            tracker=self.embedding_tracker,
            dimension=config.EMBEDDING_DIM,
        )
        self.vector_store = VectorStore(
            db_path=config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            table_name=config.MEMORY_TABLE_NAME,
        )

        if clear_db:
            print("\nClearing existing database...")
            self.vector_store.clear()

        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            window_size=config.WINDOW_SIZE,
            enable_parallel_processing=config.ENABLE_PARALLEL_PROCESSING,
            max_parallel_workers=config.MAX_PARALLEL_WORKERS,
        )
        self.hybrid_retriever = HybridRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            semantic_top_k=config.SEMANTIC_TOP_K,
            keyword_top_k=config.KEYWORD_TOP_K,
            structured_top_k=config.STRUCTURED_TOP_K,
            enable_planning=config.ENABLE_PLANNING,
            enable_reflection=config.ENABLE_REFLECTION,
            max_reflection_rounds=config.MAX_REFLECTION_ROUNDS,
            enable_parallel_retrieval=config.ENABLE_PARALLEL_RETRIEVAL,
            max_retrieval_workers=config.MAX_RETRIEVAL_WORKERS,
        )
        self.answer_generator = AnswerGenerator(llm_client=self.llm_client)

        print("\nSystem initialization complete!")
        print("=" * 60)

    def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None,
        source_log_id: Optional[str] = None,
    ) -> None:
        dialogue_id = self.memory_builder.processed_count + len(self.memory_builder.dialogue_buffer) + 1
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            source_log_id=source_log_id,
        )
        self.memory_builder.add_dialogue(dialogue)

    def add_dialogues(self, dialogues: Sequence[Dialogue]) -> None:
        self.memory_builder.add_dialogues(list(dialogues))

    def finalize(self):
        return self.memory_builder.process_remaining()

    def retrieve(self, question: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        return self.hybrid_retriever.retrieve(question, enable_reflection=enable_reflection)

    def answer_with_contexts(
        self,
        question: str,
        contexts: List[MemoryEntry],
        *,
        prompt_override: Optional[str] = None,
        return_debug: bool = False,
    ) -> Any:
        raw_output = self.answer_generator.generate_answer(
            question,
            contexts,
            prompt_override=prompt_override,
        )
        if not return_debug:
            return raw_output
        return {
            "raw_output": raw_output,
            "prompt": self.answer_generator.last_prompt,
            "contexts": [entry.to_dict() for entry in contexts],
        }

    def ask(
        self,
        question: str,
        *,
        answer_prompt_override: Optional[str] = None,
        enable_reflection: Optional[bool] = None,
        precomputed_contexts: Optional[List[MemoryEntry]] = None,
        return_debug: bool = False,
    ) -> Any:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        contexts = list(precomputed_contexts) if precomputed_contexts is not None else self.retrieve(
            question,
            enable_reflection=enable_reflection,
        )
        answer = self.answer_with_contexts(
            question,
            contexts,
            prompt_override=answer_prompt_override,
            return_debug=return_debug,
        )
        print("=" * 60 + "\n")
        return answer

    def get_all_memories(self) -> List[MemoryEntry]:
        return self.vector_store.get_all_entries()

    def export_builder_state(self) -> Dict[str, Any]:
        return self.memory_builder.export_state()

    def import_builder_state(self, payload: Dict[str, Any]) -> None:
        self.memory_builder.import_state(payload)

    def llm_usage_summary(self, phase: Optional[str] = None) -> Dict[str, Any]:
        return self.llm_client.usage_summary(phase=phase)

    def close(self) -> None:
        self.llm_client.close()


def create_system(**kwargs: Any) -> SimpleMemSystem:
    return SimpleMemSystem(**kwargs)
