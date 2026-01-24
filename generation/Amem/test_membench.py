import argparse
import json
import os
import re
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

import nltk

from memory_layer import AgenticMemorySystem, LLMController, MemoryNote, SimpleEmbeddingRetriever

GENERATION_DIR = Path(__file__).resolve().parent.parent
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import build_membench_memory_from_event, load_membench_dataset, MemBenchSample  # type: ignore


def _ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        raise RuntimeError(
            "Missing NLTK data 'punkt'. Install it with: python -m nltk.downloader punkt"
        ) from e


_PUNCT_RE = re.compile(r"[^0-9a-zA-Z]+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("membench_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _parse_cached_document(doc: str) -> dict:
    content_marker = "content:"
    context_marker = " context:"
    keywords_marker = " keywords:"
    tags_marker = " tags:"

    if content_marker not in doc:
        return {"content": doc, "context": None, "keywords": None, "tags": None}

    content_part, _, remainder = doc.partition(context_marker)
    content = content_part[len(content_marker):].strip()
    context = None
    keywords = None
    tags = None

    if remainder:
        context_part, _, remainder = remainder.partition(keywords_marker)
        context = context_part.strip() or None
        if remainder:
            keywords_part, _, tags_part = remainder.partition(tags_marker)
            keywords = [item.strip() for item in keywords_part.split(",") if item.strip()]
            if tags_part:
                tags = [item.strip() for item in tags_part.split(",") if item.strip()]

    return {"content": content, "context": context, "keywords": keywords, "tags": tags}


def _hydrate_memories_from_retriever(
    memory_system: AgenticMemorySystem,
    retriever: SimpleEmbeddingRetriever,
) -> None:
    memory_system.memories = {}
    for doc in retriever.corpus:
        parsed = _parse_cached_document(doc)
        note = MemoryNote(
            content=parsed["content"],
            keywords=parsed.get("keywords"),
            context=parsed.get("context"),
            tags=parsed.get("tags"),
            timestamp="unknown",
            llm_controller=None,
        )
        memory_system.memories[note.id] = note


class MemBenchAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        temperature: float,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        self.memory_system = AgenticMemorySystem(
            model_name="text-embedding-3-large",
            embedding_backend="litellm",
            api_base="https://api.aimlapi.com/v1",
            api_key=os.getenv("AIML_API_KEY"),
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.llm = LLMController(
            backend=backend,
            model=model,
            api_base="https://api.aimlapi.com/v1",
            api_key=os.getenv("AIML_API_KEY"),
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature = temperature

    def add_memory(self, content: str, time: Optional[str] = None) -> None:
        self.memory_system.add_note(content, time=time)

    def add_memory_batch(self, items: list[tuple[str, Optional[str]]]) -> None:
        messages = [{"content": content, "time": time} for content, time in items]
        self.memory_system.add(messages)

    def retrieve_memory(self, query: str) -> str:
        return self.memory_system.find_related_memories_raw(query, k=self.retrieve_k)

    def answer_question(self, question: str) -> tuple[str, str, str]:
        raw_context = self.retrieve_memory(question)
        prompt = f"""Context:
{raw_context}

Question: {question}

Return a short answer based only on the context. Respond as JSON with key 'answer'."""
        response = self.llm.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        prediction = response
        try:
            prediction = json.loads(response)["answer"]
        except Exception:
            prediction = response.strip()
        return prediction, prompt, raw_context


def evaluate_membench(
    app_log_path: str,
    qa_path: str,
    size: str,
    model: str,
    backend: str,
    retrieve_k: int,
    temperature: float,
    batch_size: int,
    output_path: Optional[str],
    sglang_host: str,
    sglang_port: int,
) -> dict:
    _ensure_nltk()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_membench_{backend}_{timestamp}.log"
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, log_filename))
    cache_dir = Path(log_dir) / "retriever_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    total = 0
    correct = 0
    results = []
    sample_summaries = []

    for sample in load_membench_dataset(app_log_path, qa_path, size=size):
        logger.info(
            f"Loaded MemBench sample_id={sample.sample_id} "
            f"events={len(sample.app_logs)} qa={len(sample.qa)}"
        )
        agent = MemBenchAgent(
            model=model,
            backend=backend,
            retrieve_k=retrieve_k,
            temperature=temperature,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        # max_events = 5
        retriever = agent.memory_system.retriever
        if isinstance(retriever, SimpleEmbeddingRetriever):
            cache_prefix = f"retriever_{sample.sample_id}_{size}"
            cache_file = cache_dir / f"{cache_prefix}.pkl"
            cache_embeddings = cache_dir / f"{cache_prefix}.npy"
            if cache_file.exists() and cache_embeddings.exists():
                retriever.load(cache_file, cache_embeddings)
                _hydrate_memories_from_retriever(agent.memory_system, retriever)
                logger.info("Loaded retriever cache: %s (skipped ingest)", cache_prefix)
            else:
                batch: list[tuple[str, Optional[str]]] = []
                for idx, event in enumerate(sample.app_logs):
                    content, time_str = build_membench_memory_from_event(event)
                    batch.append((content, time_str))
                    if len(batch) >= batch_size:
                        agent.add_memory_batch(batch)
                        batch = []
                    print(f"idx: {idx}")
                if batch:
                    agent.add_memory_batch(batch)
                logger.info(
                    "Added %d events to memory (max_events=%d)",
                    len(sample.app_logs)
                )
                retriever.save(cache_file, cache_embeddings)
                retriever.load(cache_file, cache_embeddings)
                logger.info("Saved and reloaded retriever cache: %s", cache_prefix)
        else:
            batch: list[tuple[str, Optional[str]]] = []
            for idx, event in enumerate(sample.app_logs):
                content, time_str = build_membench_memory_from_event(event)
                batch.append((content, time_str))
                if len(batch) >= batch_size:
                    agent.add_memory_batch(batch)
                    batch = []
            if batch:
                agent.add_memory_batch(batch)
                # if idx + 1 >= max_events:
                #     break
            logger.info(
                "Added %d events to memory (max_events=%d)",
                len(sample.app_logs)
            )
            logger.info(
                "Retriever type %s does not support save/load",
                type(retriever).__name__,
            )

        sample_total = 0
        sample_correct = 0
        for qa in sample.qa:
            if not qa.question:
                continue
            sample_total += 1
            total += 1
            prediction, prompt, raw_context = agent.answer_question(qa.question)
            is_correct = normalize_answer(prediction) == normalize_answer(qa.answer)
            if is_correct:
                sample_correct += 1
                correct += 1
            qa_log = {
                "sample_id": sample.sample_id,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.answer,
                "evidence": qa.evidence,
                "correct": is_correct,
                "prompt": prompt,
                "raw_context": raw_context,
            }
            logger.info(f"QA {total}: {json.dumps(qa_log, ensure_ascii=False)}")
            results.append(qa_log)

        sample_accuracy = (sample_correct / sample_total) if sample_total else 0.0
        sample_summaries.append(
            {
                "sample_id": sample.sample_id,
                "total_questions": sample_total,
                "correct": sample_correct,
                "accuracy": sample_accuracy,
            }
        )

    accuracy = (correct / total) if total else 0.0
    summary = {
        "dataset": {"app_log_path": app_log_path, "qa_path": qa_path, "size": size},
        "model": model,
        "backend": backend,
        "retrieve_k": retrieve_k,
        "temperature": temperature,
        "batch_size": batch_size,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "samples": sample_summaries,
        "results": results,
    }
    # logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent on MemBench app-log dataset")
    parser.add_argument("--app-log", type=str, default="", help="Path to app log JSON")
    parser.add_argument("--qa", type=str, default="data/qa_samples.json", help="Path to QA JSON")
    parser.add_argument("--size", type=str, default="small", help="Dataset size: small|medium|large")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--backend", type=str, default="sglang", help="Backend (openai, ollama, sglang)")
    parser.add_argument("--retrieve-k", type=int, default=10, help="Number of retrieved memories")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for ingesting app logs")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to this path")
    parser.add_argument("--sglang-host", type=str, default="http://localhost", help="SGLang server host")
    parser.add_argument("--sglang-port", type=int, default=30000, help="SGLang server port")
    args = parser.parse_args()

    GENERATION_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = GENERATION_DIR / "data"
    qa_path = DATA_DIR / "qa_samples.json"
    if str(GENERATION_DIR) not in sys.path:
        sys.path.append(str(GENERATION_DIR))

    base_dir = os.path.dirname(__file__)
    
    
    output_path = args.output
    if output_path and not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    summary = evaluate_membench(
        app_log_path=DATA_DIR,
        qa_path=qa_path,
        size=args.size,
        model=args.model,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        temperature=args.temperature,
        batch_size=args.batch_size,
        output_path=output_path,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    )
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total_questions']})")


if __name__ == "__main__":
    main()
