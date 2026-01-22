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

from memory_layer import AgenticMemorySystem, LLMController

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
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.llm = LLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature = temperature

    def add_memory(self, content: str, time: Optional[str] = None) -> None:
        print("content:", content)
        input('enter')
        self.memory_system.add_note(content, time=time)

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
    model: str,
    backend: str,
    retrieve_k: int,
    temperature: float,
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

    sample: MemBenchSample = load_membench_dataset(app_log_path, qa_path)
    logger.info(f"Loaded MemBench sample_id={sample.sample_id} events={len(sample.app_logs)} qa={len(sample.qa)}")
    agent = MemBenchAgent(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        temperature=temperature,
        sglang_host=sglang_host,
        sglang_port=sglang_port,
    )
    max_events = 5
    for idx, event in enumerate(sample.app_logs):
        content, time_str = build_membench_memory_from_event(event)
        agent.add_memory(content, time=time_str)
        if idx + 1 >= max_events:
            break
    logger.info(f"Added {min(len(sample.app_logs), max_events)} events to memory (max_events={max_events})")

    total = 0
    correct = 0
    results = []
    for qa in sample.qa:
        if not qa.question:
            continue
        total += 1
        prediction, prompt, raw_context = agent.answer_question(qa.question)
        is_correct = normalize_answer(prediction) == normalize_answer(qa.answer)
        if is_correct:
            correct += 1
        qa_log = {
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

    accuracy = (correct / total) if total else 0.0
    summary = {
        "dataset": {"app_log_path": app_log_path, "qa_path": qa_path, "sample_id": sample.sample_id},
        "model": model,
        "backend": backend,
        "retrieve_k": retrieve_k,
        "temperature": temperature,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    # logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent on MemBench app-log dataset")
    parser.add_argument("--app-log", type=str, default="../mock_data/app_log_518.json", help="Path to app log JSON")
    parser.add_argument("--qa", type=str, default="../mock_data/qa_samples.json", help="Path to QA JSON")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--backend", type=str, default="sglang", help="Backend (openai, ollama, sglang)")
    parser.add_argument("--retrieve-k", type=int, default=10, help="Number of retrieved memories")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to this path")
    parser.add_argument("--sglang-host", type=str, default="http://localhost", help="SGLang server host")
    parser.add_argument("--sglang-port", type=int, default=30000, help="SGLang server port")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    mock_data_dir = os.path.normpath(os.path.join(base_dir, "..", "mock_data"))
    app_log_path = args.app_log
    qa_path = args.qa
    if not os.path.isabs(app_log_path):
        candidate = os.path.join(base_dir, app_log_path)
        if os.path.exists(candidate):
            app_log_path = candidate
        else:
            app_log_path = os.path.join(mock_data_dir, os.path.basename(app_log_path))
    if not os.path.isabs(qa_path):
        candidate = os.path.join(base_dir, qa_path)
        if os.path.exists(candidate):
            qa_path = candidate
        else:
            qa_path = os.path.join(mock_data_dir, os.path.basename(qa_path))
    output_path = args.output
    if output_path and not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    summary = evaluate_membench(
        app_log_path=app_log_path,
        qa_path=qa_path,
        model=args.model,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        temperature=args.temperature,
        output_path=output_path,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    )
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total_questions']})")


if __name__ == "__main__":
    main()
