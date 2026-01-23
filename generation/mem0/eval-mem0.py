import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mem0 import Memory
from openai import OpenAI


GENERATION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = GENERATION_DIR / "data"
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from load_dataset import MemBenchSample, build_membench_memory_from_event, load_membench_dataset  # type: ignore  # noqa: E402


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


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("mem0_membench_eval")
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

import os
def build_mem0_config(collection_name: str, host: str, port: int) -> Dict[str, Any]:
    return {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                # "api_key":os.getenv("GEMINI_API_KEY")
                
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "host": host,
                "port": port,
            },
        }
    }


def load_mem0_config(
    config_path: Optional[str],
    collection_name: str,
    host: str,
    port: int,
) -> Dict[str, Any]:
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_mem0_config(collection_name, host, port)


def _format_mem0_context(results: Any) -> str:
    def _coerce(entry: Any) -> str:
        if isinstance(entry, dict):
            metadata = entry.get("metadata")
            time_value = None
            if isinstance(metadata, dict):
                time_value = metadata.get("time") or metadata.get("timestamp")
            text = (
                entry.get("memory")
                or entry.get("content")
                or entry.get("text")
                or entry.get("value")
            )
            if text is None:
                return json.dumps(entry, ensure_ascii=False)
            if time_value:
                return f"DATE: {time_value} | {text}"
            return str(text)
        return str(entry)

    if isinstance(results, dict):
        if "results" in results:
            return "\n".join(_coerce(item) for item in results["results"])
        if "context" in results:
            return str(results["context"])
        return json.dumps(results, ensure_ascii=False)
    if isinstance(results, list):
        return "\n".join(_coerce(item) for item in results)
    return str(results)


def _make_prompt(question: str, context: str) -> str:
    return f"""
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION wer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:
""".strip()


def _serialize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def add_app_logs_to_memory(
    memory: Memory,
    sample: MemBenchSample,
    user_id: str,
    logger: logging.Logger,
    max_events: Optional[int] = None,
) -> int:
    added = 0
    for idx, event in enumerate(sample.app_logs):
        if max_events is not None and added >= max_events:
            break
        content, time_str = build_membench_memory_from_event(event)
        memory.add([{"role": "user", "content": content}], metadata={"time": time_str}, user_id=user_id)
        added += 1
        logger.info(f"Added event {idx + 1}/{len(sample.app_logs)} to mem0 memory")
    return added


class Mem0MemBenchAgent:
    def __init__(
        self,
        memory: Memory,
        llm_client: OpenAI,
        model: str,
        user_id: str,
        temperature: float,
    ):
        self.memory = memory
        self.llm_client = llm_client
        self.model = model
        self.user_id = user_id
        self.temperature = temperature
        self.last_search_results: Any = None

    def add_memory(self, content: str) -> None:
        self.memory.add([{"role": "user", "content": content}], user_id=self.user_id)

    def retrieve_memory(self, query: str) -> str:
        self.last_search_results = self.memory.search(query, user_id=self.user_id)
        return _format_mem0_context(self.last_search_results)

    def answer_question(self, question: str) -> tuple[str, str, str]:
        raw_context = self.retrieve_memory(question)
        prompt = _make_prompt(question, raw_context)
        response_format = {
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
        }
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=self.temperature,
            )
        except TypeError:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

        content = response.choices[0].message.content or ""
        prediction = content
        try:
            prediction = json.loads(content)["answer"]
        except Exception:
            prediction = content.strip()
        return prediction, prompt, raw_context


def evaluate_membench_with_mem0(
    app_log_path: Path,
    qa_path: Optional[Path],
    mem0_config: Dict[str, Any],
    user_id: Optional[str],
    llm_model: str,
    temperature: float,
    output_path: Optional[Path],
    log_dir: Path,
    max_events: Optional[int],
    reset_memories: bool,
    size: str,
    sample_filter: Optional[str],
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_mem0_{timestamp}.log"
    logger = setup_logger(log_file)

    memory = Memory.from_config(mem0_config)
    if reset_memories:
        try:
            memory.delete_all_memories()
            logger.info("Cleared existing mem0 collection")
        except Exception as e:
            logger.warning(f"Could not reset mem0 memories: {e}")

    summaries: List[Dict[str, Any]] = []
    for sample in load_membench_dataset(app_log_path, qa_path, size=size):
        if sample_filter and sample.sample_id != sample_filter:
            continue
        sample_user_id = user_id or sample.sample_id
        logger.info(
            f"Loaded MemBench sample_id={sample.sample_id} "
            f"events={len(sample.app_logs)} qa={len(sample.qa)} user_id={sample_user_id}"
        )

        agent = Mem0MemBenchAgent(
            memory=memory,
            llm_client=OpenAI(),
            model=llm_model,
            user_id=sample_user_id,
            temperature=temperature,
        )

        added = add_app_logs_to_memory(
            memory,
            sample,
            sample_user_id,
            logger,
            max_events=max_events,
        )
        logger.info(f"Added {added} events to mem0 memory")
        summaries.append(
            {
                "sample_id": sample.sample_id,
                "user_id": sample_user_id,
                "events": len(sample.app_logs),
                "qa": len(sample.qa),
                "added": added,
            }
        )

    total = 0
    correct = 0
    results: List[Dict[str, Any]] = []

    for idx, qa in enumerate(sample.qa):
        if not qa.question:
            continue
        total += 1
        prediction, prompt, context = agent.answer_question(qa.question)
        search_results = agent.last_search_results
        reference = qa.final_answer if qa.final_answer is not None else qa.answer
        is_correct = normalize_answer(prediction) == normalize_answer(reference)
        if is_correct:
            correct += 1
        qa_log = {
            "idx": idx,
            "question": qa.question,
            "prediction": prediction,
            "reference": reference,
            "evidence": qa.evidence,
            "correct": is_correct,
            "prompt": prompt,
            "context": context,
            "search_results": _serialize_for_json(search_results),
        }
        logger.info(f"QA {total}: {json.dumps(qa_log, ensure_ascii=False)}")
        results.append(qa_log)

    accuracy = (correct / total) if total else 0.0
    summary = {
        "dataset": {
            "app_log_path": str(app_log_path),
            "qa_path": str(qa_path),
            "sample_id": sample.sample_id,
        },
        "user_id": user_id,
        "llm_model": llm_model,
        "temperature": temperature,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # return summary
    return {"samples": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mem0 on MemBench app-log dataset")
    parser.add_argument(
        "--app-log",
        type=str,
        default="",
    )
    parser.add_argument("--qa", type=str, default=None, help="Optional path to QA JSON")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        help="Dataset size: small|medium|large (or s|m|l)",
    )
    parser.add_argument(
        "--user-folder",
        type=str,
        default=None,
        help="Optional sample id to filter when loading a data directory",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="membench_mem0",
        help="Qdrant collection name for mem0 vector store",
    )
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host for mem0 storage")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port for mem0 storage")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a mem0 config JSON (overrides default qdrant config)",
    )
    parser.add_argument("--user-id", type=str, default=None, help="User id to use inside mem0 (defaults to sample id)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model for answering")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to this path")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "logs"),
        help="Directory for eval logs",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of app log events to ingest")
    parser.add_argument(
        "--reset-memories",
        action="store_true",
        help="Wipe the mem0 collection before ingesting (affects all users in that collection)",
    )
    args = parser.parse_args()

    app_log_path = Path(args.app_log)
    if not app_log_path.is_absolute():
        app_log_path = DATA_DIR / app_log_path
    qa_path: Optional[Path] = None
    if args.qa:
        qa_path = Path(args.qa)
        if not qa_path.is_absolute():
            qa_path = DATA_DIR / qa_path
    output_path = Path(args.output) if args.output else None

    mem0_config = load_mem0_config(
        config_path=args.config,
        collection_name=args.collection_name,
        host=args.qdrant_host,
        port=args.qdrant_port,
    )
    print("app_log_path: ", app_log_path)
    input("Hey!")
    summary = evaluate_membench_with_mem0(
        app_log_path=app_log_path,
        qa_path=qa_path,
        mem0_config=mem0_config,
        user_id=args.user_id,
        llm_model=args.model,
        temperature=args.temperature,
        output_path=output_path,
        log_dir=Path(args.log_dir),
        max_events=args.max_events,
        reset_memories=args.reset_memories,
        size=args.size,
        sample_filter=args.user_folder,
    )
    if "accuracy" in summary:
        print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total_questions']})")
    else:
        print("Evaluation complete.")


if __name__ == "__main__":
    main()
