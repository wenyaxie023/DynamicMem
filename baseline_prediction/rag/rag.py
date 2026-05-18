import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from client import LLMClient
from dotenv import load_dotenv

load_dotenv()


# =========================
# Config
# =========================

class RAGConfig:
    def __init__(
        self,
        *,
        schema_path: str,
        qa_path: str,
        output_path: str = "data/rag_results.json",
        chunk_size: int = 500,   # -1 => AppLogChunker
        top_k: int = 3,
        retrieval_top_k: int = 20,
        retrieval_output_paths: Dict[int, str] | None = None,
        retriever_type: str = "contriever",   # "contriever" | "qwen" | "openai" | "sentence_transformers"
        retriever_model: str = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini",
        llm_max_workers: int = 4,
    ):
        self.schema_path = schema_path
        self.qa_path = qa_path
        self.output_path = output_path

        self.chunk_size = chunk_size
        self.top_k = top_k
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_output_paths = retrieval_output_paths or {}

        self.retriever_type = retriever_type
        self.retriever_model = retriever_model

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_max_workers = llm_max_workers


# =========================
# Chunker
# =========================

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        return [
            text[i: i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]


class AppLogChunker(Chunker):
    """
    Each item in `app_logs` array becomes ONE chunk.
    No semantic processing.
    """

    def chunk(self, text: str) -> List[str]:
        try:
            import tiktoken
        except Exception as exc:
            raise RuntimeError(
                "Token-based chunking requires tiktoken. "
                "Install it with `pip install tiktoken`."
            ) from exc

        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = 8000

        data = json.loads(text)

        if isinstance(data, list):
            logs = data
        else:
            logs = data.get("app_logs", [])
        chunks: List[dict] = []
        for log in logs:
            raw = json.dumps(log, ensure_ascii=False)
            tokens = enc.encode(raw)
            if len(tokens) <= max_tokens:
                chunks.append(log)
                continue
            # Split by tokens to avoid embedding context limits.
            for i in range(0, len(tokens), max_tokens):
                chunks.append(
                    {
                        "app_log_id": log.get("app_log_id"),
                        "chunk": enc.decode(tokens[i: i + max_tokens]),
                    }
                )
        return chunks


# =========================
# Retriever (Brute Force)
# =========================

class Retriever(ABC):
    @abstractmethod
    def build_index(self, documents: List[str]):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> str:
        pass


class BruteForceRetriever(Retriever):
    """
    Brute-force cosine similarity retriever.
    embeddings must be L2-normalized.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        *,
        batch_size: int = 64,
        length_fn: Callable[[str], int] | None = None,
        long_text_tokens: int = 4096,
    ):
        self.embed_fn = embed_fn
        self.batch_size = batch_size
        self.length_fn = length_fn or (lambda text: len(text))
        self.long_text_tokens = long_text_tokens
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def _iter_batches(self, texts: List[str]) -> List[List[str]]:
        if not texts:
            return []
        lengths = [self.length_fn(text) for text in texts]
        indexed = list(enumerate(texts))
        indexed.sort(key=lambda item: lengths[item[0]])

        batches: List[List[str]] = []
        current: List[str] = []
        for idx, text in indexed:
            if lengths[idx] >= self.long_text_tokens:
                if current:
                    batches.append(current)
                    current = []
                batches.append([text])
                continue
            current.append(text)
            if len(current) >= self.batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def build_index(self, documents: List[str]):
        self.documents = documents
        if not documents:
            self.embeddings = np.empty((0, 0), dtype="float32")
            return

        texts: List[str] = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            else:
                texts.append(json.dumps(doc, ensure_ascii=False))

        batches = []
        for batch in tqdm(
            self._iter_batches(texts),
            desc="Embedding",
        ):
            batches.append(self.embed_fn(batch))
        self.embeddings = np.vstack(batches)  # (N, D)

    def retrieve_docs(self, query: str, top_k: int) -> List[dict | str]:
        q_emb = self.embed_fn([query])[0]           # (D,)
        scores = self.embeddings @ q_emb            # (N,)

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [self.documents[i] for i in top_idx]

    def retrieve(self, query: str, top_k: int) -> str:
        docs = self.retrieve_docs(query, top_k)
        return "\n<->\n".join(
            doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            for doc in docs
        )


# =========================
# Embedders
# =========================

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass

    def length(self, text: str) -> int:
        return len(text)


class ContrieverEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.eval()

    def length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @torch.no_grad()
    def embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb.cpu().numpy().astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


class QwenEmbeddingEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @torch.no_grad()
    def embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb.float().cpu().numpy().astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


class OpenAIEmbeddingEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        *,
        max_batch_texts: int = 32,
        max_batch_chars: int = 8000,
        max_text_tokens: int = 8000,
    ):
        self.client = OpenAI()
        self.model_name = model_name
        self.max_batch_texts = max_batch_texts
        self.max_batch_chars = max_batch_chars
        self.max_text_tokens = max_text_tokens

        try:
            import tiktoken
        except Exception as exc:
            raise RuntimeError(
                "OpenAI embeddings require tiktoken for token truncation. "
                "Install it with `pip install tiktoken`."
            ) from exc

        try:
            self._encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def length(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        truncated: List[str] = []
        for text in texts:
            tokens = self._encoder.encode(text)
            if len(tokens) > self.max_text_tokens:
                tokens = tokens[: self.max_text_tokens]
                truncated.append(self._encoder.decode(tokens))
            else:
                truncated.append(text)
        return truncated

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        texts = self._truncate_texts(texts)

        batches: List[List[str]] = []
        current: List[str] = []
        current_chars = 0

        for text in texts:
            text_len = len(text)
            if (
                current
                and (
                    len(current) >= self.max_batch_texts
                    or current_chars + text_len > self.max_batch_chars
                )
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(text)
            current_chars += text_len

        if current:
            batches.append(current)

        all_embs: List[np.ndarray] = []
        for batch in batches:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_emb = np.array(
                [item.embedding for item in response.data],
                dtype="float32",
            )
            all_embs.append(batch_emb)

        emb = np.vstack(all_embs)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def length(self, text: str) -> int:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return len(text)
        return len(tokenizer.encode(text))

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return emb


# =========================
# RAG Manager
# =========================

PROMPT = """# Question:
{{ question }}

# User App Logs:
{{ context }}

Return JSON only, with this schema:
{
  "evidence": [{"app_log_id": string, "supporting_content": string}],  # List of dicts; supporting_content should preserve key supporting content as faithfully as possible, and should be the exact content of the app log
  "answer": string
}
"""


class RAGManager:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

        # ---- Chunker ----
        if cfg.chunk_size == -1:
            self.chunker = AppLogChunker()
        else:
            self.chunker = FixedSizeChunker(cfg.chunk_size)

        # ---- Embedder + Retriever ----
        if cfg.retriever_type == "qwen":
            embedder = QwenEmbeddingEmbedder(
                cfg.retriever_model or "Qwen/Qwen3-Embedding-8B"
            )
        elif cfg.retriever_type == "sentence_transformers":
            embedder = SentenceTransformerEmbedder(
                cfg.retriever_model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        elif cfg.retriever_type == "openai":
            embedder = OpenAIEmbeddingEmbedder(
                cfg.retriever_model or "text-embedding-3-large"
            )
        else:
            embedder = ContrieverEmbedder(
                cfg.retriever_model or "facebook/contriever"
            )

        self.retriever = BruteForceRetriever(
            embedder.embed,
            length_fn=embedder.length,
        )

        # ---- Reader ----
        self.llm = LLMClient(
            provider=cfg.llm_provider,
            model_name=cfg.llm_model,
            max_workers=cfg.llm_max_workers,
        )
        self.template = Template(PROMPT)

    def _safe_parse_llm_output(self, raw: object) -> tuple[dict, str | None]:
        if isinstance(raw, dict):
            return raw, None

        if isinstance(raw, str):
            try:
                return json.loads(raw), None
            except json.JSONDecodeError as exc:
                # Try client-side extraction fallback.
                try:
                    parsed = self.llm._parse_response(raw, "json")
                    return parsed, None
                except Exception:
                    # Last resort: trim to outermost JSON object.
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end > start:
                        snippet = raw[start : end + 1]
                        try:
                            return json.loads(snippet), None
                        except Exception:
                            pass
                return {"answer": "", "evidence": []}, (
                    f"LLM JSON parse failed: {exc}"
                )

        return {"answer": "", "evidence": []}, (
            f"Unexpected LLM output type: {type(raw).__name__}"
        )

    def load_schema_as_text(self) -> str:
        with open(self.cfg.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return json.dumps(schema, ensure_ascii=False)

    def answer(self, question: str, context: str) -> str:
        prompt = self.template.render(
            question=question,
            context=context,
        )
        return self.llm.ask(prompt, response_type="text")

    def run(self):
        return self.run_two_stage(skip_retrieve=False)

    def _load_qa_list(self) -> List[dict]:
        print("[RAG] Loading QA list...")
        with open(self.cfg.qa_path, "r", encoding="utf-8") as f:
            qa_list = json.load(f)
        print(f"[RAG] Loaded {len(qa_list)} QA items.")
        return qa_list

    def _build_index(self) -> None:
        print("[RAG] Loading schema...")
        schema_text = self.load_schema_as_text()
        chunks = self.chunker.chunk(schema_text)
        print(f"[RAG] Chunked schema into {len(chunks)} chunks.")
        print("[RAG] Building index (embedding corpus)...")
        self.retriever.build_index(chunks)
        print("[RAG] Index built.")

    def _retrieve_all(self, qa_list: List[dict]) -> List[dict]:
        retrieval_items: List[dict] = []
        for q in tqdm(qa_list, desc="Retrieving"):
            t1 = time.time()
            docs = self.retriever.retrieve_docs(
                q["query"], self.cfg.retrieval_top_k
            )
            t2 = time.time()

            contexts = {
                k: docs[:k]
                for k in sorted(self.cfg.retrieval_output_paths)
            }

            retrieval_items.append(
                {
                    "id": q.get("id"),
                    "query": q["query"],
                    "reference": q.get("reference"),
                    "metadata": q.get("metadata") or {},
                    "contexts": contexts,
                    "search_time": t2 - t1,
                }
            )

        return retrieval_items

    def _write_retrieval_outputs(self, retrieval_items: List[dict]) -> None:
        for k, path in self.cfg.retrieval_output_paths.items():
            output_items = []
            for item in retrieval_items:
                output_items.append(
                    {
                        "id": item.get("id"),
                        "query": item["query"],
                        "reference": item.get("reference"),
                        "metadata": item.get("metadata") or {},
                        "context": item["contexts"][k],
                        "search_time": item["search_time"],
                    }
                )

            with open(path, "w", encoding="utf-8") as f:
                json.dump(output_items, f, indent=2, ensure_ascii=False)

    def _load_retrieval_output(self, path: str) -> List[dict]:
        print(f"[RAG] Loading retrieval file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _generate(
        self,
        items: List[dict],
        *,
        output_path: str | None = None,
        write_each: bool = False,
        initial_results: List[dict] | None = None,
    ) -> List[dict]:
        max_parse_retries = 1
        results = list(initial_results or [])

        def _normalize_context(raw_context: object) -> str:
            if isinstance(raw_context, list):
                return "\n<->\n".join(
                    c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                    for c in raw_context
                )
            return raw_context if isinstance(raw_context, str) else json.dumps(raw_context, ensure_ascii=False)

        def _answer_one(item: dict) -> dict:
            context = _normalize_context(item["context"])
            prompt = self.template.render(
                question=item["query"],
                context=context,
            )
            t1 = time.time()
            try:
                raw = self.llm.ask(prompt, response_type="text")
                parse_error = None
                parsed, parse_error = self._safe_parse_llm_output(raw)
                if parse_error:
                    for _ in range(max_parse_retries):
                        try:
                            retry_raw = self.llm.ask(
                                prompt, response_type="text"
                            )
                        except Exception as exc:
                            parse_error = (
                                f"LLM retry failed: {exc}"
                            )
                            continue
                        parsed, parse_error = (
                            self._safe_parse_llm_output(retry_raw)
                        )
                        if not parse_error:
                            raw = retry_raw
                            break

                prediction = parsed.get("answer", "")
                evidence = parsed.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = []
                t2 = time.time()

                metadata = {
                    **(item.get("metadata") or {}),
                    "context": context,
                    "search_time": item.get("search_time"),
                    "response_time": t2 - t1,
                    "evidence_prediction": evidence,
                }
                if parse_error:
                    metadata["llm_parse_error"] = parse_error
                    if isinstance(raw, str):
                        metadata["llm_raw_preview"] = raw[:1000]

                return {
                    "id": item.get("id"),
                    "query": item["query"],
                    "reference": item.get("reference"),
                    "prediction": prediction,
                    "metadata": metadata,
                }
            except Exception as exc:
                t2 = time.time()
                return {
                    "id": item.get("id"),
                    "query": item.get("query", ""),
                    "reference": item.get("reference"),
                    "prediction": "",
                    "metadata": {
                        **(item.get("metadata") or {}),
                        "context": context,
                        "search_time": item.get("search_time"),
                        "response_time": t2 - t1,
                        "llm_parse_error": f"Unhandled error: {exc}",
                    },
                }

        # If write_each is enabled, generate one-by-one and write after each item.
        if write_each and output_path:
            for item in tqdm(items, desc="Answering"):
                results.append(_answer_one(item))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            return results

        prompts: List[str] = []
        contexts: List[str] = []
        start_times: List[float] = []

        for item in items:
            context = _normalize_context(item["context"])
            prompt = self.template.render(
                question=item["query"],
                context=context,
            )
            prompts.append(prompt)
            contexts.append(context)
            start_times.append(time.time())

        raw_results = self.llm.ask_many(prompts, response_type="text")

        for item, context, t1, prompt, raw in tqdm(
            zip(items, contexts, start_times, prompts, raw_results),
            total=len(items),
            desc="Answering",
        ):
            try:
                parse_error = None
                if isinstance(raw, Exception):
                    parsed = {"answer": "", "evidence": []}
                    parse_error = f"LLM request failed: {raw}"
                else:
                    parsed, parse_error = self._safe_parse_llm_output(raw)

                if parse_error:
                    for attempt in range(max_parse_retries):
                        try:
                            retry_raw = self.llm.ask(
                                prompt, response_type="text"
                            )
                        except Exception as exc:
                            parse_error = (
                                f"LLM retry failed: {exc}"
                            )
                            continue
                        parsed, parse_error = (
                            self._safe_parse_llm_output(retry_raw)
                        )
                        if not parse_error:
                            raw = retry_raw
                            break

                prediction = parsed.get("answer", "")
                evidence = parsed.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = []
                t2 = time.time()

                metadata = {
                    **(item.get("metadata") or {}),
                    "context": context,
                    "search_time": item.get("search_time"),
                    "response_time": t2 - t1,
                    "evidence_prediction": evidence,
                }
                if parse_error:
                    metadata["llm_parse_error"] = parse_error
                    if isinstance(raw, str):
                        metadata["llm_raw_preview"] = raw[:1000]

                results.append(
                    {
                        "id": item.get("id"),
                        "query": item["query"],
                        "reference": item.get("reference"),
                        "prediction": prediction,
                        "metadata": metadata,
                    }
                )
            except Exception as exc:
                t2 = time.time()
                results.append(
                    {
                        "id": item.get("id"),
                        "query": item.get("query", ""),
                        "reference": item.get("reference"),
                        "prediction": "",
                        "metadata": {
                            **(item.get("metadata") or {}),
                            "context": context,
                            "search_time": item.get("search_time"),
                            "response_time": t2 - t1,
                            "llm_parse_error": f"Unhandled error: {exc}",
                        },
                    }
                )
        return results

    def run_two_stage(
        self,
        *,
        skip_retrieve: bool = False,
        retrieve_only: bool = False,
        write_each: bool = False,
        resume: bool = False,
    ) -> List[dict]:
        if skip_retrieve and retrieve_only:
            raise ValueError("Cannot use --skip-retrieve with --retrieve-only.")
        if skip_retrieve:
            retrieval_path = self.cfg.retrieval_output_paths.get(self.cfg.top_k)
            if not retrieval_path:
                raise ValueError(
                    f"No retrieval file configured for top_k={self.cfg.top_k}"
                )
            retrieval_items = self._load_retrieval_output(retrieval_path)
        else:
            self._build_index()
            qa_list = self._load_qa_list()
            retrieval_items = self._retrieve_all(qa_list)
            self._write_retrieval_outputs(retrieval_items)
            if retrieve_only:
                return []
            retrieval_items = [
                {
                    "id": item.get("id"),
                    "query": item["query"],
                    "reference": item.get("reference"),
                    "metadata": item.get("metadata") or {},
                    "context": item["contexts"][self.cfg.top_k],
                    "search_time": item["search_time"],
                }
                for item in retrieval_items
            ]

        existing_results: List[dict] = []
        done_keys: set[tuple[str, object]] = set()

        def _item_key(item: dict) -> tuple[str, object] | None:
            if item.get("id") is not None:
                return ("id", item.get("id"))
            if item.get("query"):
                return ("query", item.get("query"))
            return None

        if resume:
            output_path = Path(self.cfg.output_path)
            if output_path.exists():
                try:
                    existing_results = json.loads(
                        output_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(existing_results, list):
                        existing_results = []
                except Exception:
                    existing_results = []

                for item in existing_results:
                    key = _item_key(item)
                    if key:
                        done_keys.add(key)

                if done_keys:
                    before = len(retrieval_items)
                    retrieval_items = [
                        item
                        for item in retrieval_items
                        if _item_key(item) not in done_keys
                    ]
                    after = len(retrieval_items)
                    print(
                        f"[RAG] Resume: skipping {before - after} already generated items."
                    )

        results = self._generate(
            retrieval_items,
            output_path=self.cfg.output_path,
            write_each=write_each,
            initial_results=existing_results if write_each else None,
        )

        if existing_results and not write_each:
            results = existing_results + results

        with open(self.cfg.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage RAG pipeline")
    parser.add_argument(
        "--user-idx",
        required=True,
        help="User index (e.g. 1/2/3) or full directory name like 001_user_001",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(
            "generation/rag/results"
        ),
        help="Root directory containing user subdirectories (legacy, used for input and output)",
    )
    parser.add_argument(
        "--input-root-dir",
        type=Path,
        default=None,
        help="Root directory containing user inputs (app logs). If unset, falls back to --root-dir.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=None,
        help="Root directory containing user outputs. If unset, falls back to --root-dir.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "qa",
        help="Directory containing qa_human_{user_id}.json",
    )
    parser.add_argument(
        "--gen-topk",
        type=int,
        choices=[5, 10, 20],
        default=5,
        help="Top-k context size for generation",
    )
    parser.add_argument(
        "--skip-retrieve",
        action="store_true",
        help="Skip retrieval and load saved contexts",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Run retrieval and save contexts only",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="Chunk size for schema (use -1 for AppLogChunker)",
    )
    parser.add_argument(
        "--write-each",
        action="store_true",
        help="Write output JSON after each answer",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation by skipping already-generated items in output JSON",
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="openai",
        choices=["contriever", "qwen", "openai", "sentence_transformers"],
    )
    parser.add_argument(
        "--retriever-model",
        type=str,
        default="text-embedding-3-large",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-mini",
    )
    parser.add_argument(
        "--llm-max-workers",
        type=int,
        default=4,
        help="Max parallel LLM requests.",
    )

    args = parser.parse_args()

    def _normalize_user_dir(user_idx: str) -> str:
        if user_idx.isdigit():
            idx = int(user_idx)
            return f"{idx:03d}_user_{idx:03d}"
        return user_idx

    def _normalize_user_id(user_idx: str) -> str:
        if user_idx.isdigit():
            return f"{int(user_idx):03d}"
        digits = "".join(ch for ch in user_idx if ch.isdigit())
        if len(digits) >= 3:
            return digits[-3:]
        return user_idx

    def _select_qa_path(
        user_dir: Path,
        qa_dir: Path,
        user_idx: str,
    ) -> Path:
        user_id = _normalize_user_id(user_idx)
        qa_path = qa_dir / f"qa_human_{user_id}.json"
        if qa_path.exists():
            return qa_path
        import pdb; pdb.set_trace()

        # Legacy naming convention fallback in user directory.
        legacy = user_dir / "QA.json"
        if legacy.exists():
            return legacy
        return user_dir / "qa_w0_w4_with_app_logs.json"

    input_root = args.input_root_dir or args.root_dir
    output_root = args.output_root_dir or args.root_dir

    input_user_dir = input_root / _normalize_user_dir(args.user_idx)
    output_user_dir = output_root / _normalize_user_dir(args.user_idx)

    output_user_dir.mkdir(parents=True, exist_ok=True)

    schema_path = input_user_dir / "app_log_large.json"
    qa_path = _select_qa_path(
        input_user_dir,
        args.qa_dir,
        args.user_idx,
    )

    retrieval_dir = output_user_dir / "memory"
    prediction_dir = output_user_dir / "prediction"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    retrieval_output_paths = {
        5: str(retrieval_dir / "rag_retrieval_top5.json"),
        10: str(retrieval_dir / "rag_retrieval_top10.json"),
        20: str(retrieval_dir / "rag_retrieval_top20.json"),
    }

    output_path = prediction_dir / f"rag_results_top{args.gen_topk}.json"

    cfg = RAGConfig(
        schema_path=str(schema_path),
        qa_path=str(qa_path),
        output_path=str(output_path),
        chunk_size=args.chunk_size,
        top_k=args.gen_topk,
        retrieval_top_k=20,
        retrieval_output_paths=retrieval_output_paths,
        retriever_type=args.retriever_type,
        retriever_model=args.retriever_model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
    )

    rag = RAGManager(cfg)
    rag.run_two_stage(
        skip_retrieve=args.skip_retrieve,
        retrieve_only=args.retrieve_only,
        write_each=args.write_each,
        resume=args.resume,
    )
