import json
import time
from typing import List, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer, AutoModel

from client import LLMClient

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
        retriever_type: str = "contriever",   # "contriever" | "qwen"
        retriever_model: str = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini",
    ):
        self.schema_path = schema_path
        self.qa_path = qa_path
        self.output_path = output_path

        self.chunk_size = chunk_size
        self.top_k = top_k

        self.retriever_type = retriever_type
        self.retriever_model = retriever_model

        self.llm_provider = llm_provider
        self.llm_model = llm_model


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
        data = json.loads(text)

        logs = data.get("app_logs", [])
        return [
            json.dumps(log, ensure_ascii=False)
            for log in logs
        ]


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

    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray]):
        self.embed_fn = embed_fn
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embed_fn(documents)  # (N, D)

    def retrieve(self, query: str, top_k: int) -> str:
        q_emb = self.embed_fn([query])[0]           # (D,)
        scores = self.embeddings @ q_emb            # (N,)

        top_idx = np.argsort(scores)[::-1][:top_k]

        return "\n<->\n".join(self.documents[i] for i in top_idx)


# =========================
# Embedders
# =========================

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass


class ContrieverEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.eval()

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


# =========================
# RAG Manager
# =========================

PROMPT = """
# Question:
{{ question }}

# Context:
{{ context }}

Return JSON only, with this schema:
{
  "answer": string
  "evidence": List[string]  # List of event IDs(not log IDs) from the context that support the answer
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
        else:
            embedder = ContrieverEmbedder(
                cfg.retriever_model or "facebook/contriever"
            )

        self.retriever = BruteForceRetriever(embedder.embed)

        # ---- Reader ----
        self.llm = LLMClient(
            provider=cfg.llm_provider,
            model_name=cfg.llm_model,
        )
        self.template = Template(PROMPT)

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
        # ---- schema ----
        schema_text = self.load_schema_as_text()
        chunks = self.chunker.chunk(schema_text)
        self.retriever.build_index(chunks)

        # ---- QA ----
        with open(self.cfg.qa_path, "r", encoding="utf-8") as f:
            qa_list = json.load(f)

        results = []

        for q in tqdm(qa_list, desc="Answering"):
            t1 = time.time()
            context = self.retriever.retrieve(
                q["query"], self.cfg.top_k
            )
            t2 = time.time()

            t3 = time.time()
            raw = self.answer(q["query"], context)
            prediction = json.loads(raw)["answer"]
            evidence = json.loads(raw)["evidence"]
            t4 = time.time()

            results.append(
                {
                    "id": q.get("id"),
                    "query": q["query"],
                    "reference": q.get("reference"),
                    "prediction": prediction,
                    "metadata": {
                        **(q.get("metadata") or {}),
                        "context": context,
                        "search_time": t2 - t1,
                        "response_time": t4 - t3,
                        "evidence_prediction": evidence,
                    },
                }
            )

        with open(self.cfg.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    cfg = RAGConfig(
        schema_path="vannila-rag/data/schema.json",
        qa_path="vannila-rag/data/QA.json",
        output_path="vannila-rag/data/rag_results.json",

        chunk_size=-1,
        top_k=5,

        retriever_type="qwen",
        retriever_model=None,

        llm_provider="openai",
        llm_model="gpt-5-mini",
    )

    rag = RAGManager(cfg)
    rag.run()
