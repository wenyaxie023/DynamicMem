import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .llm_controller import record_usage

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None

class TrackedOpenAIEmbeddingFunction:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        api_base: Optional[str],
        model_name: str,
    ):
        if OpenAI is None:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.api_base = api_base
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
            if os.getenv("AZURE_OPENAI_API_KEY") or "azure" in api_base.lower():
                client_kwargs["default_headers"] = {"api-key": api_key}
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name

    @staticmethod
    def name() -> str:
        return "openai"

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    @staticmethod
    def default_space() -> str:
        return "cosine"

    @staticmethod
    def supported_spaces() -> List[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
        }

    def __call__(self, input: List[str]) -> List[List[float]]:
        values = list(input)
        if not values:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=values,
        )
        record_usage(
            request_kind="embedding",
            provider="openai",
            model=self.model_name,
            usage=getattr(response, "usage", None),
        )
        return [item.embedding for item in response.data]

class SimpleEmbeddingRetriever:
    """Original A-mem style in-memory embedding retriever with snapshot helpers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_backend: str = "sentence-transformers",
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        directory: Optional[str] = None,
        extend: bool = False,
        collection_name: str = "memories",
    ):
        self.model_name = model_name
        self.embedding_backend = embedding_backend
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.collection_name = collection_name
        self.directory = Path(directory) if directory else None
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model: Any = None
        self._embedding_function: Any = None
        if extend and self.directory is not None:
            self._load_from_directory(self.directory)

    def _get_sentence_transformer(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "sentence-transformers package not found. Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_openai_embedding_function(self) -> TrackedOpenAIEmbeddingFunction:
        if self._embedding_function is None:
            self._embedding_function = TrackedOpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                api_base=self.openai_api_base,
                model_name=self.model_name,
            )
        return self._embedding_function

    def _embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, 0), dtype="float32")
        if self.embedding_backend == "openai":
            embeddings = self._get_openai_embedding_function()(documents)
            return np.asarray(embeddings, dtype="float32")
        model = self._get_sentence_transformer()
        return np.asarray(model.encode(documents), dtype="float32")

    @staticmethod
    def _cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.zeros((0,), dtype="float32")
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        corpus = np.asarray(embeddings, dtype="float32")
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        corpus_norm = np.linalg.norm(corpus, axis=1, keepdims=True)
        denom = np.clip(query_norm * corpus_norm.T, 1e-12, None)
        scores = (query @ corpus.T) / denom
        return scores[0]

    @staticmethod
    def _enhanced_document(document: str, metadata: Optional[Dict[str, Any]]) -> str:
        if not isinstance(metadata, dict):
            return "content:" + str(document)
        enhanced_document = "content:" + str(document)
        context = metadata.get("context")
        if isinstance(context, str) and context.strip() and context.strip() != "General":
            enhanced_document += f" context: {context.strip()}"
        keywords = metadata.get("keywords")
        if isinstance(keywords, list) and keywords:
            enhanced_document += " keywords: {}".format(", ".join(str(item) for item in keywords))
        tags = metadata.get("tags")
        if isinstance(tags, list) and tags:
            enhanced_document += " tags: {}".format(", ".join(str(item) for item in tags))
        return enhanced_document

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return
        new_embeddings = self._embed_documents(documents)
        self.corpus.extend(list(documents))
        next_offset = len(self.doc_ids)
        self.doc_ids.extend([f"note_{next_offset + idx + 1}" for idx in range(len(documents))])
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings]).astype("float32")

    def add_document(self, document: str, metadata: Optional[Dict[str, Any]] = None, doc_id: str = "") -> None:
        enhanced = self._enhanced_document(document, metadata)
        new_embedding = self._embed_documents([enhanced])
        self.corpus.append(enhanced)
        self.doc_ids.append(str(doc_id or f"note_{len(self.doc_ids) + 1}"))
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = new_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding]).astype("float32")

    def delete_document(self, doc_id: str) -> None:
        if not self.doc_ids:
            return
        target = str(doc_id)
        keep_indices = [idx for idx, existing in enumerate(self.doc_ids) if str(existing) != target]
        if len(keep_indices) == len(self.doc_ids):
            return
        self.corpus = [self.corpus[idx] for idx in keep_indices]
        self.doc_ids = [self.doc_ids[idx] for idx in keep_indices]
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = None
        elif keep_indices:
            self.embeddings = self.embeddings[keep_indices]
        else:
            self.embeddings = None

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus:
            return []
        query_embedding = self._embed_documents([query])
        if query_embedding.size == 0 or self.embeddings is None or self.embeddings.size == 0:
            return []
        scores = self._cosine_scores(query_embedding[0], self.embeddings)
        top_k = min(max(int(k), 0), len(self.corpus))
        if top_k <= 0:
            return []
        indices = np.argsort(scores)[-top_k:][::-1]
        return [int(idx) for idx in indices.tolist()]

    def clone_collection_to_directory(
        self,
        dest_directory: Union[str, Path],
        dest_collection_name: Optional[str] = None,
        overwrite: bool = False,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        del batch_size
        dest_directory = Path(dest_directory)
        if dest_directory.exists() and not overwrite and any(dest_directory.iterdir()):
            raise ValueError(
                f"Destination collection '{dest_collection_name or self.collection_name}' already exists in "
                f"{dest_directory}. Set overwrite=True to replace it."
            )
        dest_directory.mkdir(parents=True, exist_ok=True)
        (dest_directory / "documents.json").write_text(
            json.dumps(list(self.corpus), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (dest_directory / "doc_ids.json").write_text(
            json.dumps(list(self.doc_ids), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.embeddings is not None and self.embeddings.size > 0:
            np.save(dest_directory / "embeddings.npy", self.embeddings)
        (dest_directory / "retriever_meta.json").write_text(
            json.dumps(
                {
                    "embedding_backend": self.embedding_backend,
                    "model_name": self.model_name,
                    "collection_name": dest_collection_name or self.collection_name,
                    "count": len(self.corpus),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "dest_directory": str(dest_directory),
            "collection_name": dest_collection_name or self.collection_name,
            "count": len(self.corpus),
        }

    def _load_from_directory(self, directory: Path) -> None:
        docs_path = directory / "documents.json"
        ids_path = directory / "doc_ids.json"
        embeddings_path = directory / "embeddings.npy"
        if not docs_path.exists():
            return
        payload = json.loads(docs_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Invalid retriever snapshot at {docs_path}: expected a JSON list.")
        self.corpus = [str(doc) for doc in payload]
        if ids_path.exists():
            raw_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            self.doc_ids = [str(doc_id) for doc_id in raw_ids] if isinstance(raw_ids, list) else []
        if len(self.doc_ids) != len(self.corpus):
            self.doc_ids = [f"note_{idx + 1}" for idx in range(len(self.corpus))]
        if embeddings_path.exists():
            self.embeddings = np.asarray(np.load(embeddings_path), dtype="float32")
        elif self.corpus:
            self.embeddings = self._embed_documents(self.corpus)
        else:
            self.embeddings = None
