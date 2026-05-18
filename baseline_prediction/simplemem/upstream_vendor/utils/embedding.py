"""Embedding wrapper aligned to shared TCE retriever config."""

from typing import List, Optional

import numpy as np

from . import usage
from .. import config


class EmbeddingModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        batch_size: Optional[int] = None,
        tracker: Optional[usage.UsageTracker] = None,
        dimension: Optional[int] = None,
        use_optimization: bool = True,
    ):
        del use_optimization
        self.provider = str(provider or config.EMBEDDING_PROVIDER)
        self.model_name = str(model_name or config.EMBEDDING_MODEL)
        self.batch_size = int(batch_size or config.RETRIEVER_BATCH_SIZE or 64)
        self.dimension = int(
            dimension
            or config.EMBEDDING_DIM
            or (3072 if "large" in self.model_name.lower() else 1536)
        )
        self.model_type = "openai_compatible_embedding"
        self.supports_query_prompt = False
        self.tracker = tracker
        self._client = usage.EmbeddingClient(
            provider=self.provider,
            model=self.model_name,
            batch_size=self.batch_size,
            tracker=self.tracker,
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        del is_query
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(self._client.encode_documents(list(texts)), dtype=np.float32)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        del is_query
        return np.asarray(self._client.encode_single(str(text or "")), dtype=np.float32)

    def encode_query(self, queries: List[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)
