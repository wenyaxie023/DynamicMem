import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def log_to_text(log: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(log.get("app_name", "")),
            str(log.get("api_name", "")),
            json.dumps(log.get("request", {}), ensure_ascii=False),
            json.dumps(log.get("response", {}), ensure_ascii=False),
        ]
    )


def log_to_doc(log: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(log.get("app_log_id", "")).strip(),
        "text": log_to_text(log),
        "payload": log,
    }


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _score_overlap(query: str, text: str) -> float:
    q = Counter(_tokenize(query))
    t = Counter(_tokenize(text))
    if not q:
        return 0.0
    overlap = sum(min(v, t.get(k, 0)) for k, v in q.items())
    norm = sum(q.values())
    return overlap / max(norm, 1)


@dataclass
class RetrievedDoc:
    id: str
    text: str
    payload: Dict[str, Any]
    score: float


class LettaRetriever:
    """
    Letta retriever adapter.

    Notes:
    - `mode=sdk`: tries to use installed Letta Python SDK.
    - if SDK is unavailable or API shape is different, optionally falls back to local lexical retrieval.
    """

    def __init__(
        self,
        *,
        namespace: str,
        mode: str = "sdk",
        allow_local_fallback: bool = True,
    ):
        self.namespace = namespace
        self.mode = mode
        self.allow_local_fallback = allow_local_fallback
        self._docs: List[Dict[str, Any]] = []
        self._doc_by_id: Dict[str, Dict[str, Any]] = {}
        self._sdk_client = None
        self._sdk_ready = False

    def index_logs(self, logs: List[Dict[str, Any]]) -> None:
        self._docs = []
        self._doc_by_id = {}
        for log in logs:
            if not isinstance(log, dict):
                continue
            doc = log_to_doc(log)
            if not doc["id"]:
                continue
            self._docs.append(doc)
            self._doc_by_id[doc["id"]] = doc

        if self.mode == "sdk":
            self._setup_sdk()
            if self._sdk_ready:
                self._index_docs_to_sdk(self._docs)
            elif not self.allow_local_fallback:
                raise RuntimeError(
                    "Letta SDK initialization failed and local fallback is disabled. "
                    "Install/configure Letta SDK or enable fallback."
                )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_ids: Optional[set[str]] = None,
    ) -> List[RetrievedDoc]:
        if top_k <= 0:
            return []

        if self._sdk_ready:
            try:
                return self._sdk_retrieve(query=query, top_k=top_k, allowed_ids=allowed_ids)
            except Exception:
                if not self.allow_local_fallback:
                    raise
        return self._local_retrieve(query=query, top_k=top_k, allowed_ids=allowed_ids)

    def _setup_sdk(self) -> None:
        candidates: List[Tuple[str, str]] = [
            ("letta_client", "Letta"),
            ("letta_client", "Client"),
            ("letta", "Letta"),
            ("letta", "Client"),
        ]
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name, None)
                if cls is None:
                    continue
                self._sdk_client = self._instantiate_sdk_client(cls)
                self._sdk_ready = self._sdk_client is not None
                if self._sdk_ready:
                    return
            except Exception:
                continue
        self._sdk_ready = False

    def _instantiate_sdk_client(self, cls: Any) -> Any:
        base_url = os.getenv("LETTA_BASE_URL")
        api_key = os.getenv("LETTA_API_KEY")
        kwargs = {}
        if api_key:
            for key in ("api_key", "token", "bearer_token"):
                kwargs[key] = api_key
                try:
                    return cls(**kwargs)
                except Exception:
                    kwargs.pop(key, None)
        if base_url:
            for key in ("base_url", "url", "endpoint"):
                kwargs[key] = base_url
                try:
                    return cls(**kwargs)
                except Exception:
                    kwargs.pop(key, None)
        try:
            return cls()
        except Exception:
            return None

    def _index_docs_to_sdk(self, docs: List[Dict[str, Any]]) -> None:
        """
        Best-effort indexing against different possible SDK method names.
        If no known method exists, we keep `_sdk_ready=False` and local fallback is used.
        """
        if self._sdk_client is None:
            self._sdk_ready = False
            return

        try:
            if hasattr(self._sdk_client, "upsert_memory"):
                for doc in docs:
                    self._sdk_client.upsert_memory(
                        namespace=self.namespace,
                        memory_id=doc["id"],
                        text=doc["text"],
                        metadata={"app_log_id": doc["id"]},
                    )
                return
            if hasattr(self._sdk_client, "add_memory"):
                for doc in docs:
                    self._sdk_client.add_memory(
                        text=doc["text"],
                        metadata={"namespace": self.namespace, "app_log_id": doc["id"]},
                    )
                return
            # Unknown SDK shape; switch to local.
            self._sdk_ready = False
        except Exception:
            self._sdk_ready = False

    def _sdk_retrieve(
        self,
        *,
        query: str,
        top_k: int,
        allowed_ids: Optional[set[str]],
    ) -> List[RetrievedDoc]:
        if self._sdk_client is None:
            return []

        raw_items = None
        if hasattr(self._sdk_client, "search"):
            raw_items = self._sdk_client.search(query=query, top_k=top_k, namespace=self.namespace)
        elif hasattr(self._sdk_client, "search_memory"):
            raw_items = self._sdk_client.search_memory(query=query, top_k=top_k, namespace=self.namespace)
        else:
            return self._local_retrieve(query=query, top_k=top_k, allowed_ids=allowed_ids)

        docs: List[RetrievedDoc] = []
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("results", [])
        if not isinstance(raw_items, list):
            raw_items = []

        for item in raw_items:
            doc_id = ""
            score = 0.0
            if isinstance(item, dict):
                doc_id = str(item.get("memory_id") or item.get("id") or "").strip()
                try:
                    score = float(item.get("score", 0.0))
                except Exception:
                    score = 0.0
            if not doc_id:
                continue
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            doc = self._doc_by_id.get(doc_id)
            if not doc:
                continue
            docs.append(
                RetrievedDoc(
                    id=doc["id"],
                    text=doc["text"],
                    payload=doc["payload"],
                    score=score,
                )
            )
            if len(docs) >= top_k:
                break
        return docs

    def _local_retrieve(
        self,
        *,
        query: str,
        top_k: int,
        allowed_ids: Optional[set[str]],
    ) -> List[RetrievedDoc]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in self._docs:
            doc_id = doc["id"]
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            score = _score_overlap(query, doc["text"])
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[RetrievedDoc] = []
        for score, doc in scored[:top_k]:
            out.append(
                RetrievedDoc(
                    id=doc["id"],
                    text=doc["text"],
                    payload=doc["payload"],
                    score=score,
                )
            )
        return out

