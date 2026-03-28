#!/usr/bin/env python3
"""SimpleMem TCE Baseline - Hybrid Vector (LanceDB) + BM25 lexical retrieval.

Incremental indexing: each log is indexed one by one when it becomes visible.
"""

import json
import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "..")))

# Load .env from SimpleMem directory and repo root
from dotenv import load_dotenv
simplemem_root = Path(os.path.abspath(os.path.join(script_dir, "..")))
env_path = simplemem_root / ".env"
load_dotenv(override=True)  # Load from repo root first
load_dotenv(env_path, override=True)  # Then load from SimpleMem directory with override

import lancedb
from rank_bm25 import BM25Okapi
from openai import OpenAI

from generation.common.provider_config import setup_provider_env
from tce_core.pipeline import run_pipeline
from tce_core.retrieval_query import build_retrieval_query

REPO_ROOT_DIR = Path(__file__).resolve().parents[3]


@dataclass
class SimpleMemConfig:
    db_path: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    batch_size: int = 64


class SimpleMemIndex:
    def __init__(self, config: SimpleMemConfig):
        self.config = config
        self.db: Optional[lancedb.DBConnection] = None
        self.table: Optional[lancedb.table.Table] = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self.indexed_log_ids: Set[str] = set()
        self.docid_to_log: Dict[int, Dict[str, Any]] = {}
        self._client: Optional[OpenAI] = None
        self._query_cache: Dict[str, List[float]] = {}
        
    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            
            self._client = OpenAI(**client_kwargs)
        return self._client
    
    def _log_to_text(self, log: Dict[str, Any]) -> str:
        return " ".join([
            str(log.get("app_name", "")),
            str(log.get("api_name", "")),
            json.dumps(log.get("request", {}), ensure_ascii=False),
            json.dumps(log.get("response", {}), ensure_ascii=False),
        ])
    
    def _embed_single(self, text: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=[text],
        )
        return response.data[0].embedding

    def initialize(self):
        print(f"[SimpleMem] Initializing LanceDB at: {self.config.db_path}")
        
        if os.path.exists(self.config.db_path):
            shutil.rmtree(self.config.db_path, ignore_errors=True)
        
        self.db = lancedb.connect(self.config.db_path)
        
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("log_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
            pa.field("log_data", pa.string()),
        ])
        self.table = self.db.create_table("logs", schema=schema, mode="overwrite")
        
        self.corpus_tokens = []
        self.bm25 = None
        self.indexed_log_ids = set()
        self.docid_to_log = {}
        
        print("[SimpleMem] Database initialized")

    def load_from_existing(self) -> bool:
        if not os.path.exists(self.config.db_path):
            print(f"[SimpleMem] No existing database at {self.config.db_path}")
            return False
        
        print(f"[SimpleMem] Loading existing database from {self.config.db_path}")
        
        try:
            self.db = lancedb.connect(self.config.db_path)
            
            try:
                self.table = self.db.open_table("logs")
            except Exception as e:
                print(f"[SimpleMem] Failed to open table: {e}")
                return False
            
            rows = self.table.search().limit(None).to_list()
            
            self.indexed_log_ids = set()
            self.corpus_tokens = []
            self.docid_to_log = {}
            
            for i, row in enumerate(rows):
                log_id = str(row.get("log_id", ""))
                if log_id:
                    self.indexed_log_ids.add(log_id)
                
                text = row.get("text", "")
                self.corpus_tokens.append(text.split())
                
                try:
                    log_data = json.loads(row.get("log_data", "{}"))
                    self.docid_to_log[i] = log_data
                except:
                    pass
            
            if self.corpus_tokens:
                self.bm25 = BM25Okapi(self.corpus_tokens)
            
            print(f"[SimpleMem] Loaded {len(self.indexed_log_ids)} indexed logs from existing database")
            return True
            
        except Exception as e:
            print(f"[SimpleMem] Failed to load existing database: {e}")
            return False

    def index_log(self, log: Dict[str, Any]) -> bool:
        """Index a single log. Returns True if indexed, False if already exists."""
        log_id = str(log.get("app_log_id", ""))
        
        if log_id in self.indexed_log_ids:
            return False
        
        if self.table is None:
            raise RuntimeError("Database not initialized")
        
        text = self._log_to_text(log)
        
        print(f"[SimpleMem] Indexing log {log_id}...")
        
        embedding = self._embed_single(text)
        
        doc_id = len(self.indexed_log_ids)
        self.docid_to_log[doc_id] = log
        self.indexed_log_ids.add(log_id)
        
        tokens = text.split()
        self.corpus_tokens.append(tokens)
        
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        
        self.table.add([{
            "id": f"doc_{doc_id}",
            "text": text,
            "log_id": log_id,
            "vector": embedding,
            "log_data": json.dumps(log, ensure_ascii=False),
        }])
        
        return True

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.table is None or len(self.indexed_log_ids) == 0:
            return []
        
        if query in self._query_cache:
            query_embedding = self._query_cache[query]
        else:
            query_embedding = self._embed_single(query)
            self._query_cache[query] = query_embedding
        
        vector_results = self.table.search(query_embedding).limit(top_k * 2).to_list()
        
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens) if self.bm25 else np.array([])
        
        if len(bm25_scores) > 0:
            bm25_top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k * 2]
        else:
            bm25_top_indices = []
        
        vector_ranks: Dict[str, int] = {}
        for rank, result in enumerate(vector_results, 1):
            doc_id = result["id"]
            vector_ranks[doc_id] = rank
        
        bm25_ranks: Dict[str, int] = {}
        for rank, doc_idx in enumerate(bm25_top_indices, 1):
            doc_id = f"doc_{doc_idx}"
            bm25_ranks[doc_id] = rank
        
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: Dict[str, float] = {}
        
        k = 60
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 0.5 / (k + vector_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += 0.5 / (k + bm25_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            row = self.table.search().where(f"id = '{doc_id}'").limit(1).to_list()
            if row:
                log_data = json.loads(row[0]["log_data"])
                results.append({**log_data, "retrieval_score": score})
        
        return results


def _retrieve_context(
    checkpoint: Dict[str, Any],
    memory_pool: List[Dict[str, Any]],
    target_keys: List[str],
    index: SimpleMemIndex,
    retrieval_top_k: int = 10,
    task_text: Optional[str] = None,
    retrieval_top_k_override: Optional[int] = None,
) -> Dict[str, Any]:
    checkpoint_id = checkpoint.get("checkpoint_id", "unknown")
    
    new_count = 0
    for log in memory_pool:
        log_id = str(log.get("app_log_id"))
        if log_id not in index.indexed_log_ids:
            if index.index_log(log):
                new_count += 1
    
    if new_count > 0:
        print(f"[SimpleMem] CP {checkpoint_id}: indexed {new_count} new logs, total: {len(index.indexed_log_ids)}")
    
    query = task_text or build_retrieval_query(checkpoint, target_keys)
    top_k = retrieval_top_k_override if retrieval_top_k_override is not None else retrieval_top_k
    
    retrieved = index.hybrid_search(query, top_k=top_k)
    
    pool_ids = {str(log.get("app_log_id")) for log in memory_pool}
    selected = [log for log in retrieved if str(log.get("app_log_id")) in pool_ids][:top_k]
    
    return {
        "context_logs": selected,
        "context_note": f"SimpleMem hybrid retrieved {len(selected)} logs",
        "retrieval_query": query,
        "metadata": {
            "simplemem_indexed_count": len(index.indexed_log_ids),
            "new_indexed_count": new_count,
            "retrieved_app_log_ids": [log.get("app_log_id") for log in selected],
        },
    }


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    retrieval_top_k: int = 10,
    embedding_model: str = "text-embedding-3-large",
    embedding_dim: Optional[int] = None,
    batch_size: int = 64,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
) -> Dict[str, Any]:
    setup_provider_env(llm_provider, repo_root=REPO_ROOT_DIR)
    
    from generation.rag.client import LLMClient
    
    print(f"[SimpleMem] === Configuration ===")
    print(f"[SimpleMem] LLM Provider: {llm_provider}")
    print(f"[SimpleMem] LLM Model: {llm_model}")
    print(f"[SimpleMem] Embedding Model: {embedding_model}")
    print(f"[SimpleMem] Retrieval Top-K: {retrieval_top_k}")
    print(f"[SimpleMem] Resume: {resume}")
    print(f"[SimpleMem] =======================")
    
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )
    
    db_dir = output_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "simplemem_db")
    
    if embedding_dim is None:
        embedding_dim = 3072 if "large" in embedding_model else 1536
        print(f"[SimpleMem] Auto-detected embedding dimension: {embedding_dim}")
    
    config = SimpleMemConfig(
        db_path=db_path,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )
    
    index = SimpleMemIndex(config)
    
    if resume:
        loaded = index.load_from_existing()
        if not loaded:
            print("[SimpleMem] Resume failed, initializing new database")
            index.initialize()
    else:
        index.initialize()
    
    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    all_logs = all_logs_payload.get("app_logs", all_logs_payload) if isinstance(all_logs_payload, dict) else all_logs_payload
    all_logs = [x for x in all_logs if isinstance(x, dict)]
    
    print(f"[SimpleMem] Loaded {len(all_logs)} app logs")
    
    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")
    
    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)
    
    def close() -> None:
        client.close()
    
    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
        task_text: Optional[str] = None,
        retrieval_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _retrieve_context(
            cp, memory_pool, target_keys, index, retrieval_top_k,
            task_text=task_text,
            retrieval_top_k_override=retrieval_top_k_override,
        )
    
    print(f"[SimpleMem] Starting TCE pipeline")
    
    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="simplemem",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        enable_rq3_apply_service_qa=True,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
    )


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="SimpleMem baseline for TCE")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    
    args = parser.parse_args()
    
    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=None,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=None,
        save_prompt_and_raw=args.save_prompt_and_raw,
        retrieval_top_k=args.retrieval_top_k,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()