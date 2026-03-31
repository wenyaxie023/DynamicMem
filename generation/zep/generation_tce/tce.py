#!/usr/bin/env python3
"""Zep (Graphiti) TCE Baseline - Temporal knowledge graph with serial indexing."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from functools import wraps

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "..")))

# Use local Graphiti clone for easier debugging and modification
graphiti_dir = os.path.abspath(os.path.join(script_dir, "..", "graphiti"))
sys.path.insert(0, graphiti_dir)

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
from graphiti_core.utils.bulk_utils import RawEpisode
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from datetime import datetime, timezone

from generation.common.provider_config import setup_provider_env
from tce_core.pipeline import run_pipeline

REPO_ROOT_DIR = Path(__file__).resolve().parents[3]


def async_retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    rate_limit_delay: float = 60.0,
):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay cap
        rate_limit_delay: Special delay for rate limit errors (429)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except RateLimitError as e:
                    # Rate limit - use special longer delay
                    delay = rate_limit_delay
                    print(f"[Zep] Rate limit hit (429). Waiting {delay}s before retry...")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
                except APIStatusError as e:
                    # Other API errors - exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"[Zep] API error (status {e.status_code}). Waiting {delay}s before retry...")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
                except Exception as e:
                    # Generic errors - exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"[Zep] Error: {type(e).__name__}: {e}. Waiting {delay}s before retry...")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise
            return None
        return wrapper
    return decorator


def rate_limited_sleep(min_requests_per_minute: int = 20):
    """
    Sleep to ensure we don't exceed rate limit.
    Azure free tier: 24 requests/minute -> use 20 to be safe.
    """
    delay = 60.0 / min_requests_per_minute  # ~3 seconds per request
    return delay


@dataclass
class GraphitiConfig:
    db_path: str
    embedding_model: str
    llm_model: str
    batch_size: int
    max_coroutines: int
    max_tokens: int = 16384
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class ZepGraphitiIndex:
    """Zep index wrapper using Graphiti for temporal knowledge graph management."""
    
    def __init__(self, config: GraphitiConfig):
        self.config = config
        self.graphiti: Optional[Graphiti] = None
        self.indexed_log_ids: Set[str] = set()
        
    async def initialize(self):
        from dotenv import load_dotenv
        
        # Force reload root .env to override any existing env vars (e.g., from shell)
        root_env_path = REPO_ROOT_DIR / ".env"
        if root_env_path.exists():
            load_dotenv(root_env_path, override=True)
        
        # Prefer explicitly passed config, then env vars
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("OpenAI credentials not found")
        
        print(f"[Zep] Initializing Graphiti with Kuzu database: {self.config.db_path}")
        print(f"[Zep] Using OpenAI endpoint: {base_url}")
        print(f"[Zep] LLM model: {self.config.llm_model}")
        print(f"[Zep] Embedding model: {self.config.embedding_model}")
        print(f"[Zep] Max coroutines: {self.config.max_coroutines}")
        
        openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        embedder_config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=self.config.embedding_model,
            base_url=base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config, client=openai_client)
        
        llm_config = LLMConfig(
            api_key=api_key,
            model=self.config.llm_model,
            base_url=base_url,
        )
        llm_client = OpenAIGenericClient(
            config=llm_config,
            client=openai_client,
            max_tokens=self.config.max_tokens,
        )
        
        cross_encoder = OpenAIRerankerClient(
            config=llm_config,
            client=openai_client,
        )
        
        graph_driver = KuzuDriver(db=self.config.db_path)
        
        self.graphiti = Graphiti(
            uri=None,
            user=None,
            password=None,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=True,
            graph_driver=graph_driver,
            max_coroutines=self.config.max_coroutines,
        )
        
        # Build FTS indices - KuzuDriver.build_indices_and_constraints() is a no-op,
        # so we need to execute FTS creation queries directly
        from graphiti_core.graph_queries import get_fulltext_indices
        from graphiti_core.driver.driver import GraphProvider
        
        fts_queries = get_fulltext_indices(GraphProvider.KUZU)
        for q in fts_queries:
            try:
                await graph_driver.execute_query(q)
            except Exception as e:
                if "already exists" in str(e):
                    pass
                else:
                    raise
        print("[Zep] FTS indices built successfully")
        
    async def close(self):
        if self.graphiti:
            await self.graphiti.close()
            print("[Zep] Graphiti connection closed")
    
    async def add_episodes_bulk(self, episodes: List[Dict[str, Any]], batch_size: int = 10):
        if not self.graphiti:
            raise RuntimeError("Graphiti not initialized")
        
        print(f"[Zep] Adding {len(episodes)} episodes in batches of {batch_size}")
        total_batches = (len(episodes) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(episodes), batch_size)):
            batch = episodes[i:i + batch_size]
            raw_episodes = []
            
            for j, episode in enumerate(batch):
                episode_id = str(episode.get("app_log_id", f"ep_{i}_{j}"))
                episode_content = json.dumps(episode, ensure_ascii=False)
                
                timestamp = episode.get("timestamp", "")
                reference_time = datetime.now(timezone.utc)
                if timestamp:
                    try:
                        reference_time = _parse_timestamp(timestamp)
                    except ValueError:
                        pass
                
                raw_episodes.append(RawEpisode(
                    name=episode_id,
                    content=episode_content,
                    source_description="App log entry",
                    source=EpisodeType.json,
                    reference_time=reference_time,
                ))
            
            if raw_episodes:
                batch_start = time.time()
                try:
                    await self._add_batch_with_retry(raw_episodes)
                    for ep in raw_episodes:
                        self.indexed_log_ids.add(ep.name)
                    batch_duration = time.time() - batch_start
                    print(f"[Zep] Batch {batch_idx + 1}/{total_batches} indexed ({len(raw_episodes)} episodes) in {batch_duration:.1f}s")
                except Exception as e:
                    print(f"[Zep] Failed to index batch {batch_idx + 1}: {e}")
                    raise
                
                if batch_idx < total_batches - 1:
                    await asyncio.sleep(rate_limited_sleep(min_requests_per_minute=15))

    @async_retry_with_backoff(max_attempts=5, base_delay=10.0, max_delay=120.0, rate_limit_delay=60.0)
    async def _add_batch_with_retry(self, raw_episodes: List[RawEpisode]):
        await self.graphiti.add_episode_bulk(raw_episodes)
    
    @async_retry_with_backoff(max_attempts=5, base_delay=10.0, max_delay=120.0, rate_limit_delay=60.0)
    async def search(self, query: str, top_k: int = 5, checkpoint_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.graphiti:
            raise RuntimeError("Graphiti not initialized")
        
        # Option A: Removed temporal filter entirely
        # Rationale: checkpoint indexing order already ensures no future leakage
        # The original filter used ComparisonOperator.greater_than which retrieved
        # logs AFTER checkpoint (wrong semantics). Fixed by removing filter.
        search_filters = None
        
        if search_filters:
            results = await self.graphiti.search_(
                query=query,
                search_filter=search_filters,
            )
        else:
            results = await self.graphiti.search_(query=query)
        
        retrieved_episodes = []
        
        if hasattr(results, 'episodes') and results.episodes:
            for ep in results.episodes[:top_k]:
                try:
                    episode_data = json.loads(ep.content)
                    retrieved_episodes.append(episode_data)
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        if hasattr(results, 'edges') and results.edges:
            seen_episode_ids = set()
            for edge in results.edges[:top_k]:
                if hasattr(edge, 'episodes') and edge.episodes:
                    for ep_uuid in edge.episodes:
                        if ep_uuid not in seen_episode_ids:
                            seen_episode_ids.add(ep_uuid)
                            if hasattr(edge, 'source_episode'):
                                try:
                                    episode_data = json.loads(edge.source_episode)
                                    retrieved_episodes.append(episode_data)
                                except (json.JSONDecodeError, AttributeError):
                                    continue
        
        return retrieved_episodes


def _parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {ts}") from exc


def _build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    as_of = checkpoint.get("as_of", {})
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])
    
    return (
        f"Predict values for provided state keys at checkpoint time {ts}. "
        f"Target keys include: {keys_hint}"
    )


async def _retrieve_context_async(
    checkpoint: Dict[str, Any],
    memory_pool: List[Dict[str, Any]],
    target_keys: List[str],
    index: ZepGraphitiIndex,
    log_id_to_content: Dict[str, Dict[str, Any]],
    batch_size: int = 10,
    task_text: Optional[str] = None,
    retrieval_top_k_override: Optional[int] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    checkpoint_id = checkpoint.get("checkpoint_id", "unknown")
    
    new_logs_to_index = []
    for log in memory_pool:
        lid = str(log.get("app_log_id"))
        if lid not in index.indexed_log_ids:
            new_logs_to_index.append(log)
    
    if new_logs_to_index:
        print(f"[ONLINE] Indexing {len(new_logs_to_index)} new logs for CP {checkpoint_id}...")
        t_idx_start = time.time()
        await index.add_episodes_bulk(new_logs_to_index, batch_size=batch_size)
        t_idx_end = time.time()
        print(f"[ONLINE] Indexed {len(new_logs_to_index)} logs in {t_idx_end - t_idx_start:.2f}s")
    
    query = task_text or _build_retrieval_query(checkpoint, target_keys)
    
    as_of = checkpoint.get("as_of", {})
    checkpoint_ts = as_of.get("timestamp", "")
    
    top_k = retrieval_top_k_override if retrieval_top_k_override is not None else 5
    
    print(f"\n[DEBUG] CP: {checkpoint_id} | Keys: {len(target_keys)} | Query: {query[:100]}...")
    print(f"[DEBUG] CP: {checkpoint_id} | Temporal filter: {checkpoint_ts}")
    
    retrieval_start = time.time()
    retrieved_logs = await index.search(query, top_k=top_k, checkpoint_timestamp=checkpoint_ts)
    retrieval_end = time.time()
    
    retrieval_duration = retrieval_end - retrieval_start
    print(f"[DEBUG] CP: {checkpoint_id} | Graphiti retrieval took {retrieval_duration:.2f}s")
    
    pool_ids = {str(log.get("app_log_id")) for log in memory_pool}
    
    selected_logs = []
    for log in retrieved_logs:
        log_id = str(log.get("app_log_id"))
        if log_id in pool_ids:
            selected_logs.append(log)
    
    total_duration = time.time() - start_time
    print(f"[DEBUG] CP: {checkpoint_id} | Selected {len(selected_logs)} logs from retrieval")
    if selected_logs:
        log_ids_str = ", ".join([str(x.get("app_log_id")) for x in selected_logs])
        print(f"[DEBUG] CP: {checkpoint_id} | Selected IDs: {log_ids_str}")
    print(f"[DEBUG] CP: {checkpoint_id} | Total retrieval took {total_duration:.2f}s\n")
    
    return {
        "context_logs": selected_logs,
        "context_note": f"Graphiti retrieved (top-{len(selected_logs)} from memory pool)",
        "retrieval_query": query,
        "metadata": {
            "graphiti_indexed_count": len(index.indexed_log_ids),
            "retrieved_app_log_ids": [x.get("app_log_id") for x in selected_logs],
            "retrieval_duration_s": retrieval_duration,
            "total_retrieval_context_duration_s": total_duration,
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
    embedding_model: str,  # 必须传入
    batch_size: int = 10,
    max_coroutines: int = 5,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
) -> Dict[str, Any]:
    import nest_asyncio
    nest_asyncio.apply()
    
    # Force reload root .env to override any existing env vars (e.g., from shell)
    from dotenv import load_dotenv
    root_env_path = REPO_ROOT_DIR / ".env"
    if root_env_path.exists():
        load_dotenv(root_env_path, override=True)
    
    _, resolved_base_url = setup_provider_env(llm_provider, repo_root=REPO_ROOT_DIR)
    
    from generation.rag.client import LLMClient
    
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )
    
    db_dir = output_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "zep_kuzu.db")
    
    if not resume and os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)
        print(f"[Zep] Cleaned existing database: {db_path}")
    
    config = GraphitiConfig(
        db_path=db_path,
        embedding_model=embedding_model,
        llm_model=llm_model,
        batch_size=batch_size,
        max_coroutines=max_coroutines,
        base_url=resolved_base_url,
    )
    
    index = ZepGraphitiIndex(config)
    
    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(all_logs_payload, dict):
        all_logs = all_logs_payload.get("app_logs", [])
    elif isinstance(all_logs_payload, list):
        all_logs = all_logs_payload
    else:
        all_logs = []
    all_logs = [x for x in all_logs if isinstance(x, dict)]
    
    log_id_to_content = {}
    for log in all_logs:
        lid = str(log.get("app_log_id"))
        if lid:
            log_id_to_content[lid] = log
    
    print(f"[Zep] Loaded {len(all_logs)} app logs for content lookup")
    
    print("[Zep] Initializing Graphiti index...")
    asyncio.get_event_loop().run_until_complete(index.initialize())
    print("[Zep] Graphiti index initialized")
    
    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")
    
    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)
    
    def close() -> None:
        client.close()
        try:
            asyncio.get_event_loop().run_until_complete(index.close())
        except Exception as e:
            print(f"[Zep] Warning: Error closing Graphiti: {e}")
    
    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
        task_text: Optional[str] = None,
        retrieval_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(
            _retrieve_context_async(
                checkpoint=cp,
                memory_pool=memory_pool,
                target_keys=target_keys,
                index=index,
                log_id_to_content=log_id_to_content,
                batch_size=batch_size,
                task_text=task_text,
                retrieval_top_k_override=retrieval_top_k_override,
            )
        )
    
    print(f"[Zep] Starting TCE pipeline with {max_checkpoints or 'all'} checkpoints...")
    
    result = run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="zep",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
    )
    
    print("[Zep] TCE pipeline completed")
    return result


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Zep (Graphiti) baseline generation for TCE.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--embedding-model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-coroutines", type=int, default=5)
    
    args = parser.parse_args()
    
    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        max_coroutines=args.max_coroutines,
    )


if __name__ == "__main__":
    main()
