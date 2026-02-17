#!/usr/bin/env python3
import argparse
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Add HippoRAG and project roots to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(os.path.join(root_dir, "HippoRAG/src"))
sys.path.append(os.path.abspath(os.path.join(root_dir, "..", ".."))) # For generation.rag etc.

# Monkeypatch for torch load safety (needed for HippoRAG dependencies)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

from generation.rag.client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline

# Load environment variables
load_dotenv("generation/HippoRAG2/.env", override=True)

# Azure Compatibility: Map Azure credentials to OpenAI standard env vars
if os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
if os.getenv("AZURE_OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("AZURE_OPENAI_BASE_URL")

def _build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    as_of = checkpoint.get("as_of", {})
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])
    return (
        f"Predict values for provided state keys at checkpoint time {ts}. "
        f"Target keys include: {keys_hint}"
    )

def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    hipporag_dir: Path,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
) -> Dict[str, Any]:
    # 1. Initialize LLM Client for Generation (Unified Client)
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )

    # 2. Initialize HippoRAG for Retrieval
    print(f"Initializing HippoRAG from {hipporag_dir}...")
    if not hipporag_dir.exists():
        raise FileNotFoundError(f"HippoRAG directory not found: {hipporag_dir}")

    # Use a dummy API key if not present, but we loaded .env so it should be fine.
    # HippoRAG requires an OpenAI key even if we only use it for retrieval/indexing?
    # Actually we just use it for retrieval here.
    
    # We assume the directory name contains config info like "gpt-5-mini_text-embedding-3-small"
    # But optimal usage is to pass config explicitly.
    # Let's try to infer or use defaults.
    
    # Configure HippoRAG
    # We use high temperature? No, retrieval doesn't use temperature usually, 
    # but the config expects it.
    
    # Restore legacy behavior: propagate base URLs from environment
    llm_base_url = os.getenv("OPENAI_BASE_URL")
    embedding_base_url = os.getenv("OPENAI_BASE_URL")
    
    config = BaseConfig(
        temperature=0,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
    )
    
    # We must match the embedding model used during strict indexing.
    # Usually it's in the folder name.
    # We will let HippoRAG infer or we hardcode common one? 
    # Better to allow arguments, but for now we follow the structure of 
    # run_hipporag_generation.py which sets working_dir.
    
    # We can use "gpt-5-mini" and "text-embedding-3-small" as reasonable defaults 
    # if not provided, but ideally we match the folder.
    # For now, we initialize with defaults and override working_dir which is the critical part.
    # hipporag_dir is like ".../outputs/user_expr/gpt-5-mini_text-embedding-3-small"
    # save_dir should be ".../outputs/user_expr"
    real_save_dir = hipporag_dir.parent
    
    hipporag = HippoRAG(
        global_config=config,
        save_dir=str(real_save_dir), 
        llm_model_name=llm_model,
        embedding_model_name=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large"), 
    )
    # We still enforce working_dir to be exactly what was passed, just in case
    hipporag.working_dir = str(hipporag_dir.resolve())
    
    # Re-initialize graph and stores
    print("Loading HippoRAG graph and embeddings...")
    hipporag.graph = hipporag.initialize_graph()
    
    from hipporag.embedding_store import EmbeddingStore
    # We need to re-init these because they point to save_dir by default
    # And we want them to point to working_dir
    hipporag.chunk_embedding_store = EmbeddingStore(
        hipporag.embedding_model,
        os.path.join(hipporag.working_dir, "chunk_embeddings"),
        hipporag.global_config.embedding_batch_size, 
        'chunk'
    )
    hipporag.entity_embedding_store = EmbeddingStore(
        hipporag.embedding_model,
        os.path.join(hipporag.working_dir, "entity_embeddings"),
        hipporag.global_config.embedding_batch_size, 
        'entity'
    )
    hipporag.fact_embedding_store = EmbeddingStore(
        hipporag.embedding_model,
        os.path.join(hipporag.working_dir, "fact_embeddings"),
        hipporag.global_config.embedding_batch_size, 
        'fact'
    )
    
    # Map app_log_ids for quick lookup
    # We need to know which logs are available in the memory pool.
    # HippoRAG retrieves "docs". We need to map docs back to app_logs.
    # Usually doc_node_id corresponds to 0, 1, 2... (index in original corpus).
    # The app_logs_path file typically contains the logs in order.
    # We need to load them to map index -> app_log_id.
    
    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(all_logs_payload, dict):
        all_logs = all_logs_payload.get("app_logs", [])
    elif isinstance(all_logs_payload, list):
        all_logs = all_logs_payload
    else:
        all_logs = []
    all_logs = [x for x in all_logs if isinstance(x, dict)]
    
    # Create mapping: index -> app_log (or app_log_id)
    # HippoRAG's doc indices usually match the input list order.
    # We assume the index was built from the same app_logs.
    
    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()
        
        # 1. Build Query
        query = _build_retrieval_query(cp, target_keys)
        checkpoint_id = cp.get("checkpoint_id", "unknown")
        
        print(f"\n[DEBUG] CP: {checkpoint_id} | Keys: {len(target_keys)} | Query: {query}")
        
        # 2. Retrieve using HippoRAG
        retrieval_start = time.time()
        solutions = hipporag.retrieve(queries=[query])
        retrieval_end = time.time()
        
        retrieval_duration = retrieval_end - retrieval_start
        print(f"[DEBUG] CP: {checkpoint_id} | HippoRAG retrieval took {retrieval_duration:.2f}s")

        if not solutions:
            print(f"[DEBUG] CP: {checkpoint_id} | NO solutions returned from HippoRAG")
            return {
                "context_logs": [],
                "context_note": "HippoRAG returned no solutions",
                "retrieval_query": query,
                "metadata": {"retrieval_mode": "hipporag", "retrieval_duration_s": retrieval_duration},
            }
            
        sol = solutions[0]
        
        # Parse retrieved docs directly
        retrieved_logs_candidates = []
        if hasattr(sol, 'docs'):
            print(f"[DEBUG] CP: {checkpoint_id} | Raw docs retrieved: {len(sol.docs)}")
            for doc_str in sol.docs:
                try:
                    log_obj = json.loads(doc_str)
                    retrieved_logs_candidates.append(log_obj)
                except json.JSONDecodeError:
                    pass
        
        # No more filtering by memory_pool (OLD LOGIC REMOVED)
        # We assume the index provided to HippoRAG is correct/valid for this run.
        
        selected_logs = retrieved_logs_candidates[:5] # Limit to top-5, but from whatever RAG returned
        
        total_duration = time.time() - start_time
        print(f"[DEBUG] CP: {checkpoint_id} | Selected {len(selected_logs)} logs (Raw Top-5)")
        if selected_logs:
            log_ids_str = ", ".join([str(x.get("app_log_id")) for x in selected_logs])
            print(f"[DEBUG] CP: {checkpoint_id} | Selected IDs: {log_ids_str}")
        print(f"[DEBUG] CP: {checkpoint_id} | Total retrieval took {total_duration:.2f}s\n")
        
        return {
            "context_logs": selected_logs,
            "context_note": f"HippoRAG retrieved (top-{len(selected_logs)} raw candidates)",
            "retrieval_query": query,
            "metadata": {
                "hipporag_retrieved_raw_count": len(retrieved_logs_candidates),
                "retrieved_app_log_ids": [x.get("app_log_id") for x in selected_logs],
                "retrieval_duration_s": retrieval_duration,
                "total_retrieval_context_duration_s": total_duration,
            },
        }

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
        baseline_name="hipporag",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="HippoRAG baseline generation for dynamic state prediction.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to dynamic_state_prediction_benchmark.json")
    parser.add_argument("--app-logs-path", type=Path, required=True, help="Path to raw app logs.")
    parser.add_argument("--output", type=Path, required=True, help="Output path for results.")
    parser.add_argument("--hipporag-dir", type=Path, required=True, help="Path to existing HippoRAG output directory (containing graph/embeddings).")
    
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    
    args = parser.parse_args()

    run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        hipporag_dir=args.hipporag_dir,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )

if __name__ == "__main__":
    main()
