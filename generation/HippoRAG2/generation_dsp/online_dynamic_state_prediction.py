#!/usr/bin/env python3
import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add HippoRAG and project roots to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(os.path.join(root_dir, "HippoRAG/src"))
sys.path.append(os.path.abspath(os.path.join(root_dir, "..", "..")))

# Monkeypatch for torch load safety
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import compute_mdhash_id

from generation.rag.client import LLMClient
from dynamic_state_prediction_core.pipeline import run_pipeline, observed_logs_for_checkpoint

# Load environment variables
load_dotenv(os.path.join(root_dir, ".env"), override=True)

# Azure Compatibility
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

class OnlineDSPRunner:
    def __init__(self, hipporag: HippoRAG, all_logs: List[Dict[str, Any]], batch_size: int):
        self.hipporag = hipporag
        self.all_logs = all_logs
        self.batch_size = batch_size
        self.last_indexed_idx = -1
        self.last_prefetched_idx = -1

    def prefetch_openie(self, start_idx: int, end_idx: int):
        """
        Runs OpenIE on a range of logs and saves results to disk WITHOUT adding them to the graph yet.
        """
        target_logs = self.all_logs[start_idx : end_idx + 1]
        if not target_logs:
            return
            
        print(f"[PREFETCH] Running OpenIE for logs {start_idx} to {end_idx} (Count: {len(target_logs)})...")
        t0 = time.time()
        
        # Format logs
        docs = [json.dumps(log, ensure_ascii=False) for log in target_logs]
        
        # 1. Chunk Embedding (needed for keys)
        self.hipporag.chunk_embedding_store.insert_strings(docs)
        new_chunk_keys = [compute_mdhash_id(doc, prefix='chunk-') for doc in docs]
        chunk_to_rows = self.hipporag.chunk_embedding_store.get_rows(new_chunk_keys)
        
        # 2. Check cache
        all_openie_info, chunk_keys_to_process = self.hipporag.load_existing_openie(new_chunk_keys)
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}
        
        if len(chunk_keys_to_process) > 0:
            print(f"[PREFETCH] LLM Extraction needed for {len(chunk_keys_to_process)} chunks...")
            new_ner, new_triples = self.hipporag.openie.batch_openie(new_openie_rows)
            self.hipporag.merge_openie_results(all_openie_info, new_openie_rows, new_ner, new_triples)
            if self.hipporag.global_config.save_openie:
                self.hipporag.save_openie_results(all_openie_info)
                
        self.last_prefetched_idx = end_idx
        print(f"[PREFETCH] Done. Took {time.time() - t0:.2f}s. New last_prefetched_idx: {self.last_prefetched_idx}")

    def retrieve_context(
        self,
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        checkpoint_id = cp.get("checkpoint_id", "unknown")
        
        # 1. Identify cutoff
        as_of = cp.get("as_of", {})
        cutoff_idx = as_of.get("log_index")
        
        if cutoff_idx is None:
            observed, _, _ = observed_logs_for_checkpoint(cp, self.all_logs)
            cutoff_idx = len(observed) - 1
            
        print(f"\n[ONLINE] CP: {checkpoint_id} | Cutoff Index: {cutoff_idx} | Last Indexed: {self.last_indexed_idx}")
        
        # 1.5 LOOKAHEAD PREFETCH (Hyper-Aggressive as requested)
        PREFETCH_BUFFER = self.batch_size * 4 # Buffer of 160 logs
        if self.last_prefetched_idx < cutoff_idx + PREFETCH_BUFFER and self.last_prefetched_idx < len(self.all_logs) - 1:
            pf_start = self.last_prefetched_idx + 1
            # Aggressive prefetch: fetch up to 320 logs ahead to minimize interruptions
            pf_end = min(len(self.all_logs) - 1, pf_start + (self.batch_size * 8) - 1)
            pf_end = max(pf_end, cutoff_idx)
            self.prefetch_openie(pf_start, pf_end)

        # 2. Incremental Indexing
        if cutoff_idx > self.last_indexed_idx:
            new_logs = self.all_logs[self.last_indexed_idx + 1 : cutoff_idx + 1]
            print(f"[ONLINE] Indexing {len(new_logs)} new logs...")
            docs = [json.dumps(log, ensure_ascii=False) for log in new_logs]
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i : i + self.batch_size]
                print(f"[ONLINE] Indexing batch {i//self.batch_size + 1}/{(len(docs)-1)//self.batch_size + 1}...")
                self.hipporag.index(docs=batch)
            
            self.last_indexed_idx = cutoff_idx
            self.hipporag.ready_to_retrieve = False
            print(f"[ONLINE] Indexing complete. Total indexed: {self.last_indexed_idx + 1}")
        
        # 3. Retrieve
        query = _build_retrieval_query(cp, target_keys)
        retrieval_start = time.time()
        solutions = self.hipporag.retrieve(queries=[query])
        retrieval_duration = time.time() - retrieval_start
        print(f"[ONLINE] HippoRAG retrieval took {retrieval_duration:.2f}s")
        
        if not solutions:
            return {
                "context_logs": [],
                "retrieval_query": query,
                "metadata": {"retrieval_mode": "online_hipporag", "retrieval_duration_s": retrieval_duration},
            }
            
        sol = solutions[0]
        retrieved_logs = []
        if hasattr(sol, 'docs'):
            for doc_str in sol.docs[:5]:
                try:
                    retrieved_logs.append(json.loads(doc_str))
                except:
                    pass
        
        return {
            "context_logs": retrieved_logs,
            "context_note": f"Online HippoRAG retrieved (top-5)",
            "retrieval_query": query,
            "metadata": {
                "retrieval_duration_s": retrieval_duration,
                "cutoff_idx": cutoff_idx,
                "last_indexed_idx": self.last_indexed_idx
            },
        }

def run_online_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    save_dir: Path,
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    embedding_model: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
    
    # 1. Initialize LLM Client
    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )

    # 2. Determine OpenIE Mode
    openie_mode = "Transformers-offline"
    if any(k in llm_model.lower() for k in ["gpt-", "o3-", "openai/"]):
        print(f"[*] Detected OpenAI model: {llm_model}. Using Online OpenIE.")
        openie_mode = "online"

    # 3. Initialize HippoRAG
    print(f"[*] Initializing HippoRAG in {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    llm_base_url = os.getenv("OPENAI_BASE_URL")
    embedding_base_url = os.getenv("OPENAI_BASE_URL")
    
    config = BaseConfig(temperature=0, llm_base_url=llm_base_url, embedding_base_url=embedding_base_url)
    hipporag = HippoRAG(
        global_config=config,
        save_dir=str(save_dir),
        llm_model_name=llm_model,
        embedding_model_name=embedding_model,
        openie_mode=openie_mode,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url
    )

    # 4. Load logs
    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    all_logs = all_logs_payload.get("app_logs", []) if isinstance(all_logs_payload, dict) else all_logs_payload

    # 5. Initialize Runner
    runner = OnlineDSPRunner(hipporag, all_logs, batch_size)

    # 6. Initialize State from Cache/Resume
    if os.path.isfile(hipporag.openie_results_path):
        try:
            with open(hipporag.openie_results_path, 'r') as f:
                openie_data = json.load(f)
                existing_hashes = set(d['idx'] for d in openie_data.get('docs', []))
                temp_covered = -1
                for i, log in enumerate(all_logs):
                    doc_str = json.dumps(log, ensure_ascii=False)
                    if compute_mdhash_id(doc_str, prefix='chunk-') in existing_hashes:
                        temp_covered = i
                    else:
                        break
                runner.last_prefetched_idx = temp_covered
                print(f"[*] Pre-computed OpenIE coverage: {runner.last_prefetched_idx}")
        except:
            pass

    if resume and output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text(encoding="utf-8"))
            runner.last_indexed_idx = max([p.get("metadata", {}).get("last_indexed_idx", -1) for p in existing_data.get("predictions", [])] + [-1])
            print(f"[*] Resuming from last_indexed_idx: {runner.last_indexed_idx}")
        except:
            pass


    # 7. GLOBAL PREFETCH (User Requested: Pre-compute ALL OpenIE upfront)
    print(f"[*] Prefetching ALL OpenIE results (Batch Size: {batch_size})...")
    
    total_logs = len(all_logs)
    current_start = runner.last_prefetched_idx + 1
    
    # We only prefetch if we haven't covered everything yet
    if current_start < total_logs:
        for start_idx in range(current_start, total_logs, batch_size):
            end_idx = min(start_idx + batch_size - 1, total_logs - 1)
            # Use the runner's prefetch logic which handles dedup and LLM calls
            runner.prefetch_openie(start_idx, end_idx)
            
    print(f"[*] All OpenIE results prefetched. Starting pipeline...")

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=None,
        ask_json=lambda p: client.ask(p, response_type="json"),
        ask_structured=lambda p, fmt: client.ask_structured(p, text_format=fmt),
        use_structured_response=client.supports_structured_response(),
        close=lambda: None,
        retrieve_context=runner.retrieve_context,
        baseline_name="hipporag_online",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )

def main():
    parser = argparse.ArgumentParser(description="Online Synchronous HippoRAG DSP")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    args = parser.parse_args()

    run_online_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        save_dir=args.save_dir,
        llm_provider="openai",
        llm_model=args.llm_model,
        llm_max_workers=1,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=None,
        save_prompt_and_raw=args.save_prompt_and_raw,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
