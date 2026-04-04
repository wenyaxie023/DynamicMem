#!/usr/bin/env python3
import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from generation.rag.client import LLMClient
from generation.common.provider_config import setup_provider_env
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_core.pipeline import observed_logs_for_checkpoint, run_pipeline, to_log_text

REPO_ROOT_DIR = Path(os.path.abspath(os.path.join(root_dir, "..", "..")))

class OnlineDSPRunner:
    def __init__(self, hipporag: HippoRAG, all_logs: List[Dict[str, Any]], batch_size: int):
        self.hipporag = hipporag
        self.all_logs = all_logs
        self.batch_size = batch_size
        self.last_indexed_idx = -1

    def prepare_checkpoint_state(
        self,
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
    ) -> CheckpointHandle:
        return CheckpointHandle(
            checkpoint_id=str(cp.get("checkpoint_id") or ""),
            state_kind="graph_index",
            state_ref=cp,
            metadata={
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
            },
        )

    def retrieve_context_for_query(
        self,
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        cp = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}
        checkpoint_id = cp.get("checkpoint_id", "unknown")
        
        # 1. Identify cutoff
        as_of = cp.get("as_of", {})
        cutoff_idx = as_of.get("log_index")
        
        if cutoff_idx is None:
            observed, _, _ = observed_logs_for_checkpoint(cp, self.all_logs)
            cutoff_idx = len(observed) - 1
            
        print(f"\n[ONLINE] CP: {checkpoint_id} | Cutoff Index: {cutoff_idx} | Last Indexed: {self.last_indexed_idx}")
        
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
        query = str(query_spec.retrieval_query_text or "").strip()
        if not query:
            raise ValueError("HippoRAG2 requires shared QuerySpec.retrieval_query_text.")
        retrieval_start = time.time()
        solutions = self.hipporag.retrieve(queries=[query])
        retrieval_duration = time.time() - retrieval_start
        print(f"[ONLINE] HippoRAG retrieval took {retrieval_duration:.2f}s")

        if not solutions:
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=[],
                debug_metadata={
                    "retrieval_mode": "online_hipporag",
                    "retrieval_duration_s": retrieval_duration,
                    "retrieval_query": query,
                },
            )
            
        sol = solutions[0]
        retrieved_logs = []
        if hasattr(sol, 'docs'):
            for doc_str in sol.docs[:5]:
                try:
                    retrieved_logs.append(json.loads(doc_str))
                except:
                    pass
        
        top_k_for_call = retrieval_options.common.get("top_k")
        if isinstance(top_k_for_call, int) and top_k_for_call >= 0:
            retrieved_logs = retrieved_logs[:top_k_for_call] if top_k_for_call > 0 else list(retrieved_logs)

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in retrieved_logs],
            debug_metadata={
                "retrieval_mode": "online_hipporag",
                "retrieval_duration_s": retrieval_duration,
                "cutoff_idx": cutoff_idx,
                "last_indexed_idx": self.last_indexed_idx,
                "retrieval_query": query,
                "retrieved_app_log_ids": [log.get("app_log_id") for log in retrieved_logs],
            },
        )

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
    _, resolved_base_url = setup_provider_env(llm_provider, repo_root=REPO_ROOT_DIR)
    
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
    
    llm_base_url = resolved_base_url
    embedding_base_url = resolved_base_url
    
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

    # 6. Initialize State from Resume
    if resume and output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text(encoding="utf-8"))
            runner.last_indexed_idx = max([p.get("metadata", {}).get("last_indexed_idx", -1) for p in existing_data.get("predictions", [])] + [-1])
            print(f"[*] Resuming from last_indexed_idx: {runner.last_indexed_idx}")
        except:
            pass

    print(f"[*] No pre-fetching. OpenIE will run incrementally per checkpoint. Starting pipeline...")

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=None,
        ask_json=lambda p: client.ask(p, response_type="json"),
        ask_structured=lambda p, fmt: client.ask_structured(p, text_format=fmt),
        use_structured_response=client.supports_structured_response(),
        close=lambda: None,
        prepare_checkpoint_state=runner.prepare_checkpoint_state,
        retrieve_context_for_query=runner.retrieve_context_for_query,
        baseline_name="hipporag_online",
        memory_prompt_mode="inline_memory",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        retrieval_options_backend={"embedding_model": embedding_model, "batch_size": batch_size},
    )

def main():
    parser = argparse.ArgumentParser(description="Online Synchronous HippoRAG TCE")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--app-logs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini")
    parser.add_argument("--llm-max-workers", type=int, default=1)
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
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
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
