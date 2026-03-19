import json
import argparse
import time
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

import numpy as np
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from jinja2 import Template

from langgraph.store.memory import InMemoryStore
from langgraph.func import entrypoint
from langchain.chat_models import init_chat_model

# Langmem APIS
from langmem import (
    create_memory_manager,
    create_memory_store_manager,
    create_prompt_optimizer,
    create_search_memory_tool,
    create_manage_memory_tool,
)

load_dotenv()



# =========================
# 1. Schema Definitions
# =========================

# Collection of Semantic Memory:
# From https://langchain-ai.github.io/langmem/guides/extract_semantic_memories/?h=#without-storage

class SemanticTriple(BaseModel):
    """Store all new facts, preferences, and relationships as triples."""
    subject: str = Field(description= "main subject of the memory")
    predicate: str = Field(description = "Relationship between the subject and object")
    object: str = Field(description = "object of the memory")
    context: Optional[str] = Field(default=None, description="context of the memory")


# Collection of UserProfile Memory:
# From https://langchain-ai.github.io/langmem/guides/manage_user_profile/#basic-usage

class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None

# Instruction: Episodic Memories
# From https://langchain-ai.github.io/langmem/guides/extract_episodic_memories/#without-storage
class Episode(BaseModel):  
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the memory, saving the agent's key internal thought process so it can learn over time."""

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )


# =========================
# 2. Config
# =========================

class LangmemConfig:
    def __init__(
        self,
        *,
        schema_path: str,
        qa_path: str,
        output_path: str = "data/langmem_results.json",
        mode: str = "hybrid",
        embed_model: str = "openai:text-embedding-3-small",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        query_model: str = "gpt-4o-mini",
        top_k: int = 5,
        user_id: str = "default",
        namespace: tuple = ("memories",),
        resume: bool = False,
    ):
        self.schema_path = schema_path
        self.qa_path = qa_path
        self.output_path = output_path
        self.mode = mode
        self.embed_model = embed_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.query_model = query_model
        self.top_k = top_k
        self.user_id = user_id
        self.namespace = namespace
        self.resume = resume


# =========================
# 3. Memory Managers
# =========================

class SemanticTripleManager: # a manager for semantic triples


    def __init__(self, cfg: LangmemConfig):
        self.cfg = cfg
        self.store = InMemoryStore(
            index = {"dims": 1536, "embed": cfg.embed_model}
        )

        # need to store here
        self.manager = create_memory_store_manager(
            cfg.llm_model,
            query_model = cfg.query_model,
            schemas = [SemanticTriple],
            namespace = ("memories", cfg.user_id, "semantic", "triples"),
            instructions="Extract user preferences and any other useful information as triples.",
            enable_inserts = True,
            enable_deletes = True,
            query_limit = cfg.top_k,
            store = self.store,
        )

        self.search_tool = create_search_memory_tool(
            namespace = ("memories", cfg.user_id, "semantic", "triples"),
            instructions = "Find relevant factual memories.",
            store = self.store,
        )

        # For debugging
        self.extracted_triples = []

        
    def ingest(self, app_logs: List[dict], debug: bool = False) -> None:
        """Extract Triple Memories from app_logs"""
        print(f"[Semantic Triple] Processing {len(app_logs)} logs...")
        
        if debug:
            print(f"\n[DEBUG] Raw app_logs sample (first 3):")
            for i, log in enumerate(app_logs[:3]):
                print(f"  Log {i+1}: {log.get('app_name')}: {str(log.get('request', {}))[:200]}...")
        
        messages = self._convert_logs_to_messages(app_logs)
        
        if debug:
            print(f"\n[DEBUG] Converted messages (first 1):")
            print(f"  {messages[0] if messages else 'None'}")
        
        result = self.manager.invoke({"messages": messages})
        
        if debug:
            print(f"\n[DEBUG] Extraction result (triples):")

            if hasattr(result, 'items'):
                for k, v in result.items():
                    print(f"  {k}: {v}")
            elif isinstance(result, dict):
                for k, v in result.items():
                    print(f"  {k}: {v}")
        
        # Retrieve stored triples from store
        self.extracted_triples = list(self.store.search(
            ("memories", self.cfg.user_id, "semantic", "triples"),
            query="",
            limit=100
        ))
        if debug:
            print(f"\n[DEBUG] Stored triples in memory (total {len(self.extracted_triples)}):")
            for i, item in enumerate(self.extracted_triples[:10]):
                print(f"  {i+1}. {item.value}")
        
        print(f"[Semantic Triple] Extracted and stored.")


    def retrieve(self, query:str) -> List[dict]:
        """Retrieve relevant Triple Memories"""
        results = self.manager.search(query=query, limit = self.cfg.top_k)
        return [{"id": item.key, "content": item.value} for item in results]

    def _convert_logs_to_messages(self, app_logs: List[dict]) -> List[dict]:
        messages = []
        for log in app_logs:
            content = f"[{log.get('timestamp', '')}] {log.get('app_name', '')}: {json.dumps(log.get('request', {}), ensure_ascii=False)}"
            messages.append({"role": "user", "content": content})
        return messages

class UserProfileManager:
    """User Profile"""
    
    def __init__(self, cfg: LangmemConfig):
        self.cfg = cfg
        self.store = InMemoryStore(
            index={"dims": 1536, "embed": cfg.embed_model}
        )
        
        self.manager = create_memory_store_manager(
            cfg.llm_model,
            query_model=cfg.query_model,
            schemas=[UserProfile],
            namespace=("memories", cfg.user_id, "profile"),
            instructions="Extract user profile information including name, language, and timezone.",
            enable_inserts=False,  
            query_limit=1,
            store=self.store,
        )

    def ingest(self, app_logs: List[dict], debug: bool = False) -> None:
        """Extract Userfile from app_logs"""
        print(f"[User Profile] Processing {len(app_logs)} logs...")
        
        if debug:
            print(f"\n[DEBUG] Raw app_logs sample (first 3):")
            for i, log in enumerate(app_logs[:3]):
                print(f"  Log {i+1}: {log.get('app_name')}: {str(log.get('request', {}))[:200]}...")
        
        messages = self._convert_logs_to_messages(app_logs)
        self.manager.invoke({"messages": messages})
        
        # 从 store 中读取已存储的 profile
        profiles = list(self.store.search(
            ("memories", self.cfg.user_id, "profile"),
            query="",
            limit=10
        ))
        if debug:
            print(f"\n[DEBUG] Stored profiles (total {len(profiles)}):")
            for i, item in enumerate(profiles):
                print(f"  {i+1}. {item.value}")
        
        print(f"[User Profile] Extracted/updated.")
    def _convert_logs_to_messages(self, app_logs: List[dict]) -> List[dict]:
        messages = []
        for log in app_logs:
            content = f"[{log.get('timestamp', '')}] {log.get('app_name', '')}: {json.dumps(log.get('request', {}), ensure_ascii=False)}"
            messages.append({"role": "user", "content": content})
        return messages
    def retrieve(self, query: str) -> Optional[dict]:
        """Retrieve User Profile"""
        results = self.manager.search(query = query, limit=1)
        if results:
            return {"id": results[0].key, "content": results[0].value}
        return None



class EpisodicMemoryManager:
    """Episodic Memory"""
    
    def __init__(self, cfg: LangmemConfig):
        self.cfg = cfg
        self.store = InMemoryStore(
            index={"dims": 1536, "embed": cfg.embed_model}
        )
        
        self.manager = create_memory_store_manager(
            cfg.llm_model,
            query_model=cfg.query_model,
            schemas=[Episode],
            namespace=("memories", cfg.user_id, "episodes"),
            instructions="Extract examples of successful explanations, capturing the full chain of reasoning.",
            enable_inserts=True,
            query_limit=cfg.top_k,
            store=self.store,
        )
        
        self.search_tool = create_search_memory_tool(
            ("memories", cfg.user_id, "episodes"),
            instructions="Find relevant past experiences.",
            store=self.store,
        )


    def ingest(self, app_logs: List[dict], debug: bool = False) -> None:
        """Extract episodic memory from app_logs"""
        print(f"[Episodic] Processing {len(app_logs)} logs...")
        
        if debug:
            print(f"\n[DEBUG] Raw app_logs sample (first 3):")
            for i, log in enumerate(app_logs[:3]):
                print(f"  Log {i+1}: {log.get('app_name')}: {str(log.get('request', {}))[:200]}...")
        
        messages = self._convert_logs_to_messages(app_logs)
        self.manager.invoke({"messages": messages})
        
        # Retrieve stored episodes from store
        episodes = list(self.store.search(
            ("memories", self.cfg.user_id, "episodes"),
            query="",
            limit=20
        ))
        if debug:
            print(f"\n[DEBUG] Stored episodes (total {len(episodes)}):")
            for i, item in enumerate(episodes[:5]):
                obs = item.value.get('observation', '')[:150] if isinstance(item.value, dict) else str(item.value)[:150]
                print(f"  {i+1}. Observation: {obs}...")
        
        print(f"[Episodic] Extracted.")
    def _convert_logs_to_messages(self, app_logs: List[dict]) -> List[dict]:
        messages = []
        for log in app_logs:
            content = f"[{log.get('timestamp', '')}] {log.get('app_name', '')}: {json.dumps(log.get('request', {}), ensure_ascii=False)}"
            messages.append({"role": "user", "content": content})
        return messages
    def retrieve(self, query: str) -> List[dict]:
        """Retrieve the episodic memory"""
        results = self.manager.search(query = query, limit=self.cfg.top_k)
        return [{"id": item.key, "content": item.value} for item in results]
    def build_few_shot_prompt(self, retrieved_episodes: List[dict]) -> str:
        """build few_shot_prompt"""
        if not retrieved_episodes:
            return ""
        
        examples = []
        for i, ep in enumerate(retrieved_episodes, 1):
            content = ep.get("content", {})
            # Convert to dict 
            if hasattr(content, 'model_dump'):
                content = content.model_dump()
            elif hasattr(content, 'dict'):
                content = content.dict()
            if isinstance(content, dict):
                examples.append(f"""
Example {i}:
- Situation: {content.get('observation', '')}
- Thought: {content.get('thoughts', '')}
- Action: {content.get('action', '')}
- Result: {content.get('result', '')}
""")
        return "\n".join(examples)



class HybridMemoryManager:
    """Hybrid - 3 Memories"""
    
    def __init__(self, cfg: LangmemConfig):
        self.cfg = cfg
        self.triple_mgr = SemanticTripleManager(cfg)
        self.profile_mgr = UserProfileManager(cfg)
        self.episodic_mgr = EpisodicMemoryManager(cfg)
    def ingest(self, app_logs: List[dict], debug: bool = False) -> None:
        """Ingest all memories"""
        print("[Hybrid] Starting ingestion...")
        self.triple_mgr.ingest(app_logs, debug=debug)
        self.profile_mgr.ingest(app_logs, debug=debug)
        self.episodic_mgr.ingest(app_logs, debug=debug)
        print("[Hybrid] Ingestion complete.")
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve all memories"""
        return {
            "triples": self.triple_mgr.retrieve(query),
            "profile": self.profile_mgr.retrieve(query),
            "episodes": self.episodic_mgr.retrieve(query),
        }


# =========================
# 4. QA Pipeline
# =========================

class LangmemQAPipeline:
    """Full QA Pipeline"""
    
    def __init__(self, cfg: LangmemConfig):
        self.cfg = cfg
        self.llm = init_chat_model(cfg.llm_model)
        
        # Select manager based on mode
        if cfg.mode == "triple":
            self.manager = SemanticTripleManager(cfg)
        elif cfg.mode == "profile":
            self.manager = UserProfileManager(cfg)
        elif cfg.mode == "episodic":
            self.manager = EpisodicMemoryManager(cfg)
        else:  # hybrid
            self.manager = HybridMemoryManager(cfg)

    def run_ingestion(self, app_logs: List[dict], debug: bool = False) -> None:
        """Run memory ingestion"""
        self.manager.ingest(app_logs, debug=debug)
    def answer(self, question: str, debug: bool = False) -> Dict[str, Any]:
        """Answer question"""
        # 1. Retrieve memories
        retrieved = self.manager.retrieve(question)
        
        # Debug: Print retrieved results (only when debug=True)
        if debug:
            print(f"\n{'='*60}")
            print(f"Query: {question}")
            print(f"{'='*60}")
            
            if isinstance(retrieved, dict):
                # Hybrid mode
                # Triples
                triples = retrieved.get("triples", [])
                print(f"\n[Triples] Retrieved {len(triples)} items:")
                for i, t in enumerate(triples):
                    content = t.get("content", {})
                    if hasattr(content, 'model_dump'):
                        content = content.model_dump()
                    elif hasattr(content, 'dict'):
                        content = content.dict()
                    print(f"  {i+1}. {content}")
                
                # Profile
                profile = retrieved.get("profile")
                print(f"\n[Profile] Retrieved: {profile}")
                
                # Episodes
                episodes = retrieved.get("episodes", [])
                print(f"\n[Episodes] Retrieved {len(episodes)} items:")
                for i, ep in enumerate(episodes):
                    content = ep.get("content", {})
                    if hasattr(content, 'model_dump'):
                        content = content.model_dump()
                    elif hasattr(content, 'dict'):
                        content = content.dict()
                    print(f"  {i+1}. {content}")
            else:
                print(f"Retrieved: {retrieved}")
            
            print(f"{'='*60}\n")
        
        # 2. Build prompt
        prompt = self._build_prompt(question, retrieved)
        
        # 3. Generate answer
        start_time = time.time()
        response = self.llm.invoke(prompt)
        answer_time = time.time() - start_time
        
        return {
            "answer": response.content,
            "retrieved": retrieved,
            "answer_time": answer_time,
        }

    def _build_prompt(self, question: str, retrieved: Dict[str, Any]) -> List[dict]:
        """Build QA prompt"""
        system_content = "You are a helpful assistant that answers questions based on user memories."
        
        # Add context
        context_parts = []
        
        # Triples
        if "triples" in retrieved:
            triples = retrieved["triples"]
            if triples:
                context_parts.append("### Relevant Facts:")
                for t in triples[:5]:
                    content = t.get("content", {})
                    # Convert to dict (可能是 Pydantic 模型)
                    if hasattr(content, 'model_dump'):
                        content = content.model_dump()
                    elif hasattr(content, 'dict'):
                        content = content.dict()
                    if isinstance(content, dict):
                        s = content.get("subject", "")
                        p = content.get("predicate", "")
                        o = content.get("object", "")
                        c = content.get("context", "")
                        if c:
                            context_parts.append(f"- {s} {p} {o} (context: {c})")
                        else:
                            context_parts.append(f"- {s} {p} {o}")
            
        # Profile
        if "profile" in retrieved:
            profile = retrieved["profile"]
            if profile:
                profile_content = profile.get('content', {})
                # Convert to dict
                if hasattr(profile_content, 'model_dump'):
                    profile_content = profile_content.model_dump()
                elif hasattr(profile_content, 'dict'):
                    profile_content = profile_content.dict()
                context_parts.append(f"\n### User Profile:\n{json.dumps(profile_content, ensure_ascii=False)}")
        
        # Episodes
        if "episodes" in retrieved:
            episodes = retrieved["episodes"]
            if episodes:
                context_parts.append("\n### Past Experiences:")
                for ep in episodes[:3]:
                    content = ep.get("content", {})
                    # Convert to dict
                    if hasattr(content, 'model_dump'):
                        content = content.model_dump()
                    elif hasattr(content, 'dict'):
                        content = content.dict()
                    if isinstance(content, dict):
                        context_parts.append(f"- {content.get('observation', '')}")
        
        messages = [{"role": "system", "content": system_content}]
        
        if context_parts:
            context = "\n".join(context_parts)
            messages.append({
                "role": "user",
                "content": f"{question}\n\nContext:\n{context}"
            })
        else:
            messages.append({"role": "user", "content": question})
        
        return messages


# =========================
# 5. Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Langmem Full Pipeline")
    parser.add_argument("--user-idx", required=True, help="User index (e.g., 001)")
    parser.add_argument("--mode", default="episodic",
                       choices=["triple", "profile", "episodic", "hybrid"],
                       help="Memory mode")
    parser.add_argument("--input-root-dir", type=str, default=None)
    parser.add_argument("--output-root-dir", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--schema-path", type=str, default=None)
    parser.add_argument("--qa-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="openai:text-embedding-3-small")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--query-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only run first QA and print retrieved memories")
    args = parser.parse_args()
    
    # Parse paths
    input_root = args.input_root_dir or "/home/zyli/MUSE/.data/data_construction/generated_outputs/gemini_3_flash_preview"
    output_root = args.output_root_dir or "/home/zyli/MUSE/generation/Langmem/results"
    qa_dir = args.qa_dir or "/home/zyli/MUSE/generation/qa"
    
    # Process user_id: support "001" or "001_user_001" format
    if "_user_" in args.user_idx:
        user_id = args.user_idx
        # QA filename using original number format (e.g., "001")
        qa_user_idx = args.user_idx.split("_user_")[0]
    else:
        user_id = f"{int(args.user_idx):03d}_user_{args.user_idx}"
        qa_user_idx = f"{int(args.user_idx):03d}"
    
    schema_path = Path(args.schema_path) if args.schema_path else \
                  Path(input_root) / user_id / "app_log_large.json"
    qa_path = Path(args.qa_path) if args.qa_path else \
              Path(qa_dir) / f"qa_human_{qa_user_idx}.json"
    output_path = Path(args.output_path) if args.output_path else \
                  Path(output_root) / user_id / "prediction" / f"langmem_{args.mode}_results.json"
    
    # Configuration
    cfg = LangmemConfig(
        schema_path=str(schema_path),
        qa_path=str(qa_path),
        output_path=str(output_path),
        mode=args.mode,
        embed_model=args.embed_model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        query_model=args.query_model,
        top_k=args.top_k,
        user_id=args.user_idx,
        resume=args.resume,
    )
    
    # Load data
    print(f"Loading app logs from {schema_path}...")
    with open(schema_path, "r", encoding="utf-8") as f:
        app_logs = json.load(f)
    if isinstance(app_logs, dict):
        app_logs = app_logs.get("app_logs", [])
    print(f"Loaded {len(app_logs)} logs.")
    
    print(f"Loading QA from {qa_path}...")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    print(f"Loaded {len(qa_list)} QA items.")
    
    # Initialize pipeline
    pipeline = LangmemQAPipeline(cfg)
    
    # Ingestion
    print(f"\n[{args.mode}] Running ingestion...")
    pipeline.run_ingestion(app_logs, debug=args.debug)
    
    # QA
    print(f"\n[{args.mode}] Running QA...")
    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Limit number: debug mode only run 1
    qa_to_run = [qa_list[0]] if args.debug else qa_list
    
    for qa in tqdm(qa_to_run, desc="Answering"):
        question = qa.get("query") or qa.get("question")
        if not question:
            continue
        
        result = pipeline.answer(question, debug=args.debug)
        results.append({
            "id": qa.get("uid"),
            "query": question,
            "reference": qa.get("reference"),
            "answer": result["answer"],
            "answer_time": result["answer_time"],
        })
        
        # Save intermediate results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Results saved to {output_path}")
if __name__ == "__main__":
    main()