import os
import json
import glob
import sys
from dotenv import load_dotenv

# Add HippoRAG and project roots to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(os.path.join(root_dir, "HippoRAG/src"))
sys.path.append(os.path.abspath(os.path.join(root_dir, "..", ".."))) # For generation.rag etc.

# Monkeypatch for torch load vulnerability check
# Must be done BEFORE importing transformers or libraries that use it
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True

import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

from hipporag import HippoRAG
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig
import argparse
from jinja2 import Template
from generation_prompt_legacy import PROMPT

def main():
    parser = argparse.ArgumentParser(description="Run HippoRAG Generation")
    parser.add_argument("--user_id", type=str, default="003_user_003", help="User ID (e.g., 003_user_003)")
    parser.add_argument("--llm_name", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--test_only", action="store_true", help="If set, only process the first 2 questions for testing")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to process")
    
    args = parser.parse_args()

    # 1. Load Environment Variables
    env_path = ".env"
    load_dotenv(env_path, override=True)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print(f"Error: OPENAI_API_KEY not found in {env_path}")
        return
    print(f"Loaded API Key ending in: ...{key[-4:] if key else 'None'}")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        print(f"Loaded Base URL: {base_url}")

    # 2. Configuration
    user_id = args.user_id
    llm_name = args.llm_name
    embedding_model = args.embedding_model
    
    # Existing HippoRAG output directory to reuse index/graph
    # We assume the directory structure: generation/HippoRAG2/outputs/{user_id}_large/{llm_name}_{embedding_model}
    # Note: user_id 003_user_003 has index in 003_user_003_large
    folder_user_id = user_id
    if not folder_user_id.endswith("_large") and "user_003" in folder_user_id:
        folder_user_id = f"{user_id}_large"
        
    existing_data_dir = os.path.abspath(f"outputs/{folder_user_id}/{llm_name}_{embedding_model}")
    
    if not os.path.exists(existing_data_dir):
        print(f"Error: Existing data directory not found: {existing_data_dir}")
        return
        
    print(f"Using existing data from: {existing_data_dir}")
    
    # Configure HippoRAG with Azure Base URL and temperature=1
    config = BaseConfig(
        temperature=1,
        llm_base_url=base_url,
        embedding_base_url=base_url
    )

    # Initialize HippoRAG
    # We use a dummy save_dir initially, then OVERRIDE working_dir
    hipporag = HippoRAG(
        global_config=config,
        save_dir="temp_init_dir", 
        llm_model_name=llm_name,
        embedding_model_name=embedding_model
    )
    
    # FORCE override working_dir to the existing data directory
    hipporag.working_dir = existing_data_dir
    
    # Re-initialize components that depend on working_dir
    print("Re-initializing components with existing data...")
    
    # Graph
    hipporag.graph = hipporag.initialize_graph()
    
    # Embeddings (Need to re-init stores with new path if they weren't lazy)
    # Actually, the stores are init in __init__ with working_dir. 
    # We need to manually update them or re-instantiate them.
    
    # Update paths in stores
    from hipporag.embedding_store import EmbeddingStore
    
    # Note: We need the embedding model to be ready. 
    # Since we are using BAAI/bge-m3, it should load via TransformersEmbeddingModel
    # Check if we need to set specific config for it? 
    # The init() called _get_embedding_model_class, so hipporag.embedding_model should be set.
    
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
    
    # Mark as ready to retrieve so it doesn't try to re-index or something
    hipporag.ready_to_retrieve = False 
    # Actually, prepare_retrieval_objects() checks this. We want it to run later.
    
    # 3. Process Questions
    questions_dir = "questions"
    # Find question file for this user (e.g., qa_human_003.json)
    user_num = user_id.split("_")[0] # extraction "003" from "003_user_003"
    json_files = glob.glob(os.path.join(questions_dir, f"qa_human_{user_num}.json"))
    json_files.sort()
    
    if not json_files:
        print(f"No JSON files found in {questions_dir}")
        return

    print(f"Found {len(json_files)} question files: {json_files}")

    for q_file in json_files:
        print(f"Processing {q_file}...")
        
        output_dir = os.path.join(
            ".", 
            "hipporag_gpt5mini", 
            "results", 
            user_id, 
            "prediction"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(q_file))
        
        if os.path.exists(output_file):
            print(f"Skipping {q_file} as {output_file} already exists.")
            continue
        
        with open(q_file, 'r') as f:
            questions_data = json.load(f)
            
        queries = [item['query'] for item in questions_data]
        
        if args.limit:
            print(f"Limit Mode: Processing first {args.limit} questions")
            queries = queries[:args.limit]
            questions_data = questions_data[:args.limit]
        elif args.test_only:
            print("Test Mode: Only processing first 2 questions")
            queries = queries[:2]
            questions_data = questions_data[:2]
            
        print(f"Retrieving context for {len(queries)} questions...")
        query_solutions = hipporag.retrieve(queries=queries)
        
        template = Template(PROMPT)
        output_data = []
        
        print("Generating answers with standard prompt...")
        for i, res in enumerate(query_solutions):
            original_item = questions_data[i]
            
            # Prepare context from retrieved docs 
            # res.docs contains the content strings (JSON strings)
            # Use only top K docs for QA context to save tokens and match standard HippoRAG behavior
            top_k = hipporag.global_config.qa_top_k
            context_str = "\n<->\n".join(res.docs[:top_k])
            
            # Render prompt
            prompt_text = template.render(
                question=res.question,
                context=context_str
            )
            
            # Use HippoRAG's LLM to infer
            # We need to wrap prompt in TextChatMessage if HippoRAG's llm expect it,
            # but CacheOpenAI.infer handles standard messages too if we are careful.
            # Looking at OpenAI.py, it expects List[TextChatMessage].
            from hipporag.utils.llm_utils import TextChatMessage
            messages = [TextChatMessage(role="user", content=prompt_text)]
            
            try:
                response_content, metadata, _ = hipporag.llm_model.infer(messages=messages)
                
                # Try to parse JSON from response
                # Sometimes LLM adds markdown triple backticks
                clean_json = response_content.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif clean_json.startswith("```"):
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
                res_json = json.loads(clean_json)
                prediction = res_json.get("answer", response_content)
                predicted_evidence = res_json.get("evidence", [])
                
            except Exception as e:
                print(f"Error generating answer for question {i}: {e}")
                prediction = response_content if 'response_content' in locals() else "Error"
                predicted_evidence = []
                metadata = {}

            output_item = {
                "id": original_item.get("id"),
                "query": res.question,
                "reference": original_item.get("reference", ""),
                "prediction": prediction,
                "predicted_evidence": predicted_evidence,
                "metadata": {
                    **(original_item.get("metadata", {})),
                    "hipporag_metadata": metadata
                }
            }
            output_data.append(output_item)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
