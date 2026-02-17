import os
import json
import glob
import sys
import time
from dotenv import load_dotenv
import argparse
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add HippoRAG to path
sys.path.append(os.path.abspath("HippoRAG/src"))

# Monkeypatch for torch load vulnerability check
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.embedding_store import EmbeddingStore
from hipporag.utils.llm_utils import TextChatMessage
from generation_prompt_legacy import PROMPT

def run_user_generation(user_id, questions_file, llm_name, embedding_model, base_url, batch_size, limit, qa_top_k=None, output_parent_dir=None):
    print(f"\n{'='*50}")
    print(f"Starting generation for User: {user_id}")
    print(f"Questions File: {questions_file}")
    print(f"{'='*50}")

    # 1. Setup Data Directory
    # user_id format example: 003_user_003
    # Folder format: 003_user_003_large
    folder_user_id = f"{user_id}_large" if not user_id.endswith("_large") else user_id
    existing_data_dir = os.path.abspath(f"outputs/{folder_user_id}/{llm_name}_{embedding_model}")
    
    if not os.path.exists(existing_data_dir):
        print(f"SKIP: Data directory not found for {user_id}: {existing_data_dir}")
        return

    print(f"Using existing data from: {existing_data_dir}")

    # 2. Initialize HippoRAG
    config = BaseConfig(
        temperature=1,
        llm_base_url=base_url,
        embedding_base_url=base_url,
        qa_top_k=qa_top_k if qa_top_k else 5
    )

    hipporag = HippoRAG(
        global_config=config,
        save_dir="temp_init_dir", # Dummy
        llm_model_name=llm_name,
        embedding_model_name=embedding_model
    )
    
    # Override working directory and force re-init stores
    hipporag.working_dir = existing_data_dir
    hipporag.graph = hipporag.initialize_graph()
    
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
    hipporag.ready_to_retrieve = False 

    # 3. Load Questions
    with open(questions_file, 'r') as f:
        all_questions_data = json.load(f)

    if limit:
        print(f"Applying limit: {limit}")
        all_questions_data = all_questions_data[:limit]

    # 4. Prepare Output
    # README format: generation/<baseline_name>/results/<user_id>/prediction/
    if output_parent_dir:
        # Here output_parent_dir acts as the <baseline_root>
        output_dir = os.path.join(output_parent_dir, "results", user_id, "prediction")
    else:
        output_dir = os.path.join("hipporag_gpt5mini", "results", user_id, "prediction")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(questions_file))
    
    # LOAD CHECKPOINT
    existing_results = []
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                processed_ids = {item['id'] for item in existing_results}
            print(f"Resuming: Found {len(existing_results)} existing results.")
        except json.JSONDecodeError:
            print("Warning: Output file corrupted or empty. Starting fresh.")
    
    # 5. Batch Processing
    # Filter out questions that are already done
    questions_to_process = [q for q in all_questions_data if q['id'] not in processed_ids]
    
    if not questions_to_process:
        print(f"User {user_id} already fully processed.")
        return

    print(f"Remaining questions to process: {len(questions_to_process)}")
    
    # We will append new results to existing_results and save incrementally
    all_results = existing_results
    
    template = Template(PROMPT)
    
    # Process the REMAINING questions in chunks
    total_remaining = len(questions_to_process)
    
    # Split into chunks of batch_size
    for i in range(0, total_remaining, batch_size):
        batch_slice = questions_to_process[i : i + batch_size]
        batch_queries = [item['query'] for item in batch_slice]
        
        print(f"Processing batch {i // batch_size + 1}/{(total_remaining + batch_size - 1) // batch_size} (Items {i} to {i + len(batch_slice)})")
        
        # A. Retrieval
        try:
            query_solutions = hipporag.retrieve(queries=batch_queries)
        except Exception as e:
            print(f"Error retrieving batch start {i}: {e}")
            continue

        # B. Generation (Concurrent)
        qa_top_k = hipporag.global_config.qa_top_k
        
        # Helper function (same as before)
        def generate_single_answer(idx, solution, original_item):
            try:
                # Prepare context
                context_str = "\n<->\n".join(solution.docs[:qa_top_k])
                prompt_text = template.render(question=solution.question, context=context_str)
                messages = [TextChatMessage(role="user", content=prompt_text)]
                
                # Infer
                response_content, metadata, _ = hipporag.llm_model.infer(messages=messages)
                
                # Parse JSON
                clean_json = response_content.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif clean_json.startswith("```"):
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
                res_json = json.loads(clean_json)
                prediction = res_json.get("answer", response_content)
                predicted_evidence = res_json.get("evidence", [])
                
                return {
                    "id": original_item.get("id"),
                    "query": solution.question,
                    "reference": original_item.get("reference", ""),
                    "prediction": prediction,
                    "predicted_evidence": predicted_evidence,
                    "metadata": {
                        **(original_item.get("metadata", {})),
                        "hipporag_metadata": metadata
                    }
                }
            except Exception as e:
                print(f"Error generating Q{idx}: {e}")
                return {
                    "id": original_item.get("id"),
                    "query": solution.question,
                    "reference": original_item.get("reference", ""),
                    "prediction": "Error",
                    "predicted_evidence": [],
                    "metadata": original_item.get("metadata", {})
                }

        batch_results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for j, solution in enumerate(query_solutions):
                futures.append(executor.submit(generate_single_answer, i + j, solution, batch_slice[j]))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating (Concurrent)"):
                batch_results.append(future.result())
        
        # Incremental Save
        all_results.extend(batch_results)
        # Sort to keep tidy (optional, but good)
        all_results.sort(key=lambda x: x['id'])
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {len(all_results)} total items.")

    print(f"Finished User {user_id}. Final count: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="Run Batch HippoRAG Generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions per user for testing")
    parser.add_argument("--llm_name", type=str, default="gpt-5-mini")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    
    parser.add_argument("--user_id", type=str, default=None, help="Specific user ID to process (e.g., 002_user_002)")
    parser.add_argument("--top_k", type=int, default=None, help="Override qa_top_k")
    parser.add_argument("--baseline_dir", type=str, default=None, help="Override baseline root directory (e.g. generation/baseline_x)")
    
    args = parser.parse_args()
    
    # Load Environment
    env_path = ".env"
    load_dotenv(env_path, override=True)
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # Discover Users
    if args.user_id:
        uid_num = args.user_id.split('_')[0]
        q_file = os.path.join("../../questions", f"qa_human_{uid_num}.json")
        if not os.path.exists(q_file):
            print(f"Error: Target file {q_file} not found!")
            return
        question_files = [q_file]
    else:
        questions_pattern = os.path.join("../../questions", "qa_human_*.json")
        question_files = sorted(glob.glob(questions_pattern))
    
    if not question_files:
        print("No question files found!")
        return
        
    print(f"Found {len(question_files)} user files to process.")
    
    for q_file in question_files:
        # q_file example: questions/qa_human_003.json
        basename = os.path.basename(q_file)
        user_num = basename.split("_")[2].split(".")[0] # "003"
        
        # Construct user_id: 003_user_003
        user_id = f"{user_num}_user_{user_num}"
        
        try:
            run_user_generation(
                user_id=user_id,
                questions_file=q_file,
                llm_name=args.llm_name,
                embedding_model=args.embedding_model,
                base_url=base_url,
                batch_size=args.batch_size,
                limit=args.limit,
                qa_top_k=args.top_k,
                output_parent_dir=args.baseline_dir
            )
        except Exception as e:
            print(f"CRITICAL ERROR processing user {user_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
