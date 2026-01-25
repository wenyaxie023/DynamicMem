import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_PATH = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
TENSOR_PARALLEL_SIZE = 4

# Template (Same as before)
QA_TEMPLATE_HEADER = """You are presented with a question and a user activity log. Please answer the question based on the user activity log.

<activity_log>
"""
QA_TEMPLATE_FOOTER = """
</activity_log>

Question: {question}
Answer:"""

def load_json_data(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_text_data(filepath: str):
    """Load json logs and dump to string for context"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Kimi-Linear QA Offline Benchmark")
    parser.add_argument("--log_file", required=True, help="Path to the raw app log JSON file")
    parser.add_argument("--question_file", required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", required=True, help="Path to save the results")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"[*] Loading Raw Logs from {args.log_file}...")
    context_text = load_text_data(args.log_file)
    print(f"[*] Context Length: {len(context_text)} chars")

    print(f"[*] Loading Questions from {args.question_file}...")
    questions = load_json_data(args.question_file)
    print(f"[*] Found {len(questions)} questions.")

    # 2. Initialize Model & Tokenizer
    print(f"[*] Initializing vLLM Offline Engine (TP={TENSOR_PARALLEL_SIZE})...")
    # Note: enable_prefix_caching might help even in offline mode if we re-submit the same prefix.
    # However, passing token_ids explicitly bypasses the tokenizer bottleneck effectively.
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=1000000,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # --- Load Existing Results (Resume Logic) ---
    existing_results = []
    if os.path.exists(args.output_file):
        print(f"[*] Found existing output file: {args.output_file}")
        try:
            with open(args.output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"[*] Loaded {len(existing_results)} existing results. Resuming...")
        except json.JSONDecodeError:
            print("[!] Existing file is corrupted or empty. Starting from scratch.")
    
    processed_ids = {item.get("question_id") for item in existing_results if "question_id" in item}
    # Fallback if question_id missing? Use index? Better to rely on QID.
    # Questions usually have 'question_id'.
    
    questions_to_process = []
    for q in questions:
        if q.get("question_id") in processed_ids:
            continue
        questions_to_process.append(q)
        
    if not questions_to_process:
        print("[*] All questions already processed. Exiting.")
        return 

    print(f"[*] Remaining questions to process: {len(questions_to_process)}")
    
    # 3. Pre-Tokenize Context
    # We construct the prompt as: Header + Context + Footer(Question)
    # To optimize, we tokenize Header + Context ONCE.
    print("[*] Pre-tokenizing Context (This may take a minute on CPU)...")
    start_tok = time.time()
    
    # Kimi-Linear format: Just text concatenation
    # We need to be careful about strict template matching if the model relies on it.
    # But for now, we follow the previous prompt structure.
    
    prefix_text = QA_TEMPLATE_HEADER + context_text
    prefix_ids = tokenizer.encode(prefix_text)
    
    tok_duration = time.time() - start_tok
    print(f"[+] Tokenization Complete. {len(prefix_ids)} tokens. Time: {tok_duration:.2f}s")
    

    # 4. Inference Loop
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    
    print("[*] Starting Offline QA Phase...")
    start_time = time.time()
    
    # We process sequentially or in batches? 
    # Since we need to save incrementally, let's process concurrently but keep track.
    # Actually, vLLM .generate() is blocking for the whole list.
    # To save incrementally, we should maybe pass chunks?
    # Or just save at the end? 
    # If we want to be safe against crashes, we should batch manually, e.g. 10 at a time?
    # Given the slowness (2 mins/req), processing 1 by 1 or 5 by 5 is better.

    current_results = existing_results # Start with what we have

    BATCH_SIZE = 1 # Process one by one effectively to allow saving? 
    # vLLM is efficient with batches, but if Pre-fill is the bottleneck (100%), batching 5 might mean waiting 10 mins for 5 results.
    # Let's do a small chunk size.
    
    CHUNK_SIZE = 5
    
    # Batch the remaining questions
    for i in range(0, len(questions_to_process), CHUNK_SIZE):
        chunk = questions_to_process[i : i + CHUNK_SIZE]
        
        chunk_prompts = []
        for q_item in chunk:
            question_text = q_item.get("query", "")
            suffix_text = QA_TEMPLATE_FOOTER.format(question=question_text)
            suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
            full_ids = prefix_ids + suffix_ids
            chunk_prompts.append({"prompt_token_ids": full_ids})
        
        print(f"[*] Submitting chunk {i//CHUNK_SIZE + 1} ({len(chunk)} reqs) to vLLM...")
        outputs = llm.generate(prompts=chunk_prompts, sampling_params=sampling_params)
        
        for j, output in enumerate(outputs):
            prediction = output.outputs[0].text
            result_item = chunk[j].copy()
            result_item["prediction"] = prediction
            current_results.append(result_item)
            
        # Save after every chunk
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(current_results, f, indent=2)
        print(f"[+] Saved {len(current_results)} total results.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"[+] QA Complete in {duration:.2f}s.")

if __name__ == "__main__":
    main()
