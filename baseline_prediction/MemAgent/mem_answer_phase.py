import os
import json
import time
import argparse
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
API_KEY = "EMPTY"
API_BASE = "http://localhost:8003/v1"
MODEL_NAME = "BytedTsinghua-SIA/RL-MemoryAgent-7B"

# --- Template (Inspired by MemAgent source template commonalities) ---
# Using a clear instruction format similar to standard RAG
ANSWER_TEMPLATE = """You are presented with a question and a previous memory. Please answer the question based on the previous memory.

<memory>
{memory}
</memory>

Question: {question}
Answer:"""

def load_json_data(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, # Greedy for answering to be deterministic
        top_p=1.0, 
        max_tokens=512 
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="MemAgent Answering Phase")
    parser.add_argument("--memory_file", required=True, help="Path to the memory JSON file")
    parser.add_argument("--question_file", required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", required=True, help="Path to save the results")
    
    args = parser.parse_args()

    print(f"[*] Loading Memory from {args.memory_file}...")
    memory_data = load_json_data(args.memory_file)
    final_memory = memory_data.get("final_memory", "")
    
    if not final_memory:
        print("[!] WARNING: Final memory is empty!")

    print(f"[*] Loading Questions from {args.question_file}...")
    questions = load_json_data(args.question_file)
    print(f"[*] Found {len(questions)} questions.")

    print(f"[*] Initializing Client...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    results = []
    
    print("[*] Starting Answering Phase...")
    start_time = time.time()
    
    for q_item in tqdm(questions):
        question_text = q_item.get("query", "")
        # Construct prompt
        prompt = ANSWER_TEMPLATE.format(memory=final_memory, question=question_text)
        
        # Inference
        try:
            prediction = call_llm(client, prompt)
        except Exception as e:
            print(f"[!] Error answering question: {e}")
            prediction = "ERROR"
            
        # Save result (preserve original metadata)
        result_item = q_item.copy()
        result_item["model_prediction"] = prediction
        results.append(result_item)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[+] Answering Complete in {duration:.2f}s.")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[SUCCESS] Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
