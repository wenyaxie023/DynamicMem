import os
import json
import time
import argparse
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
API_KEY = "EMPTY"
API_BASE = "http://localhost:8000/v1" # Kimi Linear default port
MODEL_NAME = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

# --- Template ---
# Standard RAG-like / Long Context prompt
QA_TEMPLATE = """You are a helpful assistant. I will provide you with a long user activity log. Please answer the question based on the log.

<activity_log>
{context}
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

def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=512 
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Kimi-Linear QA Benchmark")
    parser.add_argument("--log_file", required=True, help="Path to the raw app log JSON file")
    parser.add_argument("--question_file", required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", required=True, help="Path to save the results")
    
    args = parser.parse_args()

    print(f"[*] Loading Raw Logs from {args.log_file}...")
    context_text = load_text_data(args.log_file)
    print(f"[*] Context Length: {len(context_text)} chars")

    print(f"[*] Loading Questions from {args.question_file}...")
    questions = load_json_data(args.question_file)
    print(f"[*] Found {len(questions)} questions.")

    print(f"[*] Initializing Client...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    results = []
    
    print("[*] Starting QA Phase...")
    start_time = time.time()
    
    for q_item in tqdm(questions):
        question_text = q_item.get("query", "")
        # Construct prompt
        # Note: Kimi-Linear has huge context, so we just dump the whole log.
        prompt = QA_TEMPLATE.format(context=context_text, question=question_text)
        
        # Inference
        try:
            prediction = call_llm(client, prompt)
        except Exception as e:
            print(f"[!] Error answering: {e}")
            prediction = "ERROR"
            
        result_item = q_item.copy()
        result_item["model_prediction"] = prediction
        results.append(result_item)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[+] QA Complete in {duration:.2f}s.")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[SUCCESS] Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
