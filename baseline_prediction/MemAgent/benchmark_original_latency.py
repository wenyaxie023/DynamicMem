import os
import json
import time
import argparse
from typing import Generator
from openai import OpenAI
from transformers import AutoTokenizer

# --- Configuration ---
API_KEY = "EMPTY"  
API_BASE = "http://localhost:8003/v1"
MODEL_NAME = "BytedTsinghua-SIA/RL-MemoryAgent-7B"

# --- Template from quickstart.py (The Original Logic) ---
TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

def generate_chunks(text: str, tokenizer, chunk_size: int = 5000) -> Generator[str, None, None]:
    """Splits text into chunks based on token count."""
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i : i + chunk_size]
        yield tokenizer.decode(chunk_ids)

def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, # Greedy for benchmark stability
        top_p=1.0,
        max_tokens=1024 
    )
    return response.choices[0].message.content

def run_benchmark(file_path):
    print(f"[*] Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    context_text = json.dumps(data, indent=2)
    
    print(f"[*] Initializing Client & Tokenizer...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Dummy Question
    question = "What is the user's name and primary occupation?"
    print(f"[*] running Original Logic Benchmark...")
    print(f"    Target File: {os.path.basename(file_path)}")
    print(f"    Question: {question}")
    print(f"    Context Length: {len(context_text)} chars")
    
    start_time = time.time()
    
    memory = "No previous memory"
    chunk_count = 0
    
    # 1. Recurrent Pass
    print("[*] Starting Recurrent Pass...")
    for chunk in generate_chunks(context_text, tokenizer, chunk_size=5000):
        chunk_count += 1
        prompt = TEMPLATE.format(prompt=question, memory=memory, chunk=chunk)
        
        chunk_start = time.time()
        memory = call_llm(client, prompt)
        chunk_duration = time.time() - chunk_start
        
        print(f"    - Chunk {chunk_count}: {chunk_duration:.2f}s")
        
    # 2. Final Answer
    print("[*] Generating Final Answer...")
    final_prompt = TEMPLATE_FINAL.format(prompt=question, memory=memory)
    _ = call_llm(client, final_prompt)
    
    total_time = time.time() - start_time
    print(f"\n[SUMMARY]")
    print(f"Total Time for 1 Question: {total_time:.2f}s")
    print(f"Chunks Processed: {chunk_count}")
    print(f"Average Time per Chunk: {total_time/chunk_count if chunk_count else 0:.2f}s")

if __name__ == "__main__":
    # Use the small logs for the test
    target_file = "../../user_data/003_user_003/app_log_small.json"
    if os.path.exists(target_file):
        run_benchmark(target_file)
    else:
        print(f"Error: {target_file} not found.")
