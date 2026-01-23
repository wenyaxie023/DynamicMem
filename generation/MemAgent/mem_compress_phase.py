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

# --- Template ---
COMPRESS_TEMPLATE = """You are a helpful assistant to help me summarize the events log.
I will give you a list of events log, and you need to summarize the events log into a memory string.
The memory string should be concise and informative.
Your memory should be able to help me answer questions about the events log.
You can update your memory based on the new events log.
Here is the previous memory:
{memory}

Here is the new events log:
{chunk}

Please update the memory based on the new events log.
"""

def load_json_data(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

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
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024 
    )
    return response.choices[0].message.content

def stage_compress(client: OpenAI, tokenizer, context_text: str, chunk_size: int = 5000) -> tuple[str, float, int]:
    print(f"[*] Starting Compression Stage (Input len: {len(context_text)} chars)...")
    start_time = time.time()
    
    memory = "No previous memory."
    chunk_count = 0
    
    for chunk in generate_chunks(context_text, tokenizer, chunk_size):
        chunk_count += 1
        prompt = COMPRESS_TEMPLATE.format(memory=memory, chunk=chunk)
        memory = call_llm(client, prompt)
        # Optional: Print dynamic progress
        # print(f"    - Chunk {chunk_count} processed.")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"[+] Compression Complete. Processed {chunk_count} chunks in {duration:.2f}s.")
    return memory, duration, chunk_count

def main():
    parser = argparse.ArgumentParser(description="MemAgent Compression Phase")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--user_id", default="unknown")
    parser.add_argument("--dataset_size", default="unknown")
    
    args = parser.parse_args()

    print(f"[*] Initializing Client & Tokenizer...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    data = load_json_data(args.data_file)
    context_text = json.dumps(data, indent=2)

    final_memory, compress_time, chunks = stage_compress(client, tokenizer, context_text)

    result = {
        "user_id": args.user_id,
        "dataset_size": args.dataset_size,
        "compression_time_seconds": compress_time,
        "chunks_processed": chunks,
        "final_memory": final_memory
    }
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[SUCCESS] Memory saved to {args.output_file}")

if __name__ == "__main__":
    main()
