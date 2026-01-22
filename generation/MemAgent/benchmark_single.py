import json
import argparse
import time
import sys
import threading
import subprocess
import os
import asyncio
import aiohttp
from openai import OpenAI
from transformers import AutoTokenizer

# Global flag to stop monitoring
stop_monitoring = False
max_vram_usage = []

# MemAgent Cores
RECURRENT_CHUNK_SIZE = 5000
RECURRENT_MAX_NEW = 1024
NO_MEMORY = "No previous memory"

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

def monitor_vram():
    """Polls nvidia-smi every 1 second to get memory usage."""
    global stop_monitoring, max_vram_usage
    peak = 0
    while not stop_monitoring:
        try:
            # Query memory.used for all GPUs
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            usages = [int(x) for x in result.strip().split('\n')]
            current_total = sum(usages)
            if current_total > peak:
                peak = current_total
        except Exception as e:
            pass 
        time.sleep(1)
    max_vram_usage = [peak]

async def async_query_llm(context_text, prompt, model_name, tokenizer, url, api_key):
    # Recurrent Loop Implementation
    input_ids = tokenizer.encode(context_text)
    
    # NO LONGER CLIPPING. We want to test the full length as per MemAgent's capability.

    memory = NO_MEMORY
    
    token_count = 0 
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
        # 1. Processing Chunks
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i : i + RECURRENT_CHUNK_SIZE]
            chunk_text = tokenizer.decode(chunk)
            
            msg = TEMPLATE.format(prompt=prompt, chunk=chunk_text, memory=memory)
            
            try:
                async with session.post(
                    url=url + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": msg}], 
                        "temperature": 0.7, 
                        "top_p": 0.95, 
                        "max_tokens": RECURRENT_MAX_NEW
                    },
                ) as resp:
                    if resp.status != 200:
                        print(f"Error {resp.status} in chunk processing")
                        return None, token_count
                    
                    data = await resp.json()
                    memory = data["choices"][0]["message"]["content"]
                    
                    # Approximate output tokens
                    # token_count += len(tokenizer.encode(memory)) # Costly?
                    token_count += 0 # Don't count intermediate for now, or just track final?
                    
            except Exception as e:
                print(f"Exception during chunking: {e}")
                return None, token_count

        # 2. Final Answer
        msg = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        try:
            async with session.post(
                url=url + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name, 
                    "messages": [{"role": "user", "content": msg}], 
                    "temperature": 0.7, 
                    "top_p": 0.95, 
                    "max_tokens": RECURRENT_MAX_NEW
                },
            ) as resp:
                if resp.status != 200:
                    print(f"Error {resp.status} in final answer")
                    return None, token_count
                
                data = await resp.json()
                final_ans = data["choices"][0]["message"]["content"]
                token_count += len(tokenizer.encode(final_ans))
                return final_ans, token_count
                
        except Exception as e:
            print(f"Exception during final answer: {e}")
            return None, token_count


def run_benchmark(data_path, port):
    global stop_monitoring, max_vram_usage
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract Context and Prompt from app_log format
    # app_log format: list of dicts. 
    # Usually history + current req.
    # MemAgent expects "context" and "input" (prompt).
    # We need to construct this from app_logs.
    # Approach:
    # Context = String dump of all history events? 
    # Prompt = The last event request?
    
    app_logs = data.get("app_logs", [])
    history = app_logs[:-1]
    current_event = app_logs[-1]
    
    # Serialize history as context string
    context_str = json.dumps(history, indent=2)
    prompt_str = json.dumps(current_event, indent=2)
    
    est_input_chars = len(context_str) + len(prompt_str)
    print(f"Input Length: {est_input_chars} chars")
    
    # Model Config
    MODEL_NAME = "BytedTsinghua-SIA/RL-MemoryAgent-7B"
    URL = f"http://localhost:{port}/v1"
    API_KEY = "EMPTY"
    
    # Initialize Tokenizer (needed for chunking)
    print("Initializing tokenizer...")
    try:
        # Use venv's transformers
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer from {MODEL_NAME}: {e}")
        # Fallback? Or fail.
        # Assuming model path is valid on disk via huggingface cache or similar.
        # Since we just ran vllm serve with this name, it should be in cache.
        sys.exit(1)

    # Start VRAM monitor
    monitor_thread = threading.Thread(target=monitor_vram)
    monitor_thread.start()
    
    results = {
        "factor_file": data_path,
        "input_chars": est_input_chars,
        "status": "Failed"
    }

    try:
        start_t = time.time()
        
        # Run Async Logic
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_ans, token_count = loop.run_until_complete(
            async_query_llm(context_str, prompt_str, MODEL_NAME, tokenizer, URL, API_KEY)
        )
        loop.close()
        
        end_t = time.time()
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        if final_ans:
            duration = end_t - start_t
            peak_vram = max(max_vram_usage) if max_vram_usage else 0
            
            print(f"Total Time: {duration:.4f} s")
            print(f"Generated Tokens (Final): ~{token_count}")
            print(f"Peak VRAM: {peak_vram} MiB")
            print(f"Final Answer Preview: {final_ans[:100]}...")
            
            results = {
                "factor_file": data_path,
                "total_time_s": duration,
                "generated_tokens": token_count, # Only final answer tokens counted accurately here
                "peak_vram_mib": peak_vram,
                "status": "Success",
                "response": final_ans
            }
        else:
            print("Failed to get response.")
            results["error"] = "Empty response from agent loop"

    except Exception as e:
        print(f"Benchmark Failed: {e}")
        stop_monitoring = True
        results["error"] = str(e)
        
    # Save results json
    with open(f"{data_path}.result.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    
    run_benchmark(args.data_path, args.port)
