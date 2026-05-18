import json
import argparse
import time
import sys
import threading
import subprocess
from openai import OpenAI

# Global flag to stop monitoring
stop_monitoring = False
max_vram_usage = []

def monitor_vram():
    """Polls nvidia-smi every 1 second to get memory usage."""
    global stop_monitoring, max_vram_usage
    while not stop_monitoring:
        try:
            # Query memory.used for all GPUs
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            # result looks like:
            # 45000
            # 45000
            # ...
            usages = [int(x) for x in result.strip().split('\n')]
            current_total = sum(usages)
            max_vram_usage.append(current_total)
            
        except Exception as e:
            pass # ignore errors
        time.sleep(1)

def run_benchmark(data_path, port):
    global stop_monitoring, max_vram_usage
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    app_logs = data.get("app_logs", [])
    history = app_logs[:-1]
    current_event = app_logs[-1]
    
    messages = [
        {"role": "system", "content": "You are an intelligent operating system agent. Predict the response JSON."},
        {"role": "user", "content": f"History:\n{json.dumps(history)}\n\nReq:\n{json.dumps(current_event)}"}
    ]
    
    # Estimate input tokens broadly (chars / 4)
    prompt_str = messages[0]['content'] + messages[1]['content']
    est_input_chars = len(prompt_str)
    
    print(f"Running benchmark on {data_path}")
    print(f"Input Length: {est_input_chars} chars")
    
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

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
        
        # Use streaming to measure TTFT (Prefill) AND Token generation speed
        stream = client.chat.completions.create(
            model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
            messages=messages,
            max_tokens=50, # Generate enough to measure speed
            temperature=0.0,
            stream=True
        )
        
        first_token_time = None
        token_times = []
        token_count = 0
        
        for chunk in stream:
            now = time.time()
            content = chunk.choices[0].delta.content
            if content:
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                token_count += 1
        
        end_t = time.time()
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        # Metrics Calculation
        if first_token_time:
            prefill_time = first_token_time - start_t
        else:
            prefill_time = end_t - start_t # Fallback if no tokens
            
        # Decode Speed: (N-1) / (Time_Last - Time_First)
        if len(token_times) > 1:
            decode_duration = token_times[-1] - token_times[0]
            decode_speed = (len(token_times) - 1) / decode_duration
        else:
            decode_speed = 0
            
        peak_vram = max(max_vram_usage) if max_vram_usage else 0
        
        print(f"Prefill Time: {prefill_time:.4f} s")
        print(f"Generated {token_count} tokens")
        print(f"Decode Speed: {decode_speed:.2f} tokens/s")
        print(f"Peak VRAM (Total across GPUs): {peak_vram} MiB")

        # Get exact input tokens from usage if available? 
        # OpenAI stream chunk doesn't always contain usage stats unless requested with stream_options.
        # But vllm 0.6+ supports it. Let's try to assume we can't get it easily in stream 
        # without extra config, so we'll treat 'est_input_chars/4' as rough estimate 
        # OR just report time. Input tokens is constant for the file anyway.
        
        results = {
            "factor_file": data_path,
            "prefill_time_s": prefill_time,
            "decode_speed_tps": decode_speed,
            "generated_tokens": token_count,
            "peak_vram_mib": peak_vram,
            "status": "Success"
        }

    except Exception as e:
        print(f"Failed: {e}")
        stop_monitoring = True
        results["error"] = str(e)
        
    # Save results json
    with open(f"{data_path}.result.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    run_benchmark(args.data_path, args.port)
