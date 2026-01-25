import argparse
import subprocess
import time
import os
import signal
import sys
import threading
import requests
import shutil

# Configuration
LOG_DIR = "logs_stable_4x"
os.makedirs(LOG_DIR, exist_ok=True)
MONITOR_CONFIG = os.path.join(LOG_DIR, "monitor.conf")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def log(msg, port=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts} Port:{port}]" if port else f"[{ts} Manager]"
    print(f"{prefix} {msg}")
    sys.stdout.flush()

def check_port_ready(port, timeout=300):
    url = f"http://localhost:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False

def start_vllm(gpu_id, port, model):
    log_file = os.path.join(LOG_DIR, f"vllm_gpu{gpu_id}_port{port}.log")
    
    cmd = [
        "vllm", "serve", model,
        "--quantization", "fp8",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.7",
        "--port", str(port)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log(f"Starting vLLM on GPU {gpu_id}...", port)
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    log(f"vLLM started with PID {process.pid}", port)
    return process

def run_user_task(gpu_id, port, user_id, size):
    log_file = os.path.join(LOG_DIR, f"client_{user_id}_{size}.log")
    
    # Check if already done
    checkpoint_path = f"outputs/{user_id}_{size}/indexing_checkpoint.json"
    
    cmd = [
        "python3", "run_index_only.py",
        "--data_path", f"../../user_data/{user_id}/app_log_{size}.json",
        "--save_dir", f"outputs/{user_id}_{size}",
        "--reasoner_port", str(port)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Fix ModuleNotFoundError
    # Add Current Directory AND HippoRAG/src
    cwd = os.path.abspath(".")
    src_path = os.path.join(cwd, "HippoRAG", "src")
    
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    # Force fresh download to avoid permission/pickle issues
    env["HF_HOME"] = os.path.abspath("hf_cache")
    env["TRANSFORMERS_CACHE"] = env["HF_HOME"]
    env["SENTENCE_TRANSFORMERS_HOME"] = env["HF_HOME"]
    
    log(f"Starting Task: {user_id} - {size}", port)
    log(f"PYTHONPATH: {env['PYTHONPATH']}", port)
    log(f"HF_HOME: {env['HF_HOME']}", port)
    
    cmd[0] = sys.executable # Use same interpreter
    
    with open(log_file, "a") as f: # Append mode to keep history
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    process.wait()
    if process.returncode == 0:
        log(f"Finished Task: {user_id} - {size}", port)
        return True
    else:
        log(f"FAILED Task: {user_id} - {size} (See {log_file})", port)
        return False

def main():
    parser = argparse.ArgumentParser(description="HippoRAG Process Manager")
    parser.add_argument("--action", choices=["start", "stop"], required=True)
    parser.add_argument("--gpu", type=int, help="GPU ID")
    parser.add_argument("--port", type=int, help="Port for vLLM")
    parser.add_argument("--users", type=str, help="Space-separated list of users (e.g. '005_user_005')")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.action == "start":
        if args.gpu is None or args.port is None or not args.users:
            print("Error: --gpu, --port, and --users are required for 'start' action.")
            sys.exit(1)
            
        users = args.users.split()
        
        # 1. Start vLLM
        vllm_proc = start_vllm(args.gpu, args.port, args.model)
        
        try:
            # 2. Wait for Readiness
            log("Waiting for vLLM ready...", args.port)
            if not check_port_ready(args.port):
                log("vLLM failed to become ready. Aborting.", args.port)
                vllm_proc.terminate()
                sys.exit(1)
            
            log("vLLM Ready.", args.port)
            
            # 3. Process Users (Optimization: Large First to populate cache)
            sizes = ["large", "small", "medium"]
            global_openie_file = f"openie_results_ner_{args.model.replace('/', '_')}.json"
            
            for user in users:
                for size in sizes:
                    # Check if file exists
                    data_file = f"../../user_data/{user}/app_log_{size}.json"
                    if not os.path.exists(data_file):
                        log(f"Warning: Data file not found: {data_file}", args.port)
                        continue
                        
                    # Optimization: If small/medium, try to reuse Large's OpenIE cache
                    if size in ["small", "medium"]:
                        large_dir = f"outputs/{user}_large"
                        target_dir = f"outputs/{user}_{size}"
                        large_cache = os.path.join(large_dir, global_openie_file)
                        target_cache = os.path.join(target_dir, global_openie_file)
                        
                        if os.path.exists(large_cache):
                            os.makedirs(target_dir, exist_ok=True)
                            if not os.path.exists(target_cache):
                                log(f"Optimizing: Copying OpenIE cache for {user} {size} from Large", args.port)
                                shutil.copy2(large_cache, target_cache)
                    
                    run_user_task(args.gpu, args.port, user, size)
            
            log("All assigned tasks completed.", args.port)
            
        finally:
            log("Shutting down vLLM...", args.port)
            vllm_proc.terminate()
            vllm_proc.wait()
            log("Exiting.", args.port)

if __name__ == "__main__":
    main()
