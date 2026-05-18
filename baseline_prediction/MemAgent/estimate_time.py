import os
import json
import glob

# --- Configuration ---
DATA_ROOT = "../../user_data"
QUESTIONS_ROOT = "../../questions"

# Benchmark result
BENCHMARK_SIZE_BYTES = 248536 # app_log_small.json
BENCHMARK_TIME_SEC = 275.40

def get_file_size(path):
    try:
        return os.path.getsize(path)
    except:
        return 0

def get_question_count(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return len(data)
    except:
        return 0

def estimate():
    print(f"[*] Starting Estimation...")
    print(f"    Baseline: {BENCHMARK_SIZE_BYTES/1024:.2f} KB -> {BENCHMARK_TIME_SEC:.2f}s per question")
    print(f"    Factor: {BENCHMARK_TIME_SEC / BENCHMARK_SIZE_BYTES:.6f} sec/byte")
    
    time_per_byte = BENCHMARK_TIME_SEC / BENCHMARK_SIZE_BYTES
    
    total_estimated_seconds = 0
    total_questions = 0
    
    # Mapping for question files
    # small -> qa_w0_with_app_logs.json
    # medium -> qa_w0_w1_with_app_logs.json
    # large -> qa_w0_w4_with_app_logs.json
    size_to_qfile = {
        "small": "qa_w0_with_app_logs.json",
        "medium": "qa_w0_w1_with_app_logs.json",
        "large": "qa_w0_w4_with_app_logs.json"
    }
    
    # Walk User dirs
    user_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*_user_*")))
    
    print(f"\n{'User':<15} | {'Size':<10} | {'Log Size (MB)':<15} | {'Questions':<10} | {'Time/Q (min)':<15} | {'Subtotal (hr)':<15}")
    print("-" * 100)
    
    for user_dir in user_dirs:
        user_id_raw = os.path.basename(user_dir) # 003_user_003
        
        # Find logs
        log_files = glob.glob(os.path.join(user_dir, "app_log_*.json"))
        
        for log_file in log_files:
            filename = os.path.basename(log_file)
            size_cate = filename.replace("app_log_", "").replace(".json", "") # small
            
            if size_cate not in size_to_qfile:
                continue
                
            # Get Log Size
            log_size = get_file_size(log_file)
            
            # Get Question Count
            # Question dir: ../../questions/003_user_003/
            q_file_path = os.path.join(QUESTIONS_ROOT, user_id_raw, size_to_qfile[size_cate])
            q_count = get_question_count(q_file_path)
            
            # Estimate
            # Time per Q = log_size * time_per_byte
            estimated_latency_per_q = log_size * time_per_byte
            total_time_for_file = estimated_latency_per_q * q_count
            
            total_estimated_seconds += total_time_for_file
            total_questions += q_count
            
            print(f"{user_id_raw:<15} | {size_cate:<10} | {log_size/1024/1024:<15.2f} | {q_count:<10} | {estimated_latency_per_q/60:<15.2f} | {total_time_for_file/3600:<15.2f}")

    print("-" * 100)
    print(f"Total Questions: {total_questions}")
    print(f"Total Estimated Time: {total_estimated_seconds/3600:.2f} hours ({total_estimated_seconds/3600/24:.2f} days)")

if __name__ == "__main__":
    estimate()
