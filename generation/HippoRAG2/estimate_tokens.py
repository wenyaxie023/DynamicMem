
import os
import re
import glob

def parse_logs(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    total_prompt = 0
    total_completion = 0
    total_requests = 0
    
    user_stats = {}

    # Regex to extract tokens
    # Example line: [TIMING] openai_call model=gpt-5-mini duration_s=22.129 prompt_tokens=1687 completion_tokens=1678
    pattern = re.compile(r"prompt_tokens=(\d+)\s+completion_tokens=(\d+)")

    print(f"{'User':<15} {'Requests':<10} {'Prompt':<15} {'Completion':<15} {'Total Tokens':<15}")
    print("-" * 75)

    for log_file in sorted(log_files):
        filename = os.path.basename(log_file)
        user_id = filename.replace(".log", "")
        
        u_prompt = 0
        u_completion = 0
        u_requests = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "openai_call" in line:
                    u_requests += 1
                    match = pattern.search(line)
                    if match:
                        p = int(match.group(1))
                        c = int(match.group(2))
                        u_prompt += p
                        u_completion += c
        
        user_stats[user_id] = {'requests': u_requests, 'prompt': u_prompt, 'completion': u_completion}
        total_prompt += u_prompt
        total_completion += u_completion
        total_requests += u_requests
        
        print(f"{user_id:<15} {u_requests:<10,} {u_prompt:<15,} {u_completion:<15,} {u_prompt + u_completion:<15,}")

    print("-" * 75)
    print(f"{'TOTAL':<15} {total_requests:<10,} {total_prompt:<15,} {total_completion:<15,} {total_prompt + total_completion:<15,}")
    
    # Estimate Cost (Standard GPT-4o pricing as a proxy if gpt-5-mini is unknown, or just leave as tokens)
    # Assuming gpt-5-mini might be similar to 4o-mini
    # 4o-mini: Input $0.15 / 1M, Output $0.60 / 1M
    cost_input = (total_prompt / 1_000_000) * 0.15
    cost_output = (total_completion / 1_000_000) * 0.60
    total_cost = cost_input + cost_output
    
    print(f"\n--- Estimated Cost (Hypothetical based on GPT-4o-mini rates) ---")
    print(f"Total Cost: ${total_cost:.4f}")

if __name__ == "__main__":
    parse_logs("logs_stable_4x")
