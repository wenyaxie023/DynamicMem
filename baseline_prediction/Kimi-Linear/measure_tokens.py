
import json
import os
from transformers import AutoTokenizer

MODEL_PATH = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

def load_text_data(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    files = {
        "Small": "../../user_app_logs/003_user_003/app_log_small.json",
        "Medium": "../../user_app_logs/003_user_003/app_log_medium.json",
        "Large": "../../user_app_logs/003_user_003/app_log_large.json"
    }
    
    template_overhead = len(tokenizer.encode("Header...Footer...", add_special_tokens=False)) # Negligible
    
    print(f"{'Dataset':<10} | {'File Size (MB)':<15} | {'Chars':<15} | {'Tokens':<15}")
    print("-" * 65)
    
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"{name:<10} | Not Found")
            continue
            
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        text = load_text_data(path)
        chars = len(text)
        
        # Approximate check first to avoid long wait? No, accurate count needed.
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        
        print(f"{name:<10} | {file_size_mb:<15.2f} | {chars:<15} | {tokens:<15}")

if __name__ == "__main__":
    main()
