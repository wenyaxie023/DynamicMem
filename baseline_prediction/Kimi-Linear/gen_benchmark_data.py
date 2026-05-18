import json
import os

BASE_FILE = "../../app_log_518.json"
OUTPUT_DIR = "benchmark_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(BASE_FILE, 'r') as f:
    base_data = json.load(f)

base_logs = base_data.get("app_logs", [])

factors = [1, 2, 3, 4]

for factor in factors:
    new_logs = []
    for _ in range(factor):
        new_logs.extend(base_logs)
    
    data = base_data.copy()
    data["app_logs"] = new_logs
    data["total_available"] = len(new_logs)
    data["total_merged"] = len(new_logs)
    
    filename = os.path.join(OUTPUT_DIR, f"app_log_{factor}x.json")
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Generated {filename} with {len(new_logs)} events.")
