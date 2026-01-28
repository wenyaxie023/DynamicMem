import os
import json

def main():
    print(f"{'User':<15} {'Processed':<10} {'Total':<10} {'Progress':<10}")
    print("-" * 45)

    base_dir = "outputs"
    if not os.path.exists(base_dir):
        print("Outputs directory not found.")
        return

    users = sorted([d for d in os.listdir(base_dir) if d.endswith("_large")])
    
    total_processed = 0
    total_events = 0

    for user_dir in users:
        user_name = user_dir.replace("_large", "")
        checkpoint_path = os.path.join(base_dir, user_dir, "indexing_checkpoint.json")
        data_path = f"../../user_data/{user_name}/app_log_large.json"
        
        processed = 0
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                try:
                    processed = json.load(f).get("last_processed_index", -1) + 1
                except: pass
        
        total = 0
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                         data = data.get('app_logs', data.get('events', []))
                    if hasattr(data, 'values') and not isinstance(data, list): # Handle dict of values case
                         data = list(data.values())
                    total = len(data)
                except: pass

        progress = (processed / total * 100) if total > 0 else 0
        print(f"{user_name:<15} {processed:<10} {total:<10} {progress:.1f}%")
        
        total_processed += processed
        total_events += total

    print("-" * 45)
    total_prog = (total_processed / total_events * 100) if total_events > 0 else 0
    print(f"{'TOTAL':<15} {total_processed:<10} {total_events:<10} {total_prog:.1f}%")

if __name__ == "__main__":
    main()
