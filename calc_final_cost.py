import json
import glob

def calculate_costs():
    total_p = 0
    total_c = 0
    count = 0
    files = glob.glob('generation/HippoRAG2/hipporag_gpt5mini/results/*/prediction/*.json')
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            for item in data:
                if not isinstance(item, dict):
                    continue
                m = item.get('metadata', {}).get('hipporag_metadata', {})
                total_p += m.get('prompt_tokens', 0)
                total_c += m.get('completion_tokens', 0)
                count += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Pricing from user
    # Input: $0.6 / 1M
    # Output: $2.0 / 1M
    price_p = (total_p / 1_000_000) * 0.6
    price_c = (total_c / 1_000_000) * 2.0
    
    print(f"Total Files: {len(files)}")
    print(f"Total Queries: {count}")
    print(f"Total Prompt Tokens: {total_p}")
    print(f"Total Completion Tokens: {total_c}")
    print(f"Avg Prompt/Query: {total_p/count if count > 0 else 0:.0f}")
    print(f"Avg Completion/Query: {total_c/count if count > 0 else 0:.0f}")
    print("-" * 30)
    print(f"Cost Input ($0.6/1M): ${price_p:.2f}")
    print(f"Cost Output ($2.0/1M): ${price_c:.2f}")
    print(f"Total Cost: ${price_p + price_c:.2f}")

if __name__ == "__main__":
    calculate_costs()
