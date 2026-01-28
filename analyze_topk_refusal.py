import json
import os

def analyze_refusal(file_path):
    if not os.path.exists(file_path):
        return "Not found", 0, 0
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    refusal_keywords = ["does not contain", "no evidence", "no logs", "no information", "not found", "insufficient evidence", "not mentioned", "no entries", "no records"]
    refusal_count = 0
    total = len(data)
    
    refused_ids = []
    for item in data:
        prediction = item.get("prediction", "").lower()
        if any(kw in prediction for kw in refusal_keywords):
            refusal_count += 1
            refused_ids.append(item.get("id"))
            
    return refusal_count, total, (refusal_count/total*100) if total > 0 else 0, refused_ids

paths = {
    "Top-K 5": "generation/HippoRAG2/hipporag_gpt5mini/results/001_user_001/prediction/qa_human_001.json",
    "Top-K 10": "generation/HippoRAG2/hipporag_gpt5mini_topk10/results/001_user_001/prediction/qa_human_001.json",
    "Top-K 20": "generation/HippoRAG2/hipporag_gpt5mini_topk20/results/001_user_001/prediction/qa_human_001.json"
}

print(f"{'Version':<10} | {'Refusals':<10} | {'Total':<10} | {'Rate (%)':<10}")
print("-" * 50)
for label, path in paths.items():
    res, total, rate, ids = analyze_refusal(path)
    print(f"{label:<10} | {res:<10} | {total:<10} | {rate:<10.2f} | {ids}")
