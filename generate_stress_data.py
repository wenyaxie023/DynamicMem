import json

input_path = "app_log_518.json"
output_path = "app_log_stress_test.json"

with open(input_path, 'r') as f:
    data = json.load(f)

original_logs = data.get("app_logs", [])
print(f"Original logs count: {len(original_logs)}")

# Duplicate 5 times
stress_logs = []
for i in range(5):
    # We might want to update event_id to keep them unique, but for context length it doesn't matter much.
    # But let's keep them slightly distinct to avoid confusion if we care.
    # Actually, the user just wants to fill the context. Simple duplication is fine.
    # To be safe against deduplication mechanisms (unlikely in LLM context but possible), let's just append.
    stress_logs.extend(original_logs)

data["app_logs"] = stress_logs
data["total_available"] = len(stress_logs)
data["total_merged"] = len(stress_logs)

print(f"Stress logs count: {len(stress_logs)}")

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Saved stress data to {output_path}")
