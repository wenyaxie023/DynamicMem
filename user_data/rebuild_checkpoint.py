#!/usr/bin/env python3
"""
Script to rebuild checkpoint from debug app_log files.
This is useful when you need to rollback to a specific point.

Usage:
    python rebuild_checkpoint.py

This script will:
1. Load all app_log_*.json files from the debug directory
2. Rebuild chain_logs_history, processed_event_ids, app_states
3. Save a new checkpoint and intermediate logs file
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

# Add parent paths for imports
# Path: generated_outputs/002_user_002 -> need to go up 3 levels to data_construction
data_construction_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(data_construction_dir))

from app_system import AppRegistry

# Paths
output_dir = Path(__file__).parent
debug_dir = output_dir / "debug" / "stage3_app_logs"
checkpoint_path = output_dir / "stage3_checkpoint.json"
intermediate_path = output_dir / "app_logs_intermediate.json"

# User ID (adjust if needed)
user_id = "user_002"

print(f"Output dir: {output_dir}")
print(f"Debug dir: {debug_dir}")

# Load all app_log files
app_logs = []
for log_file in sorted(debug_dir.glob("app_log_log_*.json")):
    # Skip error files
    if log_file.name.startswith("error_"):
        continue
    try:
        with open(log_file) as f:
            log = json.load(f)
            app_logs.append(log)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")

# Sort by timestamp
app_logs.sort(key=lambda x: x.get("timestamp", "9999-12-31"))

print(f"Loaded {len(app_logs)} app logs")
if app_logs:
    print(f"First: {app_logs[0]['app_log_id']} - {app_logs[0]['timestamp']}")
    print(f"Last: {app_logs[-1]['app_log_id']} - {app_logs[-1]['timestamp']}")

# Build chain_logs_history
chain_logs_history = defaultdict(list)
for log in app_logs:
    chain_id = log.get("metadata", {}).get("chain_id", "")
    if chain_id:
        chain_logs_history[chain_id].append(log)

print(f"Built chain_logs_history with {len(chain_logs_history)} chains")

# Build processed_event_ids
processed_event_ids = []
seen_ids = set()
for log in app_logs:
    atomic_id = log.get("atomic_event_id") or log.get("event_id")
    if atomic_id and atomic_id not in seen_ids:
        processed_event_ids.append(atomic_id)
        seen_ids.add(atomic_id)

print(f"Built processed_event_ids with {len(processed_event_ids)} unique events")

# Find max app_log_counter
max_counter = 0
for log in app_logs:
    app_log_id = log.get("app_log_id", "")
    match = re.match(r"log_(\d+)$", app_log_id)
    if match:
        max_counter = max(max_counter, int(match.group(1)))

print(f"Max app_log_counter: {max_counter}")

# Build aggregate_usage
aggregate_usage = {}
for log in app_logs:
    app_log_id = log.get("app_log_id", "")
    usage = log.get("metadata", {}).get("llm_usage")
    if usage:
        aggregate_usage[app_log_id] = usage

# Rebuild app_states by replaying all logs
print("Rebuilding app_states by replaying logs...")
app_registry = AppRegistry()
for log in app_logs:
    app_name = log.get("app_name")
    api_name = log.get("api_name")
    if not app_name or not api_name:
        continue
    try:
        app = app_registry.get_app(app_name, user_id)
        event_stub = {
            "timestamp": log.get("timestamp"),
            "app_name": app_name,
            "api_name": api_name,
        }
        app.record_api_call(event_stub, log.get("request", {}), log.get("response", {}))
    except ValueError as e:
        print(f"  Warning: Could not get app {app_name}: {e}")
    except Exception as e:
        print(f"  Warning: Error replaying {app_name}.{api_name}: {e}")

# Collect app states
app_states = {}
for (app_name, uid), app in app_registry.apps.items():
    if uid == user_id:
        app_states[app_name] = deepcopy(app.state)

print(f"Rebuilt app_states for {len(app_states)} apps: {list(app_states.keys())}")

# last_processed_index: We set this to -1 to indicate we don't have a reliable index
# The resume logic will use processed_event_ids to skip already processed events
# This is slower but more reliable than trying to guess the index
last_processed_index = -1
print(f"last_processed_index set to {last_processed_index} (will rely on processed_event_ids for skipping)")

# Build checkpoint
checkpoint = {
    "last_processed_event_id": app_logs[-1].get("atomic_event_id", "") if app_logs else "",
    "last_processed_index": last_processed_index,
    "app_log_counter": max_counter,
    "chain_logs_history": dict(chain_logs_history),
    "app_states": app_states,
    "aggregate_usage": aggregate_usage,
    "processed_event_ids": processed_event_ids,
}

# Save checkpoint
with open(checkpoint_path, 'w') as f:
    json.dump(checkpoint, f, indent=2, ensure_ascii=False)
print(f"Saved checkpoint to {checkpoint_path}")

# Save intermediate logs
with open(intermediate_path, 'w') as f:
    json.dump({"app_logs": app_logs}, f, indent=2, ensure_ascii=False)
print(f"Saved intermediate logs to {intermediate_path}")

print("\n" + "="*60)
print("Done! You can now resume with resume_from_existing=True")
print("="*60)
print(f"\nCheckpoint summary:")
print(f"  - app_log_counter: {max_counter}")
print(f"  - processed_event_ids: {len(processed_event_ids)}")
print(f"  - chain_logs_history chains: {len(chain_logs_history)}")
print(f"  - app_states apps: {len(app_states)}")
print(f"  - last event: {app_logs[-1]['app_log_id']} @ {app_logs[-1]['timestamp']}" if app_logs else "  - No logs")
