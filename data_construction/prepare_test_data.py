"""
Prepare clean app_logs for LLM testing.

This script processes app_logs_final.json files and creates three versions:
- app_log_small.json: Only window 0 (w0)
- app_log_medium.json: Windows 0 and 1 (w0, w1)
- app_log_large.json: All windows

Each version contains only the clean app_log fields:
- app_log_id, timestamp, app_name, api_name, request, response
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


# Time window definitions (from stage1_dynamic_profile.py)
TIME_WINDOWS = {
    "w0": ("2023-10-01", "2023-12-31"),
    "w1": ("2024-01-01", "2024-03-31"),
    "w2": ("2024-04-01", "2024-07-01"),
    "w3": ("2024-07-02", "2024-09-30"),
    "w4": ("2024-10-01", "2024-12-31"),
}

# Fields to keep in clean app_logs
CLEAN_FIELDS = ["app_log_id", "timestamp", "app_name", "api_name", "request", "response"]


def get_window_id_from_timestamp(timestamp: str) -> str:
    """
    Determine window_id from timestamp.

    Args:
        timestamp: Timestamp string like "2023-10-01 05:15:00"

    Returns:
        Window ID (w0, w1, w2, w3, w4) or "unknown"
    """
    # Extract date part
    date_str = timestamp.split()[0] if " " in timestamp else timestamp[:10]

    for window_id, (start, end) in TIME_WINDOWS.items():
        if start <= date_str <= end:
            return window_id

    return "unknown"


def clean_app_log(app_log: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the clean fields from an app_log.

    Args:
        app_log: Original app_log with all fields

    Returns:
        Clean app_log with only essential fields
    """
    return {field: app_log[field] for field in CLEAN_FIELDS if field in app_log}


def filter_logs_by_windows(app_logs: List[Dict], target_windows: List[str]) -> List[Dict]:
    """
    Filter app_logs to only include those in the target windows.

    Args:
        app_logs: List of app_log entries
        target_windows: List of window IDs to include (e.g., ["w0", "w1"])

    Returns:
        Filtered list of clean app_logs
    """
    filtered = []
    for log in app_logs:
        # Get window_id from metadata if available, otherwise derive from timestamp
        if "metadata" in log and "window_id" in log["metadata"]:
            window_id = log["metadata"]["window_id"]
        else:
            window_id = get_window_id_from_timestamp(log["timestamp"])

        if window_id in target_windows:
            filtered.append(clean_app_log(log))

    return filtered


def process_user_data(input_path: Path, output_dir: Path) -> Dict[str, int]:
    """
    Process a single user's app_logs_final.json and create three versions.

    Args:
        input_path: Path to app_logs_final.json
        output_dir: Directory to write output files

    Returns:
        Dictionary with counts for each output file
    """
    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    app_logs = data.get("app_logs", [])

    # Define window sets for each version
    window_sets = {
        "app_log_small.json": ["w0"],
        "app_log_medium.json": ["w0", "w1"],
        "app_log_large.json": ["w0", "w1", "w2", "w3", "w4"],
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}

    for filename, windows in window_sets.items():
        filtered_logs = filter_logs_by_windows(app_logs, windows)

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_logs, f, indent=2, ensure_ascii=False)

        counts[filename] = len(filtered_logs)
        print(f"  {filename}: {len(filtered_logs)} logs (windows: {', '.join(windows)})")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean app_logs for LLM testing"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "generated_outputs",
        help="Directory containing model/user subdirectories with app_logs_final.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input user directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model directory to process (e.g., gemini_3_flash_preview)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Specific user directory to process (e.g., 001_user_001)",
    )

    args = parser.parse_args()

    # Find all app_logs_final.json files
    if args.model and args.user:
        # Process specific user
        input_path = args.input_dir / args.model / args.user / "app_logs_final.json"
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return

        output_dir = args.output_dir or input_path.parent
        print(f"Processing: {input_path}")
        import pdb; pdb.set_trace()
        process_user_data(input_path, output_dir)

    elif args.model:
        # Process all users under a specific model
        model_dir = args.input_dir / args.model
        if not model_dir.exists():
            print(f"Error: {model_dir} not found")
            return

        for user_dir in sorted(model_dir.iterdir()):
            if not user_dir.is_dir():
                continue

            input_path = user_dir / "app_logs_final.json"
            if not input_path.exists():
                continue

            output_dir = args.output_dir / user_dir.name if args.output_dir else user_dir
            print(f"Processing: {user_dir.name}")
            process_user_data(input_path, output_dir)

    else:
        # Process all models and users
        for model_dir in sorted(args.input_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            print(f"\nModel: {model_dir.name}")

            for user_dir in sorted(model_dir.iterdir()):
                if not user_dir.is_dir():
                    continue

                input_path = user_dir / "app_logs_final.json"
                if not input_path.exists():
                    continue

                output_dir = args.output_dir / model_dir.name / user_dir.name if args.output_dir else user_dir
                print(f"  Processing: {user_dir.name}")
                process_user_data(input_path, output_dir)


if __name__ == "__main__":
    main()
