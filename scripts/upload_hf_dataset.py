#!/usr/bin/env python3
"""Publish the DynamicMem benchmark data to the Hugging Face Hub.

Run this on a machine that can read the generated `outputs/` tree (e.g. the
cluster). It uploads ONLY the files a benchmark consumer needs — each user's
task packs and app-log stream — preserving the `<model>/<user_id>/...` layout
the configs expect, so `huggingface-cli download ... --local-dir outputs/`
reproduces a runnable tree.

Safe by default: prints the manifest and does nothing. Add --push to upload.

Examples
--------
# 1. See exactly what would be published (no network):
python scripts/upload_hf_dataset.py --outputs-root /path/to/outputs

# 2. Actually upload (after `huggingface-cli login`):
python scripts/upload_hf_dataset.py --outputs-root /path/to/outputs --push
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Files that make up the public benchmark. Patterns are relative to a
# <model>/<user_id>/ directory. Keep this conservative — everything else in
# outputs/ (intermediate generation artifacts, debug dumps) stays private.
INCLUDE_PATTERNS = [
    "*_task_packs.json",   # benchmark task packs (carry ground-truth state)
    "app_log_large.json",  # the activity stream baselines consume
]

DEFAULT_REPO_ID = "xiewenya/dynamicmem"
DEFAULT_OUTPUTS_ROOT = (
    "/projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/outputs"
)


def collect(outputs_root: Path) -> list[Path]:
    files: list[Path] = []
    for model_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        for user_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for pat in INCLUDE_PATTERNS:
                files.extend(sorted(user_dir.glob(pat)))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-root", default=DEFAULT_OUTPUTS_ROOT, type=Path)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--card", type=Path, default=Path(__file__).with_name("dataset_card.md"),
                    help="Markdown uploaded as the dataset README (the HF landing page).")
    ap.add_argument("--push", action="store_true", help="Actually upload (default: dry-run).")
    ap.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    args = ap.parse_args()

    root = args.outputs_root.resolve()
    if not root.is_dir():
        print(f"ERROR: outputs root not found: {root}", file=sys.stderr)
        return 2

    files = collect(root)
    if not files:
        print(f"ERROR: no benchmark files matched {INCLUDE_PATTERNS} under {root}", file=sys.stderr)
        return 2

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Repo:         {args.repo_id} (dataset)")
    print(f"Outputs root: {root}")
    print(f"Matched {len(files)} file(s), {total_mb:.1f} MB:")
    for f in files:
        print(f"  {f.relative_to(root)}")
    if args.card.is_file():
        print(f"Dataset card: {args.card}")

    if not args.push:
        print("\nDRY RUN — nothing uploaded. Re-run with --push to publish.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"\nUploading to https://huggingface.co/datasets/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=[f"*/*/{pat}" for pat in INCLUDE_PATTERNS],
        commit_message="Add DynamicMem benchmark task packs + app logs",
    )
    if args.card.is_file():
        api.upload_file(
            path_or_fileobj=str(args.card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
