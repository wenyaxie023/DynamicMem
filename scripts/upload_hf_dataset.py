#!/usr/bin/env python3
"""Publish the DynamicMem benchmark data to the Hugging Face Hub.

Run this on a machine that can read the generated benchmark tree (e.g. the
cluster). For each user it publishes exactly two files — the canonical task
packs (renamed to `task_packs.json`) and the app-log stream — under the
`<model>/<user_id>/` layout the repo configs expect, so

    hf download xiewenya/dynamicmem --repo-type dataset --local-dir outputs/

reproduces a runnable `outputs/<model>/<user_id>/{task_packs.json,
app_log_large.json}` tree.

Safe by default: prints the manifest and does nothing. Add --push to upload
(one atomic commit).

Examples
--------
# See exactly what would be published (no network):
python scripts/upload_hf_dataset.py

# Actually upload (after `hf auth login`):
python scripts/upload_hf_dataset.py --push
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The generated benchmark tree on the maintainer's host.
DEFAULT_OUTPUTS_ROOT = (
    "/projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/"
    "data_construction/generated_outputs"
)
DEFAULT_REPO_ID = "xiewenya/dynamicmem"

# The canonical benchmark version to publish, and the clean name it ships under.
CANONICAL_TASK_PACK = "tce_benchmark_vnext_20260504v2_formal_task_packs_human_revised.json"
PUBLISHED_TASK_PACK = "task_packs.json"
APP_LOG = "app_log_large.json"


def build_manifest(outputs_root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return (uploads, warnings). uploads = [(local_path, path_in_repo), ...]."""
    uploads: list[tuple[Path, str]] = []
    warnings: list[str] = []
    for model_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        for user_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            rel = f"{model_dir.name}/{user_dir.name}"
            pack = user_dir / CANONICAL_TASK_PACK
            applog = user_dir / APP_LOG
            if not pack.is_file():
                warnings.append(f"missing task pack for {rel}: {CANONICAL_TASK_PACK}")
                continue
            if not applog.is_file():
                warnings.append(f"missing app log for {rel}: {APP_LOG}")
                continue
            uploads.append((pack, f"{rel}/{PUBLISHED_TASK_PACK}"))
            uploads.append((applog, f"{rel}/{APP_LOG}"))
    return uploads, warnings


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

    uploads, warnings = build_manifest(root)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if not uploads:
        print(f"ERROR: no publishable files found under {root}", file=sys.stderr)
        return 2

    total_mb = sum(p.stat().st_size for p, _ in uploads) / 1e6
    n_users = len({repo_path.rsplit('/', 1)[0] for _, repo_path in uploads})
    print(f"Repo:         {args.repo_id} (dataset)")
    print(f"Outputs root: {root}")
    print(f"Publishing {len(uploads)} file(s) for {n_users} user(s), {total_mb:.1f} MB:")
    for local, repo_path in uploads:
        rename = "  (renamed)" if local.name != repo_path.rsplit("/", 1)[1] else ""
        print(f"  {repo_path}{rename}")
    if args.card.is_file():
        print(f"Dataset card: {args.card} -> README.md")

    if not args.push:
        print("\nDRY RUN — nothing uploaded. Re-run with --push to publish.")
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub import CommitOperationAdd

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    ops = [
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local))
        for local, repo_path in uploads
    ]
    if args.card.is_file():
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(args.card)))

    print(f"\nUploading to https://huggingface.co/datasets/{args.repo_id} ...")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message="Publish DynamicMem benchmark (task packs + app logs)",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
