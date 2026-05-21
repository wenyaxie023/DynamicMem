from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from .elite_persona_sampler import (
        DEFAULT_SAMPLE_PATH,
        DEFAULT_SEED,
        load_sampled_personas,
        sample_elite_personas,
    )
except ImportError:
    from trajectory_synthesis.elite_persona_sampler import (
        DEFAULT_SAMPLE_PATH,
        DEFAULT_SEED,
        load_sampled_personas,
        sample_elite_personas,
    )


DEFAULT_PERSONA_SAMPLE_SIZE = 10
DEFAULT_REPO_ID = "xiewenya/user-world-backgrounds"
DEFAULT_MODEL = "gemini_3_flash_preview"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _extract_user_description(persona: Dict[str, object]) -> str | None:
    persona_text = persona.get("persona") or persona.get("user_description")
    if isinstance(persona_text, str) and persona_text.strip():
        return persona_text
    return None


def _build_user_records(
    personas: Sequence[Dict[str, object]],
    *,
    max_users: int,
) -> List[Dict[str, str]]:
    users: List[Dict[str, str]] = []
    for idx, persona in enumerate(personas):
        if len(users) >= max_users:
            break
        user_description = _extract_user_description(persona)
        if not user_description:
            continue
        user_id = str(persona.get("user_id") or persona.get("id") or f"user_{idx + 1:03d}")
        users.append(
            {
                "user_id": user_id,
                "user_description": user_description,
            }
        )
    return users


def _load_or_sample_elite_personas(
    sample_path: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> List[Dict[str, object]]:
    sample_path = Path(sample_path)
    if not sample_path.exists():
        sample_elite_personas(
            output_path=sample_path,
            sample_size=sample_size,
            seed=seed,
        )
    return load_sampled_personas(sample_path)


def resolve_target_user_dirs(
    *,
    explicit_users: Sequence[str] | None,
    sample_path: str | Path,
    user_count: int,
    sample_size: int,
    seed: int,
    user_index: int | None,
) -> List[str]:
    if explicit_users:
        return [str(user).strip() for user in explicit_users if str(user).strip()]

    personas = _load_or_sample_elite_personas(
        sample_path=sample_path,
        sample_size=max(user_count, sample_size),
        seed=seed,
    )
    user_records = _build_user_records(personas, max_users=user_count)
    if len(user_records) < user_count:
        raise ValueError(
            f"Only found {len(user_records)} usable personas, expected {user_count}."
        )

    target_dirs = [
        f"{idx:03d}_{_slugify(record['user_id']) or f'user_{idx:03d}'}"
        for idx, record in enumerate(user_records, start=1)
    ]

    if user_index is not None:
        if user_index < 1 or user_index > len(target_dirs):
            raise ValueError(
                f"--user-index must be between 1 and {len(target_dirs)}"
            )
        return [target_dirs[user_index - 1]]

    return target_dirs


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)


def download_world_backgrounds(
    *,
    repo_id: str,
    model: str,
    output_root: Path,
    user_dirs: Sequence[str],
    force: bool = False,
) -> List[Path]:
    downloaded: List[Path] = []
    repo_root_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{model}"

    for user_dir in user_dirs:
        destination = output_root / model / user_dir / "world_backgrounds.json"
        if destination.exists() and not force:
            downloaded.append(destination)
            continue

        url = f"{repo_root_url}/{user_dir}/world_backgrounds.json"
        try:
            _download_file(url, destination)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(
                    f"No world_backgrounds.json found for {user_dir} in {repo_id}/{model}."
                ) from exc
            raise
        downloaded.append(destination)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download precomputed world_backgrounds.json artifacts from Hugging Face. "
            "If --users is omitted, this command mirrors the current persona-sampling "
            "selection used by batch_generation_runner.py."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model subdirectory inside the dataset repo",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs"),
        help="Local repo-root outputs/ root",
    )
    parser.add_argument(
        "--users",
        nargs="*",
        default=None,
        help="Explicit user directories such as 001_user_001 002_user_002",
    )
    parser.add_argument(
        "--user-count",
        type=int,
        default=10,
        help="Number of sampled users to mirror when --users is omitted",
    )
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="Optional 1-based sampled user index to mirror",
    )
    parser.add_argument(
        "--persona-sample-path",
        type=str,
        default=str(DEFAULT_SAMPLE_PATH),
        help="Path to the PersonaHub sample JSONL used by the runner",
    )
    parser.add_argument(
        "--persona-sample-size",
        type=int,
        default=DEFAULT_PERSONA_SAMPLE_SIZE,
        help="Persona sample size to use when the sample file does not yet exist",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Persona sampling seed to use when the sample file does not yet exist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local world_backgrounds.json files if they already exist",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the resolved user directories without downloading",
    )
    args = parser.parse_args()

    user_dirs = resolve_target_user_dirs(
        explicit_users=args.users,
        sample_path=args.persona_sample_path,
        user_count=args.user_count,
        sample_size=args.persona_sample_size,
        seed=args.persona_seed,
        user_index=args.user_index,
    )

    if args.list_only:
        print(json.dumps({"users": user_dirs}, indent=2))
        return

    downloaded = download_world_backgrounds(
        repo_id=args.repo_id,
        model=args.model,
        output_root=Path(args.output_root),
        user_dirs=user_dirs,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "model": args.model,
                "users": user_dirs,
                "downloaded_files": [str(path) for path in downloaded],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
