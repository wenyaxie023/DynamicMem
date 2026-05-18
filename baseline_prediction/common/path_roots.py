from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "results"


def repo_root() -> Path:
    return REPO_ROOT


def artifact_root() -> Path:
    return ARTIFACT_ROOT


def resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()
