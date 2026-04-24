from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def _path_has_existing_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file() or path.is_symlink():
        return True
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            return True
        if child.is_dir() and _path_has_existing_artifact(child):
            return True
    return False


def existing_artifact_details(paths: Optional[Iterable[Path]] = None) -> List[str]:
    details: List[str] = []
    for raw_path in paths or ():
        path = Path(raw_path)
        if not _path_has_existing_artifact(path):
            continue
        if path.is_dir():
            details.append("{} (non-empty directory)".format(path))
        else:
            details.append("{} (existing file)".format(path))
    return details


def ensure_destructive_rebuild_allowed(
    *,
    allow_destructive_rebuild: bool,
    operation: str,
    paths: Optional[Sequence[Path]] = None,
    details: Optional[Iterable[str]] = None,
) -> None:
    evidence = existing_artifact_details(paths)
    for item in details or ():
        text = str(item or "").strip()
        if text:
            evidence.append(text)
    if allow_destructive_rebuild or not evidence:
        return
    preview = "; ".join(evidence[:4])
    if len(evidence) > 4:
        preview = "{}; ...".format(preview)
    raise RuntimeError(
        "Refusing to {} because existing memory artifacts were found: {}. "
        "Set runtime.allow_destructive_rebuild=true to allow this destructive rebuild.".format(
            str(operation or "perform a destructive rebuild"),
            preview,
        )
    )
