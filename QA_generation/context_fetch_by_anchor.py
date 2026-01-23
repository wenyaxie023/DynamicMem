# fetch_context_by_anchor.py
from typing import Any, Dict, Iterable, List, Optional


# =========================
# Time window helpers
# =========================

def _time_window_rank(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value == "init":
        return 0
    if value.startswith("w") and value[1:].isdigit():
        return int(value[1:])
    return None


# =========================
# Schema path resolution
# =========================

def _get_by_path(schema: Dict[str, Any], path: str) -> Optional[Any]:
    if not path:
        return None
    cur: Any = schema
    for seg in [s for s in path.split(".") if s]:
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list):
            if not seg.isdigit():
                return None
            idx = int(seg)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
        if cur is None:
            return None
    return cur


# =========================
# Core context fetcher
# =========================

def fetch_context_by_anchor(
    *,
    anchor: str,
    atoms: Iterable[Dict[str, Any]],
    schema: Dict[str, Any],
    max_window: Optional[str] = None,
) -> List[Any]:
    """
    Collect context objects from schema for atoms sharing the same anchor.

    If max_window is provided, only atoms whose time_window rank
    is <= max_window will be considered.
    """

    max_rank: Optional[int] = (
        _time_window_rank(max_window) if max_window is not None else None
    )

    targets: List[str] = []
    seen = set()

    for atom in atoms:
        if atom.get("anchor") != anchor:
            continue

        # --- time window filtering ---
        if max_rank is not None:
            atom_rank = _time_window_rank(atom.get("time_window"))
            if atom_rank is None or atom_rank > max_rank:
                continue

        target = atom.get("anchor_target")
        if not target or not isinstance(target, str):
            continue

        if target in seen:
            continue

        seen.add(target)
        targets.append(target)

    contexts: List[Any] = []
    for target in targets:
        value = _get_by_path(schema, target)
        if value is None:
            continue
        contexts.append(value)

    return contexts
