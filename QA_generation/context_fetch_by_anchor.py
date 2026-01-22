# fetch_context_by_anchor.py
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def fetch_context_by_anchor(
    anchor: str,
    atoms: Iterable[Dict[str, Any]],
    schema: Dict[str, Any],
) -> List[Any]:
    targets: List[str] = []
    seen = set()
    for atom in atoms:
        if atom.get("anchor") != anchor:
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
