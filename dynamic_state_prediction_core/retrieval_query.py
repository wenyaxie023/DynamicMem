"""Shared retrieval-query builder for retrieval-based DSP baselines."""

from typing import Any, Dict, List


def _flatten_snapshot(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in snapshot.keys()):
        return {str(k): v for k, v in snapshot.items()}
    flat: Dict[str, Any] = {}
    for category, content in snapshot.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def _collect_leaf_paths(value: Any, prefix: str, out: List[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _collect_leaf_paths(v, f"{prefix}.{k}", out, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(value, list):
        if not value:
            out.append(prefix)
            return
        head = value[0]
        if isinstance(head, (dict, list)):
            _collect_leaf_paths(head, f"{prefix}[]", out, limit)
        else:
            out.append(prefix)
        return
    out.append(prefix)


def build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    """Build a consistent retrieval query text across retrieval-based baselines."""
    as_of = checkpoint.get("as_of") or {}
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])

    expected_snapshot = checkpoint.get("expected_snapshot_state") or {}
    flat_snapshot = _flatten_snapshot(expected_snapshot)
    target_values = {k: flat_snapshot[k] for k in target_keys if k in flat_snapshot}

    leaf_paths: List[str] = []
    for key, value in list(target_values.items())[:10]:
        _collect_leaf_paths(value, key, leaf_paths, limit=40)
        if len(leaf_paths) >= 40:
            break
    nested_hint = ", ".join(leaf_paths[:30])

    hints: List[str] = []
    if keys_hint:
        hints.append(keys_hint)
    if nested_hint:
        hints.append(nested_hint)
    merged_hint = ", ".join(hints)

    parts: List[str] = [
        f"Retrieve logs that help infer user state as of checkpoint time {ts}."
    ]
    if merged_hint:
        parts.append(f"Relevant target hints: {merged_hint}.")
    return " ".join(parts)
