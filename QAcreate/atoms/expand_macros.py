import argparse
from copy import deepcopy
from typing import Any, Dict, List

try:
    import yaml
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required to expand atom macros") from exc


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": rule["id"],
        "match": rule["match"],
        "emit": rule.get("emit", {}) or {},
    }


def expand_group(group: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_match = group["base_match"]
    base_emit = group.get("emit", {}) or {}
    expanded: List[Dict[str, Any]] = []
    for rule in group.get("rules", []) or []:
        path = rule.get("path", "")
        match = base_match if not path else f"{base_match}.{path}"
        emit = base_emit
        if rule.get("emit"):
            emit = deep_merge(base_emit, rule["emit"])
        expanded.append(
            {
                "id": rule["id"],
                "match": match,
                "emit": emit,
            }
        )
    return expanded


def expand_macros(data: Dict[str, Any]) -> Dict[str, Any]:
    rules = [normalize_rule(r) for r in (data.get("rules", []) or [])]
    for group in data.get("groups", []) or []:
        rules.extend(expand_group(group))
    return {"atoms": rules}


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand atom macros into atom_ir.yaml")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="atom_macros.yaml",
        help="Path to atom_macros.yaml",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="atom_ir.yaml",
        help="Path to atom_ir.yaml",
    )
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    expanded = expand_macros(data)
    with open(args.out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(expanded, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
