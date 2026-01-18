import argparse
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required to load atom_ir.yaml") from exc


def emit_signature(emit: Dict[str, Any]) -> str:
    return yaml.safe_dump(emit, sort_keys=True)


def common_prefix(parts_list: List[List[str]]) -> List[str]:
    if not parts_list:
        return []
    prefix = parts_list[0][:]
    for parts in parts_list[1:]:
        i = 0
        while i < min(len(prefix), len(parts)) and prefix[i] == parts[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def split_id_prefix(rule_id: str) -> str:
    return rule_id.split(".", 1)[0]


def to_macros(data: Dict[str, Any]) -> Dict[str, Any]:
    atoms = data.get("atoms", []) or []
    items = []
    for atom in atoms:
        items.append(
            {
                "id": atom["id"],
                "match": atom["match"],
                "emit": atom.get("emit", {}) or {},
                "id_prefix": split_id_prefix(atom["id"]),
                "emit_sig": emit_signature(atom.get("emit", {}) or {}),
            }
        )

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in items:
        key = (item["id_prefix"], item["emit_sig"])
        groups.setdefault(key, []).append(item)

    eligible_group: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, group_items in groups.items():
        if len(group_items) <= 1:
            continue
        parts_list = [g["match"].split(".") for g in group_items]
        prefix = common_prefix(parts_list)
        if not prefix:
            continue
        if any(len(parts) == len(prefix) for parts in parts_list):
            continue
        eligible_group[key] = {
            "name": key[0],
            "base_match": ".".join(prefix),
            "emit": group_items[0]["emit"],
            "rules": [],
        }

    rules_out: List[Dict[str, Any]] = []
    groups_out: List[Dict[str, Any]] = []
    seen_groups: Set[Tuple[str, str]] = set()

    for item in items:
        key = (item["id_prefix"], item["emit_sig"])
        if key in eligible_group:
            group = eligible_group[key]
            if key not in seen_groups:
                groups_out.append(group)
                seen_groups.add(key)
            base_parts = group["base_match"].split(".")
            match_parts = item["match"].split(".")
            remainder = match_parts[len(base_parts) :]
            group["rules"].append(
                {
                    "id": item["id"],
                    "path": ".".join(remainder),
                }
            )
        else:
            rules_out.append(
                {
                    "id": item["id"],
                    "match": item["match"],
                    "emit": item["emit"],
                }
            )

    return {
        "version": 1,
        "rules": rules_out,
        "groups": groups_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert atom_ir.yaml to atom_macros.yaml")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="atom_ir.yaml",
        help="Path to atom_ir.yaml",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="atom_macros.yaml",
        help="Path to atom_macros.yaml",
    )
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    macros = to_macros(data)
    with open(args.out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(macros, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
