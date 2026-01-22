import argparse
import json
from typing import Any, Dict, Iterable, List


def iter_state_refs(schema: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for domain_windows in schema.values():
        if not isinstance(domain_windows, list):
            continue
        for window in domain_windows:
            if not isinstance(window, dict):
                continue
            for chain in window.get("event_chains", []) or []:
                if not isinstance(chain, dict):
                    continue
                for ref in chain.get("state_refs", []) or []:
                    if isinstance(ref, dict):
                        yield ref


def collect_keys_by_category(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for ref in iter_state_refs(schema):
        category = ref.get("state_category") or "UNKNOWN"
        bucket = result.setdefault(
            category, {"keys": set(), "items": 0, "refs": 0}
        )
        bucket["refs"] += 1
        resolved_items = ref.get("resolved_items") or []
        if not isinstance(resolved_items, list):
            continue
        for item in resolved_items:
            if not isinstance(item, dict):
                continue
            bucket["items"] += 1
            for key in item.keys():
                bucket["keys"].add(key)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect resolved_items keys grouped by state_category."
    )
    parser.add_argument(
        "--schema",
        default="schema.json",
        help="Path to schema.json (default: schema.json)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for JSON summary (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    summary = collect_keys_by_category(schema)

    output: Dict[str, Any] = {}
    for category, info in summary.items():
        output[category] = {
            "refs": info["refs"],
            "items": info["items"],
            "keys": sorted(info["keys"]),
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
