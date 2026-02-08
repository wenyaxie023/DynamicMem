from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from shared.config import GenerationConfig


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str) and x]


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_edges_from_raw(raw_items: List[Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    event_to_logs: Dict[str, Set[str]] = {}
    log_to_events: Dict[str, Set[str]] = {}

    for item in raw_items:
        source = item.get("from")
        context = item.get("context")
        if not isinstance(context, dict):
            continue

        if source == "event":
            event_id = context.get("event_id")
            if isinstance(event_id, str) and event_id:
                app_log_ids = _normalize_str_list(item.get("app_log_ids"))
                event_to_logs.setdefault(event_id, set()).update(app_log_ids)
        elif source == "log":
            log_id = context.get("app_log_id")
            if isinstance(log_id, str) and log_id:
                event_ids = _normalize_str_list(item.get("event_ids"))
                log_to_events.setdefault(log_id, set()).update(event_ids)

    return event_to_logs, log_to_events


def remap_qa_links(
    *,
    raw_items: List[Dict[str, Any]],
    qa_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    event_to_logs, log_to_events = _build_edges_from_raw(raw_items)
    updated = 0

    for item in qa_items:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue

        qtype = metadata.get("qtype")
        if not isinstance(qtype, int):
            continue

        changed = False
        if qtype in {1, 2, 3, 4}:
            event_ids = _normalize_str_list(metadata.get("event_ids"))
            mapped_logs: Set[str] = set()
            for event_id in event_ids:
                mapped_logs.update(event_to_logs.get(event_id, set()))
            new_ref = sorted(mapped_logs)
            if new_ref != _normalize_str_list(metadata.get("reference_evidence")):
                metadata["reference_evidence"] = new_ref
                changed = True
        elif qtype in {5, 6}:
            log_ids = _normalize_str_list(metadata.get("reference_evidence"))
            mapped_events: Set[str] = set()
            for log_id in log_ids:
                mapped_events.update(log_to_events.get(log_id, set()))
            new_events = sorted(mapped_events)
            if new_events != _normalize_str_list(metadata.get("event_ids")):
                metadata["event_ids"] = new_events
                changed = True

        if changed:
            updated += 1

    return qa_items, updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-map QA event/log links from raw.json")
    parser.add_argument("--user-id", type=int, default=10)
    parser.add_argument("--qa", type=str, default=None, help="Override qa.json path")
    parser.add_argument("--raw", type=str, default=None, help="Override raw.json path")
    parser.add_argument("--output", type=str, default=None, help="Default: qa_remapped.json")
    parser.add_argument("--inplace", action="store_true", help="Write back to qa.json")
    args = parser.parse_args()

    config = GenerationConfig(user_id=args.user_id)
    qa_path = Path(args.qa) if args.qa else config.qa_output_path
    raw_path = Path(args.raw) if args.raw else config.raw_path
    output_path = qa_path if args.inplace else (Path(args.output) if args.output else qa_path.with_name("qa_remapped.json"))

    raw_items = _read_json_list(raw_path)
    qa_items = _read_json_list(qa_path)
    remapped, updated = remap_qa_links(raw_items=raw_items, qa_items=qa_items)
    _write_json(output_path, remapped)

    print(f"Raw: {raw_path}")
    print(f"Input QA: {qa_path}")
    print(f"Output QA: {output_path}")
    print(f"Updated items: {updated}/{len(remapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
