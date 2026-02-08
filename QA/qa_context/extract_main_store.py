from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import ContextConfig


@dataclass
class WindowInfo:
    domain: str
    window_id: str | None
    window_obj: Dict[str, Any]


def _iter_windows(schema: Dict[str, Any]) -> Iterable[WindowInfo]:
    for domain, domain_windows in schema.items():
        if not isinstance(domain_windows, list):
            continue
        for window in domain_windows:
            if not isinstance(window, dict):
                continue
            window_id = window.get("window_id") or window.get("time_window")
            yield WindowInfo(domain=str(domain), window_id=window_id, window_obj=window)


def _iter_chains(window: WindowInfo) -> Iterable[Tuple[int, Dict[str, Any]]]:
    chains = window.window_obj.get("event_chains", [])
    if not isinstance(chains, list):
        return
    for idx, chain in enumerate(chains):
        if isinstance(chain, dict):
            yield idx, chain


def extract_states(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for window in _iter_windows(schema):
        for _, chain in _iter_chains(window):
            state_refs = chain.get("state_refs", [])
            if not isinstance(state_refs, list):
                continue
            for state_ref in state_refs:
                if not isinstance(state_ref, dict):
                    continue
                records.append(state_ref)
    return records


def extract_events(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for window in _iter_windows(schema):
        for _, chain in _iter_chains(window):
            events = chain.get("events", [])
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                records.append(event)
    return records


def load_schema(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid schema file (expected object): {path}")
    return data


def load_logs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid app_log_large file (expected list): {path}")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract states/events/logs JSON from schema and app_log_large."
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to schema.json (default: config.schema_path)",
    )
    parser.add_argument(
        "--app-log-large",
        type=str,
        default=None,
        help="Path to app_log_large.json (default: config.data_dir/app_log_large.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: config.data_dir)",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="User id (default: config.user_id)",
    )
    args = parser.parse_args()

    config = (
        ContextConfig(user_id=args.user_id)
        if args.user_id is not None
        else ContextConfig()
    )
    schema_path = Path(args.schema) if args.schema else config.schema_path
    app_log_large_path = (
        Path(args.app_log_large)
        if args.app_log_large
        else config.data_dir / "app_log_large.json"
    )
    out_dir = Path(args.out_dir) if args.out_dir else config.data_dir

    schema = load_schema(schema_path)
    states = extract_states(schema)
    events = extract_events(schema)
    logs = load_logs(app_log_large_path)

    write_json(out_dir / "states.json", states)
    write_json(out_dir / "events.json", events)
    write_json(out_dir / "logs.json", logs)

    print(f"states: {len(states)}")
    print(f"events: {len(events)}")
    print(f"logs: {len(logs)}")
    print(f"output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

