"""Context builder: merge states/events/logs into raw.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import ContextConfig


def _load_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}. "
            "Run `python -m qa_context extract --user-id <id>` first, or pass --states/--events/--logs."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _load_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _stable_key(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iter_chains(schema: Dict[str, Any]) -> Iterable[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]]:
    for domain, domain_windows in schema.items():
        if not isinstance(domain_windows, list):
            continue
        for window in domain_windows:
            if not isinstance(window, dict):
                continue
            chains = window.get("event_chains", [])
            if not isinstance(chains, list):
                continue
            for chain in chains:
                if not isinstance(chain, dict):
                    continue
                state_refs = chain.get("state_refs", [])
                events = chain.get("events", [])
                if not isinstance(state_refs, list) or not isinstance(events, list):
                    continue
                states = [s for s in state_refs if isinstance(s, dict)]
                evs = [e for e in events if isinstance(e, dict)]
                yield str(domain), states, evs


def _build_relations(
    schema: Dict[str, Any],
    app_logs_final: Dict[str, Any],
) -> Tuple[
    Dict[str, Set[str]],  # state_key -> event_ids
    Dict[str, Set[str]],  # state_key -> state_names
    Dict[str, Set[str]],  # state_key -> domains
    Dict[str, Set[str]],  # event_id -> state_names
    Dict[str, Set[str]],  # event_id -> domains
    Dict[str, Set[str]],  # event_id -> log_ids
    Dict[str, str],       # event_key -> event_id
    Dict[str, str],       # log_id -> event_id
]:
    state_to_events: Dict[str, Set[str]] = {}
    state_to_names: Dict[str, Set[str]] = {}
    state_to_domains: Dict[str, Set[str]] = {}
    event_to_states: Dict[str, Set[str]] = {}
    event_to_domains: Dict[str, Set[str]] = {}
    event_key_to_id: Dict[str, str] = {}

    for domain, states, events in _iter_chains(schema):
        event_ids = []
        for event in events:
            event_id = event.get("event_id")
            if isinstance(event_id, str) and event_id:
                event_ids.append(event_id)
                event_key_to_id[_stable_key(event)] = event_id

        chain_state_names: Set[str] = set()
        for state in states:
            name = state.get("state_name")
            if isinstance(name, str) and name:
                chain_state_names.add(name)

        for state in states:
            state_key = _stable_key(state)
            state_to_events.setdefault(state_key, set()).update(event_ids)
            state_to_names.setdefault(state_key, set())
            state_to_domains.setdefault(state_key, set()).add(domain)
            name = state.get("state_name")
            if isinstance(name, str) and name:
                state_to_names[state_key].add(name)

        for event_id in event_ids:
            event_to_states.setdefault(event_id, set()).update(chain_state_names)
            event_to_domains.setdefault(event_id, set()).add(domain)

    event_to_logs: Dict[str, Set[str]] = {}
    log_to_event: Dict[str, str] = {}
    app_logs = app_logs_final.get("app_logs", [])
    if isinstance(app_logs, list):
        for item in app_logs:
            if not isinstance(item, dict):
                continue
            log_id = item.get("app_log_id")
            event_id = item.get("event_id")
            if isinstance(log_id, str) and log_id and isinstance(event_id, str) and event_id:
                log_to_event[log_id] = event_id
                event_to_logs.setdefault(event_id, set()).add(log_id)

    return (
        state_to_events,
        state_to_names,
        state_to_domains,
        event_to_states,
        event_to_domains,
        event_to_logs,
        event_key_to_id,
        log_to_event,
    )


def _as_sorted_list(values: Set[str]) -> List[str]:
    return sorted(values)


def build_raw_store(
    *,
    states: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    idx = 1
    for item in states:
        raw.append({"id": idx, "from": "state", "context": item})
        idx += 1
    for item in events:
        raw.append({"id": idx, "from": "event", "context": item})
        idx += 1
    for item in logs:
        raw.append({"id": idx, "from": "log", "context": item})
        idx += 1
    return raw


def enrich_raw_store(
    *,
    raw_items: List[Dict[str, Any]],
    schema: Dict[str, Any],
    app_logs_final: Dict[str, Any],
) -> List[Dict[str, Any]]:
    (
        state_to_events,
        state_to_names,
        state_to_domains,
        event_to_states,
        event_to_domains,
        event_to_logs,
        event_key_to_id,
        log_to_event,
    ) = _build_relations(schema, app_logs_final)

    enriched: List[Dict[str, Any]] = []
    for item in raw_items:
        source = item.get("from")
        context = item.get("context")
        if not isinstance(context, dict):
            enriched.append(item)
            continue

        state_names: Set[str] = set()
        event_ids: Set[str] = set()
        log_ids: Set[str] = set()
        categories: Set[str] = set()

        if source == "state":
            state_key = _stable_key(context)
            state_names.update(state_to_names.get(state_key, set()))
            event_ids.update(state_to_events.get(state_key, set()))
            categories.update(state_to_domains.get(state_key, set()))
            for event_id in event_ids:
                log_ids.update(event_to_logs.get(event_id, set()))
        elif source == "event":
            event_id = context.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                event_id = event_key_to_id.get(_stable_key(context), "")
            if event_id:
                event_ids.add(event_id)
                state_names.update(event_to_states.get(event_id, set()))
                log_ids.update(event_to_logs.get(event_id, set()))
                categories.update(event_to_domains.get(event_id, set()))
        elif source == "log":
            log_id = context.get("app_log_id")
            if isinstance(log_id, str) and log_id:
                log_ids.add(log_id)
                event_id = log_to_event.get(log_id)
                if event_id:
                    event_ids.add(event_id)
                    state_names.update(event_to_states.get(event_id, set()))
                    categories.update(event_to_domains.get(event_id, set()))

        enriched.append(
            {
                **item,
                "category": _as_sorted_list(categories),
                "state_names": _as_sorted_list(state_names),
                "event_ids": _as_sorted_list(event_ids),
                "app_log_ids": _as_sorted_list(log_ids),
            }
        )
    return enriched


def write_raw_store(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def build_context_store(
    *,
    states_path: Path,
    events_path: Path,
    logs_path: Path,
    schema_path: Path,
    app_logs_final_path: Path,
    output_path: Path,
) -> int:
    states = _load_list(states_path)
    events = _load_list(events_path)
    logs = _load_list(logs_path)
    schema = _load_object(schema_path)
    app_logs_final = _load_object(app_logs_final_path)

    items = build_raw_store(states=states, events=events, logs=logs)
    items = enrich_raw_store(
        raw_items=items,
        schema=schema,
        app_logs_final=app_logs_final,
    )
    write_raw_store(output_path, items)
    return len(items)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build raw.json from tables")
    parser.add_argument("--states", type=str, default=None)
    parser.add_argument("--events", type=str, default=None)
    parser.add_argument("--logs", type=str, default=None)
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--app-logs-final", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--user-id", type=int, default=None)
    args = parser.parse_args()

    config = (
        ContextConfig(user_id=args.user_id)
        if args.user_id is not None
        else ContextConfig()
    )
    states_path = Path(args.states) if args.states else config.states_path
    events_path = Path(args.events) if args.events else config.events_path
    logs_path = Path(args.logs) if args.logs else config.logs_path
    schema_path = Path(args.schema) if args.schema else config.schema_path
    app_logs_final_path = (
        Path(args.app_logs_final) if args.app_logs_final else config.app_logs_final_path
    )
    output_path = Path(args.output) if args.output else config.raw_path

    count = build_context_store(
        states_path=states_path,
        events_path=events_path,
        logs_path=logs_path,
        schema_path=schema_path,
        app_logs_final_path=app_logs_final_path,
        output_path=output_path,
    )
    print(f"Raw items: {count}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
