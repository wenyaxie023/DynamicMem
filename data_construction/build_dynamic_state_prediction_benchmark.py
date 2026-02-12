#!/usr/bin/env python3
"""
Build dynamic-state-prediction checkpoints from app_logs_final.json.

This script creates evaluation checkpoints at chain-completion time points:
- snapshot_state: full reconstructed user state at this time point
- observability: evidence-count metadata per state item
"""

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def _parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {ts}") from exc


def _state_key(state_category: str, state_name: str) -> str:
    return f"{state_category}:{state_name}"


def _split_state_key(key: str) -> Tuple[str, str]:
    if ":" not in key:
        return "unknown_state_category", key
    return key.split(":", 1)


def _expand_snapshot(flat_snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    expanded: Dict[str, Dict[str, Any]] = {}
    for key, value in flat_snapshot.items():
        cat, name = _split_state_key(key)
        expanded.setdefault(cat, {})[name] = value
    return expanded


def _expand_observability(flat_obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    expanded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in flat_obs.items():
        cat, name = _split_state_key(key)
        expanded.setdefault(cat, {})[name] = value
    return expanded


def _normalize_evidence_field(field: str) -> str:
    if field == "new_value":
        return "current_value"
    if field.startswith("new_value."):
        return "current_value." + field[len("new_value.") :]
    if field == "delta.to":
        return "current_value"
    if field.startswith("delta.to."):
        return "current_value." + field[len("delta.to.") :]
    return field


def _path_is_covered(required_path: str, evidence_fields: Set[str]) -> bool:
    if "current_value" in evidence_fields or "new_value" in evidence_fields or "delta.to" in evidence_fields:
        return True
    for ef in evidence_fields:
        if required_path == ef:
            return True
        if required_path.startswith(ef + "."):
            return True
    return False


def _iter_event_chains(all_events_chains: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(all_events_chains, list):
        for item in all_events_chains:
            if isinstance(item, dict):
                if isinstance(item.get("event_chains"), list):
                    for c in item["event_chains"]:
                        if isinstance(c, dict):
                            yield c
                else:
                    yield item
        return

    if isinstance(all_events_chains, dict):
        for value in all_events_chains.values():
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    if isinstance(item.get("event_chains"), list):
                        for c in item["event_chains"]:
                            if isinstance(c, dict):
                                yield c
                    else:
                        yield item


def _extract_state_value_from_resolved_item(item: Dict[str, Any]) -> Any:
    if "current_value" in item:
        return item.get("current_value")
    if "new_value" in item:
        return item.get("new_value")
    delta = item.get("delta")
    if isinstance(delta, dict) and "to" in delta:
        return delta.get("to")
    if "previous_value" in item:
        return item.get("previous_value")
    return None


def _to_date_set(values: Any) -> Set[str]:
    if not isinstance(values, list):
        return set()
    out: Set[str] = set()
    for x in values:
        sx = str(x).strip()
        if sx:
            out.add(sx)
    return out


def _validate_state_ref_with_chain_events(
    state_ref: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    state_category = str(state_ref.get("state_category", "")).strip()
    state_name = str(state_ref.get("state_name", "")).strip()
    if not state_category or not state_name:
        return []

    evidence_fields_by_state_name: Dict[str, Set[str]] = defaultdict(set)
    event_schedule_dates_by_state_name: Dict[str, Set[str]] = defaultdict(set)

    for event in events:
        time_spec = event.get("time_specification") or {}
        event_dates = _to_date_set(time_spec.get("schedule_dates"))
        for e in event.get("evidence_for_states", []) or []:
            if not isinstance(e, dict):
                continue
            e_state_name = str(e.get("state_name", "")).strip()
            if not e_state_name:
                continue
            raw_fields = e.get("evidenced_fields", [])
            if isinstance(raw_fields, list):
                for f in raw_fields:
                    evidence_fields_by_state_name[e_state_name].add(_normalize_evidence_field(str(f)))
            if event_dates:
                event_schedule_dates_by_state_name[e_state_name] |= event_dates

    resolved_items = state_ref.get("resolved_items", [])
    if not isinstance(resolved_items, list):
        resolved_items = []

    outputs: List[Dict[str, Any]] = []
    for item in resolved_items:
        if not isinstance(item, dict):
            continue
        resolved_name = str(item.get("name", state_name)).strip() or state_name
        state_key = _state_key(state_category, state_name)
        reasons: List[str] = []

        required_fields_raw = item.get("required_observable_fields", [])
        required_fields = {
            _normalize_evidence_field(str(x))
            for x in required_fields_raw
            if isinstance(required_fields_raw, list)
        }
        covered_fields = evidence_fields_by_state_name.get(resolved_name, set())

        if required_fields and not all(_path_is_covered(rf, covered_fields) for rf in required_fields):
            reasons.append("required_observable_fields_not_fully_covered")

        value_obj = _extract_state_value_from_resolved_item(item)
        value_schedule_dates = set()
        if isinstance(value_obj, dict):
            maybe_dates = value_obj.get("schedule_dates")
            value_schedule_dates = _to_date_set(maybe_dates)

        event_schedule_dates = event_schedule_dates_by_state_name.get(resolved_name, set())
        if value_schedule_dates or event_schedule_dates:
            if value_schedule_dates != event_schedule_dates:
                reasons.append("schedule_dates_mismatch_with_event_union")

        outputs.append(
            {
                "state_key": state_key,
                "resolved_name": resolved_name,
                "is_valid": len(reasons) == 0,
                "reasons": reasons,
                "required_observable_fields": sorted(required_fields),
                "covered_evidence_fields": sorted(covered_fields),
                "resolved_schedule_dates": sorted(value_schedule_dates),
                "event_schedule_dates_union": sorted(event_schedule_dates),
            }
        )

    return outputs


def build_chain_state_validity(all_events_chains: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for chain in _iter_event_chains(all_events_chains):
        chain_id = str(chain.get("chain_id", "")).strip()
        if not chain_id:
            continue
        events = chain.get("events", [])
        if not isinstance(events, list):
            events = []
        state_refs = chain.get("state_refs", [])
        if not isinstance(state_refs, list):
            state_refs = []

        for state_ref in state_refs:
            if not isinstance(state_ref, dict):
                continue
            for result in _validate_state_ref_with_chain_events(state_ref, events):
                key = result["state_key"]
                prev = out[chain_id].get(key)
                if prev is None:
                    out[chain_id][key] = result
                    continue
                # Conservative merge: if any resolved item is invalid, mark invalid.
                merged = dict(prev)
                merged["is_valid"] = bool(prev.get("is_valid", False)) and bool(result.get("is_valid", False))
                merged["reasons"] = sorted(set(list(prev.get("reasons", [])) + list(result.get("reasons", []))))
                merged["required_observable_fields"] = sorted(
                    set(list(prev.get("required_observable_fields", [])) + list(result.get("required_observable_fields", [])))
                )
                merged["covered_evidence_fields"] = sorted(
                    set(list(prev.get("covered_evidence_fields", [])) + list(result.get("covered_evidence_fields", [])))
                )
                merged["resolved_schedule_dates"] = sorted(
                    set(list(prev.get("resolved_schedule_dates", [])) + list(result.get("resolved_schedule_dates", [])))
                )
                merged["event_schedule_dates_union"] = sorted(
                    set(list(prev.get("event_schedule_dates_union", [])) + list(result.get("event_schedule_dates_union", [])))
                )
                out[chain_id][key] = merged
    return out


def _sort_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(log: Dict[str, Any]) -> Tuple[datetime, str]:
        ts = log.get("timestamp")
        if not ts:
            # push no-timestamp logs to the end deterministically
            return datetime.max, str(log.get("app_log_id", ""))
        return _parse_timestamp(ts), str(log.get("app_log_id", ""))

    return sorted(logs, key=sort_key)


def build_checkpoints(
    app_logs_final: Dict[str, Any],
    chain_state_validity: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    app_logs = _sort_logs(app_logs_final.get("app_logs", []))

    chain_last_index: Dict[str, int] = {}
    chain_ids_by_index: Dict[int, List[str]] = defaultdict(list)
    for idx, log in enumerate(app_logs):
        chain_id = (log.get("metadata") or {}).get("chain_id", "")
        if chain_id:
            chain_last_index[chain_id] = idx
    for chain_id, idx in chain_last_index.items():
        chain_ids_by_index[idx].append(chain_id)

    completion_indices = sorted(chain_ids_by_index.keys())

    current_state: Dict[str, Any] = {}
    observability: Dict[str, Dict[str, Any]] = {}
    state_provenance: Dict[str, Dict[str, Any]] = {}
    # Hold evidence until the corresponding chain is completed.
    pending_chain_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    checkpoints: List[Dict[str, Any]] = []
    skipped_checkpoints_no_valid_states = 0
    source_total_checkpoints = len(completion_indices)

    for idx, log in enumerate(app_logs):
        chain_id = (log.get("metadata") or {}).get("chain_id", "")
        if chain_id:
            for ev in log.get("golden_evidence", []):
                pending_chain_events[chain_id].append(
                    {
                        "event": ev,
                        "timestamp": log.get("timestamp"),
                        "app_log_id": log.get("app_log_id"),
                        "chain_id": chain_id,
                    }
                )

        # Apply evidence only when chain(s) are completed at this index.
        for completed_chain in chain_ids_by_index.get(idx, []):
            for wrapped in pending_chain_events.get(completed_chain, []):
                ev = wrapped.get("event") or {}
                state_category = ev.get("state_category")
                state_name = ev.get("state_name")
                if not state_category or not state_name:
                    continue
                key = _state_key(state_category, state_name)
                change_type = str(ev.get("change_type", "unchanged")).lower()
                raw_evidence_fields = ev.get("evidenced_fields", [])
                if not isinstance(raw_evidence_fields, list):
                    raw_evidence_fields = []
                evidence_fields = {_normalize_evidence_field(str(x)) for x in raw_evidence_fields}

                if change_type in {"drop", "remove", "deleted"}:
                    current_state.pop(key, None)
                    state_provenance.pop(key, None)
                elif "state_value" in ev:
                    current_state[key] = copy.deepcopy(ev["state_value"])
                    prev = state_provenance.get(key) or {}
                    prev_chain = str(prev.get("chain_id", ""))
                    if prev_chain == completed_chain:
                        merged_fields = set(prev.get("evidenced_fields", set())) | evidence_fields
                    else:
                        merged_fields = set(evidence_fields)
                    state_provenance[key] = {
                        "chain_id": completed_chain,
                        "evidenced_fields": merged_fields,
                    }

                info = observability.setdefault(key, {"evidence_count": 0})
                info["evidence_count"] += 1
                info["last_timestamp"] = wrapped.get("timestamp")
                info["last_app_log_id"] = wrapped.get("app_log_id")
                info["last_change_type"] = ev.get("change_type", "unchanged")
                evidence_ids = info.setdefault("evidence_app_log_ids", [])
                app_log_id = wrapped.get("app_log_id")
                if app_log_id is not None:
                    app_log_id = str(app_log_id)
                    if app_log_id and app_log_id not in evidence_ids:
                        evidence_ids.append(app_log_id)

        if idx not in completion_indices:
            continue

        valid_snapshot_flat: Dict[str, Any] = {}
        valid_observability_flat: Dict[str, Dict[str, Any]] = {}
        invalid_state_keys: List[str] = []
        for state_key, state_value in current_state.items():
            prov = state_provenance.get(state_key, {})
            provenance_chain_id = str(prov.get("chain_id", "")).strip()
            chain_states = chain_state_validity.get(provenance_chain_id, {})
            validity_detail = chain_states.get(state_key)
            valid = bool(validity_detail and validity_detail.get("is_valid"))
            if not valid:
                invalid_state_keys.append(state_key)
                continue

            valid_snapshot_flat[state_key] = copy.deepcopy(state_value)
            obs_info = copy.deepcopy(observability.get(state_key, {}))
            obs_info["is_valid"] = True
            obs_info["provenance_chain_id"] = provenance_chain_id
            obs_info["provenance_evidenced_fields"] = sorted(set(prov.get("evidenced_fields", set())))
            if validity_detail:
                obs_info["validation"] = {
                    "required_observable_fields": validity_detail.get("required_observable_fields", []),
                    "covered_evidence_fields": validity_detail.get("covered_evidence_fields", []),
                    "resolved_schedule_dates": validity_detail.get("resolved_schedule_dates", []),
                    "event_schedule_dates_union": validity_detail.get("event_schedule_dates_union", []),
                }
            valid_observability_flat[state_key] = obs_info

        if not valid_snapshot_flat:
            skipped_checkpoints_no_valid_states += 1
            continue

        snapshot = _expand_snapshot(valid_snapshot_flat)
        md = log.get("metadata") or {}

        checkpoints.append(
            {
                "checkpoint_id": f"cp_{len(checkpoints) + 1:04d}",
                "as_of": {
                    "log_index": idx,
                    "app_log_id": log.get("app_log_id"),
                    "timestamp": log.get("timestamp"),
                    "window_id": md.get("window_id"),
                    "domain": md.get("domain"),
                    "completed_chain_ids": sorted(chain_ids_by_index[idx]),
                },
                "expected_snapshot_state": snapshot,
                "state_observability": _expand_observability(valid_observability_flat),
                "validity": {
                    "valid_state_count": len(valid_snapshot_flat),
                    "invalid_state_count": len(invalid_state_keys),
                    "invalid_state_keys": sorted(invalid_state_keys),
                },
            }
        )

    return {
        "user_id": app_logs_final.get("user_id"),
        "source_total_app_logs": len(app_logs),
        "total_chains": len(chain_last_index),
        "source_total_checkpoints": source_total_checkpoints,
        "skipped_checkpoints_no_valid_states": skipped_checkpoints_no_valid_states,
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dynamic state prediction benchmark from app_logs_final.json."
    )
    parser.add_argument(
        "--app-logs-final",
        type=Path,
        required=True,
        help="Path to app_logs_final.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <same_dir>/dynamic_state_prediction_benchmark.json",
    )
    parser.add_argument(
        "--all-events-chains",
        type=Path,
        default=None,
        help="Path to all_events_chains.json. Defaults to <same_dir>/all_events_chains.json",
    )
    args = parser.parse_args()

    if not args.app_logs_final.exists():
        raise FileNotFoundError(f"File not found: {args.app_logs_final}")

    payload = json.loads(args.app_logs_final.read_text(encoding="utf-8"))
    all_events_chains_path = args.all_events_chains
    if all_events_chains_path is None:
        all_events_chains_path = args.app_logs_final.parent / "all_events_chains.json"
    if not all_events_chains_path.exists():
        raise FileNotFoundError(f"File not found: {all_events_chains_path}")
    all_events_chains = json.loads(all_events_chains_path.read_text(encoding="utf-8"))

    chain_state_validity = build_chain_state_validity(all_events_chains)
    benchmark = build_checkpoints(payload, chain_state_validity)

    output_path = args.output
    if output_path is None:
        output_path = args.app_logs_final.parent / "dynamic_state_prediction_benchmark.json"

    output_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved benchmark: {output_path}")
    print(
        "Summary: "
        f"user_id={benchmark.get('user_id')}, "
        f"checkpoints={benchmark.get('total_checkpoints')}, "
        f"source_checkpoints={benchmark.get('source_total_checkpoints')}, "
        f"skipped_no_valid={benchmark.get('skipped_checkpoints_no_valid_states')}, "
        f"chains={benchmark.get('total_chains')}"
    )


if __name__ == "__main__":
    main()
