#!/usr/bin/env python3
"""
Build TCE benchmark from app_logs_final.json.

Pipeline:
1) Reconstruct chain-completion checkpoints with validity filtering.
2) Apply sampling strategy at benchmark-build stage (default: calendar quarterly).
"""

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
        has_schedule_dates_field = False
        if isinstance(value_obj, dict):
            has_schedule_dates_field = "schedule_dates" in value_obj
            maybe_dates = value_obj.get("schedule_dates")
            value_schedule_dates = _to_date_set(maybe_dates)

        event_schedule_dates = event_schedule_dates_by_state_name.get(resolved_name, set())
        # Only enforce schedule_dates consistency when the resolved item
        # explicitly defines schedule_dates (typically time-anchored habits).
        if has_schedule_dates_field:
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


def build_chain_change_reasons(all_events_chains: Any) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for chain in _iter_event_chains(all_events_chains):
        chain_id = str(chain.get("chain_id", "")).strip()
        if not chain_id:
            continue
        state_refs = chain.get("state_refs", [])
        if not isinstance(state_refs, list):
            state_refs = []
        for state_ref in state_refs:
            if not isinstance(state_ref, dict):
                continue
            state_category = str(state_ref.get("state_category", "")).strip()
            state_name = str(state_ref.get("state_name", "")).strip()
            if not state_category or not state_name:
                continue
            reason_text = ""
            for item in state_ref.get("resolved_items", []) or []:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("change_reason", "")).strip()
                if candidate:
                    reason_text = candidate
            if reason_text:
                out[chain_id][_state_key(state_category, state_name)] = reason_text
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
    chain_change_reasons: Optional[Dict[str, Dict[str, str]]] = None,
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
    skipped_checkpoints_no_new_valid_observation = 0
    source_total_checkpoints = len(completion_indices)
    last_exported_snapshot_flat: Dict[str, Any] = {}

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
                change_reason = str(
                    ((chain_change_reasons or {}).get(completed_chain, {}) or {}).get(key, "")
                ).strip()
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
                if change_reason:
                    info["last_change_reason"] = change_reason
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

        # Keep checkpoints only when there is new evaluable information.
        # If valid snapshot is identical to the last exported checkpoint,
        # this chain completion does not change prediction targets.
        if checkpoints and valid_snapshot_flat == last_exported_snapshot_flat:
            skipped_checkpoints_no_new_valid_observation += 1
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
        last_exported_snapshot_flat = copy.deepcopy(valid_snapshot_flat)

    return {
        "user_id": app_logs_final.get("user_id"),
        "source_total_app_logs": len(app_logs),
        "total_chains": len(chain_last_index),
        "source_total_checkpoints": source_total_checkpoints,
        "skipped_checkpoints_no_valid_states": skipped_checkpoints_no_valid_states,
        "skipped_checkpoints_no_new_valid_observation": skipped_checkpoints_no_new_valid_observation,
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


def _normalize_app_logs_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    elif isinstance(payload, list):
        logs = payload
    else:
        logs = []
    return _sort_logs([x for x in logs if isinstance(x, dict)])


def _parse_exposure_anchors(raw: str) -> List[int]:
    out: List[int] = []
    for x in str(raw or "").split(","):
        sx = str(x).strip()
        if not sx:
            continue
        out.append(int(sx))
    return out


def _build_sampled_checkpoints(
    *,
    benchmark_path: Path,
    app_logs_large: List[Dict[str, Any]],
    sampling_mode: str,
    calendar_anchor_freq: str,
    exposure_anchors: List[int],
    tokenizer_model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    # Lazy import to avoid unnecessary dependency cost when sampling_mode=none.
    from tce_core.exposure_checkpoint_builder import (
        build_calendar_anchor_checkpoints,
        build_token_exposure_checkpoints,
    )

    mode = str(sampling_mode or "none").strip().lower()
    if mode == "none":
        return [], {}, {"mode": "none"}
    if mode == "calendar":
        cps, meta = build_calendar_anchor_checkpoints(
            benchmark_path=benchmark_path,
            app_logs_large=app_logs_large,
            calendar_anchor_freq=calendar_anchor_freq,
            tokenizer_model=tokenizer_model,
        )
        return (
            cps,
            meta,
            {
                "mode": "calendar_time",
                "calendar_anchor_freq": str(calendar_anchor_freq),
                "tokenizer_model": str(tokenizer_model),
            },
        )
    if mode == "exposure":
        if not exposure_anchors:
            raise ValueError("sampling_mode=exposure requires --exposure-anchors")
        cps, meta = build_token_exposure_checkpoints(
            benchmark_path=benchmark_path,
            app_logs_large=app_logs_large,
            exposure_anchors=exposure_anchors,
            tokenizer_model=tokenizer_model,
        )
        return (
            cps,
            meta,
            {
                "mode": "exposure_token",
                "exposure_anchors": exposure_anchors,
                "tokenizer_model": str(tokenizer_model),
            },
        )
    raise ValueError("sampling_mode must be one of: none|calendar|exposure")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TCE benchmark from app_logs_final.json."
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
        help="Output path. Defaults to <same_dir>/tce_benchmark.json",
    )
    parser.add_argument(
        "--all-events-chains",
        type=Path,
        default=None,
        help="Path to all_events_chains.json. Defaults to <same_dir>/all_events_chains.json",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="calendar",
        help="none|calendar|exposure. Default=calendar (prebuild sampled checkpoints).",
    )
    parser.add_argument(
        "--calendar-anchor-freq",
        type=str,
        default="quarterly",
        help="weekly|monthly|quarterly for sampling_mode=calendar.",
    )
    parser.add_argument(
        "--exposure-anchors",
        type=str,
        default="",
        help="Comma-separated exposure percents for sampling_mode=exposure, e.g. 10,20,30.",
    )
    parser.add_argument(
        "--app-logs-large",
        type=Path,
        default=None,
        help="Path to app_log_large.json used by prebuild sampling. Defaults to sibling app_log_large.json, fallback app_logs_final.",
    )
    parser.add_argument(
        "--sampling-tokenizer-model",
        type=str,
        default="gpt-4o-mini",
        help="Tokenizer model used for token counting on sampled checkpoints.",
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
    chain_change_reasons = build_chain_change_reasons(all_events_chains)
    benchmark = build_checkpoints(payload, chain_state_validity, chain_change_reasons)

    output_path = args.output
    if output_path is None:
        output_path = args.app_logs_final.parent / "tce_benchmark.json"

    raw_total_checkpoints = int(benchmark.get("total_checkpoints", 0))

    # Always write base payload first so downstream sampling builders can
    # reuse the same benchmark_path and preserve consistent lookup semantics.
    output_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sampling_mode = str(args.sampling_mode or "none").strip().lower()
    if sampling_mode != "none":
        app_logs_large_path: Optional[Path] = args.app_logs_large
        if app_logs_large_path is None:
            default_large = args.app_logs_final.parent / "app_log_large.json"
            app_logs_large_path = default_large if default_large.exists() else args.app_logs_final
        if not app_logs_large_path.exists():
            raise FileNotFoundError(f"File not found: {app_logs_large_path}")
        app_logs_large_payload = json.loads(app_logs_large_path.read_text(encoding="utf-8"))
        app_logs_large = _normalize_app_logs_payload(app_logs_large_payload)
        exposure_anchors = _parse_exposure_anchors(args.exposure_anchors)
        sampled_checkpoints, sampled_meta, sampled_cfg = _build_sampled_checkpoints(
            benchmark_path=output_path,
            app_logs_large=app_logs_large,
            sampling_mode=sampling_mode,
            calendar_anchor_freq=str(args.calendar_anchor_freq or "quarterly").strip().lower(),
            exposure_anchors=exposure_anchors,
            tokenizer_model=str(args.sampling_tokenizer_model or "gpt-4o-mini"),
        )
        benchmark["checkpoints"] = sampled_checkpoints
        benchmark["total_checkpoints"] = len(sampled_checkpoints)
        benchmark["sampling_strategy"] = {
            "stage": "benchmark_build",
            "config": sampled_cfg,
            "source_total_checkpoints_before_sampling": raw_total_checkpoints,
            "total_checkpoints_after_sampling": len(sampled_checkpoints),
        }
        for cp in benchmark.get("checkpoints", []):
            if not isinstance(cp, dict):
                continue
            cid = str(cp.get("checkpoint_id", ""))
            cmeta = sampled_meta.get(cid) if isinstance(sampled_meta, dict) else None
            if not isinstance(cmeta, dict):
                continue
            cp["sampling"] = {
                "mode": str(cmeta.get("sampling_mode", "")),
                "params": dict(cmeta.get("sampling_params") or {}),
            }

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
        f"skipped_no_new_valid_obs={benchmark.get('skipped_checkpoints_no_new_valid_observation')}, "
        f"chains={benchmark.get('total_chains')}, "
        f"sampling_stage={((benchmark.get('sampling_strategy') or {}).get('stage') if isinstance(benchmark.get('sampling_strategy'), dict) else 'none')}"
    )


if __name__ == "__main__":
    main()
