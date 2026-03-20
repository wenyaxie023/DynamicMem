"""Build sampled TCE checkpoints from exposure/calendar sampling strategies.

Terminology:
- sampling strategy: full selection rule (mode + params)
- sampling mode: strategy type, e.g. "exposure_token" / "calendar_time"
- sampling point: one concrete selected cutoff candidate (internal-only concept)
- checkpoint: final evaluation unit rebuilt at a selected cutoff
"""

import copy
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

from data_construction.build_tce_benchmark import (
    _expand_observability,
    _expand_snapshot,
    _normalize_evidence_field,
    _sort_logs,
    _state_key,
    build_chain_change_reasons,
    build_chain_state_validity,
)


def _log_token_text(log: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "app_log_id": log.get("app_log_id"),
            "timestamp": log.get("timestamp"),
            "app_name": log.get("app_name"),
            "api_name": log.get("api_name"),
            "request": log.get("request"),
            "response": log.get("response"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _encoding_for_model(model_name: str):
    if tiktoken is None:
        class _FallbackEncoding:
            @staticmethod
            def encode(text: str) -> List[int]:
                # Deterministic fallback when tiktoken is unavailable.
                return list(text.encode("utf-8"))

        return _FallbackEncoding()
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.max


def _add_months(dt: datetime, months: int) -> datetime:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=y, month=m, day=day)


def _build_token_anchor_specs(
    app_logs_large: List[Dict[str, Any]],
    exposure_anchors: Sequence[int],
    tokenizer_model: str,
) -> List[Dict[str, Any]]:
    if not app_logs_large:
        return []
    enc = _encoding_for_model(tokenizer_model)
    per_log_tokens: List[int] = []
    cumulative_tokens: List[int] = []
    total_tokens = 0
    for log in app_logs_large:
        n = len(enc.encode(_log_token_text(log)))
        per_log_tokens.append(n)
        total_tokens += n
        cumulative_tokens.append(total_tokens)

    out: List[Dict[str, Any]] = []
    seen = set()
    for raw_pct in exposure_anchors:
        try:
            pct = max(0, min(100, int(raw_pct)))
        except Exception:
            continue
        target_tokens = int(round((pct / 100.0) * total_tokens))
        cut_idx = 0
        while cut_idx < len(cumulative_tokens) and cumulative_tokens[cut_idx] < target_tokens:
            cut_idx += 1
        if cut_idx >= len(cumulative_tokens):
            cut_idx = len(cumulative_tokens) - 1

        app_log_id = str(app_logs_large[cut_idx].get("app_log_id", "")).strip()
        key = (pct, app_log_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "anchor_percent": pct,
                "target_tokens": target_tokens,
                "total_tokens": total_tokens,
                "actual_tokens_at_cutoff": cumulative_tokens[cut_idx],
                "cutoff_log_tokens": per_log_tokens[cut_idx],
                "tokenizer_model": tokenizer_model,
                "cutoff_large_log_index": cut_idx,
                "cutoff_app_log_id": app_log_id,
            }
        )
    out.sort(key=lambda x: int(x["anchor_percent"]))
    return out

def _build_calendar_anchor_specs(
    app_logs_large: List[Dict[str, Any]],
    calendar_anchor_freq: str,
    tokenizer_model: str,
) -> List[Dict[str, Any]]:
    if not app_logs_large:
        return []
    enc = _encoding_for_model(tokenizer_model)
    per_log_tokens: List[int] = []
    cumulative_tokens: List[int] = []
    total_tokens = 0
    for log in app_logs_large:
        n = len(enc.encode(_log_token_text(log)))
        per_log_tokens.append(n)
        total_tokens += n
        cumulative_tokens.append(total_tokens)
    freq = str(calendar_anchor_freq or "weekly").strip().lower()
    if freq not in {"weekly", "monthly", "quarterly"}:
        freq = "weekly"

    rows: List[Tuple[int, datetime, Dict[str, Any]]] = []
    for i, log in enumerate(app_logs_large):
        ts = _parse_ts(str(log.get("timestamp", "")))
        if ts == datetime.max:
            continue
        rows.append((i, ts, log))
    if not rows:
        return []
    rows.sort(key=lambda x: (x[1], x[0]))

    start_ts = rows[0][1]
    end_ts = rows[-1][1]
    anchors: List[datetime] = []
    cur = start_ts
    while cur <= end_ts:
        anchors.append(cur)
        if freq == "weekly":
            cur = cur + timedelta(days=7)
        elif freq == "monthly":
            cur = _add_months(cur, 1)
        else:
            cur = _add_months(cur, 3)
    if not anchors or anchors[-1] < end_ts:
        anchors.append(end_ts)

    out: List[Dict[str, Any]] = []
    seen_ids = set()
    for i, anchor_ts in enumerate(anchors, start=1):
        prior = [r for r in rows if r[1] <= anchor_ts]
        if prior:
            cut_idx, cut_ts, cut_log = prior[-1]
        else:
            cut_idx, cut_ts, cut_log = rows[0]
        app_log_id = str(cut_log.get("app_log_id", "")).strip()
        if app_log_id in seen_ids:
            continue
        seen_ids.add(app_log_id)
        out.append(
            {
                "anchor_index": i,
                "calendar_anchor_freq": freq,
                "anchor_timestamp": anchor_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "cutoff_large_log_index": cut_idx,
                "cutoff_app_log_id": app_log_id,
                "cutoff_timestamp": cut_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "actual_tokens_at_cutoff": cumulative_tokens[cut_idx],
                "total_tokens": total_tokens,
                "cutoff_log_tokens": per_log_tokens[cut_idx],
                "tokenizer_model": tokenizer_model,
            }
        )
    return out


def _map_anchors_to_final_indices(
    *,
    benchmark_path: Path,
    anchors: List[Dict[str, Any]],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Map internal sampling points to app_logs_final indices via app_log_id."""
    if not anchors:
        return []
    app_logs_final_path = benchmark_path.parent / "app_logs_final.json"
    app_logs_final_payload = json.loads(app_logs_final_path.read_text(encoding="utf-8"))
    app_logs_final = _sort_logs(app_logs_final_payload.get("app_logs", []))

    final_idx_by_log_id: Dict[str, int] = {}
    for i, log in enumerate(app_logs_final):
        app_log_id = str(log.get("app_log_id", "")).strip()
        if app_log_id and app_log_id not in final_idx_by_log_id:
            final_idx_by_log_id[app_log_id] = i

    export_items: List[Tuple[int, Dict[str, Any]]] = []
    for anchor in anchors:
        app_log_id = str(anchor.get("cutoff_app_log_id", "")).strip()
        idx = final_idx_by_log_id.get(app_log_id)
        if idx is not None:
            export_items.append((idx, anchor))
    return export_items


def _rebuild_checkpoints_from_export_items(
    *,
    benchmark_path: Path,
    export_items: List[Tuple[int, Dict[str, Any]]],
    sampling_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Rebuild final checkpoints at selected cutoffs.

    Notes:
    - `export_items` carries selected sampling points (idx + sampling metadata).
    - Returned `meta_by_checkpoint_id` only exposes strategy-oriented fields.
    """
    if not export_items:
        return [], {}

    user_dir = benchmark_path.parent
    app_logs_final_path = user_dir / "app_logs_final.json"
    all_events_chains_path = user_dir / "all_events_chains.json"
    app_logs_final_payload = json.loads(app_logs_final_path.read_text(encoding="utf-8"))
    all_events_chains = json.loads(all_events_chains_path.read_text(encoding="utf-8"))
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    chain_state_validity = build_chain_state_validity(all_events_chains)
    chain_change_reasons = build_chain_change_reasons(all_events_chains)

    app_logs_final = _sort_logs(app_logs_final_payload.get("app_logs", []))
    source_cp_rq3_by_ts: List[Tuple[datetime, Dict[str, Any]]] = []
    source_cp_questionability_by_ts: List[Tuple[datetime, Dict[str, Any]]] = []
    for cp in benchmark_payload.get("checkpoints", []) if isinstance(benchmark_payload, dict) else []:
        if not isinstance(cp, dict):
            continue
        rq3 = cp.get("rq3_apply_service_qa")
        if not isinstance(rq3, dict):
            rq3 = {}
        sq = cp.get("state_questionability")
        if not isinstance(sq, dict):
            sq = {}
        ts = str((cp.get("as_of") or {}).get("timestamp", "")).strip()
        if not ts:
            continue
        parsed_ts = _parse_ts(ts)
        source_cp_rq3_by_ts.append((parsed_ts, rq3))
        source_cp_questionability_by_ts.append((parsed_ts, sq))
    source_cp_rq3_by_ts.sort(key=lambda x: x[0])
    source_cp_questionability_by_ts.sort(key=lambda x: x[0])

    def _lookup_rq3_pack_by_cutoff_ts(cutoff_ts: str) -> Dict[str, Any]:
        """Pick nearest source rq3 apply pack at or before cutoff timestamp."""
        if not source_cp_rq3_by_ts:
            return {}
        target = _parse_ts(str(cutoff_ts or ""))
        chosen: Optional[Dict[str, Any]] = None
        for ts, payload in source_cp_rq3_by_ts:
            if ts <= target:
                chosen = payload
            else:
                break
        if chosen is None:
            chosen = source_cp_rq3_by_ts[0][1]
        return copy.deepcopy(chosen or {})

    def _lookup_questionability_by_cutoff_ts(cutoff_ts: str) -> Dict[str, Any]:
        if not source_cp_questionability_by_ts:
            return {}
        target = _parse_ts(str(cutoff_ts or ""))
        chosen: Optional[Dict[str, Any]] = None
        for ts, payload in source_cp_questionability_by_ts:
            if ts <= target:
                chosen = payload
            else:
                break
        if chosen is None:
            chosen = source_cp_questionability_by_ts[0][1]
        return copy.deepcopy(chosen or {})

    export_items.sort(key=lambda x: x[0])
    export_indices = sorted({x[0] for x in export_items})
    anchor_by_export_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, anchor in export_items:
        anchor_by_export_idx[idx].append(anchor)

    chain_last_index: Dict[str, int] = {}
    chain_ids_by_index: Dict[int, List[str]] = defaultdict(list)
    for idx, log in enumerate(app_logs_final):
        chain_id = (log.get("metadata") or {}).get("chain_id", "")
        if chain_id:
            chain_last_index[str(chain_id)] = idx
    for chain_id, idx in chain_last_index.items():
        chain_ids_by_index[idx].append(chain_id)
    completion_indices = set(chain_ids_by_index.keys())

    current_state: Dict[str, Any] = {}
    observability: Dict[str, Dict[str, Any]] = {}
    state_provenance: Dict[str, Dict[str, Any]] = {}
    pending_chain_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    checkpoints: List[Dict[str, Any]] = []
    meta_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    export_set = set(export_indices)
    calendar_retained_index = 0

    for idx, log in enumerate(app_logs_final):
        chain_id = (log.get("metadata") or {}).get("chain_id", "")
        if chain_id:
            for ev in log.get("golden_evidence", []) or []:
                pending_chain_events[chain_id].append(
                    {
                        "event": ev,
                        "timestamp": log.get("timestamp"),
                        "app_log_id": log.get("app_log_id"),
                    }
                )

        for completed_chain in chain_ids_by_index.get(idx, []):
            for wrapped in pending_chain_events.get(completed_chain, []):
                ev = wrapped.get("event") or {}
                state_category = ev.get("state_category")
                state_name = ev.get("state_name")
                if not state_category or not state_name:
                    continue
                key = _state_key(str(state_category), str(state_name))
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

        if idx not in export_set:
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
            valid_observability_flat[state_key] = obs_info

        if not valid_snapshot_flat:
            continue

        md = log.get("metadata") or {}
        for sampling_point in anchor_by_export_idx.get(idx, []):
            if sampling_mode == "exposure_token":
                sampled_id = f"exp_{int(sampling_point['anchor_percent']):03d}"
            else:
                calendar_retained_index += 1
                sampled_id = (
                    f"cal_{str(sampling_point['calendar_anchor_freq'])}_"
                    f"{int(calendar_retained_index):03d}"
                )

            checkpoints.append(
                {
                    "checkpoint_id": sampled_id,
                    "as_of": {
                        "log_index": idx,
                        "app_log_id": log.get("app_log_id"),
                        "timestamp": log.get("timestamp"),
                        "window_id": md.get("window_id"),
                        "domain": md.get("domain"),
                        "completed_chain_ids": sorted(chain_ids_by_index[idx]) if idx in completion_indices else [],
                    },
                    "expected_snapshot_state": _expand_snapshot(valid_snapshot_flat),
                    "state_observability": _expand_observability(valid_observability_flat),
                    "validity": {
                        "valid_state_count": len(valid_snapshot_flat),
                        "invalid_state_count": len(invalid_state_keys),
                        "invalid_state_keys": sorted(invalid_state_keys),
                    },
                    "rq3_apply_service_qa": _lookup_rq3_pack_by_cutoff_ts(str(log.get("timestamp", ""))),
                    "state_questionability": _lookup_questionability_by_cutoff_ts(str(log.get("timestamp", ""))),
                }
            )

            meta: Dict[str, Any] = {
                "sampling_mode": sampling_mode,
                "cutoff_final_log_index": idx,
                "cutoff_timestamp": str(log.get("timestamp", "")),
            }
            if sampling_mode == "exposure_token":
                meta["sampling_params"] = {
                    "exposure_percent": int(sampling_point.get("anchor_percent", 0)),
                    "tokenizer_model": str(sampling_point.get("tokenizer_model", "")),
                }
            else:
                meta["sampling_params"] = {
                    "calendar_anchor_freq": str(sampling_point.get("calendar_anchor_freq", "")),
                    # Renumber retained calendar checkpoints to be contiguous from 1.
                    "anchor_index": int(calendar_retained_index),
                    # Keep original anchor index for traceability/debug.
                    "source_anchor_index": int(sampling_point.get("anchor_index", 0)),
                    "anchor_timestamp": str(sampling_point.get("anchor_timestamp", "")),
                }
                if "actual_tokens_at_cutoff" in sampling_point:
                    meta["sampling_params"]["actual_tokens_at_cutoff"] = int(
                        sampling_point.get("actual_tokens_at_cutoff", 0)
                    )
                if "total_tokens" in sampling_point:
                    meta["sampling_params"]["total_tokens"] = int(sampling_point.get("total_tokens", 0))
                if "cutoff_log_tokens" in sampling_point:
                    meta["sampling_params"]["cutoff_log_tokens"] = int(
                        sampling_point.get("cutoff_log_tokens", 0)
                    )
                if "tokenizer_model" in sampling_point:
                    meta["sampling_params"]["tokenizer_model"] = str(sampling_point.get("tokenizer_model", ""))
            meta_by_checkpoint_id[sampled_id] = meta

    checkpoints.sort(key=lambda x: str(x.get("checkpoint_id", "")))
    return checkpoints, meta_by_checkpoint_id


def build_token_exposure_checkpoints(
    *,
    benchmark_path: Path,
    app_logs_large: List[Dict[str, Any]],
    exposure_anchors: Sequence[int],
    tokenizer_model: str = "gpt-5-mini",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sampling_points = _build_token_anchor_specs(app_logs_large, exposure_anchors, tokenizer_model)
    export_items = _map_anchors_to_final_indices(
        benchmark_path=benchmark_path,
        anchors=sampling_points,
    )
    return _rebuild_checkpoints_from_export_items(
        benchmark_path=benchmark_path,
        export_items=export_items,
        sampling_mode="exposure_token",
    )


def build_calendar_anchor_checkpoints(
    *,
    benchmark_path: Path,
    app_logs_large: List[Dict[str, Any]],
    calendar_anchor_freq: str,
    tokenizer_model: str = "gpt-5-mini",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sampling_points = _build_calendar_anchor_specs(app_logs_large, calendar_anchor_freq, tokenizer_model)
    export_items = _map_anchors_to_final_indices(
        benchmark_path=benchmark_path,
        anchors=sampling_points,
    )
    return _rebuild_checkpoints_from_export_items(
        benchmark_path=benchmark_path,
        export_items=export_items,
        sampling_mode="calendar_time",
    )
