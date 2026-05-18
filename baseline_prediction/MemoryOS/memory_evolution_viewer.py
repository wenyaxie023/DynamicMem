#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_viewer_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _trim_text(text: str, limit: int = 240) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _parse_user_input_as_log(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    prefix = "[APP_LOG] "
    if raw.startswith(prefix):
        raw = raw[len(prefix) :].strip()
    first_line = raw.splitlines()[0].strip()
    try:
        payload = json.loads(first_line)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_short_term_item(item: Dict[str, Any]) -> Dict[str, Any]:
    app_log = _parse_user_input_as_log(item.get("user_input"))
    if app_log:
        return {
            "app_log_id": str(app_log.get("app_log_id") or ""),
            "native_timestamp": str(item.get("timestamp") or ""),
            "app_log_timestamp": str(app_log.get("timestamp") or ""),
            "app_name": str(app_log.get("app_name") or ""),
            "api_name": str(app_log.get("api_name") or ""),
            "summary": _trim_text(json.dumps(app_log, ensure_ascii=False), 280),
        }
    return {
        "app_log_id": "",
        "native_timestamp": str(item.get("timestamp") or ""),
        "app_log_timestamp": "",
        "app_name": "",
        "api_name": "",
        "summary": _trim_text(str(item.get("user_input") or ""), 280),
    }


def _short_term_key(item: Dict[str, Any]) -> str:
    summary = _summarize_short_term_item(item)
    app_log_id = str(summary.get("app_log_id") or "").strip()
    if app_log_id:
        return app_log_id
    return "{}|{}".format(summary.get("native_timestamp") or "", summary.get("summary") or "")


def _summarize_mid_term_session(session: Dict[str, Any], access_frequency: Dict[str, Any]) -> Dict[str, Any]:
    details = session.get("details") or []
    detail_items = [x for x in details if isinstance(x, dict)]
    detail_summaries = [_summarize_short_term_item(x) for x in detail_items[:5]]
    app_log_ids = [str(x.get("app_log_id") or "") for x in detail_summaries if str(x.get("app_log_id") or "")]
    detail_native_timestamps = [
        str(x.get("native_timestamp") or "") for x in detail_summaries if str(x.get("native_timestamp") or "")
    ]
    session_id = str(session.get("id") or "")
    return {
        "session_id": session_id,
        "native_session_timestamp": str(session.get("timestamp") or ""),
        "last_visit_time": str(session.get("last_visit_time") or ""),
        "summary": _trim_text(str(session.get("summary") or ""), 320),
        "summary_keywords": [str(x) for x in (session.get("summary_keywords") or []) if str(x or "").strip()],
        "detail_count": len(detail_items),
        "detail_app_log_ids": app_log_ids[:12],
        "detail_native_timestamps": detail_native_timestamps[:12],
        "details_preview": detail_summaries,
        "L_interaction": int(session.get("L_interaction") or 0),
        "N_visit": int(session.get("N_visit") or 0),
        "access_count_lfu": int(session.get("access_count_lfu") or 0),
        "access_frequency_count": int(access_frequency.get(session_id) or 0),
        "H_segment": float(session.get("H_segment") or 0.0),
        "R_recency": float(session.get("R_recency") or 0.0),
    }


def _build_mid_term_overview(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    detail_counts = [int(x.get("detail_count") or 0) for x in items]
    keyword_counter: Counter = Counter()
    for item in items:
        for keyword in item.get("summary_keywords") or []:
            if str(keyword or "").strip():
                keyword_counter[str(keyword).strip()] += 1
    return {
        "single_detail_sessions": sum(1 for x in detail_counts if x == 1),
        "multi_detail_sessions": sum(1 for x in detail_counts if x > 1),
        "max_detail_count": max(detail_counts) if detail_counts else 0,
        "avg_detail_count": (sum(detail_counts) / len(detail_counts)) if detail_counts else 0.0,
        "unique_summary_keyword_count": len(keyword_counter),
        "top_summary_keywords": [
            {"keyword": keyword, "count": count} for keyword, count in keyword_counter.most_common(20)
        ],
    }


def _collect_mid_term_summary(mid_term_payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    sessions = mid_term_payload.get("sessions") or {}
    access_frequency = mid_term_payload.get("access_frequency") or {}
    access_frequency = access_frequency if isinstance(access_frequency, dict) else {}
    items: List[Dict[str, Any]] = []
    total_detail_count = 0
    if isinstance(sessions, dict):
        for session in sessions.values():
            if not isinstance(session, dict):
                continue
            summary = _summarize_mid_term_session(session, access_frequency)
            items.append(summary)
            total_detail_count += int(summary.get("detail_count") or 0)
    items.sort(key=lambda x: (str(x.get("native_session_timestamp") or ""), str(x.get("session_id") or "")))
    return items, total_detail_count, _build_mid_term_overview(items)


def _summarize_knowledge_item(item: Dict[str, Any], text_key: str) -> Dict[str, Any]:
    return {
        "native_timestamp": str(item.get("timestamp") or ""),
        "text": _trim_text(str(item.get(text_key) or ""), 280),
    }


def _normalize_user_profiles(user_profiles: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for profile_id, value in user_profiles.items():
        if isinstance(value, dict):
            items.append(
                {
                    "profile_id": str(profile_id or ""),
                    "data": str(value.get("data") or ""),
                    "last_updated": str(value.get("last_updated") or ""),
                }
            )
            continue
        items.append(
            {
                "profile_id": str(profile_id or ""),
                "data": str(value or ""),
                "last_updated": "",
            }
        )
    return items


def _build_snapshot_card(snapshot_root: Path, entry: Dict[str, Any], include_full_lists: bool = False) -> Dict[str, Any]:
    checkpoint_path = snapshot_root / str(entry.get("checkpoint_path") or "")
    short_term = load_json(snapshot_root / str(entry.get("short_term_path") or ""))
    mid_term = load_json(snapshot_root / str(entry.get("mid_term_path") or ""))
    long_term_user = load_json(snapshot_root / str(entry.get("long_term_path") or ""))
    long_term_assistant = load_json(snapshot_root / str(entry.get("assistant_long_term_path") or ""))
    checkpoint_payload = load_json(checkpoint_path) if checkpoint_path.exists() else {}

    short_term_items = [x for x in short_term if isinstance(x, dict)] if isinstance(short_term, list) else []
    short_term_summaries = [_summarize_short_term_item(x) for x in short_term_items]

    mid_term_sessions, mid_term_detail_count, mid_term_overview = _collect_mid_term_summary(
        mid_term if isinstance(mid_term, dict) else {}
    )
    user_profiles = {}
    user_knowledge = []
    assistant_knowledge = []
    if isinstance(long_term_user, dict):
        raw_profiles = long_term_user.get("user_profiles")
        if isinstance(raw_profiles, dict):
            user_profiles = raw_profiles
        raw_knowledge = long_term_user.get("knowledge_base")
        if isinstance(raw_knowledge, list):
            user_knowledge = [x for x in raw_knowledge if isinstance(x, dict)]
    if isinstance(long_term_assistant, dict):
        raw_assistant = long_term_assistant.get("assistant_knowledge")
        if isinstance(raw_assistant, list):
            assistant_knowledge = [x for x in raw_assistant if isinstance(x, dict)]

    normalized_user_profiles = _normalize_user_profiles(user_profiles)

    return {
        "snapshot_id": str(entry.get("snapshot_id") or ""),
        "created_at": str(entry.get("created_at") or ""),
        "trigger": str(entry.get("trigger") or ""),
        "last_event_idx": int(entry.get("last_event_idx") or checkpoint_payload.get("last_event_idx") or -1),
        "checkpoint_id": str(entry.get("checkpoint_id") or checkpoint_payload.get("checkpoint_id") or ""),
        "checkpoint_app_log_id": str(
            entry.get("checkpoint_app_log_id") or checkpoint_payload.get("checkpoint_app_log_id") or ""
        ),
        "saved_at": str(checkpoint_payload.get("saved_at") or ""),
        "counts": {
            "short_term_items": len(short_term_items),
            "mid_term_sessions": len(mid_term_sessions),
            "mid_term_details": mid_term_detail_count,
            "user_knowledge_items": len(user_knowledge),
            "assistant_knowledge_items": len(assistant_knowledge),
            "user_profile_count": len(user_profiles),
        },
        "short_term_items": short_term_summaries,
        "short_term_preview": short_term_summaries[-8:],
        "mid_term_sessions": mid_term_sessions,
        "mid_term_sessions_preview": mid_term_sessions[-8:],
        "mid_term_overview": mid_term_overview,
        "long_term_user_items": [_summarize_knowledge_item(x, "knowledge") for x in user_knowledge],
        "long_term_user_preview": [_summarize_knowledge_item(x, "knowledge") for x in user_knowledge[-8:]],
        "long_term_assistant_items": [_summarize_knowledge_item(x, "knowledge") for x in assistant_knowledge],
        "long_term_assistant_preview": [_summarize_knowledge_item(x, "knowledge") for x in assistant_knowledge[-8:]],
        "user_profiles": normalized_user_profiles,
        "user_profiles_preview": [_trim_text(str(x.get("data") or ""), 280) for x in normalized_user_profiles[:8]],
        "user_profiles_raw": user_profiles if include_full_lists else {},
        "builder_state": checkpoint_payload.get("builder_state") if isinstance(checkpoint_payload, dict) else None,
        "_diff_sources": {
            "short_term_keys": [_short_term_key(x) for x in short_term_items],
            "mid_term_session_ids": [str(x.get("session_id") or "") for x in mid_term_sessions if str(x.get("session_id") or "")],
            "user_knowledge_texts": [str(x.get("knowledge") or "") for x in user_knowledge if str(x.get("knowledge") or "")],
            "assistant_knowledge_texts": [
                str(x.get("knowledge") or "") for x in assistant_knowledge if str(x.get("knowledge") or "")
            ],
        },
    }


def _compute_diff(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if previous is None:
        return {
            "short_term_delta": int(current["counts"]["short_term_items"]),
            "mid_term_session_delta": int(current["counts"]["mid_term_sessions"]),
            "mid_term_detail_delta": int(current["counts"]["mid_term_details"]),
            "user_knowledge_delta": int(current["counts"]["user_knowledge_items"]),
            "assistant_knowledge_delta": int(current["counts"]["assistant_knowledge_items"]),
            "new_short_term_keys": list(current["_diff_sources"]["short_term_keys"][-8:]),
            "new_mid_term_session_ids": list(current["_diff_sources"]["mid_term_session_ids"][-8:]),
            "new_user_knowledge_texts": [
                _trim_text(x, 200) for x in current["_diff_sources"]["user_knowledge_texts"][-8:]
            ],
            "new_assistant_knowledge_texts": [
                _trim_text(x, 200) for x in current["_diff_sources"]["assistant_knowledge_texts"][-8:]
            ],
        }
    prev_counts = previous.get("counts") or {}
    return {
        "short_term_delta": int(current["counts"]["short_term_items"]) - int(prev_counts.get("short_term_items") or 0),
        "mid_term_session_delta": int(current["counts"]["mid_term_sessions"]) - int(prev_counts.get("mid_term_sessions") or 0),
        "mid_term_detail_delta": int(current["counts"]["mid_term_details"]) - int(prev_counts.get("mid_term_details") or 0),
        "user_knowledge_delta": int(current["counts"]["user_knowledge_items"]) - int(prev_counts.get("user_knowledge_items") or 0),
        "assistant_knowledge_delta": int(current["counts"]["assistant_knowledge_items"]) - int(prev_counts.get("assistant_knowledge_items") or 0),
        "new_short_term_keys": sorted(
            set(current["_diff_sources"]["short_term_keys"]) - set(previous.get("_diff_sources", {}).get("short_term_keys") or [])
        )[:8],
        "new_mid_term_session_ids": sorted(
            set(current["_diff_sources"]["mid_term_session_ids"])
            - set(previous.get("_diff_sources", {}).get("mid_term_session_ids") or [])
        )[:8],
        "new_user_knowledge_texts": [
            _trim_text(x, 200)
            for x in sorted(
                set(current["_diff_sources"]["user_knowledge_texts"])
                - set(previous.get("_diff_sources", {}).get("user_knowledge_texts") or [])
            )[:8]
        ],
        "new_assistant_knowledge_texts": [
            _trim_text(x, 200)
            for x in sorted(
                set(current["_diff_sources"]["assistant_knowledge_texts"])
                - set(previous.get("_diff_sources", {}).get("assistant_knowledge_texts") or [])
            )[:8]
        ],
    }


def build_memory_evolution_payload(snapshot_root: Path) -> Dict[str, Any]:
    snapshot_root = Path(snapshot_root)
    manifest_path = snapshot_root / "manifest.json"
    manifest = load_json(manifest_path)
    entries = manifest.get("snapshots") or []
    snapshots: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    checkpoint_ids: List[str] = []
    latest_snapshot_id = ""
    if entries and isinstance(entries[-1], dict):
        latest_snapshot_id = str(entries[-1].get("snapshot_id") or "")
    latest_snapshot_id = str(manifest.get("latest_snapshot") or latest_snapshot_id or "")

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        card = _build_snapshot_card(
            snapshot_root,
            entry,
            include_full_lists=(str(entry.get("snapshot_id") or "") == latest_snapshot_id),
        )
        card["diff_from_previous"] = _compute_diff(card, previous)
        snapshots.append(card)
        checkpoint_id = str(card.get("checkpoint_id") or "")
        if checkpoint_id and checkpoint_id not in checkpoint_ids:
            checkpoint_ids.append(checkpoint_id)
        previous = card

    latest = snapshots[-1] if snapshots else {}
    for card in snapshots:
        card.pop("_diff_sources", None)

    return {
        "meta": {
            "snapshot_root": str(snapshot_root),
            "manifest_path": str(manifest_path),
            "snapshot_count": len(snapshots),
            "checkpoint_ids": checkpoint_ids,
            "latest_snapshot_id": str(latest.get("snapshot_id") or ""),
            "latest_event_idx": int(latest.get("last_event_idx") or -1),
        },
        "snapshots": snapshots,
    }
