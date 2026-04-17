"""State questionability checks for apply-service QA generation."""

from typing import Any, Dict, List, Optional, Set

REASON_EMPTY = "empty_value"
REASON_NOISE = "noisy_value"
REASON_MISSING_SCHEDULE_DATES = "missing_schedule_dates"
REASON_SCHEDULE_DATES_EVIDENCE_UNDERCOVERAGE = "schedule_dates_evidence_undercoverage"


def _collect_field_paths(value: Any, out: List[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            name = str(k).strip().lower()
            if name:
                child = name if not prefix else f"{prefix}.{name}"
                _collect_field_paths(v, out, child)
        return
    if isinstance(value, list):
        if prefix:
            out.append(prefix)
        elif value:
            out.append("current_value")
        # For object-list values, also expose a light field hint path.
        for x in value[:3]:
            if isinstance(x, dict):
                _collect_field_paths(x, out, f"{prefix}[]")
        return
    if prefix:
        out.append(prefix)
    else:
        out.append("current_value")


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_is_empty_value(v) for v in value.values()) if value else True
    if isinstance(value, list):
        return all(_is_empty_value(v) for v in value) if value else True
    return False


def _is_noisy_value(value: Any) -> bool:
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"unknown", "n/a", "none", "null", "tbd", "na"}
    return False


def _missing_schedule_dates_for_habit(state_key: str, value: Any) -> bool:
    if not str(state_key or "").startswith("habits_state:"):
        return False
    if not isinstance(value, dict):
        return False
    if "schedule" not in value:
        return False
    dates = value.get("schedule_dates")
    if not isinstance(dates, list):
        return True
    return len([x for x in dates if str(x).strip()]) == 0


def _normalize_date_values(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        svalue = str(value).strip()
        if not svalue:
            continue
        normalized = svalue[:10]
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _extract_observed_evidence_dates(
    *,
    state_observability: Optional[Dict[str, Any]],
    app_logs_by_id: Optional[Dict[str, Dict[str, Any]]],
) -> List[str]:
    obs = state_observability if isinstance(state_observability, dict) else {}
    ids = obs.get("evidence_app_log_ids")
    out: List[str] = []
    seen: Set[str] = set()
    if isinstance(ids, list):
        for app_log_id in ids:
            sid = str(app_log_id).strip()
            if not sid:
                continue
            log = (app_logs_by_id or {}).get(sid) if isinstance(app_logs_by_id, dict) else None
            timestamp = str((log or {}).get("timestamp") or "").strip()
            if len(timestamp) >= 10:
                date_value = timestamp[:10]
                if date_value not in seen:
                    seen.add(date_value)
                    out.append(date_value)
    if out:
        return out

    if isinstance(ids, list):
        for idx, _ in enumerate(ids):
            pseudo = f"count_only_{idx + 1:04d}"
            out.append(pseudo)
    return out


def _schedule_date_coverage_reason(
    *,
    state_key: str,
    state_value: Any,
    state_observability: Optional[Dict[str, Any]],
    app_logs_by_id: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[str]:
    if not str(state_key or "").startswith("habits_state:"):
        return None
    if not isinstance(state_value, dict):
        return None
    if "schedule" not in state_value:
        return None
    if _missing_schedule_dates_for_habit(state_key, state_value):
        return REASON_MISSING_SCHEDULE_DATES

    expected_dates = _normalize_date_values(state_value.get("schedule_dates"))
    if not expected_dates:
        return REASON_MISSING_SCHEDULE_DATES

    observed_dates = _extract_observed_evidence_dates(
        state_observability=state_observability,
        app_logs_by_id=app_logs_by_id,
    )
    if not observed_dates:
        return REASON_SCHEDULE_DATES_EVIDENCE_UNDERCOVERAGE

    # Prefer exact date overlap when log timestamps are available. If only ID-count
    # fallback is available, still require one evidence item per expected instance.
    if observed_dates and observed_dates[0].startswith("count_only_"):
        observed_count = len(observed_dates)
    else:
        observed_count = len(set(expected_dates) & set(observed_dates))
    if observed_count < len(expected_dates):
        return REASON_SCHEDULE_DATES_EVIDENCE_UNDERCOVERAGE
    return None


def evaluate_state_questionability(
    *,
    state_key: str,
    state_value: Any,
    state_observability: Optional[Dict[str, Any]] = None,
    app_logs_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    validator_version: str = "qv1",
) -> Dict[str, Any]:
    """Return deterministic L1 questionability payload for one state key/value."""
    reason_codes: List[str] = []
    askable_fields: List[str] = []

    if _is_empty_value(state_value):
        reason_codes.append(REASON_EMPTY)
    if _is_noisy_value(state_value):
        reason_codes.append(REASON_NOISE)

    schedule_reason = _schedule_date_coverage_reason(
        state_key=state_key,
        state_value=state_value,
        state_observability=state_observability,
        app_logs_by_id=app_logs_by_id,
    )
    if schedule_reason:
        reason_codes.append(schedule_reason)

    fields: List[str] = []
    _collect_field_paths(state_value, fields)
    seen = set()
    for f in fields:
        if f in seen:
            continue
        seen.add(f)
        askable_fields.append(f)

    is_questionable = len(reason_codes) == 0
    return {
        "state_key": state_key,
        "is_questionable": is_questionable,
        "reason_codes": reason_codes,
        "askable_fields": askable_fields[:16],
        "validator_version": validator_version,
    }
