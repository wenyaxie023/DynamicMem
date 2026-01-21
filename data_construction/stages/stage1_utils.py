from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _normalize_window_id_label(window_id: object) -> str:
    """
    Normalize window identifiers for consistent comparisons (e.g., initial_state -> initial).
    """
    if window_id is None:
        return "initial"
    normalized = str(window_id).strip()
    lower = normalized.lower()
    if lower in {"", "initial", "initial_state", "initialstate", "init"}:
        return "initial"
    return normalized


def _path_key(key: object) -> object:
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.isdigit():
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _location_to_path(location: object) -> List[object]:
    """
    Convert dotted + bracket notation paths (e.g., time_windows[0].summary) into
    a list/tuple path usable by the patch applier. If the input is already a
    list/tuple, it is returned as-is.
    """
    if isinstance(location, (list, tuple)):
        return list(location)
    parts: List[object] = []
    if location is None:
        return parts
    for segment in str(location).split("."):
        if not segment:
            continue
        remainder = segment
        while remainder:
            if "[" in remainder:
                before, after = remainder.split("[", 1)
                if before:
                    parts.append(before)
                idx_str, remainder = after.split("]", 1)
                parts.append(_path_key(idx_str))
            else:
                parts.append(remainder)
                remainder = ""
    return parts


def _traverse_path(
    root: Any, path: Sequence[object], *, create_missing: bool = False
) -> Any:
    """
    Walk the object graph following a path. Optionally create missing containers
    (dict or list) when create_missing is True.
    """
    target: Any = root
    for idx, raw_key in enumerate(path):
        key = _path_key(raw_key)
        if isinstance(target, list):
            if not isinstance(key, int):
                return None
            if 0 <= key < len(target):
                target = target[key]
            elif create_missing and key == len(target):
                next_key = path[idx + 1] if idx + 1 < len(path) else None
                target.append({} if isinstance(next_key, str) else [])
                target = target[key]
            else:
                return None
        elif isinstance(target, dict):
            if key not in target:
                if create_missing:
                    next_key = path[idx + 1] if idx + 1 < len(path) else None
                    target[key] = {} if isinstance(next_key, str) else []
                else:
                    return None
            target = target[key]
        else:
            return None
    return target


def _apply_patch_ops(base: Dict, patch_ops: List[Dict]) -> Dict:
    patched = deepcopy(base)
    for op in patch_ops or []:
        path = _location_to_path(op.get("path"))
        action = op.get("action")
        value = op.get("value")
        key_name = op.get("key")
        if action not in {"add", "replace", "remove", "append", "add_key"}:
            continue
        if not isinstance(path, list) or len(path) == 0:
            continue

        if action == "append":
            target_list = _traverse_path(patched, path, create_missing=True)
            if isinstance(target_list, list):
                target_list.append(value)
            continue

        # Allow nested field updates without replacing the whole object when a key is provided
        if key_name is not None and action in {"add", "replace", "remove", "add_key"}:
            target_dict = _traverse_path(
                patched, path, create_missing=action in {"add", "replace", "add_key"}
            )
            if isinstance(target_dict, dict):
                if action == "remove":
                    target_dict.pop(key_name, None)
                else:
                    target_dict[key_name] = value
                continue

        if action == "add_key":
            target_dict = _traverse_path(patched, path, create_missing=True)
            if isinstance(target_dict, dict) and key_name is not None:
                target_dict[key_name] = value
            continue

        parent = _traverse_path(
            patched, path[:-1], create_missing=action in {"add", "replace"}
        )
        if parent is None:
            continue

        last_key = _path_key(path[-1])
        if isinstance(parent, list):
            if not isinstance(last_key, int):
                continue
            if action == "add":
                if 0 <= last_key <= len(parent):
                    parent.insert(last_key, value)
                else:
                    parent.append(value)
            elif action == "replace":
                if 0 <= last_key < len(parent):
                    parent[last_key] = value
                elif last_key == len(parent):
                    parent.append(value)
            elif action == "remove" and 0 <= last_key < len(parent):
                parent.pop(last_key)
        elif isinstance(parent, dict):
            if action in {"add", "replace"}:
                parent[last_key] = value
            elif action == "remove":
                parent.pop(last_key, None)
    return patched


def _apply_profile_revision(original_profile: Dict, review_payload: Dict) -> Dict:
    if not isinstance(review_payload, dict):
        return deepcopy(original_profile)
    patch_ops = review_payload.get("patch") or review_payload.get("patch_ops") or []
    fixes = review_payload.get("fixes") or []
    revised = review_payload.get("revised_profile")
    base_from_singleton_list = (
        isinstance(original_profile, list)
        and len(original_profile) == 1
        and isinstance(original_profile[0], dict)
    )
    base_obj = (
        deepcopy(original_profile[0])
        if base_from_singleton_list
        else deepcopy(original_profile)
    )

    violations = review_payload.get("violations_and_fixes") or []
    if violations and not patch_ops:
        translated_ops: List[Dict] = []
        for violation in violations:
            for patch in violation.get("patches") or []:
                loc = patch.get("path") or patch.get("location")
                if loc is None:
                    continue
                translated_ops.append(
                    {
                        "path": _location_to_path(loc),
                        "action": patch.get("action") or "replace",
                        "value": patch.get("value"),
                        "key": patch.get("key"),
                    }
                )
        patch_ops = translated_ops

    if fixes and not patch_ops:
        translated_ops: List[Dict] = []
        for fix in fixes:
            loc = fix.get("location") or fix.get("path")
            if not loc:
                continue
            path = _location_to_path(loc)
            action = fix.get("action") or "replace"
            value = fix.get("fix")
            if "value" in fix and value is None:
                value = fix.get("value")
            translated_ops.append({"path": path, "action": action, "value": value})
        patch_ops = translated_ops

    patched = deepcopy(base_obj)
    if patch_ops:
        try:
            patched = _apply_patch_ops(patched, patch_ops)
        except Exception:
            if revised is not None:
                patched = deepcopy(revised)
            else:
                patched = deepcopy(base_obj)
    elif revised is not None:
        patched = deepcopy(revised)
    if base_from_singleton_list and isinstance(patched, dict):
        return [patched]
    return patched


def _append_issue(
    issues: List[Dict[str, str]],
    path: str,
    message: str,
    window_id: str | None = None,
    issue_id: str | None = None,
) -> None:
    entry: Dict[str, str] = {"path": path, "message": message}
    if window_id:
        entry["window_id"] = window_id
    if issue_id:
        entry["id"] = issue_id
    issues.append(entry)


def _validate_schedule_structure(schedule: Any) -> List[str]:
    if not isinstance(schedule, dict):
        return ["schedule must be an object with frequency_type"]
    missing: List[str] = []
    freq = schedule.get("frequency_type")
    allowed = {"daily", "weekly", "biweekly", "monthly_by_date", "monthly_nth_weekday"}
    if not freq:
        missing.append("schedule.frequency_type missing")
    elif freq not in allowed:
        missing.append(f"schedule.frequency_type '{freq}' is invalid")
    else:
        if freq == "weekly" and "days_of_week" not in schedule:
            missing.append("schedule.days_of_week missing for weekly")
        if freq == "biweekly":
            if "days_of_week" not in schedule:
                missing.append("schedule.days_of_week missing for biweekly")
            if "start_date" not in schedule:
                missing.append("schedule.start_date missing for biweekly")
        if freq == "monthly_by_date" and "days_of_month" not in schedule:
            missing.append("schedule.days_of_month missing for monthly_by_date")
        if freq == "monthly_nth_weekday":
            if "week_of_month" not in schedule:
                missing.append("schedule.week_of_month missing for monthly_nth_weekday")
            if "day_of_week" not in schedule:
                missing.append("schedule.day_of_week missing for monthly_nth_weekday")
    return missing


def _validate_timing_structure(timing: Any) -> List[str]:
    if not isinstance(timing, dict):
        return ["timing must be an object with start_time and end_time"]
    missing: List[str] = []
    if "start_time" not in timing:
        missing.append("timing.start_time missing")
    if "end_time" not in timing:
        missing.append("timing.end_time missing")
    return missing


def _validate_habit_object(habit_obj: Any) -> List[str]:
    if not isinstance(habit_obj, dict):
        return ["habit must be an object with required fields"]
    required_fields = ["schedule", "timing", "location", "priority"]
    missing = [f for f in required_fields if f not in habit_obj]
    missing += _validate_schedule_structure(habit_obj.get("schedule"))
    missing += _validate_timing_structure(habit_obj.get("timing"))
    return missing


def _validate_preference_object(pref_obj: Any, context: str = "") -> List[str]:
    """
    Validate a preference object according to the prompt requirements.

    A valid preference object must have:
    - statement: A concrete preference statement (10-30 words)
    - signals: An array of 2-4 observable behaviors/evidence

    Args:
        pref_obj: The preference object to validate
        context: Optional context string for error messages (e.g., "for shift/refine delta")

    Returns:
        List of validation error messages
    """
    context_suffix = f" {context}" if context else ""

    if not isinstance(pref_obj, dict):
        return [f"preference must be an object with statement and signals{context_suffix}"]

    missing: List[str] = []

    # Validate statement
    statement = pref_obj.get("statement")
    if not statement:
        missing.append(f"statement missing{context_suffix}")
    elif not isinstance(statement, str):
        missing.append(f"statement must be a string{context_suffix}")
    else:
        # Check word count (10-30 words recommended)
        word_count = len(statement.split())
        if word_count < 5:
            missing.append(f"statement too short ({word_count} words, recommend 10-30){context_suffix}")

    # Validate signals
    signals = pref_obj.get("signals")
    if signals is None:
        missing.append(f"signals missing{context_suffix}")
    elif not isinstance(signals, list):
        missing.append(f"signals must be an array{context_suffix}")
    elif len(signals) < 2:
        missing.append(f"signals too few ({len(signals)}, need 2-4){context_suffix}")
    elif len(signals) > 4:
        missing.append(f"signals too many ({len(signals)}, need 2-4){context_suffix}")
    else:
        # Check each signal is a non-empty string
        for idx, sig in enumerate(signals):
            if not isinstance(sig, str) or not sig.strip():
                missing.append(f"signals[{idx}] must be a non-empty string{context_suffix}")

    return missing


def _detect_rule2_prior_existence_issues(profile: Dict) -> List[Dict[str, str]]:
    # Some generators output a singleton list; normalize to dict
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}
    issues: List[Dict[str, str]] = []
    initial_state = profile.get("initial_state") or {}
    user_attributes_state = initial_state.get("user_attributes_state") or {}
    known_singular = set((user_attributes_state.get("singular") or {}).keys())
    known_collections = set((user_attributes_state.get("collections") or {}).keys())
    habits_state = initial_state.get("habits_state") or {}
    known_habits = set(habits_state.keys())
    preferences_state = initial_state.get("preferences_state") or {}
    known_preferences = set(preferences_state.keys())

    time_windows = profile.get("time_windows") or []
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"

        attr_ops = _get_delta_changes(window.get("user_attributes_delta"))
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("change_type") or op.get("op")
            attr_type = op.get("attribute_type")

            # Check 1: singular attributes can ONLY use "modify"
            if attr_type == "singular" and op_type != "modify":
                name = op.get("attribute_name")
                _append_issue(
                    issues,
                    f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]",
                    f"invalid change '{op_type}' on singular attribute '{name}' (singular only supports 'modify')",
                    window_id,
                )

            # Check 2: collections can ONLY use "add" or "remove"
            if attr_type == "collections" and op_type not in {"add", "remove"}:
                name = op.get("collection_name")
                _append_issue(
                    issues,
                    f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]",
                    f"invalid change '{op_type}' on collection '{name}' (collections only support 'add' or 'drop')",
                    window_id,
                )

            # Check 3: modify on singular requires prior existence
            if op_type == "modify" and attr_type == "singular":
                name = op.get("attribute_name")
                if name and name not in known_singular:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]",
                        f"modify '{name}' before it exists in prior state",
                        window_id,
                    )
                    known_singular.add(name)

            # Track collections
            elif op_type == "add" and attr_type == "collections":
                name = op.get("collection_name")
                if name:
                    known_collections.add(name)

            # Check 4: drop on collection requires prior existence
            elif op_type == "remove" and attr_type == "collections":
                name = op.get("collection_name")
                if name and name not in known_collections:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]",
                        f"drop collection '{name}' before it exists",
                        window_id,
                    )

        habit_ops = _get_delta_changes(window.get("habits_delta"))
        for op_idx, op in enumerate(habit_ops):
            op_type = op.get("change_type") or op.get("op")
            habit_name = op.get("habit_name")
            if op_type == "acquire":
                if habit_name:
                    known_habits.add(habit_name)
            elif op_type in {"adjust", "drop"}:
                if habit_name and habit_name not in known_habits:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].habits_delta.changes[{op_idx}]",
                        f"{op_type} '{habit_name}' before it exists in prior state",
                        window_id,
                    )
                    known_habits.add(habit_name)

                # Check 5: adjust on habit must modify schedule/timing/location, not just priority
                if op_type == "adjust":
                    delta = op.get("delta") or {}
                    if isinstance(delta, dict):
                        # Check if only priority is being modified
                        has_structural_change = any(
                            key in delta for key in ["schedule", "timing", "location"]
                        )
                        if not has_structural_change and "priority" in delta:
                            _append_issue(
                                issues,
                                f"time_windows[{w_idx}].habits_delta.changes[{op_idx}]",
                                f"adjust '{habit_name}' only modifies priority (must modify at least one of: schedule, timing, location, context)",
                                window_id,
                            )

                if op_type == "drop" and habit_name in known_habits:
                    known_habits.remove(habit_name)

        pref_ops = _get_delta_changes(window.get("preferences_delta"))
        for op_idx, op in enumerate(pref_ops):
            op_type = op.get("change_type") or op.get("op")
            pref_name = op.get("preference_name")
            if op_type in {"shift", "refine"} and pref_name:
                if pref_name not in known_preferences:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].preferences_delta.changes[{op_idx}]",
                        f"{op_type} '{pref_name}' before it exists in prior state",
                        window_id,
                    )
                    known_preferences.add(pref_name)

    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule2_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"

    return issues


def _detect_rule1_required_field_issues(profile: Dict) -> List[Dict[str, str]]:
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}
    issues: List[Dict[str, str]] = []
    initial_state = profile.get("initial_state") or {}
    if "summary" not in initial_state:
        _append_issue(issues, "initial_state.summary", "initial_state missing summary")

    user_attributes_state = initial_state.get("user_attributes_state")
    if not isinstance(user_attributes_state, dict):
        _append_issue(
            issues,
            "initial_state.user_attributes_state",
            "user_attributes_state missing or not an object",
        )
    else:
        if "singular" not in user_attributes_state:
            _append_issue(
                issues,
                "initial_state.user_attributes_state.singular",
                "singular attributes missing (can be empty object)",
            )
        if "collections" not in user_attributes_state:
            _append_issue(
                issues,
                "initial_state.user_attributes_state.collections",
                "collections missing (can be empty object)",
            )

    habits_state = initial_state.get("habits_state") or {}
    for habit_name, habit_obj in habits_state.items():
        missing = _validate_habit_object(habit_obj)
        if missing:
            _append_issue(
                issues,
                f"initial_state.habits_state.{habit_name}",
                "; ".join(missing),
                "initial",
            )

    pref_state = initial_state.get("preferences_state") or {}
    for pref_name, pref_obj in pref_state.items():
        missing = _validate_preference_object(pref_obj)
        if missing:
            _append_issue(
                issues,
                f"initial_state.preferences_state.{pref_name}",
                "; ".join(missing),
                "initial",
            )

    time_windows = profile.get("time_windows") or []
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"
        if "window_description" not in window:
            _append_issue(
                issues,
                f"time_windows[{w_idx}].window_description",
                "window_description missing",
                window_id,
            )
        if "summary" not in window:
            _append_issue(
                issues,
                f"time_windows[{w_idx}].summary",
                "summary missing",
                window_id,
            )

        attr_ops = _get_delta_changes(window.get("user_attributes_delta"))
        for op_idx, op in enumerate(attr_ops):
            op_path = f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]"
            op_type = op.get("change_type") or op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "change missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)
            attr_type = op.get("attribute_type")
            if not attr_type:
                _append_issue(
                    issues, f"{op_path}.attribute_type", "attribute_type missing", window_id
                )
            if op_type == "modify":
                if not op.get("attribute_name"):
                    _append_issue(
                        issues, f"{op_path}.attribute_name", "attribute_name missing", window_id
                    )
                if op.get("delta") in [None, ""]:
                    _append_issue(
                        issues, f"{op_path}.delta", "delta missing for modify", window_id
                    )
            elif op_type in {"add", "remove"}:
                if not op.get("attribute_name"):
                    _append_issue(
                        issues,
                        f"{op_path}.attribute_name",
                        "attribute_name missing",
                        window_id,
                    )
                delta = op.get("delta")
                if not isinstance(delta, list) or len(delta) == 0:
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        f"delta list missing/empty for {op_type}",
                        window_id,
                    )

        habit_ops = _get_delta_changes(window.get("habits_delta"))
        for op_idx, op in enumerate(habit_ops):
            op_path = f"time_windows[{w_idx}].habits_delta.changes[{op_idx}]"
            op_type = op.get("change_type") or op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "change missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)

            habit_name = op.get("habit_name")
            if not habit_name:
                _append_issue(
                    issues, f"{op_path}.habit_name", "habit_name missing", window_id
                )

            if op_type == "acquire":
                delta = op.get("delta")
                missing = _validate_habit_object(delta)
                if missing:
                    _append_issue(
                        issues, f"{op_path}.delta", "; ".join(missing), window_id
                    )
            elif op_type == "adjust":
                delta = op.get("delta")
                if not isinstance(delta, dict):
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        "delta must be an object for adjust",
                        window_id,
                    )
                else:
                    substantive_change = any(
                        key in delta for key in ("schedule", "timing", "location", "priority")
                    )
                    if not substantive_change:
                        _append_issue(
                            issues,
                            f"{op_path}.delta",
                            "adjust must change schedule/timing/location",
                            window_id,
                        )
                    if "schedule" in delta:
                        missing = _validate_schedule_structure(delta.get("schedule"))
                        if missing:
                            _append_issue(
                                issues,
                                f"{op_path}.delta.schedule",
                                "; ".join(missing),
                                window_id,
                            )
                    if "timing" in delta:
                        missing = _validate_timing_structure(delta.get("timing"))
                        if missing:
                            _append_issue(
                                issues,
                                f"{op_path}.delta.timing",
                                "; ".join(missing),
                                window_id,
                            )
            elif op_type == "drop":
                if op.get("delta") is not None:
                    _append_issue(
                        issues,
                        f"{op_path}.delta",
                        "drop change must use JSON null",
                        window_id,
                    )

        pref_ops = _get_delta_changes(window.get("preferences_delta"))
        for op_idx, op in enumerate(pref_ops):
            op_path = f"time_windows[{w_idx}].preferences_delta.changes[{op_idx}]"
            op_type = op.get("change_type") or op.get("op")
            if not op_type:
                _append_issue(issues, op_path, "change missing op", window_id)
            if not op.get("reason"):
                _append_issue(issues, f"{op_path}.reason", "reason missing", window_id)
            pref_name = op.get("preference_name")
            if not pref_name:
                _append_issue(
                    issues, f"{op_path}.preference_name", "preference_name missing", window_id
                )

            delta = op.get("delta")
            if op_type in {"shift", "refine"}:
                # Both shift and refine require complete preference object (statement + signals)
                missing = _validate_preference_object(
                    delta, context=f"(required for {op_type} change)"
                )
                if missing:
                    _append_issue(
                        issues, f"{op_path}.delta", "; ".join(missing), window_id
                    )

    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule1_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"

    return issues


def _detect_rule4_short_term_issues(profile: Dict) -> List[Dict[str, str]]:
    """
    Detect short-term changes (seasonal, special events, temporary circumstances)
    that lack appropriate follow-up changes (rollback or permanence reasoning).

    Only checks habits and attributes (NOT preferences).
    Ignores short-term changes in the LAST window (no follow-up window available).
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    issues: List[Dict[str, str]] = []

    # Keywords indicating short-term reasons
    short_term_keywords = [
        "winter", "summer", "spring", "fall", "autumn", "seasonal",
        "holiday", "vacation", "olympics", "event", "festival",
        "heatwave", "cold snap", "temporary", "temporarily",
        "this month", "this week", "during", "special occasion"
    ]

    # Track short-term changes: (window_idx, type, name, reason, keywords_found)
    short_term_changes: List[tuple] = []

    time_windows = profile.get("time_windows") or []
    last_window_idx = len(time_windows) - 1

    # First pass: identify short-term changes (but NOT in the last window)
    for w_idx, window in enumerate(time_windows):
        # Skip the last window - no follow-up window available
        if w_idx == last_window_idx:
            continue

        window_id = window.get("window_id") or f"w{w_idx + 1}"

        # Check habits_delta changes
        habit_ops = _get_delta_changes(window.get("habits_delta"))
        for op_idx, op in enumerate(habit_ops):
            op_type = op.get("change_type") or op.get("op")
            habit_name = op.get("habit_name")
            reason = (op.get("reason") or "").lower()

            # Check for short-term keywords in acquire or adjust changes
            if op_type in {"acquire", "adjust"} and reason:
                found_keywords = [kw for kw in short_term_keywords if kw in reason]
                if found_keywords:
                    short_term_changes.append((
                        w_idx, "habit", habit_name, reason, found_keywords
                    ))

        # Check user_attributes_delta changes
        attr_ops = _get_delta_changes(window.get("user_attributes_delta"))
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("change_type") or op.get("op")
            reason = (op.get("reason") or "").lower()

            if op_type in {"modify", "add"} and reason:
                found_keywords = [kw for kw in short_term_keywords if kw in reason]
                if found_keywords:
                    attr_name = op.get("attribute_name") or op.get("collection_name")
                    short_term_changes.append((
                        w_idx, "attribute", attr_name, reason, found_keywords
                    ))

    # Second pass: check for follow-ups
    issue_id_counter = 0
    for orig_w_idx, change_type, change_name, change_reason, keywords in short_term_changes:
        has_followup = False

        # Check subsequent windows for rollback or permanence mention
        for check_w_idx in range(orig_w_idx + 1, len(time_windows)):
            window = time_windows[check_w_idx]

            if change_type == "habit":
                habit_ops = _get_delta_changes(window.get("habits_delta"))
                for op in habit_ops:
                    if op.get("habit_name") == change_name:
                        # Found a follow-up change (adjust or drop)
                        if (op.get("change_type") or op.get("op")) in {"adjust", "drop"}:
                            has_followup = True
                            break
                        # Or explicit reasoning about permanence
                        reason = (op.get("reason") or "").lower()
                        if any(kw in reason for kw in ["permanent", "decided to keep", "became habit"]):
                            has_followup = True
                            break

            elif change_type == "attribute":
                attr_ops = _get_delta_changes(window.get("user_attributes_delta"))
                for op in attr_ops:
                    attr_name = op.get("attribute_name") or op.get("collection_name")
                    if attr_name == change_name:
                        # Found a follow-up change (modify or remove)
                        if (op.get("change_type") or op.get("op")) in {"modify", "remove"}:
                            has_followup = True
                            break
                        reason = (op.get("reason") or "").lower()
                        if any(kw in reason for kw in ["permanent", "decided to keep"]):
                            has_followup = True
                            break

            if has_followup:
                break

        if not has_followup:
            window_id = time_windows[orig_w_idx].get("window_id") or f"w{orig_w_idx + 1}"
            issue_id = f"rule4_{issue_id_counter:03d}"
            issue_id_counter += 1
            _append_issue(
                issues,
                f"time_windows[{orig_w_idx}]",
                f"[ID: {issue_id}] Short-term {change_type} '{change_name}' (keywords: {', '.join(keywords)}) maybe lacks follow-up in later windows. "
                f"Maybe need rollback change (drop/adjust/modify/remove) or explicit reasoning about permanence.",
                window_id,
            )
            # Add ID to the issue dict
            if issues:
                issues[-1]["id"] = issue_id

    return issues


def _detect_rule3_first_add_issues(profile: Dict) -> List[Dict[str, str]]:
    """
    Detect collection-type attributes that first appear via 'add' change
    instead of being initialized in initial_state.

    This catches unrealistic scenarios like getting a first smartphone in w3,
    when it should already exist in initial_state.
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    issues: List[Dict[str, str]] = []

    # Track what collection attributes exist in initial_state
    initial_state = profile.get("initial_state") or {}
    user_attributes_state = initial_state.get("user_attributes_state") or {}
    known_collections = set((user_attributes_state.get("collections") or {}).keys())

    time_windows = profile.get("time_windows") or []

    # Track first 'add' changes for collection attributes
    for w_idx, window in enumerate(time_windows):
        window_id = window.get("window_id") or f"w{w_idx + 1}"

        # Check user_attributes_delta for 'add' changes
        attr_ops = _get_delta_changes(window.get("user_attributes_delta"))
        for op_idx, op in enumerate(attr_ops):
            op_type = op.get("change_type") or op.get("op")

            # Check for first 'add' to a collection
            if op_type == "add":
                collection_name = op.get("collection_name") or ""

                # If this collection doesn't exist in initial_state, it's a first-time add
                if collection_name and collection_name not in known_collections:
                    _append_issue(
                        issues,
                        f"time_windows[{w_idx}].user_attributes_delta.changes[{op_idx}]",
                        f"Collection '{collection_name}' first appears via 'add' change in {window_id}. "
                        f"If this is an essential collection for the user, it should be initialized in initial_state "
                        f"with realistic baseline items.",
                        window_id,
                    )
                    # Mark as known to avoid duplicate reports
                    known_collections.add(collection_name)

    # Add unique IDs to all issues
    for idx, issue in enumerate(issues):
        if "id" not in issue:
            issue["id"] = f"rule3_{idx:03d}"
            issue["message"] = f"[ID: {issue['id']}] {issue['message']}"

    return issues


PRIORITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _safe_parse_date(value: object) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _parse_window_date_range(time_range: object) -> Tuple[Optional[date], Optional[date]]:
    if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        start = _safe_parse_date(time_range[0])
        end = _safe_parse_date(time_range[1])
        if start and end:
            return start, end
    return None, None


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _coerce_days_of_week(raw: object) -> List[int]:
    days: List[int] = []
    for item in raw or []:
        try:
            val = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= val <= 6:
            days.append(val)
    return days


def _iter_month_starts(start_date: date, end_date: date):
    current = date(start_date.year, start_date.month, 1)
    last_month = date(end_date.year, end_date.month, 1)
    while current <= last_month:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _dates_from_schedule(schedule: Dict[str, Any], start_date: date, end_date: date) -> List[date]:
    freq = (schedule.get("frequency_type") or "").lower()
    if not freq:
        return []
    if freq == "daily":
        return list(_iter_dates(start_date, end_date))
    if freq == "weekly":
        days = _coerce_days_of_week(schedule.get("days_of_week"))
        days = days or list(range(7))
        return [dt for dt in _iter_dates(start_date, end_date) if dt.weekday() in days]
    if freq == "biweekly":
        days = _coerce_days_of_week(schedule.get("days_of_week"))
        days = days or list(range(7))
        anchor = _safe_parse_date(schedule.get("start_date")) or start_date
        matches: List[date] = []
        for dt in _iter_dates(start_date, end_date):
            if dt.weekday() not in days:
                continue
            if anchor and dt >= anchor and (dt - anchor).days % 14 == 0:
                matches.append(dt)
        return matches
    if freq == "monthly_by_date":
        days_of_month = []
        for dom in schedule.get("days_of_month") or []:
            try:
                dom_int = int(dom)
            except (TypeError, ValueError):
                continue
            if 1 <= dom_int <= 31:
                days_of_month.append(dom_int)
        return [
            dt
            for dt in _iter_dates(start_date, end_date)
            if days_of_month and dt.day in days_of_month
        ]
    if freq == "monthly_nth_weekday":
        week_of_month = schedule.get("week_of_month")
        day_of_week = schedule.get("day_of_week")
        try:
            target_week = int(week_of_month)
        except (TypeError, ValueError):
            target_week = None
        if isinstance(week_of_month, str) and week_of_month.lower() == "last":
            target_week = -1
        try:
            target_dow = int(day_of_week)
        except (TypeError, ValueError):
            target_dow = None
        if target_dow is None:
            return []

        matches: List[date] = []
        for month_start in _iter_month_starts(start_date, end_date):
            days_in_month: List[date] = []
            current = month_start
            while current.month == month_start.month and current <= end_date:
                if current >= start_date and current.weekday() == target_dow:
                    days_in_month.append(current)
                current += timedelta(days=1)
            if not days_in_month:
                continue
            if target_week == -1:
                candidate = days_in_month[-1]
            elif target_week and 1 <= target_week <= len(days_in_month):
                candidate = days_in_month[target_week - 1]
            else:
                candidate = days_in_month[0]
            if start_date <= candidate <= end_date:
                matches.append(candidate)
        return matches
    return []


def _time_to_minutes(value: str) -> Optional[int]:
    value = value.strip().lower()
    match = re.match(r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)?", value)
    if not match:
        return None
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    ampm = match.group("ampm")
    if ampm:
        if hours == 12:
            hours = 0
        if ampm == "pm":
            hours += 12
    return hours * 60 + minutes


def _parse_time_range_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(
        r"(\d{1,2}:\d{2}\s*(?:am|pm)?)\s*(?:-|\u2013)\s*(\d{1,2}:\d{2}\s*(?:am|pm)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    start = _time_to_minutes(match.group(1))
    end = _time_to_minutes(match.group(2))
    if start is None or end is None:
        return None, None
    if end <= start:
        end += 24 * 60
    return start, end


def _parse_structured_timing(timing: object) -> Tuple[Optional[int], Optional[int], str]:
    if isinstance(timing, dict):
        start_text = str(timing.get("start_time", "")).strip()
        end_text = str(timing.get("end_time", "")).strip()
        label = f"{start_text}-{end_text}".strip("-") if start_text or end_text else ""
        start = _time_to_minutes(start_text) if start_text else None
        end = _time_to_minutes(end_text) if end_text else None
        if start is None or end is None:
            return None, None, label
        if end <= start:
            end += 24 * 60
        return start, end, label
    if isinstance(timing, str):
        start, end = _parse_time_range_text(timing)
        return start, end, timing.strip()
    return None, None, ""


def _materialize_habit_snapshots_for_conflicts(domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    initial = (domain.get("initial_state", {}) or {})
    base_habits = deepcopy(initial.get("habits_state") or {})

    # Track where each habit is defined (for path construction in Rule 5)
    habit_sources: Dict[str, str] = {}
    for habit_name in base_habits.keys():
        habit_sources[habit_name] = "initial_state.habits_state"

    snapshots = [
        {
            "window_id": "initial_state",
            "time_range": initial.get("time_range"),
            "habits": deepcopy(base_habits),
            "habit_sources": deepcopy(habit_sources),
        }
    ]
    current = deepcopy(base_habits)

    for w_idx, window in enumerate(domain.get("time_windows") or []):
        window_id = window.get("window_id") or f"w{w_idx + 1}"
        for op in _get_delta_changes(window.get("habits_delta")):
            name = op.get("habit_name") or "unnamed_habit"
            op_type = (op.get("change_type") or op.get("op") or "").lower()
            delta = op.get("delta")
            if op_type == "acquire" and isinstance(delta, dict):
                current[name] = deepcopy(delta)
                habit_sources[name] = f"time_windows[{w_idx}].habits_delta.changes[?habit_name='{name}']"
            elif op_type == "adjust" and isinstance(delta, dict):
                existing = current.get(name, {})
                if not isinstance(existing, dict):
                    existing = {}
                merged = deepcopy(existing)
                merged.update(delta)
                current[name] = merged
                # For adjust, the habit was defined earlier, but we track the latest modification
                if name not in habit_sources:
                    habit_sources[name] = f"time_windows[{w_idx}].habits_delta.changes[?habit_name='{name}']"
            elif op_type == "drop":
                current.pop(name, None)
                habit_sources.pop(name, None)
        snapshots.append(
            {
                "window_id": window_id,
                "time_range": window.get("time_range"),
                "habits": deepcopy(current),
                "habit_sources": deepcopy(habit_sources),
            }
        )
    return snapshots


def _collect_temporal_events(dynamic_profiles: Dict[str, Dict]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for domain_name, profile in dynamic_profiles.items():
        for snapshot in _materialize_habit_snapshots_for_conflicts(profile or {}):
            window_id = snapshot.get("window_id") or "unknown_window"
            normalized_window_id = _normalize_window_id_label(window_id)

            start_date, end_date = _parse_window_date_range(snapshot.get("time_range"))

            # For initial_state without time_range, use a default sampling period
            if not start_date or not end_date:
                if normalized_window_id == "initial":
                    # Default initial_state to the pre-2024 window for conflict detection
                    start_date = date(2023, 10, 1)
                    end_date = date(2023, 12, 31)
                else:
                    continue

            for habit_name, habit in (snapshot.get("habits") or {}).items():
                start_min, end_min, timing_label = _parse_structured_timing(
                    habit.get("timing")
                )
                if start_min is None or end_min is None:
                    continue
                schedule = habit.get("schedule") or {}
                occurrences = _dates_from_schedule(schedule, start_date, end_date)
                if not occurrences:
                    occurrences = list(_iter_dates(start_date, end_date))
                location = habit.get("location", "")
                for dt in occurrences:
                    events.append(
                        {
                            "domain": domain_name,
                            "window_id": normalized_window_id,
                            "window_range": snapshot.get("time_range"),
                            "habit": habit_name,
                            "priority": (habit.get("priority") or "").lower(),
                            "timing": timing_label,
                            "start_min": start_min,
                            "end_min": end_min,
                            "location": location,
                            "date": dt,
                        }
                    )
    return events


def _conflict_habit_summary(event: Dict[str, object]) -> Dict[str, object]:
    return {
        "domain": event.get("domain"),
        "habit": event.get("habit"),
        "priority": event.get("priority"),
        "timing": event.get("timing"),
        "location": event.get("location", ""),
        "window_id": event.get("window_id"),
    }


def _canonical_pair(
    a: Dict[str, object], b: Dict[str, object], window_id: str | None
) -> Tuple[Dict[str, object], Dict[str, object], Tuple[str | None, str, str, str, str]]:
    """
    Order pair deterministically so we don't duplicate conflicts for swapped pairs.
    """
    key_a = (str(a.get("domain") or ""), str(a.get("habit") or ""))
    key_b = (str(b.get("domain") or ""), str(b.get("habit") or ""))
    if key_a <= key_b:
        return a, b, (window_id, key_a[0], key_a[1], key_b[0], key_b[1])
    return b, a, (window_id, key_b[0], key_b[1], key_a[0], key_a[1])


def detect_temporal_conflicts(dynamic_profiles: Dict[str, Dict]) -> Dict[str, object]:
    events = _collect_temporal_events(dynamic_profiles)
    aggregated: Dict[
        Tuple[str | None, str, str, str, str], Dict[str, object]
    ] = {}
    grouped: Dict[Tuple[str, date], List[Dict[str, object]]] = {}
    for ev in events:
        dt = ev.get("date")
        if not isinstance(dt, date):
            continue
        key = (ev.get("window_id") or "unknown_window", dt)
        grouped.setdefault(key, []).append(ev)

    for (window_id, dt), bucket in grouped.items():
        bucket = sorted(bucket, key=lambda e: e["start_min"])
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]

                # Get locations
                loc_a = str(a.get("location") or "").strip()
                loc_b = str(b.get("location") or "").strip()

                # Determine if there's a conflict based on location
                has_conflict = False
                conflict_type = ""

                if loc_a and loc_b and loc_a == loc_b:
                    # Same location: conflict if time overlaps
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "same_location_time_overlap"
                elif loc_a and loc_b and loc_a != loc_b:
                    # Different locations: conflict if less than 30 min gap
                    gap_min = b["start_min"] - a["end_min"]
                    if gap_min < 30:
                        has_conflict = True
                        conflict_type = "different_location_insufficient_gap"
                else:
                    # One or both locations are empty, treat as same location
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "time_overlap_no_location"

                if has_conflict:
                    left, right, key = _canonical_pair(a, b, window_id)
                    overlap_range = f"{_format_minutes(max(a['start_min'], b['start_min']))}-{_format_minutes(min(a['end_min'], b['end_min']))}"

                    entry = aggregated.setdefault(
                        key,
                        {
                            "window_id": window_id,
                            "window_range": bucket[0].get("window_range"),
                            "habit_a": _conflict_habit_summary(left),
                            "habit_b": _conflict_habit_summary(right),
                            "conflict_type": conflict_type,
                            "occurrences": 0,
                            "sample_dates": [],
                            "overlap_examples": [],
                        },
                    )
                    entry["occurrences"] = entry.get("occurrences", 0) + 1
                    if len(entry["sample_dates"]) < 3:
                        entry["sample_dates"].append(dt.isoformat())
                    if len(entry["overlap_examples"]) < 3:
                        gap_info = ""
                        if conflict_type == "different_location_insufficient_gap":
                            gap_min = b["start_min"] - a["end_min"]
                            gap_info = f", gap: {gap_min} min"
                        entry["overlap_examples"].append(
                            {"date": dt.isoformat(), "overlap": overlap_range + gap_info}
                        )
    conflicts = list(aggregated.values())

    # Build per-window conflict graph summary (node degree by habit).
    graph_by_window: Dict[str, Dict[str, object]] = {}
    TOP_N = 8
    for entry in conflicts:
        window_id = entry.get("window_id") or "unknown_window"
        occ = int(entry.get("occurrences") or 1)
        window_graph = graph_by_window.setdefault(
            window_id, {"total_conflicts": 0, "node_degrees": {}}
        )
        window_graph["total_conflicts"] += occ
        for habit_key in ("habit_a", "habit_b"):
            habit_info = entry.get(habit_key) or {}
            node_key = (habit_info.get("domain") or "", habit_info.get("habit") or "")
            node_degrees: Dict[Tuple[str, str], int] = window_graph["node_degrees"]  # type: ignore
            node_degrees[node_key] = node_degrees.get(node_key, 0) + occ

    # Convert node_degrees to sorted top list.
    for window_id, data in graph_by_window.items():
        node_degrees = data.get("node_degrees", {}) or {}
        top_nodes = sorted(
            (
                {
                    "domain": domain,
                    "habit": habit,
                    "degree": degree,
                }
                for (domain, habit), degree in node_degrees.items()
            ),
            key=lambda x: (-x["degree"], x["domain"], x["habit"]),
        )[:TOP_N]
        data["top_nodes"] = top_nodes
        data.pop("node_degrees", None)

    return {"conflicts": conflicts, "graph_by_window": graph_by_window}


def _format_minutes(minutes: int) -> str:
    """Convert minutes since midnight to time string like '6:30 AM'."""
    hours = minutes // 60
    mins = minutes % 60

    if hours >= 12:
        period = "PM"
        display_hour = hours if hours == 12 else hours - 12
    else:
        period = "AM"
        display_hour = hours if hours > 0 else 12

    return f"{display_hour}:{mins:02d} {period}"


def _detect_rule5_time_conflict_issues(profile: Dict) -> Dict[str, object]:
    """
    Detect time conflicts using date-aware temporal event expansion (same logic as cross-domain).

    Key improvements over old implementation:
    1. Considers actual dates - habits on different dates don't conflict
    2. Respects schedule frequency (daily/weekly/monthly)
    3. Counts actual occurrence conflicts, not just habit pair conflicts
    4. Groups events by (window_id, date) before checking overlaps

    Returns:
        {
            "conflicts": [...],  # List of conflict entries with occurrence counts
            "graph_by_window": {  # Conflict graph analysis
                "window_id": {
                    "total_conflicts": int (total occurrences),
                    "top_nodes": [{"habit": str, "degree": int}, ...]
                }
            }
        }
    """
    if isinstance(profile, list):
        profile = profile[0] if profile and isinstance(profile[0], dict) else {}

    # Collect all temporal events (expanded by date)
    events: List[Dict[str, object]] = []
    for snapshot in _materialize_habit_snapshots_for_conflicts(profile or {}):
        window_id = snapshot.get("window_id") or "unknown_window"
        habit_sources = snapshot.get("habit_sources") or {}
        start_date, end_date = _parse_window_date_range(snapshot.get("time_range"))
        if not start_date or not end_date:
            continue

        for habit_name, habit in (snapshot.get("habits") or {}).items():
            start_min, end_min, timing_label = _parse_structured_timing(habit.get("timing"))
            if start_min is None or end_min is None:
                continue

            schedule = habit.get("schedule") or {}
            occurrences = _dates_from_schedule(schedule, start_date, end_date)
            if not occurrences:
                # No schedule specified, assume daily
                occurrences = list(_iter_dates(start_date, end_date))

            location = habit.get("location", "")
            habit_source_path = habit_sources.get(habit_name, "")

            for dt in occurrences:
                events.append({
                    "window_id": window_id,
                    "window_range": snapshot.get("time_range"),
                    "habit": habit_name,
                    "timing": timing_label,
                    "start_min": start_min,
                    "end_min": end_min,
                    "location": location,
                    "date": dt,
                    "habit_source_path": habit_source_path,
                })

    # Group events by (window_id, date) and detect conflicts
    aggregated: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    grouped: Dict[Tuple[str, date], List[Dict[str, object]]] = {}

    for ev in events:
        dt = ev.get("date")
        if not isinstance(dt, date):
            continue
        key = (ev.get("window_id") or "unknown_window", dt)
        grouped.setdefault(key, []).append(ev)

    # Check for overlaps within each (window, date) bucket
    for (window_id, dt), bucket in grouped.items():
        bucket = sorted(bucket, key=lambda e: e["start_min"])
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]

                # Get locations
                loc_a = str(a.get("location") or "").strip()
                loc_b = str(b.get("location") or "").strip()

                # Determine if there's a conflict based on location
                has_conflict = False
                conflict_type = ""

                if loc_a and loc_b and loc_a == loc_b:
                    # Same location: conflict if time overlaps
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "same_location_time_overlap"
                elif loc_a and loc_b and loc_a != loc_b:
                    # Different locations: conflict if less than 30 min gap
                    # Gap is the time between end of earlier event and start of later event
                    gap_min = b["start_min"] - a["end_min"]
                    if gap_min < 30:
                        has_conflict = True
                        conflict_type = "different_location_insufficient_gap"
                else:
                    # One or both locations are empty, treat as same location
                    if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                        has_conflict = True
                        conflict_type = "time_overlap_no_location"

                if has_conflict:
                    # Canonical pair key to aggregate same habit pairs
                    habit_a = str(a.get("habit") or "")
                    habit_b = str(b.get("habit") or "")
                    if habit_a <= habit_b:
                        pair_key = (window_id, habit_a, habit_b)
                        left, right = a, b
                    else:
                        pair_key = (window_id, habit_b, habit_a)
                        left, right = b, a

                    overlap_range = f"{_format_minutes(max(a['start_min'], b['start_min']))}-{_format_minutes(min(a['end_min'], b['end_min']))}"

                    # Get the source paths for both habits
                    path_a = str(a.get("habit_source_path") or "")
                    path_b = str(b.get("habit_source_path") or "")

                    # If source paths are not available, construct default paths
                    if not path_a:
                        if window_id == "initial_state":
                            path_a = f"initial_state.habits_state.{habit_a}"
                        else:
                            path_a = f"{window_id}.habits.{habit_a}"

                    if not path_b:
                        if window_id == "initial_state":
                            path_b = f"initial_state.habits_state.{habit_b}"
                        else:
                            path_b = f"{window_id}.habits.{habit_b}"

                    # Use a combined path for the conflict
                    path = f"{path_a} <-> {path_b}"

                    # Aggregate by pair key
                    entry = aggregated.setdefault(pair_key, {
                        "window_id": window_id,
                        "window_range": bucket[0].get("window_range"),
                        "habit_a": habit_a,
                        "habit_b": habit_b,
                        "habit_a_timing": left.get("timing"),
                        "habit_b_timing": right.get("timing"),
                        "habit_a_location": left.get("location", ""),
                        "habit_b_location": right.get("location", ""),
                        "conflict_type": conflict_type,
                        "path": path,
                        "occurrences": 0,
                        "overlap_examples": [],
                    })
                    entry["occurrences"] = entry.get("occurrences", 0) + 1
                    if len(entry["overlap_examples"]) < 3:
                        gap_info = ""
                        if conflict_type == "different_location_insufficient_gap":
                            gap_min = b["start_min"] - a["end_min"]
                            gap_info = f", gap: {gap_min} min"
                        entry["overlap_examples"].append({
                            "date": dt.isoformat(),
                            "overlap": overlap_range + gap_info,
                        })

    conflicts = list(aggregated.values())

    # Add unique IDs to conflicts
    for idx, conflict in enumerate(conflicts):
        if "id" not in conflict:
            conflict["id"] = f"rule5_{idx:03d}"
            # Build message with occurrence info
            habit_a = conflict.get("habit_a")
            habit_b = conflict.get("habit_b")
            timing_a = conflict.get("habit_a_timing")
            timing_b = conflict.get("habit_b_timing")
            loc_a = conflict.get("habit_a_location", "")
            loc_b = conflict.get("habit_b_location", "")
            conflict_type = conflict.get("conflict_type", "")
            occ_count = conflict.get("occurrences", 1)
            window_id = conflict.get("window_id")

            # Build location description
            loc_desc = ""
            if conflict_type == "same_location_time_overlap":
                loc_desc = f" (both at '{loc_a}')"
            elif conflict_type == "different_location_insufficient_gap":
                loc_desc = f" ('{habit_a}' at '{loc_a}', '{habit_b}' at '{loc_b}' - insufficient travel time)"
            elif loc_a or loc_b:
                loc_desc = f" ('{habit_a}' at '{loc_a}', '{habit_b}' at '{loc_b}')"

            conflict["message"] = (
                f"[ID: {conflict['id']}] Time conflict in {window_id}: "
                f"'{habit_a}' ({timing_a}) overlaps with '{habit_b}' ({timing_b}){loc_desc}. "
                f"Occurs {occ_count} time(s) in this window."
            )

    return {
        "conflicts": conflicts,
    }


ATTRIBUTE_CANONICAL_KEYS: Dict[str, str] = {
    # Device/possession clusters
    "user_material_possessions": "user_devices_and_possessions",
    "user_tech_gadgets": "user_devices_and_possessions",
    "user_owned_devices": "user_devices_and_possessions",
    "user_smart_home_devices": "user_devices_and_possessions",
}


def _get_delta_changes(delta_section: Dict | None) -> List[Dict[str, object]]:
    """Get changes from delta section, supporting both 'changes' and legacy 'changes' field names."""
    if isinstance(delta_section, dict):
        changes = delta_section.get("changes") or delta_section.get("changes")
        if isinstance(changes, list):
            return changes
    return []


def _extract_initial_state_entries(
    entries: Dict[str, object] | List[Dict[str, object]] | None,
) -> Dict[str, Dict[str, object]]:
    """
    Extract initial state entries from the new format.
    Each entry is a dict like {"attribute_name": "attribute_value"} or a map of
    names to values.
    """
    state: Dict[str, Dict[str, object]] = {}
    # Some generators produce an object map instead of a list of singleton dicts.
    if isinstance(entries, dict):
        for key, value in entries.items():
            state[key] = {"current_value": value}
        return state

    for entry in entries or []:
        if not isinstance(entry, dict):
            # Treat bare strings as keys with unknown value to avoid crashing.
            if isinstance(entry, str):
                state[entry] = {"current_value": None}
            continue
        for key, value in entry.items():
            state[key] = {"current_value": value}
    return state


def _init_user_attributes_state(
    initial_state: Dict[str, object] | None,
) -> Dict[str, Dict[str, object]]:
    """
    Build a {singular, collections} map from the new schema, with a legacy
    fallback to user_attributes_state.initial when present.
    """
    attrs = {"singular": {}, "collections": {}}
    user_attrs_state = (initial_state or {}).get("user_attributes_state") or {}
    if isinstance(user_attrs_state, dict):
        singular = user_attrs_state.get("singular")
        collections = user_attrs_state.get("collections")
        if isinstance(singular, dict):
            attrs["singular"] = deepcopy(singular)
        if isinstance(collections, dict):
            attrs["collections"] = deepcopy(collections)

        # Legacy fallback: user_attributes_state.initial (dict or list of dicts)
        if not attrs["singular"] and not attrs["collections"]:
            legacy_entries = _extract_initial_state_entries(
                user_attrs_state.get("initial")
            )
            if legacy_entries:
                attrs["singular"] = {
                    name: entry.get("current_value")
                    for name, entry in legacy_entries.items()
                }
    return attrs


def _apply_user_attribute_changes(
    state: Dict[str, Dict[str, object]],
    changes: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[tuple[str, str], Dict[str, object]]]:
    """
    Apply attribute deltas using the new singular/collections semantics.
    """
    updated = {
        "singular": deepcopy(state.get("singular") or {}),
        "collections": deepcopy(state.get("collections") or {}),
    }
    changes: Dict[tuple[str, str], Dict[str, object]] = {}
    if not changes:
        return updated, changes

    for change in changes:
        if not isinstance(change, dict):
            continue
        raw_attr_type = (change.get("attribute_type") or "").lower()
        inferred_type = "collections" if change.get("collection_name") else "singular"
        attr_type = "collections" if raw_attr_type in {"collection", "collections"} else raw_attr_type or inferred_type
        name = change.get("attribute_name") or change.get("collection_name")
        if not name:
            continue

        op_type = (change.get("change_type") or change.get("op") or "").lower()
        reason = change.get("reason", "")
        prev_value = deepcopy(updated.get(attr_type, {}).get(name))
        delta_payload = change.get("delta")
        if delta_payload is None:
            delta_payload = change.get("new_state")

        if attr_type == "collections":
            prev_list = (
                prev_value
                if isinstance(prev_value, list)
                else ([] if prev_value is None else [prev_value])
            )
            if op_type == "remove":
                if delta_payload is None:
                    new_list = []
                else:
                    removals = set()
                    if isinstance(delta_payload, list):
                        removals = set(delta_payload)
                    elif delta_payload is not None:
                        removals = {delta_payload}
                    new_list = [item for item in prev_list if item not in removals]
            elif op_type == "add":
                additions = (
                    delta_payload
                    if isinstance(delta_payload, list)
                    else ([] if delta_payload is None else [delta_payload])
                )
                new_list = list(prev_list)
                for item in additions:
                    if item not in new_list:
                        new_list.append(item)
            elif op_type in {"modify", "replace", "set"}:
                new_list = (
                    delta_payload if isinstance(delta_payload, list) else prev_list
                )
            else:
                new_list = prev_list
            new_value = new_list
        else:
            # singular attribute
            new_value = delta_payload if delta_payload is not None else prev_value

        updated.setdefault(attr_type, {})[name] = new_value
        changes[(attr_type, name)] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "change_type": op_type,
            "attribute_type": attr_type,
        }

    return updated, changes


def _user_attributes_state_to_list(
    state: Dict[str, Dict[str, object]],
    changes: Dict[tuple[str, str], Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """
    Convert user_attributes state map into a list with change metadata.
    """
    changes = changes or {}
    result: List[Dict[str, object]] = []
    for attr_type in ("singular", "collections"):
        entries = state.get(attr_type) or {}
        for name, value in sorted(entries.items()):
            item: Dict[str, object] = {
                "name": name,
                "attribute_type": attr_type,
                "current_value": value,
            }
            change = changes.get((attr_type, name))
            if change:
                item["change_type"] = change.get("change_type") or change.get("op")
                if "previous_value" in change:
                    item["previous_value"] = change.get("previous_value")
                if change.get("change_reason"):
                    item["change_reason"] = change.get("change_reason")
            result.append(item)
    return result


def _init_generic_state(entries: Dict[str, object] | None) -> Dict[str, Dict[str, object]]:
    """
    Convert an initial {name: value} mapping into {name: {current_value: value}}.
    """
    state: Dict[str, Dict[str, object]] = {}
    if isinstance(entries, dict):
        for key, value in entries.items():
            state[key] = {"current_value": deepcopy(value)}
    return state


def _apply_habit_changes(
    state: Dict[str, Dict[str, object]],
    changes: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Apply habit deltas; adjust merges partial fields into the previous habit.
    """
    updated = deepcopy(state)
    changes: Dict[str, Dict[str, object]] = {}
    if not changes:
        return updated, changes

    for change in changes:
        if not isinstance(change, dict):
            continue
        name = change.get("habit_name")
        if not name:
            continue

        op_type = (change.get("change_type") or change.get("op") or "").lower()
        reason = change.get("reason", "")
        prev_value = deepcopy((updated.get(name) or {}).get("current_value"))
        delta_payload = change.get("delta")
        if delta_payload is None:
            delta_payload = change.get("new_state")

        if delta_payload is None:
            candidate = {
                k: v
                for k, v in change.items()
                if k
                not in {
                    "habit_name",
                    "op",
                    "reason",
                    "before",
                    "after",
                    "change_reason",
                    "delta",
                    "new_state",
                    "attribute_type",
                }
            }
            if candidate:
                delta_payload = candidate

        if op_type == "drop":
            new_value = None
            updated.pop(name, None)
        elif op_type == "adjust":
            base = deepcopy(prev_value) if isinstance(prev_value, dict) else {}
            if isinstance(delta_payload, dict):
                base.update(delta_payload)
            elif delta_payload is not None:
                base = delta_payload
            new_value = base
        else:
            # acquire or fallback
            new_value = delta_payload if delta_payload is not None else prev_value

        if op_type != "drop":
            updated[name] = {"current_value": new_value}
        changes[name] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "change_type": op_type,
        }

    return updated, changes


def _apply_preference_changes(
    state: Dict[str, Dict[str, object]],
    changes: List[Dict[str, object]] | None,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Apply preference deltas with the new delta field.
    """
    updated = deepcopy(state)
    changes: Dict[str, Dict[str, object]] = {}
    if not changes:
        return updated, changes

    for change in changes:
        if not isinstance(change, dict):
            continue
        name = change.get("preference_name")
        if not name:
            continue
        op_type = (change.get("change_type") or change.get("op") or "").lower()
        reason = change.get("reason", "")
        prev_value = deepcopy((updated.get(name) or {}).get("current_value"))
        delta_payload = change.get("delta")
        if delta_payload is None:
            delta_payload = change.get("new_state")

        if op_type in {"drop", "remove"}:
            new_value = None
            updated.pop(name, None)
        else:
            new_value = delta_payload if delta_payload is not None else prev_value

        if op_type not in {"drop", "remove"}:
            updated[name] = {"current_value": new_value}
        changes[name] = {
            "previous_value": prev_value,
            "change_reason": reason,
            "change_type": op_type,
        }

    return updated, changes


def _state_dict_to_list(
    state: Dict[str, Dict[str, object]],
    changes: Dict[str, Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """
    Convert state dict to list format, including change information if available.

    Args:
        state: Current state dict
        changes: Optional dict of changes (from _apply_delta_changes)
    """
    if changes is None:
        changes = {}

    result = []
    for key, entry in sorted(state.items()):
        current_value = (
            entry.get("current_value") if isinstance(entry, dict) else entry
        )
        item: Dict[str, object] = {
            "name": key,
            "current_value": current_value,
        }
        if isinstance(entry, dict) and "attribute_type" in entry:
            item["attribute_type"] = entry.get("attribute_type")

        # Add change information if this item changed
        if key in changes:
            change_info = changes[key]
            item["previous_value"] = change_info.get("previous_value")
            item["change_reason"] = change_info.get("change_reason")
            item["change_type"] = change_info.get("change_type") or change_info.get("op")

        result.append(item)

    missing_keys = [key for key in changes.keys() if key not in state]
    for key in sorted(missing_keys):
        change_info = changes.get(key, {})
        item = {
            "name": key,
            "current_value": None,
            "change_type": change_info.get("change_type") or change_info.get("op"),
        }
        if "previous_value" in change_info:
            item["previous_value"] = change_info.get("previous_value")
        if change_info.get("change_reason"):
            item["change_reason"] = change_info.get("change_reason")
        result.append(item)

    return result


def _resolve_window_states(dynamic_profile_data: Dict) -> List[Dict]:
    """
    Resolve initial state + deltas into full window states for downstream prompts.
    Works with the newer dynamic_profile format that has:
      - top-level "initial_state"
      - per-window *_delta sections with {op, *_name, delta, reason}
    Preserves change information (previous_value, change_reason, op) for each state item.

    Now includes w0 (initial window) representing 2023 Q4 with all initial states marked as "acquire".
    """
    resolved: List[Dict] = []
    # Build initial snapshots from the top-level initial_state
    initial_state = dynamic_profile_data.get("initial_state") or {}
    current_attributes = _init_user_attributes_state(initial_state)
    current_habits = _init_generic_state(
        initial_state.get("habits_state")
    )
    current_preferences = _init_generic_state(
        initial_state.get("preferences_state")
    )

    # Create w0 (initial window) for 2023 Q4 with all initial states as "acquire"
    w0_attributes_changes: Dict[tuple[str, str], Dict[str, object]] = {}
    for attr_type in ("singular", "collections"):
        entries = current_attributes.get(attr_type) or {}
        for name, value in entries.items():
            w0_attributes_changes[(attr_type, name)] = {
                "change_type": "acquire",
                "previous_value": None,
                "change_reason": "Initial state at the start of tracking period.",
            }

    w0_habits_changes: Dict[str, Dict[str, object]] = {}
    for name, entry in current_habits.items():
        w0_habits_changes[name] = {
            "change_type": "acquire",
            "previous_value": None,
            "change_reason": "Initial habit at the start of tracking period.",
        }

    w0_preferences_changes: Dict[str, Dict[str, object]] = {}
    for name, entry in current_preferences.items():
        w0_preferences_changes[name] = {
            "change_type": "acquire",
            "previous_value": None,
            "change_reason": "Initial preference at the start of tracking period.",
        }

    # Get time_range from initial_state if available, otherwise default to 2023 Q4
    w0_time_range = initial_state.get("time_range") or ["2023-10-01", "2023-12-31"]
    w0_description = initial_state.get("summary") or "Initial state window representing the last quarter of 2023."

    resolved.append(
        {
            "window_id": "w0",
            "time_range": w0_time_range,
            "window_description": w0_description,
            "summary": w0_description,
            "user_attributes_state": _user_attributes_state_to_list(
                current_attributes, changes=w0_attributes_changes
            ),
            "habits_state": _state_dict_to_list(
                current_habits, changes=w0_habits_changes
            ),
            "preferences_state": _state_dict_to_list(
                current_preferences, changes=w0_preferences_changes
            ),
        }
    )

    for window in dynamic_profile_data.get("time_windows", []):
        window_id = window.get("window_id")
        if not window_id:
            continue

        # Track changes for each state type
        attributes_snapshot, attributes_changes = _apply_user_attribute_changes(
            current_attributes,
            _get_delta_changes(window.get("user_attributes_delta")),
        )

        habits_snapshot, habits_changes = _apply_habit_changes(
            current_habits,
            _get_delta_changes(window.get("habits_delta")),
        )

        preferences_snapshot, preferences_changes = _apply_preference_changes(
            current_preferences,
            _get_delta_changes(window.get("preferences_delta")),
        )

        resolved.append(
            {
                "window_id": window_id,
                "time_range": window.get("time_range"),
                "window_description": window.get("window_description"),
                "summary": window.get("summary", ""),
                "user_attributes_state": _user_attributes_state_to_list(
                    attributes_snapshot, changes=attributes_changes
                ),
                "habits_state": _state_dict_to_list(
                    habits_snapshot, changes=habits_changes
                ),
                "preferences_state": _state_dict_to_list(
                    preferences_snapshot, changes=preferences_changes
                ),
            }
        )

        current_attributes = attributes_snapshot
        current_habits = habits_snapshot
        current_preferences = preferences_snapshot

    return resolved


def _collect_cross_domain_attribute_conflicts(
    dynamic_profiles: Dict[str, Dict],
) -> List[Dict[str, object]]:
    """
    Collect all shared attributes across domains after key alignment.

    Simply lists all attributes that appear in multiple domains, showing their values
    in each domain. Does NOT judge whether there are conflicts - that's for the LLM.

    Returns: List of shared attributes, each containing:
    - shared_key: the aligned attribute name
    - attribute_type: "singular" or "collections"
    - windows: list of windows where this attribute appears, with domain values
    """
    # Group by (attr_type, attr_name) across domains
    # After key alignment, attribute names should already be normalized
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, object]]]] = {}

    def _record_entry(
        *,
        window_id: str,
        time_range: object,
        attr_type: str,
        attr_name: str,
        value: object,
        domain: str,
    ) -> None:
        key = (("collections" if attr_type == "collections" else "singular"), attr_name)
        entry = {
            "domain": domain,
            "value": deepcopy(value),
            "time_range": time_range,
        }
        grouped.setdefault(key, {}).setdefault(window_id or "initial", []).append(entry)

    for domain_name, profile in dynamic_profiles.items():
        if not isinstance(profile, dict):
            continue

        user_attrs_state = (profile.get("initial_state") or {}).get("user_attributes_state") or {}
        for attr_type in ("singular", "collections"):
            entries = user_attrs_state.get(attr_type)
            if isinstance(entries, dict):
                for attr_name, value in entries.items():
                    _record_entry(
                        window_id="initial",
                        time_range=None,
                        attr_type=attr_type,
                        attr_name=attr_name,
                        value=value,
                        domain=domain_name,
                    )

        # Legacy fallback: user_attributes_state.initial
        legacy_entries = _extract_initial_state_entries(user_attrs_state.get("initial"))
        for attr_name, entry in legacy_entries.items():
            _record_entry(
                window_id="initial",
                time_range=None,
                attr_type="singular",
                attr_name=attr_name,
                value=entry.get("current_value"),
                domain=domain_name,
            )

        for window in _resolve_window_states(profile):
            window_id = window.get("window_id") or "unknown_window"
            time_range = window.get("time_range")
            for item in window.get("user_attributes_state", []) or []:
                attr_name = item.get("name")
                if not attr_name:
                    continue
                attr_type = (item.get("attribute_type") or "singular").lower()
                _record_entry(
                    window_id=window_id,
                    time_range=time_range,
                    attr_type=attr_type,
                    attr_name=attr_name,
                    value=item.get("current_value"),
                    domain=domain_name,
                )

    # Collect all shared attributes (appearing in multiple domains)
    shared_attributes: List[Dict[str, object]] = []
    for (attr_type, attr_name), windows_map in grouped.items():
        windows_payload: List[Dict[str, object]] = []

        for window_id, entries in windows_map.items():
            # Only include windows where this attribute appears in multiple domains
            if len(entries) <= 1:
                continue

            time_range = next(
                (entry.get("time_range") for entry in entries if entry.get("time_range") is not None),
                None,
            )

            # Simply list all domain values
            domains_list = [
                {
                    "domain": entry["domain"],
                    "value": entry["value"],
                }
                for entry in entries
            ]

            windows_payload.append(
                {
                    "window_id": window_id,
                    "time_range": time_range,
                    "domains": domains_list,
                }
            )

        # Only include attributes that appear in multiple domains in at least one window
        if windows_payload:
            shared_attributes.append(
                {
                    "shared_key": attr_name,
                    "attribute_type": attr_type,
                    "windows": windows_payload,
                }
            )

    return shared_attributes


def _apply_key_alignment(
    dynamic_profiles: Dict[str, Dict],
    alignment_payload: Dict[str, object],
) -> tuple[Dict[str, Dict], Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Apply LLM-suggested key alignment mappings and deterministic renames (no value merging).
    """
    resolved_profiles = deepcopy(dynamic_profiles)
    if not isinstance(alignment_payload, dict):
        return resolved_profiles, {}, {}

    # Normalize the LLM output into a per-domain mapping:
    # domain -> original_key -> canonical_key
    domain_key_mapping: Dict[str, Dict[str, str]] = {}
    canonical_descriptions: Dict[str, str] = {}

    canonical_entries = alignment_payload.get("canonical_key_mappings") or alignment_payload.get("canonical_keys") or {}
    if isinstance(canonical_entries, dict):
        for canonical_key, entry in canonical_entries.items():
            if not isinstance(entry, dict):
                continue
            description = entry.get("description") or entry.get("note")
            if description:
                canonical_descriptions[canonical_key] = description

            for domain_entry in entry.get("domains") or []:
                if not isinstance(domain_entry, dict):
                    continue
                domain_name = domain_entry.get("domain")
                original_key = (
                    domain_entry.get("original_key")
                    or domain_entry.get("attribute_name")
                    or domain_entry.get("key")
                )
                if not domain_name or not original_key:
                    continue
                domain_mapping = domain_key_mapping.setdefault(domain_name, {})
                domain_mapping[original_key] = canonical_key
                if "." in original_key:
                    base_key = original_key.rsplit(".", 1)[-1]
                    if base_key:
                        domain_mapping.setdefault(base_key, canonical_key)

    def _rename_initial_entries(entries: object, mapping: Dict[str, str]) -> object:
        """
        Rename keys in initial_state.user_attributes_state.initial while preserving shape (dict vs list).
        """
        if entries is None:
            return entries

        aggregated: Dict[str, object] = {}

        # Support both dict and list-of-dicts forms.
        items = []
        if isinstance(entries, dict):
            items = list(entries.items())
        elif isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    items.extend(entry.items())
        else:
            return entries

        for key, value in items:
            target_key = mapping.get(key, key)
            aggregated[target_key] = deepcopy(value)

        if isinstance(entries, dict):
            return aggregated
        return [{k: v} for k, v in aggregated.items()]

    def _drop_prefixed_duplicates(entries: object, mapping: Dict[str, str]) -> object:
        """
        Remove prefixed duplicates such as specific_<key>/user_specific_<key> when the base key is present.
        """
        prefixes = ("specific_", "user_specific_")

        def should_drop(key: str, container: Dict[str, object]) -> bool:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                base = key[len(prefix) :]
                canonical_base = mapping.get(base, base)
                canonical_key = mapping.get(key, key)
                # Drop if the base (or canonical base) is already present.
                if base in container or canonical_base in container or canonical_key == canonical_base:
                    return True
            return False

        if isinstance(entries, dict):
            return {k: v for k, v in entries.items() if not should_drop(k, entries)}

        if isinstance(entries, list):
            cleaned: List[object] = []
            for entry in entries:
                if isinstance(entry, dict):
                    cleaned.append({k: v for k, v in entry.items() if not should_drop(k, entry)})
                else:
                    cleaned.append(entry)
            return cleaned
        return entries

    for domain_name, profile in resolved_profiles.items():
        # Some LLM responses wrap the profile object in a singleton list; unwrap to keep a consistent shape.
        if isinstance(profile, list):
            if len(profile) == 1 and isinstance(profile[0], dict):
                profile = profile[0]
                resolved_profiles[domain_name] = profile
            else:
                raise ValueError(
                    f"Dynamic profile for {domain_name} should be an object, got list (len={len(profile)})"
                )
        if not isinstance(profile, dict):
            raise ValueError(
                f"Dynamic profile for {domain_name} should be a dict, got {type(profile).__name__}"
            )
        mapping = dict(ATTRIBUTE_CANONICAL_KEYS)
        mapping.update(domain_key_mapping.get(domain_name, {}))

        # Rename in initial_state.user_attributes_state (singular/collections)
        initial_state = (profile.get("initial_state") or {}).get(
            "user_attributes_state"
        ) or {}
        for section in ("singular", "collections"):
            entries = initial_state.get(section)
            if isinstance(entries, dict):
                renamed_section = {
                    mapping.get(key, key): deepcopy(value)
                    for key, value in entries.items()
                }
                initial_state[section] = _drop_prefixed_duplicates(
                    renamed_section, mapping
                )

        # Legacy fallback: user_attributes_state.initial
        if "initial" in initial_state:
            renamed = _rename_initial_entries(initial_state.get("initial"), mapping)
            initial_state["initial"] = _drop_prefixed_duplicates(renamed, mapping)

        # Rename in each window's user_attributes_delta.changes
        for window in profile.get("time_windows", []):
            delta_ops = _get_delta_changes(
                (window.get("user_attributes_delta") or {})
            )
            for op in delta_ops:
                if not isinstance(op, dict):
                    continue
                attr_type_raw = (op.get("attribute_type") or "").lower()
                inferred_type = (
                    "collections" if op.get("collection_name") else "singular"
                )
                attr_type = (
                    "collections"
                    if attr_type_raw in {"collection", "collections"}
                    else attr_type_raw
                    or inferred_type
                )
                name_field = "collection_name" if attr_type == "collections" else "attribute_name"
                attr_name = (
                    op.get(name_field)
                    or op.get("attribute_name")
                    or op.get("collection_name")
                )
                if not attr_name:
                    continue
                mapped = mapping.get(attr_name)
                if mapped:
                    op[name_field] = mapped
                op["attribute_type"] = attr_type

    return resolved_profiles, domain_key_mapping, canonical_descriptions


def _apply_conflict_resolution_to_profiles(
    dynamic_profiles: Dict[str, Dict],
    resolution_payload: Dict[str, object] | None,
) -> Dict[str, Dict]:
    resolved_profiles = deepcopy(dynamic_profiles)
    if not isinstance(resolution_payload, dict):
        return resolved_profiles

    def _parse_path_tokens(path: str | None) -> List[object]:
        """
        Split a dotted path like "habits_delta.changes[0].new_state.timing"
        into ["habits_delta", "changes", 0, "new_state", "timing"].
        Supports selectors: [window_id=w3] or [habit_name=foo] -> {"_selector_key": "...", "_selector_value": "..."}.
        """
        if not isinstance(path, str):
            return []

        def _make_selector(raw: str) -> Dict[str, str] | str | int:
            raw = raw.strip()
            if raw.isdigit():
                return int(raw)
            if "=" in raw:
                key, value = raw.split("=", 1)
                return {"_selector_key": key.strip(), "_selector_value": value.strip()}
            return raw

        tokens: List[object] = []
        for segment in path.split("."):
            if not segment:
                continue
            for match in re.finditer(r"([^\[\]]+)|\[(.*?)\]", segment):
                plain, bracket = match.groups()
                if plain:
                    tokens.append(plain)
                elif bracket:
                    selector = _make_selector(bracket)
                    tokens.append(selector)
        return tokens

    def _is_selector_token(token: object) -> bool:
        return isinstance(token, dict) and "_selector_key" in token and "_selector_value" in token

    def _normalize_window_id(window_id: str | None) -> str:
        """
        Normalize window identifiers to match stored window_ids.
        Treats None/""/initial_state as "initial".
        """
        if window_id is None:
            return "initial"
        normalized = str(window_id).strip()
        lower = normalized.lower()
        if lower in {"", "initial", "initial_state", "initialstate", "init"}:
            return "initial"
        return normalized

    def _get_window_root(profile: Dict, window_id: str | None) -> Dict | None:
        normalized = _normalize_window_id(window_id)
        if normalized == "initial":
            return profile.setdefault("initial_state", {})
        for window in profile.get("time_windows", []):
            wid = window.get("window_id")
            if isinstance(wid, str) and wid.lower() == normalized.lower():
                return window
        return None

    def _strip_redundant_window_tokens(
        tokens: List[object], window_id: str | None, root: Dict | None
    ) -> List[object]:
        """
        Some patch formats repeat the window_id inside the path (e.g., habits_state.*).
        Strip that marker since we already route to the correct window root.
        """
        if not tokens or not isinstance(window_id, str):
            return tokens
        normalized = _normalize_window_id(window_id).lower()

        def _matches_window(token_str: str) -> bool:
            base = token_str.lower()
            if base.endswith("_state"):
                base = base[: -len("_state")]
            return base == normalized

        # Drop a leading window_id marker, e.g., "initial.habits_state..."
        if isinstance(tokens[0], str) and _matches_window(tokens[0]):
            tokens = tokens[1:]

        # Drop a window_id marker immediately after a state section.
        if len(tokens) >= 2 and isinstance(tokens[0], str) and isinstance(tokens[1], str):
            parent_container = None
            if isinstance(root, dict):
                parent_container = root.get(tokens[0])
            has_window_key = False
            if isinstance(parent_container, dict):
                for key in parent_container.keys():
                    if isinstance(key, str) and _matches_window(key):
                        has_window_key = True
                        break
            if _matches_window(tokens[1]) and tokens[0] in {
                "user_attributes_state",
                "habits_state",
                "preferences_state",
            } and not has_window_key:
                tokens = [tokens[0]] + tokens[2:]

        # Drop leading explicit container references like time_windows[2] or initial_state.*
        if len(tokens) >= 2 and tokens[0] == "time_windows" and (
            isinstance(tokens[1], int) or _is_selector_token(tokens[1])
        ):
            tokens = tokens[2:]
        if tokens:
            first = tokens[0]
            if isinstance(first, str) and first.lower() in {"initial_state", "time_windows"}:
                tokens = tokens[1:]

        return tokens

    def _strip_context_prefix(tokens: List[object]) -> List[object]:
        """
        Remove leading context markers (e.g., 'Dynamic profiles') that aren't part of the domain path.
        """
        prefixes = {
            "dynamic profiles",
            "dynamic profile",
            "dynamic_profiles",
            "dynamic_profile",
            "dynamicprofiles",
        }
        while tokens and isinstance(tokens[0], str) and tokens[0].strip().lower() in prefixes:
            tokens = tokens[1:]
        return tokens

    def _infer_window_id_from_tokens(tokens: List[object], profile: Dict | None) -> str | None:
        """
        Deduce the window_id using explicit markers (w3, initial) or time_windows indices/selectors.
        """
        for token in tokens:
            if isinstance(token, str):
                token_lower = token.lower()
                if token_lower in {"initial", "initial_state", "init"}:
                    return "initial"
                if re.fullmatch(r"w\d+", token_lower):
                    return token
            elif _is_selector_token(token):
                key = str(token.get("_selector_key", "")).lower()
                if key == "window_id":
                    return token.get("_selector_value")

        if not isinstance(profile, dict):
            return None
        windows = profile.get("time_windows")
        for idx, token in enumerate(tokens):
            if token == "time_windows" and idx + 1 < len(tokens):
                win_token = tokens[idx + 1]
                if isinstance(win_token, int):
                    if isinstance(windows, list) and 0 <= win_token < len(windows):
                        window_entry = windows[win_token]
                        if isinstance(window_entry, dict):
                            return window_entry.get("window_id") or f"w{win_token + 1}"
                        return f"w{win_token + 1}"
                elif _is_selector_token(win_token):
                    selector_key = str(win_token.get("_selector_key", "")).lower()
                    selector_val = win_token.get("_selector_value")
                    if selector_key == "window_id":
                        return selector_val
                    if isinstance(windows, list):
                        for window in windows:
                            if isinstance(window, dict) and str(window.get(selector_key)) == str(selector_val):
                                return window.get("window_id") or selector_val
                break
        return None

    def _apply_path_update(
        root: Dict, tokens: List[object], op: str, value: object, *, value_provided: bool
    ) -> None:
        """
        Generic setter/deleter/append that can walk dicts/lists, creating containers as needed for updates.
        """
        if not tokens:
            return
        op_lower = (op or "update").lower()
        is_delete = op_lower in {"delete", "remove"}
        is_append = op_lower in {"append", "add", "extend", "push"}

        parent = root
        for idx, token in enumerate(tokens[:-1]):
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None

            if _is_selector_token(token):
                key = token["_selector_key"]  # type: ignore
                val = token["_selector_value"]  # type: ignore
                if not isinstance(parent, list):
                    return
                match_idx = None
                for i, item in enumerate(parent):
                    if isinstance(item, dict) and str(item.get(key)) == str(val):
                        match_idx = i
                        break
                if match_idx is None:
                    if is_delete:
                        return
                    new_item: Dict[str, object] = {key: val}
                    if isinstance(next_token, int):
                        new_item[key] = []
                    parent.append(new_item)
                    match_idx = len(parent) - 1
                if not isinstance(parent[match_idx], (dict, list)):
                    if is_delete:
                        return
                    parent[match_idx] = {} if isinstance(next_token, (str, dict)) else []  # type: ignore
                parent = parent[match_idx]  # type: ignore
            elif isinstance(token, str):
                if not isinstance(parent, dict):
                    if is_delete:
                        return
                    return
                if token not in parent or not isinstance(parent[token], (dict, list)):
                    if is_delete:
                        return
                    parent[token] = [] if isinstance(next_token, int) else {}
                parent = parent[token]
            else:  # token is int
                if not isinstance(parent, list):
                    return
                while len(parent) <= token:  # type: ignore
                    parent.append({} if isinstance(next_token, str) else None)  # type: ignore
                if not isinstance(parent[token], (dict, list)) and isinstance(next_token, (str, int, dict)):
                    if not is_delete:
                        parent[token] = {} if isinstance(next_token, (str, dict)) else []  # type: ignore
                parent = parent[token]  # type: ignore

        last = tokens[-1]
        if _is_selector_token(last):
            return  # selectors should not be terminal
        if isinstance(last, str):
            if not isinstance(parent, dict):
                return
            if is_delete:
                parent.pop(last, None)
                return
            if is_append:
                existing = parent.get(last)
                if isinstance(existing, list):
                    existing.append(value)
                elif existing is None:
                    parent[last] = [value]
                else:
                    parent[last] = [existing, value] if value_provided else existing
                return
            if op_lower in {"replace", "update", "modify", "set"} and value_provided and (
                value == [] or value == {}
            ):
                # Treat explicit empty replacements as removal to avoid leaving empty containers.
                parent.pop(last, None)
                return
            parent[last] = value
        else:
            if not isinstance(parent, list):
                return
            while len(parent) <= last:
                parent.append(None)
            if is_delete:
                if 0 <= last < len(parent):
                    parent.pop(last)
                return
            if is_append:
                if last < len(parent) and isinstance(parent[last], list):
                    parent[last].append(value)
                elif last == len(parent):
                    parent.append(value)
                else:
                    parent[last] = value
                return
            parent[last] = value

    if not isinstance(resolution_payload, dict):
        return resolved_profiles

    def _iter_patch_lists(payload: Dict[str, object]) -> List[List[Dict[str, object]]]:
        patch_lists: List[List[Dict[str, object]]] = []
        direct_patches = payload.get("patches")
        if isinstance(direct_patches, list):
            patch_lists.append(direct_patches)
        resolution_block = payload.get("resolution")
        if isinstance(resolution_block, dict):
            nested = resolution_block.get("patches")
            if isinstance(nested, list):
                patch_lists.append(nested)
        conflicts_and_resolutions = payload.get("conflicts_and_resolutions")
        if isinstance(conflicts_and_resolutions, list):
            for conflict_entry in conflicts_and_resolutions:
                if not isinstance(conflict_entry, dict):
                    continue
                entry_patches = None
                resolution_section = conflict_entry.get("resolution")
                if isinstance(resolution_section, dict):
                    entry_patches = resolution_section.get("patches")
                if entry_patches is None:
                    entry_patches = conflict_entry.get("patches")
                if isinstance(entry_patches, list):
                    patch_lists.append(entry_patches)
        return patch_lists

    patch_batches = _iter_patch_lists(resolution_payload)
    if not patch_batches:
        return resolved_profiles

    for patches in patch_batches:
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            domain_name = patch.get("domain") or patch.get("domain_name")
            profile = resolved_profiles.get(domain_name) if domain_name else None
            tokens = _strip_context_prefix(_parse_path_tokens(patch.get("path")))
            if profile is None:
                # Attempt to infer domain from the path prefix (e.g., "Work & Education.initial_state...")
                if tokens and isinstance(tokens[0], str):
                    token_domain = tokens[0]
                    # exact match first
                    if token_domain in resolved_profiles:
                        domain_name = token_domain
                        tokens = tokens[1:]
                        profile = resolved_profiles.get(domain_name)
                    else:
                        # case-insensitive match
                        for candidate in resolved_profiles.keys():
                            if candidate.lower() == token_domain.lower():
                                domain_name = candidate
                                tokens = tokens[1:]
                                profile = resolved_profiles.get(domain_name)
                                break
                if profile is None:
                    continue
            else:
                if len(tokens) < 1:
                    continue

            if tokens and isinstance(tokens[0], str) and domain_name and tokens[0].lower() == str(domain_name).lower():
                tokens = tokens[1:]

            op = str(patch.get("change") or patch.get("action") or "update").lower()
            value = None
            value_provided = False
            for key in ("new_value", "value", "resolved_value", "new_state", "delta", "replacement"):
                if key in patch:
                    value = patch.get(key)
                    value_provided = True
                    break

            window_id = patch.get("window_id") or patch.get("window")
            if not window_id:
                window_id = _infer_window_id_from_tokens(tokens, profile)

            normalized_window_id = _normalize_window_id(window_id)
            root = _get_window_root(profile, normalized_window_id)
            if root is None:
                continue

            if op in {"delete", "remove"}:
                value_provided = True  # allow deletion without explicit value
            if not value_provided and op in {
                "append",
                "add",
                "extend",
                "push",
                "replace",
                "update",
                "modify",
                "add_key",
                "set",
            }:
                # Skip patches that would overwrite with None when no explicit value is provided.
                continue

            tokens = _strip_redundant_window_tokens(tokens, normalized_window_id, root)
            _apply_path_update(root, tokens, op, value, value_provided=value_provided)

    return resolved_profiles
