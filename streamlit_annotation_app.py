"""Streamlit app for annotators to spot structure gaps and time conflicts in dynamic profiles.

The app keeps only two tasks:
- Structural field coverage: detect missing required keys/fields relative to the dynamic_profile_template.
- Time conflict view: aggregate habits across domains, render a calendar-like grid, and flag overlaps.

Run with:
  streamlit run behavior_and_conversation/streamlit_annotation_app.py
    -- --dataset-root behavior_and_conversation/test
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import altair as alt
import pandas as pd
import streamlit as st


REQUIRED_HABIT_FIELDS = ("action", "frequency", "timing", "context", "description")
DAY_ALIASES = {
    "monday": "Mon",
    "tuesday": "Tue",
    "wednesday": "Wed",
    "thursday": "Thu",
    "friday": "Fri",
    "saturday": "Sat",
    "sunday": "Sun",
    "mon": "Mon",
    "tue": "Tue",
    "tues": "Tue",
    "wed": "Wed",
    "thu": "Thu",
    "thur": "Thu",
    "thurs": "Thu",
    "fri": "Fri",
    "sat": "Sat",
    "sun": "Sun",
}
ALL_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Unspecified"]
DEFAULT_DATASET_ROOT = Path("behavior_and_conversation/test")
DEFAULT_FILENAME = "dynamic_profiles_conflict_resolved.json"


@dataclass
class Issue:
    domain: str
    window: str
    path: str
    message: str
    kind: str = "missing_field"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_user_context(base_dir: Path) -> Dict[str, Any]:
    """Lightweight loader mirroring the original editor to keep context visible."""
    desc_path = base_dir / "context_user_profile.txt"
    basic_profile_path = base_dir / "user_basic_profile.json"
    world_bg_path = base_dir / "context_world_background_2024.txt"
    fallback_world_bg = Path(__file__).resolve().parent / "context_world_background_2024.txt"

    description = desc_path.read_text(encoding="utf-8").strip() if desc_path.exists() else ""
    basic_profile = _read_json(basic_profile_path) if basic_profile_path.exists() else {}
    if world_bg_path.exists():
        world_background = world_bg_path.read_text(encoding="utf-8").strip()
    elif fallback_world_bg.exists():
        world_background = fallback_world_bg.read_text(encoding="utf-8").strip()
    else:
        world_background = ""

    return {
        "user_description": description,
        "basic_profile": basic_profile,
        "world_background": world_background,
    }


@st.cache_data(show_spinner=False)
def list_candidate_files(dataset_root: str, target_filename: str) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    matches = sorted(root.glob(f"**/{target_filename}"))
    return [str(p) for p in matches]


@st.cache_data(show_spinner=False)
def load_profile(path_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path_str)
    payload = _read_json(path)
    profile = payload.get("profiles", payload)
    return profile, payload


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


def parse_time_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract start/end minutes from timing text like '6:30 AM - 7:15 AM'."""
    match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)?)\s*[-–]\s*(\d{1,2}:\d{2}\s*(?:am|pm)?)", text, re.IGNORECASE)
    if not match:
        return None, None
    start = _time_to_minutes(match.group(1))
    end = _time_to_minutes(match.group(2))
    if start is None or end is None:
        return None, None
    if end <= start:
        end += 24 * 60
    return start, end


def format_time_range(value: Any) -> str:
    """Render the time_range array or string into a human-readable label."""
    if isinstance(value, list) and len(value) == 2:
        return f"{value[0]} -> {value[1]}"
    if isinstance(value, str):
        return value
    return ""


def extract_days(frequency: str, timing: str) -> Set[str]:
    text = f"{frequency} {timing}".lower().replace("-", "_")
    # Special normalizations before generic rules
    if "daily_on_weekends" in text or ("daily" in text and "weekend" in text):
        return {"Sat", "Sun"}
    if "daily_on_weekdays" in text or ("daily" in text and ("weekday" in text or "weekdays" in text)):
        return {"Mon", "Tue", "Wed", "Thu", "Fri"}

    tokens: Set[str] = set()
    for raw in DAY_ALIASES:
        if raw in text:
            tokens.add(DAY_ALIASES[raw])
    if "weekday" in text or "weekdays" in text:
        tokens.update({"Mon", "Tue", "Wed", "Thu", "Fri"})
    if "weekend" in text or "weekends" in text:
        tokens.update({"Sat", "Sun"})
    if "daily" in text or "every day" in text:
        tokens.update(ALL_DAYS)
    if "per day" in text and not tokens:
        tokens.update(ALL_DAYS)
    if not tokens:
        tokens.add("Unspecified")
    return tokens


def parse_date_range(window_range: Any) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if isinstance(window_range, list) and len(window_range) == 2:
        try:
            start = pd.to_datetime(window_range[0])
            end = pd.to_datetime(window_range[1])
            if pd.isna(start) or pd.isna(end):
                return None, None
            if end < start:
                start, end = end, start
            return start.normalize(), end.normalize()
        except Exception:
            return None, None
    return None, None


def _candidate_dates(start: pd.Timestamp, end: pd.Timestamp, days: Set[str]) -> List[pd.Timestamp]:
    all_days = pd.date_range(start, end, freq="D")
    if "Unspecified" in days or not days:
        return list(all_days)
    keep = []
    for dt in all_days:
        if dt.day_name()[:3] in days:
            keep.append(dt)
    return keep


def _sample_evenly(seq: List[pd.Timestamp], count: int) -> List[pd.Timestamp]:
    if count <= 0:
        return []
    if count >= len(seq):
        return list(seq)
    if count == 1:
        return [seq[0]]
    chosen = []
    last_idx = len(seq) - 1
    for i in range(count):
        idx = round(i * last_idx / (count - 1))
        chosen.append(seq[idx])
    seen: Set[pd.Timestamp] = set()
    unique = []
    for dt in chosen:
        if dt not in seen:
            unique.append(dt)
            seen.add(dt)
    return unique


def _sample_weekly(seq: List[pd.Timestamp], per_week: int, seed: int) -> List[pd.Timestamp]:
    """Pick up to per_week items per calendar week (Mon-Sun), using deterministic randomness."""
    if per_week <= 0:
        return []
    buckets: Dict[pd.Timestamp, List[pd.Timestamp]] = {}
    for dt in seq:
        week_start = dt - pd.Timedelta(days=dt.weekday())
        buckets.setdefault(week_start, []).append(dt)
    rng = random.Random(seed)
    chosen: List[pd.Timestamp] = []
    for week_start, dates in buckets.items():
        if not dates:
            continue
        dates_sorted = sorted(dates)
        if len(dates_sorted) <= per_week:
            chosen.extend(dates_sorted)
        else:
            indices = list(range(len(dates_sorted)))
            rng.shuffle(indices)
            picked_idx = sorted(indices[:per_week])
            chosen.extend(dates_sorted[i] for i in picked_idx)
    return sorted(chosen)


def _target_count(frequency: str, total_days: int, candidate_len: int) -> int:
    freq = (frequency or "").lower().replace("-", "_")
    weeks = max(1, math.ceil(total_days / 7)) if total_days else 1
    months = max(1, math.ceil(total_days / 30)) if total_days else 1
    # Special cases
    if "daily_on_weekends" in freq or ("daily" in freq and "weekend" in freq):
        return candidate_len
    if "daily_on_weekdays" in freq or ("daily" in freq and ("weekday" in freq or "weekdays" in freq)):
        return candidate_len

    if "daily" in freq or "every day" in freq:
        return total_days
    if "weekday" in freq or "weekdays" in freq:
        return max(1, weeks * 5)
    m = re.search(r"(\\d+)_times_per_week", freq) or re.search(r"(\\d+)\\s*times\\s*per\\s*week", freq)
    if m:
        return int(m.group(1)) * weeks
    m = re.search(r"(\\d+)_times_per_month", freq) or re.search(r"(\\d+)\\s*times\\s*per\\s*month", freq)
    if m:
        return int(m.group(1)) * months
    if "weekly" in freq:
        return weeks
    if "biweekly" in freq:
        return weeks * 2
    if "monthly" in freq:
        return months
    if "once a week" in freq:
        return weeks
    return total_days if candidate_len <= 0 else candidate_len


def _merge_days(d1: Set[str], d2: Set[str]) -> Set[str]:
    left = ALL_DAYS if "Unspecified" in d1 else d1
    right = ALL_DAYS if "Unspecified" in d2 else d2
    return left & right


def _format_minutes(minutes: int) -> str:
    hours = (minutes // 60) % 24
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def materialize_habits(domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create snapshots of habits per window, applying operations cumulatively."""
    initial = (domain.get("initial_state", {}) or {})
    base_habits = ((initial.get("habits_state") or {}).get("initial") or {}).copy()
    snapshots = [
        {
            "window_id": "initial_state",
            "time_range": initial.get("time_range"),
            "habits": base_habits.copy(),
        }
    ]

    for window in domain.get("time_windows") or []:
        current = base_habits.copy()
        for op in (window.get("habits_delta") or {}).get("operations") or []:
            name = op.get("habit_name") or "unnamed_habit"
            op_type = (op.get("op") or "").lower()
            delta = op.get("delta")
            if op_type == "acquire" and isinstance(delta, dict):
                current[name] = delta
            elif op_type == "adjust" and isinstance(delta, dict):
                if name not in current:
                    current[name] = delta
                else:
                    current[name] = {**current[name], **delta}
            elif op_type == "drop":
                current.pop(name, None)
        base_habits = current
        snapshots.append(
            {
                "window_id": window.get("window_id") or "unknown_window",
                "time_range": window.get("time_range"),
                "habits": current.copy(),
            }
        )
    return snapshots


def _missing_fields(payload: Dict[str, Any], required: Iterable[str]) -> List[str]:
    return [field for field in required if not payload.get(field)]


def check_structural_gaps(domain_name: str, domain: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    initial = domain.get("initial_state")
    if not initial:
        issues.append(Issue(domain_name, "initial_state", "initial_state", "Missing initial_state block", "missing_section"))
        return issues
    if not initial.get("summary"):
        issues.append(Issue(domain_name, "initial_state", "initial_state.summary", "Missing summary", "missing_field"))

    for habit_name, habit in ((initial.get("habits_state") or {}).get("initial") or {}).items():
        missing = _missing_fields(habit, REQUIRED_HABIT_FIELDS)
        if missing:
            issues.append(
                Issue(
                    domain_name,
                    "initial_state",
                    f"habits_state.initial.{habit_name}",
                    f"Missing fields: {', '.join(missing)}",
                    "missing_field",
                )
            )

    current_habits = ((initial.get("habits_state") or {}).get("initial") or {}).copy()
    for idx, window in enumerate(domain.get("time_windows") or []):
        window_id = window.get("window_id") or f"window_{idx + 1}"
        if not window.get("summary"):
            issues.append(Issue(domain_name, window_id, f"{window_id}.summary", "Missing summary", "missing_field"))
        if not window.get("window_description"):
            issues.append(
                Issue(domain_name, window_id, f"{window_id}.window_description", "Missing window_description", "missing_field")
            )
        if not window.get("time_range"):
            issues.append(Issue(domain_name, window_id, f"{window_id}.time_range", "Missing time_range", "missing_field"))

        for section in ("user_attributes_delta", "habits_delta", "preferences_delta"):
            operations = (window.get(section) or {}).get("operations") or []
            for op_idx, op in enumerate(operations):
                if not op.get("reason"):
                    issues.append(
                        Issue(
                            domain_name,
                            window_id,
                            f"{window_id}.{section}.operations[{op_idx}]",
                            "Missing reason",
                            "missing_field",
                        )
                    )

        for op_idx, op in enumerate((window.get("habits_delta") or {}).get("operations") or []):
            name = op.get("habit_name") or f"habit_{op_idx}"
            op_type = (op.get("op") or "").lower()
            delta = op.get("delta")
            path = f"{window_id}.habits_delta.operations[{op_idx}]"
            if op_type == "acquire":
                missing = _missing_fields(delta or {}, REQUIRED_HABIT_FIELDS) if isinstance(delta, dict) else REQUIRED_HABIT_FIELDS
                if missing:
                    issues.append(Issue(domain_name, window_id, f"{path}.{name}", f"New habit missing fields: {', '.join(missing)}", "missing_field"))
                if isinstance(delta, dict):
                    current_habits[name] = delta
            elif op_type == "adjust":
                if name not in current_habits:
                    issues.append(Issue(domain_name, window_id, f"{path}.{name}", "Adjusting a habit that was not present before", "missing_prior"))
                    current_habits[name] = delta or {}
                merged = {**current_habits.get(name, {}), **(delta or {})}
                missing = _missing_fields(merged, REQUIRED_HABIT_FIELDS)
                if missing:
                    issues.append(
                        Issue(
                            domain_name,
                            window_id,
                            f"{path}.{name}",
                            f"Adjusted habit missing fields: {', '.join(missing)}",
                            "missing_field",
                        )
                    )
                current_habits[name] = merged
            elif op_type == "drop":
                if delta is not None:
                    issues.append(Issue(domain_name, window_id, f"{path}.{name}", "For drop, delta must be null", "format_error"))
                if name not in current_habits:
                    issues.append(Issue(domain_name, window_id, f"{path}.{name}", "Dropping a habit that was not present before", "missing_prior"))
                current_habits.pop(name, None)
            else:
                issues.append(Issue(domain_name, window_id, path, f"Unknown op: {op_type or 'missing'}", "format_error"))
    return issues


def collect_structural_issues(profile: Dict[str, Any], domains: List[str]) -> List[Issue]:
    issues: List[Issue] = []
    for domain_name, domain_payload in profile.items():
        if domains and domain_name not in domains:
            continue
        issues.extend(check_structural_gaps(domain_name, domain_payload or {}))
    return issues


def build_events(profile: Dict[str, Any], domains: List[str], windows: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for domain_name, domain_payload in profile.items():
        if domains and domain_name not in domains:
            continue
        for snapshot in materialize_habits(domain_payload or {}):
            window_id = snapshot["window_id"]
            if windows and window_id not in windows:
                continue
            window_range = snapshot.get("time_range")
            start_date, end_date = parse_date_range(window_range)
            if not start_date or not end_date:
                continue  # cannot place on calendar without a real window range
            total_days = (end_date - start_date).days + 1
            for habit_name, habit in (snapshot.get("habits") or {}).items():
                frequency = habit.get("frequency", "")
                timing = habit.get("timing", "")
                start_min, end_min = parse_time_range(timing)
                if start_min is None or end_min is None:
                    continue
                days = extract_days(frequency, timing)
                candidate_dates = _candidate_dates(start_date, end_date, days)
                if not candidate_dates:
                    continue
                seed = hash((domain_name, window_id, habit_name))
                weekly_match = re.search(r"(\d+)_times_per_week", frequency.replace("-", "_")) or re.search(
                    r"(\d+)\\s*times\\s*per\\s*week", frequency.lower()
                )
                if weekly_match:
                    per_week = int(weekly_match.group(1))
                    occurrences = _sample_weekly(candidate_dates, per_week, seed)
                else:
                    target = _target_count(frequency, total_days, len(candidate_dates))
                    occurrences = _sample_evenly(candidate_dates, min(target, len(candidate_dates)))
                for dt in occurrences:
                    start_ts = dt + pd.to_timedelta(start_min, unit="m")
                    end_ts = dt + pd.to_timedelta(end_min, unit="m")
                    display_start = start_min % (24 * 60)
                    display_end = min(end_min, 24 * 60 - 1)
                    events.append(
                        {
                            "domain": domain_name,
                            "window_id": window_id,
                            "window_range": window_range,
                            "habit": habit_name,
                            "action": habit.get("action", ""),
                            "timing": timing,
                            "frequency": frequency,
                            "description": habit.get("description", ""),
                            "start_min": start_min,
                            "end_min": end_min,
                            "display_start": display_start,
                            "display_end": display_end,
                            "date": dt.normalize(),
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "day_label": dt.day_name()[:3],
                        }
                    )
    return events


def detect_conflicts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, pd.Timestamp], List[Dict[str, Any]]] = {}
    for ev in events:
        if ev.get("start_min") is None or ev.get("end_min") is None or "date" not in ev:
            continue
        key = (ev["window_id"], ev["date"])
        by_key.setdefault(key, []).append(ev)

    for (window_id, date), group in by_key.items():
        window_range = format_time_range(group[0].get("window_range"))
        group = sorted(group, key=lambda e: e["start_min"])
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["start_min"] < b["end_min"] and b["start_min"] < a["end_min"]:
                    conflicts.append(
                        {
                            "window_id": window_id,
                            "window_range": window_range,
                            "date": date.date().isoformat(),
                            "time": f"{_format_minutes(max(a['start_min'], 0))}-{_format_minutes(min(a['end_min'], 24 * 60 + 59))}",
                            "habit_a": f"{a['domain']} · {a['habit']}",
                            "habit_b": f"{b['domain']} · {b['habit']}",
                            "timing_a": a["timing"],
                            "timing_b": b["timing"],
                        }
                    )
    return conflicts


def _build_calendar_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        if ev.get("start_min") is None or ev.get("end_min") is None or "date" not in ev:
            continue
        rows.append(
            {
                "domain": ev["domain"],
                "habit": ev["habit"],
                "window_id": ev["window_id"],
                "window_range": ev.get("window_range"),
                "date": ev["date"],
                "day": ev.get("day_label"),
                "start_min": ev.get("display_start", ev["start_min"]),
                "end_min": ev.get("display_end", ev["end_min"]),
                "start_label": _format_minutes(ev.get("display_start", ev["start_min"])),
                "end_label": _format_minutes(ev.get("display_end", ev["end_min"])),
                "timing": ev["timing"],
                "frequency": ev["frequency"],
                "description": ev["description"],
            }
        )
    return pd.DataFrame(rows)


def _domain_colors(domains: Iterable[str]) -> Dict[str, str]:
    palette = [
        "#6E44FF",
        "#FF7F50",
        "#2EC4B6",
        "#FFD166",
        "#EF476F",
        "#118AB2",
        "#073B4C",
        "#8AC926",
        "#FF9F1C",
    ]
    domain_list = list(domains)
    return {name: palette[i % len(palette)] for i, name in enumerate(domain_list)}


def main() -> None:
    st.set_page_config(page_title="Dynamic Profile QA (Structure + Time)", layout="wide")
    st.title("Dynamic Profile QA · Missing Fields + Time Conflicts")
    st.caption("Surface missing required fields and visualize cross-domain habit conflicts. Window IDs (w1, w2...) mean time windows, not weeks.")

    st.sidebar.header("Data source")
    dataset_root = st.sidebar.text_input("Dataset root", str(DEFAULT_DATASET_ROOT))
    target_filename = st.sidebar.text_input("Target filename", DEFAULT_FILENAME)
    candidates = list_candidate_files(dataset_root, target_filename)
    selected_path = ""
    if candidates:
        selected_label = st.sidebar.selectbox("Pick dataset", options=candidates, format_func=lambda p: str(Path(p).parent.name))
        selected_path = selected_label
    manual_path = st.sidebar.text_input("Or enter a JSON file path", selected_path or "")
    uploaded = st.sidebar.file_uploader("Or upload a JSON file", type=["json"])

    profile: Dict[str, Any] = {}
    raw_payload: Dict[str, Any] = {}
    base_dir = Path(dataset_root)
    if uploaded:
        try:
            payload = json.loads(uploaded.read())
            profile = payload.get("profiles", payload)
            raw_payload = payload
            st.sidebar.success("Uploaded file loaded")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.sidebar.error(f"Failed to parse uploaded file: {exc}")
            st.stop()
    elif manual_path:
        path = Path(manual_path)
        if not path.exists():
            st.sidebar.error(f"File not found: {path}")
            st.stop()
        profile, raw_payload = load_profile(str(path))
        base_dir = path.parent
        st.sidebar.info(f"Loaded: {path.name}")
    else:
        st.sidebar.warning("Select or enter a dynamic profile JSON file")
        st.stop()

    context = _load_user_context(base_dir)
    domain_names = sorted(profile.keys())
    if not domain_names:
        st.error("No domains found in file.")
        st.stop()

    st.sidebar.header("Filters")
    domain_filter = st.sidebar.multiselect("Domains to include", domain_names, default=domain_names)

    available_windows = {"initial_state"}
    for domain_payload in profile.values():
        for tw in domain_payload.get("time_windows") or []:
            if tw.get("window_id"):
                available_windows.add(tw["window_id"])
    window_filter = st.sidebar.multiselect("Time windows to include", sorted(available_windows), default=sorted(available_windows))

    tabs = st.tabs(["Missing Fields", "Time Conflicts Calendar"])

    with st.expander("Context materials", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**User Description**")
            st.text_area("user_description", context["user_description"], height=160, label_visibility="collapsed")
            st.markdown("**World Background**")
            st.text_area("world_background", context["world_background"], height=220, label_visibility="collapsed")
        with col2:
            st.markdown("**Basic Profile (read-only)**")
            st.json(context["basic_profile"], expanded=False)

    with tabs[0]:
        st.subheader("Auto-detect missing fields")
        issues = collect_structural_issues(profile, domain_filter)
        if not issues:
            st.success("No missing fields detected.")
        else:
            summary = {}
            for iss in issues:
                summary.setdefault(iss.domain, 0)
                summary[iss.domain] += 1
            st.write("Counts by domain")
            st.dataframe(
                pd.DataFrame(
                    [{"domain": d, "issues": c} for d, c in sorted(summary.items(), key=lambda x: x[1], reverse=True)]
                ),
                use_container_width=True,
                hide_index=True,
            )
            for domain in domain_filter:
                domain_issues = [iss for iss in issues if iss.domain == domain]
                if not domain_issues:
                    continue
                with st.expander(f"{domain} · {len(domain_issues)} issue(s)", expanded=False):
                    df = pd.DataFrame(
                        [
                            {
                                "window": iss.window,
                                "path": iss.path,
                                "issue": iss.message,
                                "type": iss.kind,
                            }
                            for iss in domain_issues
                        ]
                    )
                    st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Time conflict detection")
        st.caption(
            "Window IDs (w1/w2/...) are multi-week/month time windows. Habits are expanded across the full window by frequency (e.g., 2_times_per_month picks evenly spaced matching weekdays within that 90-day window)."
        )
        selected_windows = set(window_filter)
        events = build_events(profile, domain_filter, selected_windows)
        conflict_list = detect_conflicts(events)
        if conflict_list:
            st.warning(f"Found {len(conflict_list)} potential conflicts")
            st.dataframe(pd.DataFrame(conflict_list), use_container_width=True, hide_index=True)
        else:
            st.success("No time conflicts found (among parseable time ranges)")

        calendar_df = _build_calendar_df(events)
        if calendar_df.empty:
            st.info("No habits with parseable time ranges under the current filters; cannot draw schedule.")
        else:
            colors = _domain_colors(domain_filter)
            for window_id in sorted(calendar_df["window_id"].unique()):
                window_df = calendar_df[calendar_df["window_id"] == window_id]
                range_label = format_time_range(window_df["window_range"].iloc[0] if "window_range" in window_df else "")
                label = f"{window_id} ({range_label})" if range_label else window_id
                st.markdown(f"**{label}**")
                max_end = float(window_df["end_min"].max())
                bars = (
                    alt.Chart(window_df)
                    .mark_bar(cornerRadius=4)
                    .encode(
                        x=alt.X(
                            "date:T",
                            title="Date",
                            axis=alt.Axis(format="%Y-%m-%d"),
                        ),
                        xOffset=alt.value(0),
                        y=alt.Y(
                            "start_min:Q",
                            title="Time of day (HH:MM)",
                            scale=alt.Scale(domain=[24 * 60, 0]),
                            axis=alt.Axis(
                                values=[0, 180, 360, 540, 720, 900, 1080, 1260, 1440],
                                labelExpr="timeFormat(datetime(0,0,0, floor(datum.value/60), datum.value%60), '%H:%M')",
                            ),
                        ),
                        y2="end_min:Q",
                        color=alt.Color(
                            "domain:N",
                            scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
                            legend=alt.Legend(title="Domain"),
                        ),
                        tooltip=[
                            alt.Tooltip("domain:N", title="Domain"),
                            alt.Tooltip("habit:N", title="Habit"),
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("timing:N", title="Timing text"),
                            alt.Tooltip("frequency:N", title="Frequency"),
                            alt.Tooltip("description:N", title="Description"),
                            alt.Tooltip("start_label:N", title="Start"),
                            alt.Tooltip("end_label:N", title="End"),
                            alt.Tooltip("window_range:N", title="Window time_range"),
                        ],
                    )
                )
                noon_rule = alt.Chart(pd.DataFrame({"y": [720]})).mark_rule(strokeDash=[4, 4], color="#888", opacity=0.6).encode(
                    y="y:Q"
                )
                chart = alt.layer(bars, noon_rule).properties(height=260)
                st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
