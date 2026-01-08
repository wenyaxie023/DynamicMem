"""Generate a quick visualization report for dynamic user profiles across domains.

This script reads the generated dynamic profiles (preferring conflict-resolved
versions) and produces a lightweight HTML report that makes it easy to review:
  - Cross-domain conflicts and their resolutions.
  - Per-domain timelines for user attributes, habits, and preferences.
  - Change operations (add/modify/remove) with reasons.

Run:
  python visualize_dynamic_profiles.py --output-dir generated_outputs_debug

No external dependencies are required beyond the standard library.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import types

# Allow running as a standalone script without installing the package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# generation_pipeline depends on python-dotenv; provide a tiny stub if it's absent
# so this utility can run in lightweight environments.
if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

# Stub google.generativeai so importing generation_pipeline does not require the SDK.
if "google" not in sys.modules:
    google_stub = types.ModuleType("google")
    genai_stub = types.ModuleType("google.generativeai")

    def _noop(*_args, **_kwargs) -> None:
        return None

    class _DummyGenerativeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate_content(self, *_args, **_kwargs):  # pragma: no cover - guard rail
            raise RuntimeError("This is a stub GenerativeModel (no SDK available).")

    genai_stub.configure = _noop
    genai_stub.GenerativeModel = _DummyGenerativeModel

    google_stub.generativeai = genai_stub
    sys.modules["google"] = google_stub
    sys.modules["google.generativeai"] = genai_stub

from mem_bench.behavior_and_conversation.generation_pipeline import (
    _reorganize_by_key,
    _resolve_window_states,
)


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _find_dynamic_profiles(base_dir: Path) -> Tuple[Dict[str, dict], str]:
    """
    Discover dynamic profiles, preferring the conflict-resolved bundle.
    Returns (profiles, source_description).
    """
    conflict_path = base_dir / "dynamic_profiles_conflict_resolved.json"
    if conflict_path.exists():
        return _load_json(conflict_path), conflict_path.name

    pipeline_path = base_dir / "pipeline_output.json"
    if pipeline_path.exists():
        payload = _load_json(pipeline_path)
        domains = payload.get("domains", {}) or {}
        profiles = {
            name: domain_payload.get("dynamic_profile", {})
            for name, domain_payload in domains.items()
            if isinstance(domain_payload, dict) and domain_payload.get("dynamic_profile")
        }
        if profiles:
            return profiles, pipeline_path.name

    # Fallback: collect individual "*_dynamic_profile.json" files.
    profiles: Dict[str, dict] = {}
    for path in base_dir.glob("*_dynamic_profile.json"):
        data = _load_json(path)
        domain_name = (
            data.get("life_domain")
            or data.get("domain_name")
            or path.name.replace("_dynamic_profile.json", "")
        )
        profiles[domain_name] = data
    if profiles:
        return profiles, "individual dynamic_profile files"

    raise FileNotFoundError(
        f"No dynamic profiles found under {base_dir}. "
        "Expected dynamic_profiles_conflict_resolved.json or pipeline_output.json."
    )


def _load_conflict_data(base_dir: Path) -> Tuple[List[dict], Dict[str, dict]]:
    conflicts_path = base_dir / "cross_domain_conflicts.json"
    resolution_path = base_dir / "cross_domain_conflict_resolution.json"
    conflicts = []
    resolutions: Dict[str, dict] = {}
    if conflicts_path.exists():
        conflicts = _load_json(conflicts_path).get("conflicts", []) or []
    if resolution_path.exists():
        resolutions = _load_json(resolution_path)
    return conflicts, resolutions


def _load_user_context(base_dir: Path) -> Tuple[str, dict]:
    """
    Returns (user_description_text, user_basic_profile_dict).
    """
    description = ""
    basic_profile: dict = {}

    basic_path = base_dir / "user_basic_profile.json"
    if basic_path.exists():
        basic_profile = _load_json(basic_path)

    desc_path = base_dir / "context_user_profile.txt"
    if desc_path.exists():
        description = desc_path.read_text().strip()
    else:
        pipeline_path = base_dir / "pipeline_output.json"
        if pipeline_path.exists():
            payload = _load_json(pipeline_path)
            description = payload.get("user_profile", "") or ""

    return description, basic_profile


def _format_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 6px;border-radius:6px;font-size:12px;">'
        f'{html.escape(label)}</span>'
    )


def _render_conflicts(conflicts: List[dict], resolutions: Dict[str, dict]) -> str:
    if not conflicts:
        return "<p>No cross-domain conflicts detected.</p>"

    rows = []
    for conflict in conflicts:
        canonical = conflict.get("canonical_key", "unknown")
        domains = conflict.get("domains", [])
        resolution_entry = resolutions.get("canonical_attributes", {}).get(canonical, {})
        resolved_value = resolution_entry.get("resolved_value")
        reason = resolution_entry.get("reason", "")

        domain_lines = []
        for entry in domains:
            domain_lines.append(
                f"<div><strong>{html.escape(entry.get('domain',''))}</strong>: "
                f"{html.escape(entry.get('attribute_name',''))} "
                f"&rarr; {html.escape(_format_value(entry.get('value')))}</div>"
            )
        resolved_html = (
            html.escape(_format_value(resolved_value))
            if resolved_value is not None
            else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(canonical)}</td>"
            f"<td>{''.join(domain_lines)}</td>"
            f"<td>{resolved_html}</td>"
            f"<td>{html.escape(reason)}</td>"
            "</tr>"
        )

    return (
        "<h2>Cross-domain Conflicts</h2>"
        "<table><thead><tr>"
        "<th>Canonical Key</th><th>Domain Values</th><th>Resolved Value</th><th>Reason</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_timeline_rows(timeline: List[dict]) -> str:
    rows = []
    for entry in timeline:
        op = entry.get("op")
        badge = ""
        if op:
            color = {
                "add": "#188038",
                "modify": "#d93025",
                "remove": "#c5221f",
                "drop": "#c5221f",
            }.get(op, "#5f6368")
            badge = _badge(op, color)
        rows.append(
            "<tr>"
            f"<td>{html.escape(' to '.join(entry.get('time_range', [])))}</td>"
            f"<td>{html.escape(_format_value(entry.get('current_value')))}</td>"
            f"<td>{badge}</td>"
            f"<td>{html.escape(_format_value(entry.get('previous_value','')))}</td>"
            f"<td>{html.escape(_format_value(entry.get('change_reason','')))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_state_section(state_type: str, state_map: Dict[str, dict]) -> str:
    if not state_map:
        return ""
    blocks = []
    for name, data in sorted(state_map.items()):
        blocks.append(
            "<div class='card'>"
            f"<div class='card-title'>{html.escape(name)}</div>"
            "<table class='timeline'><thead><tr>"
            "<th>Time Range</th><th>Value</th><th>Op</th><th>Previous</th><th>Reason</th>"
            "</tr></thead><tbody>"
            f"{_render_timeline_rows(data.get('timeline', []))}"
            "</tbody></table>"
            "</div>"
        )
    return f"<h3>{html.escape(state_type)}</h3>" + "".join(blocks)


def _render_domain(domain_name: str, profile: dict) -> str:
    resolved_windows = _resolve_window_states(profile)
    reorganized = _reorganize_by_key(resolved_windows)

    stats = {
        "windows": len(resolved_windows),
        "attributes": len(reorganized.get("user_attributes_state", {})),
        "habits": len(reorganized.get("habits_state", {})),
        "preferences": len(reorganized.get("preferences_state", {})),
    }

    return (
        f"<section><h2>{html.escape(domain_name)}</h2>"
        f"<p class='meta'>windows: {stats['windows']} | "
        f"user attributes: {stats['attributes']} | "
        f"habits: {stats['habits']} | "
        f"preferences: {stats['preferences']}</p>"
        f"{_render_state_section('User Attributes', reorganized.get('user_attributes_state', {}))}"
        f"{_render_state_section('Habits', reorganized.get('habits_state', {}))}"
        f"{_render_state_section('Preferences', reorganized.get('preferences_state', {}))}"
        "</section>"
    )


def _build_window_grids(
    resolved_windows: List[dict],
    initial_state: dict | None = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, List[dict]]]]:
    """
    Build per-window timelines for each state type.
    Returns:
      headers: list of (window_id, time_range_str)
      grid: {state_type: {name: [ {value, op, reason} per window ]}}
    """
    headers: List[Tuple[str, str]] = []
    valid_windows: List[dict] = []
    for window in resolved_windows:
        wid = window.get("window_id")
        if not wid:
            continue
        tr = window.get("time_range") or []
        headers.append((wid, " to ".join(tr) if tr else ""))
        valid_windows.append(window)

    add_init = initial_state is not None
    if add_init:
        headers.insert(0, ("init", "Initial state"))

    num_windows = len(headers)
    grid: Dict[str, Dict[str, List[dict]]] = {
        "user_attributes_state": {},
        "habits_state": {},
        "preferences_state": {},
    }

    def _iter_initial_entries(entries: dict | list | None):
        """Yield (name, value) pairs from either dict or list-of-dicts."""
        if isinstance(entries, dict):
            for name, val in entries.items():
                yield name, val
            return
        for entry in entries or []:
            if isinstance(entry, dict):
                for name, val in entry.items():
                    yield name, val
            elif isinstance(entry, str):
                # Preserve the key with unknown value instead of dropping it.
                yield entry, None

    def _ensure_row(state_type: str, name: str) -> List[dict]:
        return grid[state_type].setdefault(
            name,
            [{"value": None, "op": "", "reason": ""} for _ in range(num_windows)],
        )

    if add_init:
        for state_key in ["user_attributes_state", "habits_state", "preferences_state"]:
            if state_key == "user_attributes_state":
                entries = (initial_state.get(state_key) or {}).get("initial")
            else:
                entries = initial_state.get(state_key)
            for name, val in _iter_initial_entries(entries):
                rows = _ensure_row(state_key, name)
                rows[0] = {"value": val, "op": "", "reason": "initial"}

    for idx, window in enumerate(valid_windows):
        offset = 1 if add_init else 0
        for state_type in grid.keys():
            state_list = window.get(state_type, []) or []
            # If the same key appears multiple times in one window, keep the last entry.
            by_name: Dict[str, dict] = {}
            for item in state_list:
                name = item.get("name")
                if not name:
                    continue
                by_name[name] = item

            for name, item in by_name.items():
                rows = _ensure_row(state_type, name)
                rows[idx + offset] = {
                    "value": item.get("current_value"),
                    "op": item.get("op", ""),
                    "reason": item.get("change_reason", ""),
                }

    return headers, grid


def _render_timeline_grid(
    label: str,
    headers: List[Tuple[str, str]],
    rows: Dict[str, List[dict]],
) -> str:
    if not rows:
        return ""

    header_cells = "".join(
        f"<th>{html.escape(wid)}<div class='sub'>{html.escape(tr)}</div></th>"
        for wid, tr in headers
    )

    body_rows = []
    for name, entries in sorted(rows.items()):
        cells = []
        for entry in entries:
            op = entry.get("op")
            badge = ""
            if op:
                color = {
                    "add": "#188038",
                    "modify": "#d93025",
                    "remove": "#c5221f",
                    "drop": "#c5221f",
                }.get(op, "#5f6368")
                badge = _badge(op, color)
            reason = entry.get("reason") or ""

            def _render_value(val: object) -> str:
                if isinstance(val, list):
                    items = "".join(
                        f"<li>{html.escape(_format_value(v))}</li>" for v in val
                    )
                    return f"<ul class='compact'>{items}</ul>"
                if isinstance(val, dict):
                    items = "".join(
                        f"<div><strong>{html.escape(str(k))}:</strong> {html.escape(_format_value(v))}</div>"
                        for k, v in val.items()
                    )
                    return f"<div class='dict'>{items}</div>"
                return html.escape(_format_value(val))

            cells.append(
                "<td>"
                f"{badge}<div>{_render_value(entry.get('value'))}</div>"
                f"<div class='sub'>{html.escape(reason)}</div>"
                "</td>"
            )
        body_rows.append(
            f"<tr><th class='row-h'>{html.escape(name)}</th>{''.join(cells)}</tr>"
        )

    return (
        f"<h3>{html.escape(label)} (per window)</h3>"
        "<table class='grid'><thead><tr><th></th>"
        f"{header_cells}</tr></thead><tbody>"
        f"{''.join(body_rows)}"
        "</tbody></table>"
    )


def _build_html(
    conflicts: List[dict],
    resolutions: Dict[str, dict],
    profiles: Dict[str, dict],
    source: str,
    *,
    show_conflicts: bool = False,
    user_description: str = "",
    user_basic_profile: dict | None = None,
) -> str:
    domain_sections = []
    for name, profile in profiles.items():
        resolved_windows = _resolve_window_states(profile)
        headers, grid = _build_window_grids(
            resolved_windows, profile.get("initial_state")
        )
        reorganized = _reorganize_by_key(resolved_windows)

        stats = {
            "windows": len(resolved_windows),
            "attributes": len(reorganized.get("user_attributes_state", {})),
            "habits": len(reorganized.get("habits_state", {})),
            "preferences": len(reorganized.get("preferences_state", {})),
        }

        domain_sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<p class='meta'>windows: {stats['windows']} | "
            f"user attributes: {stats['attributes']} | "
            f"habits: {stats['habits']} | "
            f"preferences: {stats['preferences']}</p>"
            f"{_render_timeline_grid('User Attributes', headers, grid.get('user_attributes_state', {}))}"
            f"{_render_timeline_grid('Habits', headers, grid.get('habits_state', {}))}"
            f"{_render_timeline_grid('Preferences', headers, grid.get('preferences_state', {}))}"
            "</section>"
        )
    domain_sections_html = "\n".join(domain_sections)
    conflicts_html = _render_conflicts(conflicts, resolutions) if show_conflicts else ""

    def _render_basic_profile(profile: dict) -> str:
        if not profile:
            return "<p>No basic profile found.</p>"

        def render_dict(d: dict) -> str:
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.append(
                        f"<li><strong>{html.escape(str(k))}</strong><ul>{render_dict(v)}</ul></li>"
                    )
                elif isinstance(v, list):
                    list_items = "".join(
                        f"<li>{html.escape(_format_value(item))}</li>" for item in v
                    )
                    items.append(
                        f"<li><strong>{html.escape(str(k))}</strong><ul>{list_items}</ul></li>"
                    )
                else:
                    items.append(
                        f"<li><strong>{html.escape(str(k))}</strong>: {html.escape(_format_value(v))}</li>"
                    )
            return "".join(items)

        return f"<ul>{render_dict(profile)}</ul>"

    description_html = (
        f"<div class='card'><div class='card-title'>User Description</div><div>{html.escape(user_description)}</div></div>"
        if user_description
        else ""
    )
    basic_html = (
        f"<div class='card'><div class='card-title'>User Basic Profile</div>{_render_basic_profile(user_basic_profile)}</div>"
        if user_basic_profile
        else "<div class='card'><div class='card-title'>User Basic Profile</div><div>Not found.</div></div>"
    )
    return f"""
<html>
<head>
<meta charset="UTF-8" />
<title>Dynamic Profile QA Report</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f7f7f7; color:#202124; margin:0; padding:24px; }}
section {{ background:#fff; border-radius:8px; padding:16px; margin-bottom:20px; box-shadow:0 2px 6px rgba(0,0,0,0.06); }}
table {{ width:100%; border-collapse: collapse; margin-top:8px; }}
th, td {{ border:1px solid #e0e0e0; padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ background:#fafafa; font-weight:600; }}
.card {{ border:1px solid #e0e0e0; border-radius:8px; padding:10px; margin-top:12px; background:#fafafa; }}
.card-title {{ font-weight:700; margin-bottom:6px; }}
.timeline td:nth-child(3) {{ text-align:center; width:70px; }}
.meta {{ color:#5f6368; margin-top:-6px; }}
h3 {{ margin-bottom:6px; }}
.grid th.row-h {{ background:#fafafa; width:180px; }}
.grid .sub {{ color:#5f6368; font-size:12px; margin-top:2px; }}
.compact {{ margin:0; padding-left:16px; }}
.dict div {{ margin-bottom:2px; }}
h1 {{ margin-top:0; }}
</style>
</head>
<body>
<h1>Dynamic Profile QA Report</h1>
<p>Source: {html.escape(source)}</p>
{description_html}
{basic_html}
{conflicts_html}
{domain_sections_html}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dynamic profiles across domains for quick QA."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_outputs_debug_v3" / "gemini_3_pro_preview",
        help="Directory containing generated output JSON files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_outputs_debug_v3" / "gemini_3_pro_preview" / "dynamic_profile_report.html",
        help="Where to write the HTML report.",
    )
    parser.add_argument(
        "--show-conflicts",
        action="store_true",
        help="Include cross-domain conflict/resolution section (hidden by default for annotators).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    profiles, source = _find_dynamic_profiles(output_dir)
    conflicts, resolutions = _load_conflict_data(output_dir)
    user_desc, user_basic = _load_user_context(output_dir)

    html_report = _build_html(
        conflicts,
        resolutions,
        profiles,
        source,
        show_conflicts=args.show_conflicts,
        user_description=user_desc,
        user_basic_profile=user_basic,
    )
    args.report_path.write_text(html_report, encoding="utf-8")
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    main()
