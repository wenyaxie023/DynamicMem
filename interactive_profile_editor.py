"""Serve an interactive, autosaving editor for dynamic profile JSON bundles.

This spins up a tiny local HTTP server with a single-page app that lets an
annotator:
- browse dynamic profiles by domain
- edit either the whole file, a single domain, or just a section (initial
  state or time windows)
- autosave changes back to disk (with an optional one-time backup)

Usage:
  python interactive_profile_editor.py \\
    --output-dir behavior_and_conversation/generated_outputs_debug_v2/gemini_3_flash_preview \\
    --file dynamic_profiles_conflict_resolved.json \\
    --port 8765 --open-browser
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
import webbrowser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_user_context(base_dir: Path) -> Tuple[str, dict, str]:
    """
    Returns (user_description_text, user_basic_profile_dict, original_user_description_text).
    Mirrors visualize_dynamic_profiles._load_user_context to avoid importing it.
    """
    description = ""
    basic_profile: dict = {}

    basic_path = base_dir / "user_basic_profile.json"
    if basic_path.exists():
        basic_profile = _read_json(basic_path)

    # Per-run description if present
    desc_path = base_dir / "context_user_profile.txt"
    if desc_path.exists():
        description = desc_path.read_text(encoding="utf-8").strip()
    else:
        pipeline_path = base_dir / "pipeline_output.json"
        if pipeline_path.exists():
            payload = _read_json(pipeline_path)
            description = payload.get("user_profile", "") or ""

    # Global/original description if present
    orig_desc_path = Path(__file__).resolve().parent / "context_user_description.txt"
    original_description = _safe_read(orig_desc_path).strip()

    return description, basic_profile, original_description


def _load_world_background(base_dir: Path) -> str:
    for name in ["context_world_background_2024.txt", "context_world_background.txt"]:
        candidate = base_dir / name
        if candidate.exists():
            return _safe_read(candidate).strip()
    # Fallback to repo-level context if dataset folder doesn't include one.
    repo_candidate = Path(__file__).resolve().parent / "context_world_background_2024.txt"
    if repo_candidate.exists():
        return _safe_read(repo_candidate).strip()
    legacy = Path(__file__).resolve().parent / "context_world_background.txt"
    return _safe_read(legacy).strip()


def _resolve_target_path(base_dir: Path, file_arg: str | None) -> Path:
    """Pick the JSON file to edit, defaulting to dynamic_profiles_conflict_resolved.json."""
    if file_arg:
        candidate = Path(file_arg)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
    else:
        candidate = base_dir / "dynamic_profiles_conflict_resolved.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find target JSON: {candidate}. "
            "Pass --file to point at the profile bundle you want to edit."
        )
    return candidate.resolve()


class AssignmentRegistry:
    """Track which annotator has claimed each dataset under a shared root."""

    def __init__(self, dataset_root: Path, target_filename: str, registry_name: str = "annotation_assignments.json") -> None:
        self.dataset_root = dataset_root
        self.target_filename = target_filename
        self.registry_path = dataset_root / registry_name
        self.lock = threading.Lock()
        self.assignments: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        text = json.dumps(self.assignments, ensure_ascii=False, indent=2)
        self.registry_path.write_text(text + "\n", encoding="utf-8")

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _validate_dataset(self, dataset_id: str) -> Tuple[Path, Path]:
        dataset_dir = (self.dataset_root / dataset_id).resolve()
        try:
            if not dataset_dir.is_relative_to(self.dataset_root):
                raise ValueError("Dataset outside configured root")
        except AttributeError:  # pragma: no cover - Python <3.9 guard
            if str(self.dataset_root.resolve()) not in str(dataset_dir):
                raise ValueError("Dataset outside configured root")
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(f"Dataset not found: {dataset_dir}")
        target_path = dataset_dir / self.target_filename
        if not target_path.exists():
            raise ValueError(f"Target file missing: {target_path}")
        return dataset_dir, target_path

    def _detect_existing_assignment(self, dataset_id: str, dataset_dir: Path) -> None:
        """Bootstrap registry entries from existing annotated files."""
        if dataset_id in self.assignments:
            return
        target_stem = Path(self.target_filename).stem
        for path in dataset_dir.glob(f"{target_stem}_annotated*.json"):
            try:
                payload = _read_json(path)
            except Exception:
                continue
            annotator = payload.get("annotator") or payload.get("__meta", {}).get("annotator")
            if annotator:
                self.assignments[dataset_id] = {
                    "annotator": annotator,
                    "claimed_at": self._now(),
                    "detected_from": path.name,
                }
                self._save()
                return

    def claim(self, dataset_id: str, annotator: str) -> Dict[str, Any]:
        with self.lock:
            dataset_dir, target_path = self._validate_dataset(dataset_id)
            self._detect_existing_assignment(dataset_id, dataset_dir)
            entry = self.assignments.get(dataset_id)
            if entry and entry.get("annotator") and entry["annotator"] != annotator:
                raise ValueError(f"Dataset already assigned to {entry['annotator']}")
            if not entry:
                entry = {"annotator": annotator, "claimed_at": self._now()}
                self.assignments[dataset_id] = entry
                self._save()
            entry["target_path"] = str(target_path)
            return entry

    def describe_datasets(self, current_user: str | None = None) -> Dict[str, Any]:
        datasets = []
        if not self.dataset_root.exists():
            return {"datasets": datasets}
        for child in sorted(self.dataset_root.iterdir()):
            if not child.is_dir():
                continue
            target_path = child / self.target_filename
            if not target_path.exists():
                continue
            dataset_id = child.name
            self._detect_existing_assignment(dataset_id, child)
            assigned_to = (self.assignments.get(dataset_id) or {}).get("annotator")
            status = "available"
            if assigned_to and assigned_to != current_user:
                status = "locked"
            elif assigned_to and assigned_to == current_user:
                status = "yours"
            datasets.append(
                {
                    "id": dataset_id,
                    "path": str(target_path),
                    "assigned_to": assigned_to,
                    "status": status,
                    "claimed_at": (self.assignments.get(dataset_id) or {}).get("claimed_at"),
                }
            )
        return {
            "datasets": datasets,
            "registry_path": str(self.registry_path),
            "dataset_root": str(self.dataset_root),
            "target_filename": self.target_filename,
        }


class EditorState:
    def __init__(
        self,
        base_dir: Path | None,
        target_path: Path | None,
        backup: bool,
        save_as: Path | None = None,
        dataset_root: Path | None = None,
        target_filename: str = "dynamic_profiles_conflict_resolved.json",
    ) -> None:
        self.base_dir = base_dir
        self.target_path = target_path
        self.default_save_base: Path | None = None
        self.save_path: Path | None = None
        self.save_as_override = save_as
        self.lock = threading.Lock()
        self.backup_enabled = backup
        self._backup_made = False
        self.current_user: str | None = None
        self.active_dataset_id: str | None = None
        self.dataset_root = dataset_root
        self.target_filename = target_filename
        self.assignment_registry = AssignmentRegistry(dataset_root, target_filename) if dataset_root else None

        self.user_description = ""
        self.user_basic_profile: dict = {}
        self.original_user_description = _safe_read(
            Path(__file__).resolve().parent / "context_user_description.txt"
        ).strip()
        self.world_background = ""
        instructions_dir = Path(__file__).resolve().parent
        brief_path = instructions_dir / "dynamic_profile_brief_instructions.txt"
        detailed_path = instructions_dir / "dynamic_profile_detailed_instructions.txt"
        legacy_path = instructions_dir / "dynamic_profile_cross_domain_conflict_resolution.txt"
        self.data_verify_instructions_brief = _safe_read(brief_path).strip()
        self.data_verify_instructions_detailed = _safe_read(detailed_path).strip()
        legacy = _safe_read(legacy_path).strip()
        # Keep legacy field for backward compatibility; prefer brief if present.
        self.data_verify_instructions = self.data_verify_instructions_brief or legacy
        if not self.data_verify_instructions_detailed:
            self.data_verify_instructions_detailed = legacy

        if base_dir and target_path:
            self._configure_dataset(base_dir=base_dir, target_path=target_path, save_as=save_as)
            self._reload_context(base_dir)

    def _configure_dataset(self, base_dir: Path, target_path: Path, save_as: Path | None = None) -> None:
        self.base_dir = base_dir
        self.target_path = target_path
        self.default_save_base = save_as or target_path.with_name(target_path.stem + "_annotated.json")
        self._backup_made = False
        self._refresh_save_path()

    def _reload_context(self, base_dir: Path) -> None:
        (
            self.user_description,
            self.user_basic_profile,
            self.original_user_description,
        ) = _load_user_context(base_dir)
        self.world_background = _load_world_background(base_dir)

    def _refresh_save_path(self) -> None:
        if not self.default_save_base:
            return
        base = self.default_save_base
        stem = base.stem
        if not stem.endswith("_annotated"):
            stem = stem + "_annotated"
        if self.current_user:
            self.save_path = base.with_name(f"{stem}_{self.current_user}{base.suffix}")
        else:
            self.save_path = base.with_name(f"{stem}{base.suffix}")

    def ensure_dataset_active(self) -> None:
        if not self.base_dir or not self.target_path or not self.default_save_base:
            raise ValueError("No dataset selected. Please claim or resume a dataset first.")

    def activate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        if not self.dataset_root:
            raise ValueError("Dataset root not configured on server.")
        dataset_dir = self.dataset_root / dataset_id
        target_path = _resolve_target_path(dataset_dir, self.target_filename)
        save_as = self.save_as_override
        if save_as and not save_as.is_absolute():
            save_as = dataset_dir / save_as
        self.active_dataset_id = dataset_id
        self._configure_dataset(base_dir=dataset_dir, target_path=target_path, save_as=save_as)
        self._reload_context(dataset_dir)
        # If the user logged in before selecting a dataset, update save path now.
        self._refresh_save_path()
        return {"dataset_id": dataset_id, "target_path": str(target_path), "base_dir": str(dataset_dir)}

    def _compute_stats(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for entry in (annotations or {}).values():
            reason = entry.get("reasonType") or "Unspecified"
            counts[reason] = counts.get(reason, 0) + 1
        total = sum(counts.values())
        return {"total": total, "counts": counts}

    def load_bundle(self) -> Dict[str, Any]:
        self.ensure_dataset_active()
        with self.lock:
            path = self.save_path if self.save_path and self.save_path.exists() else self.target_path
            data = _read_json(path)

        profiles: Dict[str, Any]
        annotations: Dict[str, Any] = {}
        stats: Dict[str, Any] = {}
        review_duration_seconds: int | None = None

        if isinstance(data, dict) and "profiles" in data:
            profiles = data.get("profiles", {})
            annotations = data.get("__annotations") or data.get("annotations") or {}
            stats = data.get("__stats") or data.get("stats") or {}
            review_duration_seconds = data.get("review_duration_seconds")
        else:
            profiles = data

        if not stats:
            stats = self._compute_stats(annotations)

        return {
            "profiles": profiles,
            "annotations": annotations,
            "stats": stats,
            "review_duration_seconds": review_duration_seconds,
        }

    def save_bundle(
        self,
        profiles: Dict[str, Any],
        annotations: Dict[str, Any],
        stats: Dict[str, Any],
        review_duration_seconds: int | None,
    ) -> Dict[str, Any]:
        self.ensure_dataset_active()
        stats = stats or self._compute_stats(annotations)
        payload = {
            "profiles": profiles,
            "__annotations": annotations or {},
            "__stats": stats,
            "source_path": str(self.target_path),
            "review_duration_seconds": review_duration_seconds,
            "annotator": self.current_user,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with self.lock:
            if self.backup_enabled and not self._backup_made and self.save_path and self.save_path.exists():
                backup_path = self.save_path.with_suffix(self.save_path.suffix + ".bak")
                backup_path.write_text(self.save_path.read_text(encoding="utf-8"), encoding="utf-8")
                self._backup_made = True
            if not self.save_path:
                raise ValueError("Save path missing. Please claim a dataset first.")
            self.save_path.write_text(text + "\n", encoding="utf-8")
        return {
            "saved_to": str(self.save_path),
            "bytes": len(text.encode("utf-8")),
            "backup_created": self._backup_made,
        }

    def set_user(self, annotator: str) -> None:
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in annotator).strip("_")
        if not slug:
            raise ValueError("Annotator name is required")
        self.current_user = slug
        self._refresh_save_path()


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dynamic Profile Editor</title>
  <style>
    :root {
      --bg: #f3f1eb;
      --card: #ffffff;
      --ink: #1d1f21;
      --muted: #5c6672;
      --accent: #0f766e;
      --accent-2: #1b4b7a;
      --border: #d8d5cf;
      --danger: #c2410c;
      --success: #0f7b2c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "DM Sans", "Trebuchet MS", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 10% 20%, #f7f3ff 0, #f3f1eb 25%, #f3f1eb 100%);
      color: var(--ink);
      min-height: 100vh;
    }
    header {
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      background: #fbfaf7;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header .title {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.4px;
    }
    header .meta { color: var(--muted); font-size: 13px; }
    header .actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    button {
      border: 1px solid var(--border);
      background: var(--card);
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
      transition: transform 120ms ease, box-shadow 120ms ease;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    button.primary {
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #fff;
      border: none;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 14px;
      padding: 16px 18px 24px;
      width: 100%;
      max-width: 100vw;
      margin: 0 auto;
    }
    main, aside { min-width: 0; }
    aside {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.05);
      display: flex;
      flex-direction: column;
      min-height: 70vh;
    }
    .domain-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }
    .domain-tile {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      cursor: pointer;
      transition: border 120ms ease, transform 120ms ease, box-shadow 120ms ease;
      background: #fdfbf8;
    }
    .domain-tile:hover { transform: translateX(2px); box-shadow: 0 3px 8px rgba(0,0,0,0.04); }
    .domain-tile.active { border-color: var(--accent); box-shadow: 0 4px 12px rgba(15,118,110,0.16); }
    .domain-name { font-weight: 700; }
    details[data-domain-section] { scroll-margin-top: 72px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(15,118,110,0.12);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .pill.warn { background: rgba(194,65,12,0.12); color: var(--danger); }
    main {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 6px 14px rgba(0,0,0,0.06);
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
    }
    .card h2 { margin: 0 0 6px; }
    .muted { color: var(--muted); }
    pre {
      background: #f7f5f0;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
    }
    details { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: #fdfcf9; }
    summary { cursor: pointer; font-weight: 700; }
    .section-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }
    .status-bar {
      margin-left: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .status-bar.ok { color: var(--success); }
    .status-bar.error { color: var(--danger); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 10px;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.35);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 30;
    }
    .modal {
      background: var(--card);
      border-radius: 12px;
      padding: 14px;
      width: min(960px, 90vw);
      max-height: 90vh;
      overflow: auto;
      border: 1px solid var(--border);
      box-shadow: 0 8px 20px rgba(0,0,0,0.16);
    }
    textarea {
      width: 100%;
      min-height: 320px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      font-family: "DM Mono", "SFMono-Regular", Consolas, monospace;
      background: #f8f7f3;
      resize: vertical;
      font-size: 13px;
    }
    .flex {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .small { font-size: 12px; }
    .grid-table { width: 100%; border-collapse: collapse; margin-top: 6px; table-layout: fixed; }
    .grid-table th, .grid-table td { border: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; background: #fff; word-break: break-word; }
    .grid-table th:first-child { width: 160px; background: #f6f5f2; }
    .cell-actions { margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }
    .pill.small { font-size: 11px; padding: 2px 6px; }
    .ghost { background: #f0efec; }
    label { font-weight: 600; display: block; margin-bottom: 4px; }
    input[type="text"] { width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 10px; }
    select { width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 10px; background: #fff; }
    .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
    .window-block { border: 1px dashed var(--border); border-radius: 10px; padding: 10px; margin-top: 8px; background: #fdfbf8; }
    .window-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
    .chart { margin-top: 6px; }
    .bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
    .bar { flex: 1; background: #f0efec; border-radius: 6px; overflow: hidden; }
    .bar-fill { height: 10px; background: linear-gradient(120deg, var(--accent), var(--accent-2)); }
    .reason-buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
    .reason-btn { border: 1px solid var(--border); background: #fff; border-radius: 999px; padding: 6px 10px; cursor: pointer; font-weight: 600; }
    .reason-btn.active { border-color: var(--accent); background: rgba(15,118,110,0.12); color: var(--accent); }
    .login-overlay { position: fixed; inset:0; background: rgba(0,0,0,0.35); display:flex; align-items:center; justify-content:center; z-index:50; }
    .login-card { background:#fff; padding:16px; border-radius:12px; width: min(420px, 90vw); box-shadow:0 8px 18px rgba(0,0,0,0.15); border:1px solid var(--border); }
    .login-card h3 { margin-top:0; }
    .dataset-scroll { max-height: 280px; overflow: auto; padding-right: 4px; }
    .dataset-item { border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; background: #fdfbf8; word-break: break-word; }
    .dataset-item + .dataset-item { margin-top: 6px; }
    .dataset-item .title { font-weight: 700; word-break: break-word; }
    .dataset-item .meta { font-size: 12px; color: var(--muted); }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      aside { min-height: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="title">Dynamic Profile Editor</div>
      <div class="meta">Dataset: <span id="target-path"></span></div>
    </div>
    <div class="actions">
      <button id="reload">Reload from disk</button>
      <button class="primary" id="save-btn">Save now (Ctrl/Cmd+S)</button>
      <div id="save-status" class="status-bar">Idle</div>
    </div>
  </header>
  <div class="layout">
    <div class="login-overlay" id="login-overlay" style="display:flex;">
      <div class="login-card">
        <h3>Login</h3>
        <p class="muted">Enter annotator name to resume your session and claim a dataset.</p>
        <input type="text" id="annotator-input" placeholder="Annotator name" />
        <div class="flex" style="margin-top:10px; justify-content:flex-end;">
          <button class="primary" id="login-btn">Login</button>
        </div>
      </div>
    </div>
    <aside>
      <div class="card" id="dataset-card" style="display:none; margin-bottom:10px;">
        <div class="section-header">
          <div style="font-weight:700;">Datasets</div>
          <div class="flex">
            <span class="pill small" id="dataset-mode-pill" style="display:none;">Claim-only</span>
            <button class="ghost" id="refresh-datasets">Refresh</button>
          </div>
        </div>
        <div class="muted small" id="dataset-root-hint"></div>
        <div id="dataset-yours" class="dataset-scroll"></div>
        <div id="dataset-available" class="dataset-scroll" style="margin-top:8px;"></div>
        <div id="dataset-locked" class="dataset-scroll" style="margin-top:8px;"></div>
      </div>
      <div class="flex" style="justify-content: space-between;">
        <div style="font-weight: 700;">Domains</div>
      </div>
      <div class="domain-list" id="domain-list"></div>
    </aside>
    <main>
      <div class="card" id="user-context" style="display:none;"></div>
      <div class="card" id="stats-card" style="display:none;"></div>
      <div class="card" id="domain-detail"><span class="muted">Select a domain to inspect and edit.</span></div>
    </main>
  </div>

  <div class="modal-backdrop" id="modal">
    <div class="modal">
      <div class="section-header">
        <h3 id="modal-title">Edit JSON</h3>
        <div class="flex">
          <button id="close-modal">Cancel</button>
          <button class="primary" id="apply-modal">Apply & Save</button>
        </div>
      </div>
      <div class="muted small" id="modal-hint"></div>
      <div id="json-editor-wrap">
        <textarea id="editor"></textarea>
      </div>
      <div id="cell-editor-wrap" style="display:none;">
        <div id="cell-target" class="muted small" style="margin-bottom:6px;"></div>
        <label for="cell-value">Value (JSON)</label>
        <textarea id="cell-value"></textarea>
        <div class="two-col" style="margin-top:8px;">
          <div id="op-wrap">
            <label for="cell-op">Op</label>
            <select id="cell-op"></select>
          </div>
          <div id="change-reason-wrap">
            <label for="cell-change-reason">Change reason (applied to profile)</label>
            <input type="text" id="cell-change-reason" placeholder="reason tied to op (stored in profile)" />
          </div>
        </div>
        <div style="margin-top:8px;">
          <label>Conflict Type</label>
          <div id="reason-buttons" class="reason-buttons"></div>
        </div>
        <div style="margin-top:8px;">
          <label for="cell-conflict-description">Conflict description (optional)</label>
          <input type="text" id="cell-conflict-description" placeholder="short conflict detail (optional)" />
        </div>
        <div style="margin-top:8px;">
          <label for="cell-correction-detail">If Other, add note</label>
          <input type="text" id="cell-correction-detail" placeholder="conflict/correction note" />
        </div>
      </div>
    </div>
  </div>

  <script>
    const SERVER_STATE = __SERVER_STATE__;
    const STATE_CONFIG = {
      user_attributes_state: { label: "User Attributes", deltaKey: "user_attributes_delta", nameKey: "attribute_name" },
      habits_state: { label: "Habits", deltaKey: "habits_delta", nameKey: "habit_name" },
      preferences_state: { label: "Preferences", deltaKey: "preferences_delta", nameKey: "preference_name" },
    };
    const OPS = {
      user_attributes_state: ["add", "remove", "modify"],
      habits_state: ["acquire", "adjust", "drop"],
      preferences_state: ["shift", "amplify", "attenuate"],
    };
    const DEFAULT_VALUES = {
      user_attributes_state: [""],
      habits_state: { action: "", frequency: "", timing: "", context: "", description: "" },
      preferences_state: "",
    };
    const REASON_OPTIONS = [
      "Category 1: Structural Missing Fields",
      "Category 2: Missing Prior Existence for Modifications",
      "Category 3: Missing Essential Items That Should Be Initialized",
      "Category 4: Direct Time Conflicts (Temporal Collision)",
      "Category 5: Schedule Overload / Unrealistic Time Constraints",
      "Category 6: Attribute Conflicts Across Domains (singular or collection)",
      "Category 7: Short-Term Changes Without Follow-Up",
      "Category 8: Cross-Window Inconsistencies / Narrative Coherence",
      "Other",
    ];
    const USAGE_NOTES = [
      "Use Add/Edit buttons in each cell to tweak values; Delete removes that entry for the column.",
      "Op is a dropdown with allowed values per type; Reason type must be selected; add a note when choosing Other.",
      "Apply & Save writes to a new annotated file; top Save or Cmd/Ctrl+S also saves.",
      "All domains start expanded; collapse any domain manually. Left list lets you jump to a domain.",
      "To add a new row, use the Add button in the section header. Add new windows via the windows JSON editor.",
      "Window descriptions and summaries can be edited inline in each domain section.",
    ];

    const state = {
      profiles: {},
      worldBackground: "",
      userBasicProfile: {},
      sourcePath: "",
      savePath: "",
      dataVerifyInstructions: "",
      dataVerifyInstructionsBrief: "",
      dataVerifyInstructionsDetailed: "",
      dirty: false,
      dirtyDomains: new Set(),
      saving: false,
      grids: {},
      focusedDomain: null,
      annotations: {},
      stats: {},
      sessionStart: Date.now(),
      lastActivity: Date.now(),
      reviewDurationSeconds: 0,
      annotator: "",
      loggedIn: false,
      datasetMode: Boolean(SERVER_STATE.dataset_mode),
      datasetRoot: SERVER_STATE.dataset_root || "",
      datasets: [],
      activeDataset: null,
      assignmentRegistryPath: "",
    };

    const dom = {
      list: document.getElementById("domain-list"),
      detail: document.getElementById("domain-detail"),
      context: document.getElementById("user-context"),
      stats: document.getElementById("stats-card"),
      targetPath: document.getElementById("target-path"),
      saveStatus: document.getElementById("save-status"),
      saveBtn: document.getElementById("save-btn"),
      reloadBtn: document.getElementById("reload"),
      modal: document.getElementById("modal"),
      modalTitle: document.getElementById("modal-title"),
      modalHint: document.getElementById("modal-hint"),
      editor: document.getElementById("editor"),
      applyModal: document.getElementById("apply-modal"),
      closeModal: document.getElementById("close-modal"),
      jsonWrap: document.getElementById("json-editor-wrap"),
      cellWrap: document.getElementById("cell-editor-wrap"),
      cellValue: document.getElementById("cell-value"),
      cellOp: document.getElementById("cell-op"),
      changeReasonWrap: document.getElementById("change-reason-wrap"),
      cellChangeReason: document.getElementById("cell-change-reason"),
      cellConflictDescription: document.getElementById("cell-conflict-description"),
      cellCorrectionDetail: document.getElementById("cell-correction-detail"),
      reasonButtons: document.getElementById("reason-buttons"),
      cellTarget: document.getElementById("cell-target"),
      opWrap: document.getElementById("op-wrap"),
      loginOverlay: document.getElementById("login-overlay"),
      loginBtn: document.getElementById("login-btn"),
      annotatorInput: document.getElementById("annotator-input"),
      datasetCard: document.getElementById("dataset-card"),
      datasetAvailable: document.getElementById("dataset-available"),
      datasetYours: document.getElementById("dataset-yours"),
      datasetLocked: document.getElementById("dataset-locked"),
      datasetRootHint: document.getElementById("dataset-root-hint"),
      refreshDatasets: document.getElementById("refresh-datasets"),
      datasetModePill: document.getElementById("dataset-mode-pill"),
    };

    let modalState = { mode: "json", scope: "domain", domain: null };
    let idleTimer = null;

    function escapeHtml(str) {
      return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }
    function shortId(id) {
      return (id || "").split("_")[0] || "";
    }
    function shortName(path) {
      if (!path) return "";
      const parts = path.split(/[/\\\\]/);
      return parts[parts.length - 1] || path;
    }
    function targetLabel() {
      if (state.datasetMode) return shortId(state.activeDataset) || "n/a";
      return shortName(state.sourcePath) || "n/a";
    }
    function safeCSS(str) {
      if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(str);
      return String(str || "").replace(/[^a-zA-Z0-9_-]/g, "_");
    }
    function setStatus(text, variant = "") {
      dom.saveStatus.textContent = text;
      dom.saveStatus.classList.remove("ok", "error");
      if (variant === "ok") dom.saveStatus.classList.add("ok");
      if (variant === "error") dom.saveStatus.classList.add("error");
    }
    function pretty(value) {
      try {
        return JSON.stringify(value ?? {}, null, 2);
      } catch (_err) {
        return String(value);
      }
    }
    function valuePreview(value) {
      return pretty(value === undefined ? null : value);
    }

    function computeStats(annotations) {
      const entries = Object.values(annotations || {});
      const counts = {};
      entries.forEach((a) => {
        const key = a.reasonType || "Unspecified";
        counts[key] = (counts[key] || 0) + 1;
      });
      return { total: entries.length, counts };
    }

    function resetDataView(message = "Select a dataset to start.") {
      state.profiles = {};
      state.annotations = {};
      state.stats = {};
      state.dirtyDomains = new Set();
      state.dirty = false;
      state.worldBackground = "";
      state.userBasicProfile = {};
      state.sourcePath = "";
      state.savePath = "";
      state.focusedDomain = null;
      dom.context.style.display = "none";
      dom.stats.style.display = "none";
      dom.list.innerHTML = `<div class='muted'>${escapeHtml(message)}</div>`;
      dom.detail.innerHTML = `<span class='muted'>${escapeHtml(message)}</span>`;
      dom.targetPath.textContent = targetLabel();
      updateActionButtons();
    }

    function applyBundlePayload(payload) {
      if (!payload) return;
      state.profiles = payload.profiles || {};
      state.worldBackground = payload.world_background || payload.worldBackground || "";
      state.dataVerifyInstructions = payload.data_verify_instructions || state.dataVerifyInstructions;
      state.dataVerifyInstructionsBrief = payload.data_verify_instructions_brief || payload.data_verify_instructions || "";
      state.dataVerifyInstructionsDetailed = payload.data_verify_instructions_detailed || "";
      state.userBasicProfile = payload.user_basic_profile || state.userBasicProfile;
      state.sourcePath = payload.source_path || state.sourcePath;
      state.savePath = payload.save_path || state.savePath;
      state.annotations = payload.annotations || {};
      state.stats = payload.stats || computeStats(state.annotations);
      state.reviewDurationSeconds = payload.review_duration_seconds || 0;
      state.sessionStart = Date.now() - (state.reviewDurationSeconds || 0) * 1000;
      state.lastActivity = Date.now();
      state.dirtyDomains = new Set();
      state.dirty = false;
      state.activeDataset = payload.dataset_id || state.activeDataset;
      state.focusedDomain = null;
      dom.targetPath.textContent = targetLabel();
      renderContext();
      renderStats();
      renderDomainList();
      renderDomains();
      updateActionButtons();
    }

    function hydrateDatasets(payload) {
      if (!payload) return;
      const desc = payload.datasets;
      const list = Array.isArray(desc?.datasets) ? desc.datasets : Array.isArray(payload.datasets) ? payload.datasets : [];
      state.datasets = list;
      const root = desc?.dataset_root || payload.dataset_root;
      const registry = desc?.registry_path || payload.registry_path;
      if (root) state.datasetRoot = root;
      if (registry) state.assignmentRegistryPath = registry;
      renderDatasets();
    }

    function updateActionButtons() {
      const disabled = !state.loggedIn || (state.datasetMode && !state.activeDataset);
      dom.saveBtn.disabled = disabled;
      dom.reloadBtn.disabled = disabled;
    }

    function renderMarkdown(text) {
      const lines = String(text || "").split(/\\r?\\n/);
      const htmlLines = [];
      let inList = false;
      let inCode = false;
      let codeLang = "";
      let codeBuffer = [];
      lines.forEach((line) => {
        const fence = line.match(/^```(.*)$/);
        if (fence) {
          if (inCode) {
            const langClass = codeLang ? ` class="lang-${escapeHtml(codeLang)}"` : "";
            htmlLines.push(`<pre><code${langClass}>${escapeHtml(codeBuffer.join("\\n"))}</code></pre>`);
            codeBuffer = [];
            codeLang = "";
            inCode = false;
          } else {
            if (inList) {
              htmlLines.push("</ul>");
              inList = false;
            }
            inCode = true;
            codeLang = (fence[1] || "").trim();
          }
          return;
        }
        if (inCode) {
          codeBuffer.push(line);
          return;
        }
        if (/^\\s*(-|\\*)\\s+/.test(line)) {
          if (!inList) {
            htmlLines.push("<ul>");
            inList = true;
          }
          const item = escapeHtml(line.replace(/^\\s*(-|\\*)\\s+/, ""));
          htmlLines.push(`<li>${item}</li>`);
          return;
        }
        if (inList) {
          htmlLines.push("</ul>");
          inList = false;
        }
        const h3 = line.match(/^###\\s+(.*)/);
        const h2 = line.match(/^##\\s+(.*)/);
        const h1 = line.match(/^#\\s+(.*)/);
        if (h3) {
          htmlLines.push(`<h3>${escapeHtml(h3[1])}</h3>`);
          return;
        }
        if (h2) {
          htmlLines.push(`<h2>${escapeHtml(h2[1])}</h2>`);
          return;
        }
        if (h1) {
          htmlLines.push(`<h1>${escapeHtml(h1[1])}</h1>`);
          return;
        }
        const safe = escapeHtml(line);
        let rendered = safe.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
        rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
        htmlLines.push(`<div>${rendered}</div>`);
      });
      if (inList) htmlLines.push("</ul>");
      if (inCode) {
        const langClass = codeLang ? ` class="lang-${escapeHtml(codeLang)}"` : "";
        htmlLines.push(`<pre><code${langClass}>${escapeHtml(codeBuffer.join("\\n"))}</code></pre>`);
      }
      return htmlLines.join("");
    }

    function renderProfileList(obj) {
      if (!obj || typeof obj !== "object") return "<div class='muted'>No basic profile found.</div>";
      const renderNode = (val) => {
        if (Array.isArray(val)) {
          return `<ul>${val.map((v) => `<li>${escapeHtml(String(v))}</li>`).join("")}</ul>`;
        }
        if (val && typeof val === "object") {
          return `<ul>${Object.entries(val)
            .map(([k, v]) => `<li><strong>${escapeHtml(k)}</strong>: ${renderNode(v)}</li>`)
            .join("")}</ul>`;
        }
        return escapeHtml(String(val));
      };
      return renderNode(obj);
    }

    function scrollDomainIntoView(name, behavior = "smooth") {
      if (!name) return;
      const section = document.querySelector(`[data-domain-section="${safeCSS(name)}"]`);
      if (!section) return;
      section.open = true;
      document.querySelectorAll("[data-domain-section]").forEach((el) => {
        if (el !== section) el.open = false;
      });
      section.scrollIntoView({ behavior, block: "start" });
    }

    function renderStats() {
      if (state.datasetMode && !state.activeDataset) {
        dom.stats.style.display = "none";
        dom.stats.innerHTML = "";
        return;
      }
      const stats = computeStats(state.annotations);
      state.stats = stats;
      state.reviewDurationSeconds = Math.round((Date.now() - state.sessionStart) / 1000);
      if (!stats.total) {
        dom.stats.style.display = "none";
        dom.stats.innerHTML = "";
        return;
      }
      const max = Math.max(...Object.values(stats.counts));
      const rows = Object.entries(stats.counts)
        .map(
          ([k, v]) => `
            <div class="bar-row">
              <div style="width:240px;">${escapeHtml(k)}</div>
              <div class="bar"><div class="bar-fill" style="width:${Math.max(8, (v / max) * 100)}%;"></div></div>
              <div class="pill small">${v}</div>
            </div>`
        )
        .join("");
      dom.stats.style.display = "block";
      dom.stats.innerHTML = `
        <h2>Correction Stats</h2>
        <div class="muted">Annotator: ${escapeHtml(state.annotator || "n/a")} | Saved to: ${escapeHtml(state.savePath || "")} | Total corrections: ${stats.total} | Review time: ${state.reviewDurationSeconds}s</div>
        <div class="chart">${rows}</div>
      `;
    }

    function renderContext() {
      if (state.datasetMode && !state.activeDataset) {
        dom.context.style.display = "none";
        return;
      }
      const cards = [];
      const addSection = (title, body, open = true) => {
        if (!body) return;
        cards.push(
          `<details ${open ? "open" : ""}><summary>${escapeHtml(title)}</summary><div style="margin-top:6px;">${body}</div></details>`
        );
      };

      const instructions =
        state.dataVerifyInstructionsDetailed || state.dataVerifyInstructions || state.dataVerifyInstructionsBrief || "";
      if (instructions) {
        addSection(
          "Dynamic Profile Human Validation Instructions",
          `<div class="md">${renderMarkdown(instructions)}</div>`,
          true
        );
      }
      addSection(
        "How to use this editor",
        `<ul>${USAGE_NOTES.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>`,
        true
      );
      addSection("Basic profile", renderProfileList(state.userBasicProfile), false);
      addSection("World background 2024", `<div class="md">${renderMarkdown(state.worldBackground)}</div>`, false);

      if (!cards.length) {
        dom.context.style.display = "none";
        return;
      }
      dom.context.style.display = "block";
      dom.context.innerHTML = `<h2>Dynamic Profile Human Validation Instructions</h2>${cards.join("")}`;
    }

    function renderDomainList() {
      if (state.datasetMode && !state.activeDataset) {
        dom.list.innerHTML = "<div class='muted'>Select a dataset to begin.</div>";
        return;
      }
      dom.list.innerHTML = "";
      const names = Object.keys(state.profiles || {}).sort();
      if (!names.length) {
        dom.list.innerHTML = "<div class='muted'>No domains loaded.</div>";
        return;
      }
      names.forEach((name) => {
        const domain = state.profiles[name] || {};
        const windows = Array.isArray(domain.time_windows) ? domain.time_windows.length : 0;
        const tile = document.createElement("div");
        tile.className = "domain-tile" + (state.focusedDomain === name ? " active" : "");
        tile.onclick = () => {
          state.focusedDomain = name;
          renderDomainList();
          scrollDomainIntoView(name, "smooth");
        };
        const dirtyMark = state.dirtyDomains.has(name) ? `<span class="pill warn">edited</span>` : "";
        tile.innerHTML = `
          <div class="domain-name">${escapeHtml(name)}</div>
          <div class="muted">Windows: ${windows}</div>
          <div style="margin-top:4px;">${dirtyMark}</div>
        `;
        dom.list.appendChild(tile);
      });
    }

    function buildGridForDomain(domain) {
      const windows = Array.isArray(domain.time_windows) ? domain.time_windows : [];
      const headers = [{ id: "init", label: "Initial state" }];
      windows.forEach((w, idx) => {
        const tr = Array.isArray(w.time_range) ? w.time_range.join(" to ") : "";
        const label = tr || w.window_description || w.summary || "";
        headers.push({ id: w.window_id || `w${idx + 1}`, label });
      });

      const grids = {
        user_attributes_state: new Map(),
        habits_state: new Map(),
        preferences_state: new Map(),
      };

      function ensureRow(stateType, name) {
        const map = grids[stateType];
        if (!map.has(name)) {
          map.set(name, { name, cells: Array(headers.length).fill(null) });
        }
        return map.get(name);
      }

      function setCell(stateType, name, colIdx, data) {
        const row = ensureRow(stateType, name);
        row.cells[colIdx] = {
          value: data.value,
          op: data.op || "",
          reason: data.reason || "",
          source: data.source || null,
        };
      }

      function iterInitial(entries) {
        if (!entries) return [];
        if (Array.isArray(entries)) {
          const pairs = [];
          entries.forEach((item) => {
            if (typeof item === "string") {
              pairs.push([item, null]);
            } else if (item && typeof item === "object") {
              Object.entries(item).forEach(([k, v]) => pairs.push([k, v]));
            }
          });
          return pairs;
        }
        if (typeof entries === "object") {
          return Object.entries(entries);
        }
        return [];
      }

      const init = domain.initial_state || {};
      ["user_attributes_state", "habits_state", "preferences_state"].forEach((stateType) => {
        const entries = (init[stateType] || {}).initial;
        for (const [name, val] of iterInitial(entries)) {
          setCell(stateType, name, 0, { value: val, op: "", reason: "initial", source: { kind: "initial", stateType, name } });
        }
      });

      windows.forEach((w, wIdx) => {
        const colIdx = wIdx + 1;
        Object.entries(STATE_CONFIG).forEach(([stateType, cfg]) => {
          const delta = w[cfg.deltaKey];
          const ops = delta && Array.isArray(delta.operations) ? delta.operations : [];
          const byName = new Map();
          ops.forEach((opObj, idx) => {
            const name = opObj[cfg.nameKey];
            if (!name) return;
            byName.set(name, { opObj, idx });
          });
          byName.forEach(({ opObj, idx }, name) => {
            setCell(stateType, name, colIdx, {
              value: opObj.new_state,
              op: opObj.op,
              reason: opObj.reason,
              source: { kind: "op", stateType, windowIndex: wIdx, opIndex: idx, name, deltaKey: cfg.deltaKey, nameKey: cfg.nameKey },
            });
          });
        });
      });

      return {
        headers,
        grids: {
          user_attributes_state: Array.from(grids.user_attributes_state.values()),
          habits_state: Array.from(grids.habits_state.values()),
          preferences_state: Array.from(grids.preferences_state.values()),
        },
      };
    }

    function renderStateGrid(stateType, grid, domainName) {
      const cfg = STATE_CONFIG[stateType];
      const rows = grid.grids[stateType] || [];
      const headers = grid.headers || [];
      if (!headers.length) return "";
      const headerCells = headers
        .map((h, idx) => `<th>${idx === 0 ? "Initial" : escapeHtml(h.id)}<div class="sub">${escapeHtml(h.label || "")}</div></th>`)
        .join("");

      const body = rows
        .map((row) => {
          const cells = headers
            .map((_, idx) => {
              const cell = row.cells[idx] || {};
              const hasData = (cell.value !== undefined && cell.value !== null) || cell.op || cell.reason;
              const badge = cell.op ? `<span class="pill small">${escapeHtml(cell.op)}</span>` : "";
              const valueBlock = hasData ? `<pre>${escapeHtml(valuePreview(cell.value))}</pre>` : `<span class="muted">Empty</span>`;
              const reason = cell.reason ? `<div class="muted small">${escapeHtml(cell.reason)}</div>` : "";
              const actions = hasData
                ? `<div class="cell-actions">
                    <button data-action="edit" class="ghost" data-domain="${escapeHtml(domainName)}">Edit</button>
                    <button data-action="delete" data-domain="${escapeHtml(domainName)}">Delete</button>
                  </div>`
                : `<button data-action="add" class="ghost" data-domain="${escapeHtml(domainName)}">Add</button>`;
              return `<td data-state="${stateType}" data-name="${escapeHtml(row.name)}" data-col="${idx}" data-domain="${escapeHtml(domainName)}">${badge}${valueBlock}${reason}${actions}</td>`;
            })
            .join("");
          return `<tr><th><div class="flex" style="justify-content:space-between; gap:6px; align-items:center;"><span>${escapeHtml(row.name)}</span><button class="ghost" data-action="delete-row" data-state="${stateType}" data-name="${escapeHtml(row.name)}" data-domain="${escapeHtml(domainName)}">Delete row</button></div></th>${cells}</tr>`;
        })
        .join("");

      const emptyRow = rows.length ? "" : `<tr><td colspan="${headers.length + 1}" class="muted">No entries yet.</td></tr>`;

      return `
        <details open>
          <summary class="section-header" style="cursor:pointer;">
            <div style="font-weight:700;">${cfg.label}</div>
            <div class="flex">
              <button data-action="add-row" data-state="${stateType}" data-domain="${escapeHtml(domainName)}">Add ${cfg.label}</button>
            </div>
          </summary>
          <div class="card">
            <table class="grid-table" data-domain="${escapeHtml(domainName)}">
              <thead><tr><th></th>${headerCells}</tr></thead>
              <tbody>${body || emptyRow}</tbody>
            </table>
          </div>
        </details>
      `;
    }

    function renderWindowsMeta(domainName, domain) {
      const windows = Array.isArray(domain.time_windows) ? domain.time_windows : [];
      if (!windows.length) return "";
      const blocks = windows
        .map((w, idx) => {
          const label = Array.isArray(w.time_range) ? w.time_range.join(" to ") : "";
          return `
            <div class="window-block" data-domain="${escapeHtml(domainName)}" data-window="${idx}">
              <div class="muted">${escapeHtml(w.window_id || `w${idx + 1}`)} | ${escapeHtml(label || "no time range")}</div>
              <div style="margin-top:6px;">
                <div class="muted small">window_description</div>
                <div class="card ghost" style="max-height:96px; overflow:auto; padding:8px;">${escapeHtml(w.window_description || "")}</div>
                <button data-action="edit-window-field" data-domain="${escapeHtml(domainName)}" data-window="${idx}" data-field="window_description">Edit description</button>
              </div>
              <div style="margin-top:8px;">
                <div class="muted small">summary</div>
                <div class="card ghost" style="max-height:96px; overflow:auto; padding:8px;">${escapeHtml(w.summary || "")}</div>
                <button data-action="edit-window-field" data-domain="${escapeHtml(domainName)}" data-window="${idx}" data-field="summary">Edit summary</button>
              </div>
            </div>
          `;
        })
        .join("");
      return `
        <details open>
          <summary class="section-header" style="cursor:pointer;"><div style="font-weight:700;">Window descriptions & summaries</div></summary>
          <div class="card">
            <div class="window-grid">
              ${blocks}
            </div>
          </div>
        </details>
      `;
    }

    function renderDomains() {
      if (state.datasetMode && !state.activeDataset) {
        dom.detail.innerHTML = "<span class='muted'>Select a dataset to start annotating.</span>";
        return;
      }
      const names = Object.keys(state.profiles || {}).sort();
      if (!names.length) {
        dom.detail.innerHTML = "<span class='muted'>No domains loaded.</span>";
        return;
      }
      state.grids = {};
      const sections = names
        .map((name) => {
          const domain = state.profiles[name] || {};
          const grid = buildGridForDomain(domain);
          state.grids[name] = grid;
          const domainKey = safeCSS(name);
          const stats = {
            windows: Array.isArray(domain.time_windows) ? domain.time_windows.length : 0,
            attributes: grid.grids.user_attributes_state.length,
            habits: grid.grids.habits_state.length,
            preferences: grid.grids.preferences_state.length,
          };
          const gridsHtml = [
            renderWindowsMeta(name, domain),
            renderStateGrid("user_attributes_state", grid, name),
            renderStateGrid("habits_state", grid, name),
            renderStateGrid("preferences_state", grid, name),
          ].join("");
          const dirtyMark = state.dirtyDomains.has(name) ? `<span class="pill warn">edited</span>` : "";
          return `
            <details open data-domain-section="${domainKey}">
              <summary class="section-header" style="cursor:pointer;">
                <div>
                  <div style="font-weight:700;">${escapeHtml(name)}</div>
                  <div class="muted">life_domain: ${escapeHtml(domain.life_domain || "n/a")} | windows: ${stats.windows} | attributes: ${stats.attributes} | habits: ${stats.habits} | preferences: ${stats.preferences}</div>
                </div>
                <div class="flex">
                  ${dirtyMark}
                  <button data-edit-scope="domain" data-domain="${escapeHtml(name)}">Edit domain JSON</button>
                  <button data-edit-scope="initial" data-domain="${escapeHtml(name)}">Edit initial state</button>
                  <button data-edit-scope="windows" data-domain="${escapeHtml(name)}">Edit windows</button>
                  <button class="primary" data-edit-scope="full" data-domain="${escapeHtml(name)}">Edit entire file</button>
                </div>
              </summary>
              ${gridsHtml}
            </details>
          `;
        })
        .join("");
      dom.detail.innerHTML = sections;
      if (state.focusedDomain) {
        scrollDomainIntoView(state.focusedDomain, "auto");
      }
    }

    function renderDatasets() {
      if (!state.datasetMode) {
        dom.datasetCard.style.display = "none";
        return;
      }
      dom.datasetCard.style.display = "block";
      if (dom.datasetModePill) dom.datasetModePill.style.display = "inline-flex";
      dom.datasetRootHint.textContent = state.datasetRoot ? `Root: ${shortName(state.datasetRoot)}` : "";
      const datasets = Array.isArray(state.datasets) ? state.datasets : [];
      if (!state.loggedIn && !datasets.length) {
        dom.datasetYours.innerHTML = "<div class='muted small'>Login to see your assignments.</div>";
        dom.datasetAvailable.innerHTML = "<div class='muted small'>Login to view unclaimed datasets.</div>";
        dom.datasetLocked.innerHTML = "";
        return;
      }
      const yours = datasets.filter((d) => d.status === "yours");
      const available = datasets.filter((d) => d.status === "available");
      const locked = datasets.filter((d) => d.status === "locked");

      const renderList = (list, emptyText, actionLabel, actionType) => {
        if (!list.length) return `<div class="muted small">${escapeHtml(emptyText)}</div>`;
        return list
          .map((d) => {
            const isActive = d.id === state.activeDataset;
            const badge = isActive ? `<span class="pill small">Active</span>` : "";
            const metaParts = [];
            if (d.assigned_to) metaParts.push(`Assigned to ${escapeHtml(d.assigned_to)}`);
            if (d.claimed_at) metaParts.push(escapeHtml(d.claimed_at));
            const actionBtn = actionLabel
              ? `<button data-dataset-action="${actionType}" data-dataset-id="${escapeHtml(d.id)}">${escapeHtml(actionLabel)}</button>`
              : "";
            return `
              <div class="dataset-item">
                <div class="flex" style="justify-content:space-between; gap:6px;">
                  <div>
                    <div class="title" title="${escapeHtml(d.id)}">${escapeHtml(shortId(d.id) || d.id)}</div>
                    <div class="meta">${metaParts.join(" · ") || "Unassigned"}</div>
                  </div>
                  <div class="flex">
                    ${badge}
                    ${actionBtn}
                  </div>
                </div>
              </div>
            `;
          })
          .join("");
      };

      dom.datasetYours.innerHTML = renderList(yours, "No active assignments yet.", "Resume", "resume");
      dom.datasetAvailable.innerHTML = renderList(available, "No unclaimed datasets available.", "Claim & load", "claim");
      dom.datasetLocked.innerHTML = locked.length
        ? `<div class="muted small" style="margin-bottom:4px;">Taken by others</div>${renderList(locked, "", "", "")}`
        : "";
    }

    function openEditor(scope, domain) {
      modalState = { mode: "json", scope, domain };
      dom.modal.style.display = "flex";
      dom.jsonWrap.style.display = "block";
      dom.cellWrap.style.display = "none";
      dom.editor.value = "";
      if (domain) {
        state.focusedDomain = domain;
        renderDomainList();
      }
      const hintTarget =
        scope === "full" ? "entire dynamic_profiles_conflict_resolved payload" : scope === "domain" ? `domain "${domain}"` : `${scope} of "${domain}"`;
      dom.modalTitle.textContent = `Edit ${hintTarget}`;
      dom.modalHint.textContent = scope === "windows" ? "Expecting an array of windows. Keep window_id/time_range consistent." : "JSON only. Ctrl/Cmd+S will save after you apply.";

      let current;
      if (scope === "full") current = state.profiles;
      else if (scope === "domain") current = state.profiles[domain];
      else if (scope === "initial") current = state.profiles[domain]?.initial_state;
      else if (scope === "windows") current = state.profiles[domain]?.time_windows ?? [];
      if (scope === "initial" && !current) current = {};
      dom.editor.value = pretty(current);
      dom.editor.focus();
    }

    function findCell(stateType, name, colIdx, domain) {
      const grid = state.grids[domain];
      if (!grid) return null;
      const rows = grid.grids[stateType] || [];
      const row = rows.find((r) => r.name === name);
      if (!row) return null;
      return row.cells[colIdx] || null;
    }

    function defaultValueFor(stateType) {
      const val = DEFAULT_VALUES[stateType];
      return typeof val === "object" ? JSON.parse(JSON.stringify(val)) : val;
    }

    function getWindowAnnotationKey(domain, windowIdx, field) {
      return `${domain}::window::${windowIdx}::${field}`;
    }

    function openWindowEditor(domain, windowIdx, field) {
      const domainProfile = state.profiles[domain] || {};
      const win = ensureWindow(domainProfile, windowIdx);
      modalState = { mode: "window", domain, windowIdx, field };
      dom.modal.style.display = "flex";
      dom.jsonWrap.style.display = "none";
      dom.cellWrap.style.display = "block";
      dom.opWrap.style.display = "none";
      dom.changeReasonWrap.style.display = "none";
      state.focusedDomain = domain;
      renderDomainList();

      dom.modalTitle.textContent = `Edit ${field}`;
      dom.modalHint.textContent = "Free text. Reason is recorded for QA.";
      dom.cellTarget.textContent = `Domain: ${domain} | Window: ${win.window_id || `w${windowIdx + 1}`}`;

      dom.cellValue.value = win[field] || "";
      dom.cellChangeReason.value = "";

      const annKey = getWindowAnnotationKey(domain, windowIdx, field);
      const ann = state.annotations[annKey] || {};
      renderReasonButtons(ann.reasonType || REASON_OPTIONS[0]);
      dom.cellConflictDescription.value = ann.conflictDescription || "";
      dom.cellCorrectionDetail.value = ann.reasonNote || "";
      dom.cellChangeReason.value = ann.changeReason || "";
      dom.cellValue.focus();
    }

    function getAnnotationKey(domain, stateType, name, colIdx) {
      return `${domain}::${stateType}::${name}::${colIdx}`;
    }

    function setReasonSelection(selected) {
      Array.from(dom.reasonButtons.querySelectorAll(".reason-btn")).forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.reason === selected);
      });
    }

    function activeReasonSelection() {
      const btn = dom.reasonButtons.querySelector(".reason-btn.active");
      return btn ? btn.dataset.reason : REASON_OPTIONS[0];
    }

    function renderReasonButtons(selected) {
      dom.reasonButtons.innerHTML = REASON_OPTIONS.map((r) => `<button type="button" class="reason-btn${r === selected ? " active" : ""}" data-reason="${r}">${escapeHtml(r)}</button>`).join("");
    }

    function openCellEditor(action, stateType, name, colIdx, domain) {
      const cfg = STATE_CONFIG[stateType];
      modalState = { mode: "cell", action, stateType, name, colIdx, domain };
      dom.modal.style.display = "flex";
      dom.jsonWrap.style.display = "none";
      dom.cellWrap.style.display = "block";

      const header = state.grids[domain]?.headers?.[colIdx];
      const columnLabel = header ? `${header.id}${header.label ? " (" + header.label + ")" : ""}` : `column ${colIdx}`;
      dom.modalTitle.textContent = `${action === "add" ? "Add" : "Edit"} ${cfg.label}`;
      dom.modalHint.textContent = "JSON value supported. Name is fixed per row.";
      dom.cellTarget.textContent = `Domain: ${domain} | Row: ${name} | Column: ${columnLabel}`;
      state.focusedDomain = domain;
      renderDomainList();

      const cell = findCell(stateType, name, colIdx, domain) || {};
      const isInitial = colIdx === 0;
      const opList = OPS[stateType] || [];
      dom.opWrap.style.display = isInitial ? "none" : "block";
      dom.changeReasonWrap.style.display = isInitial ? "none" : "block";
      dom.cellOp.innerHTML = opList.map((op) => `<option value="${op}">${op}</option>`).join("");
      dom.cellOp.value = cell.op || opList[0] || "";

      const annKey = getAnnotationKey(domain, stateType, name, colIdx);
      const ann = state.annotations[annKey] || {};
      renderReasonButtons(ann.reasonType || REASON_OPTIONS[0]);
      dom.cellConflictDescription.value = ann.conflictDescription || "";
      dom.cellCorrectionDetail.value = ann.reasonNote || "";

      const valueForEditor = cell.value !== undefined && cell.value !== null ? cell.value : defaultValueFor(stateType);
      dom.cellValue.value = valuePreview(valueForEditor);
      dom.cellChangeReason.value = isInitial ? "" : cell.reason || ann.changeReason || "";
      dom.cellValue.focus();
    }

    function closeEditor() {
      dom.modal.style.display = "none";
      dom.editor.value = "";
      dom.cellValue.value = "";
      dom.cellOp.value = "";
      dom.cellChangeReason.value = "";
      dom.cellConflictDescription.value = "";
      dom.cellCorrectionDetail.value = "";
      dom.reasonButtons.innerHTML = "";
    }

    function ensureWindow(domain, windowIdx) {
      domain.time_windows = Array.isArray(domain.time_windows) ? domain.time_windows : [];
      while (domain.time_windows.length <= windowIdx) {
        domain.time_windows.push({});
      }
      return domain.time_windows[windowIdx];
    }

    function normalizeInitialBlock(block) {
      if (!block.initial) {
        block.initial = {};
        return;
      }
      if (Array.isArray(block.initial)) {
        const obj = {};
        block.initial.forEach((item) => {
          if (typeof item === "string") {
            obj[item] = null;
          } else if (item && typeof item === "object") {
            Object.entries(item).forEach(([k, v]) => {
              obj[k] = v;
            });
          }
        });
        block.initial = obj;
      } else if (typeof block.initial !== "object") {
        block.initial = {};
      }
    }

    function updateAnnotation(domain, stateType, name, colIdx, op, reasonType, reasonNote, changeReason, conflictDescription) {
      const key = getAnnotationKey(domain, stateType, name, colIdx);
      const prev = state.annotations[key] || {};
      state.annotations[key] = {
        ...prev,
        kind: "cell",
        domain,
        stateType,
        name,
        colIdx,
        op,
        reasonType,
        reasonNote,
        changeReason,
        conflictDescription,
      };
      renderStats();
    }

    function updateWindowAnnotation(domain, windowIdx, field, reasonType, reasonNote, changeReason, conflictDescription) {
      const key = getWindowAnnotationKey(domain, windowIdx, field);
      const prev = state.annotations[key] || {};
      state.annotations[key] = {
        ...prev,
        kind: "window",
        domain,
        windowIdx,
        field,
        reasonType,
        reasonNote,
        changeReason,
        conflictDescription,
      };
      renderStats();
    }

    function deleteCell(stateType, name, colIdx, domain) {
      if (!state.profiles[domain]) return;
      const cfg = STATE_CONFIG[stateType];

      if (!confirm(`Delete entry for "${name}" in column ${colIdx === 0 ? "initial" : colIdx}?`)) return;

      if (colIdx === 0) {
        const initState = (state.profiles[domain].initial_state = state.profiles[domain].initial_state || {});
        const block = (initState[stateType] = initState[stateType] || { initial: {} });
        normalizeInitialBlock(block);
        if (block.initial && typeof block.initial === "object") {
          delete block.initial[name];
        }
      } else {
        const windowIdx = colIdx - 1;
        const win = ensureWindow(state.profiles[domain], windowIdx);
        const delta = (win[cfg.deltaKey] = win[cfg.deltaKey] || { operations: [] });
        delta.operations = Array.isArray(delta.operations) ? delta.operations : [];
        delta.operations = delta.operations.filter((op) => op && op[cfg.nameKey] !== name);
      }

      const annKey = getAnnotationKey(domain, stateType, name, colIdx);
      delete state.annotations[annKey];
      renderStats();
      markDirty(domain);
      renderDomains();
      saveToDisk("autosave");
    }

    function deleteRow(stateType, name, domain) {
      const profile = state.profiles[domain];
      if (!profile) return;
      if (!confirm(`Delete ${STATE_CONFIG[stateType].label} "${name}" across all columns?`)) return;
      // Remove from initial
      const initState = (profile.initial_state = profile.initial_state || {});
      const block = (initState[stateType] = initState[stateType] || { initial: {} });
      normalizeInitialBlock(block);
      if (block.initial && typeof block.initial === "object") {
        delete block.initial[name];
      }
      // Remove from all windows
      const windows = Array.isArray(profile.time_windows) ? profile.time_windows : [];
      windows.forEach((win) => {
        const deltaKey = STATE_CONFIG[stateType].deltaKey;
        const nameKey = STATE_CONFIG[stateType].nameKey;
        const delta = (win[deltaKey] = win[deltaKey] || { operations: [] });
        delta.operations = Array.isArray(delta.operations) ? delta.operations : [];
        delta.operations = delta.operations.filter((op) => op && op[nameKey] !== name);
      });
      // Drop annotations for this row
      Object.keys(state.annotations || {}).forEach((key) => {
        if (key.startsWith(`${domain}::${stateType}::${name}::`)) {
          delete state.annotations[key];
        }
      });
      markDirty(domain);
      renderDomains();
      renderStats();
      saveToDisk("autosave");
    }

    async function saveToDisk(reason = "manual") {
      if (!state.loggedIn) return;
      if (state.datasetMode && !state.activeDataset) {
        alert("Please claim a dataset before saving.");
        return;
      }
      if (state.saving) return;
      state.saving = true;
      state.reviewDurationSeconds = Math.round((Date.now() - state.sessionStart) / 1000);
      setStatus("Saving...", "");
      dom.saveBtn.disabled = true;
      try {
        const res = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            profiles: state.profiles,
            annotations: state.annotations,
            stats: state.stats,
            review_duration_seconds: state.reviewDurationSeconds,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        state.dirty = false;
        state.dirtyDomains.clear();
        state.savePath = payload.saved_to || state.savePath;
        state.lastActivity = Date.now();
        setStatus(`Saved to ${payload.saved_to} - ${new Date().toLocaleTimeString()} (${reason})`, "ok");
      } catch (err) {
        console.error(err);
        setStatus(`Save failed: ${err.message}`, "error");
        alert("Save failed: " + err.message);
      } finally {
        state.saving = false;
        dom.saveBtn.disabled = false;
      }
    }

    function markDirty(domain) {
      state.dirty = true;
      if (domain) state.dirtyDomains.add(domain);
      state.lastActivity = Date.now();
      setStatus("Pending changes...", "");
      renderDomainList();
    }

    function applyCellEdit() {
      const { stateType, name, colIdx, domain } = modalState;
      const cfg = STATE_CONFIG[stateType];

      let parsedValue;
      try {
        parsedValue = JSON.parse(dom.cellValue.value || "null");
      } catch (err) {
        alert("JSON parse error: " + err.message);
        return;
      }

      const reasonType = activeReasonSelection();
      const reasonNote = dom.cellCorrectionDetail.value || "";
      const conflictDescription = dom.cellConflictDescription.value || "";
      const hasOp = colIdx !== 0;
      const changeReason = hasOp ? dom.cellChangeReason.value || "" : "";
      if (reasonType === "Other" && !reasonNote.trim()) {
        alert("Please provide a note for 'Other'.");
        return;
      }

      const profile = (state.profiles[domain] = state.profiles[domain] || {});

      if (colIdx === 0) {
        profile.initial_state = profile.initial_state || {};
        const block = (profile.initial_state[stateType] = profile.initial_state[stateType] || { initial: {} });
        normalizeInitialBlock(block);
        block.initial[name] = parsedValue;
      } else {
        const windowIdx = colIdx - 1;
        const win = ensureWindow(profile, windowIdx);
        win[cfg.deltaKey] = win[cfg.deltaKey] || { operations: [] };
        const ops = (win[cfg.deltaKey].operations = Array.isArray(win[cfg.deltaKey].operations) ? win[cfg.deltaKey].operations : []);
        let target = ops.find((op) => op && op[cfg.nameKey] === name);
        if (!target) {
          target = { [cfg.nameKey]: name };
          ops.push(target);
        }
        target.op = dom.cellOp.value || target.op || OPS[stateType]?.[0] || "";
        target.new_state = parsedValue;
        target.reason = changeReason;
      }

      updateAnnotation(
        domain,
        stateType,
        name,
        colIdx,
        colIdx === 0 ? "" : dom.cellOp.value,
        reasonType,
        reasonNote,
        changeReason,
        conflictDescription
      );
      closeEditor();
      markDirty(domain);
      renderDomains();
      saveToDisk("autosave");
    }

    function applyWindowEdit() {
      const { domain, windowIdx, field } = modalState;
      const value = dom.cellValue.value || "";
      const reasonType = activeReasonSelection();
      const reasonNote = dom.cellCorrectionDetail.value || "";
      const conflictDescription = dom.cellConflictDescription.value || "";
      const annKey = getWindowAnnotationKey(domain, windowIdx, field);
      const prevAnn = state.annotations[annKey] || {};
      const changeReason =
        dom.changeReasonWrap.style.display === "none" ? prevAnn.changeReason || "" : dom.cellChangeReason.value || "";
      if (reasonType === "Other" && !reasonNote.trim()) {
        alert("Please provide a note for 'Other'.");
        return;
      }
      const profile = (state.profiles[domain] = state.profiles[domain] || {});
      const win = ensureWindow(profile, windowIdx);
      win[field] = value;
      updateWindowAnnotation(domain, windowIdx, field, reasonType, reasonNote, changeReason, conflictDescription);
      closeEditor();
      markDirty(domain);
      renderDomains();
      saveToDisk("autosave");
    }

    async function applyEditor() {
      if (modalState.mode === "cell") {
        applyCellEdit();
        return;
      }
      if (modalState.mode === "window") {
        applyWindowEdit();
        return;
      }
      let parsed;
      try {
        parsed = JSON.parse(dom.editor.value);
      } catch (err) {
        alert("JSON parse error: " + err.message);
        return;
      }
      const { scope, domain } = modalState;
      if (scope === "full") {
        state.profiles = parsed;
      } else if (scope === "domain") {
        state.profiles[domain] = parsed;
      } else if (scope === "initial") {
        if (!state.profiles[domain]) state.profiles[domain] = {};
        state.profiles[domain].initial_state = parsed;
      } else if (scope === "windows") {
        if (!Array.isArray(parsed)) {
          alert("Windows must be an array.");
          return;
        }
        if (!state.profiles[domain]) state.profiles[domain] = {};
        state.profiles[domain].time_windows = parsed;
      }
      closeEditor();
      markDirty(scope === "full" ? null : domain);
      renderDomainList();
      renderDomains();
      await saveToDisk("autosave");
    }

    async function fetchData() {
      if (!state.loggedIn) return;
      if (state.datasetMode && !state.activeDataset) {
        alert("Please claim a dataset first.");
        return;
      }
      setStatus("Loading...", "");
      try {
        const res = await fetch("/api/data");
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        state.activeDataset = payload.dataset_id || state.activeDataset;
        applyBundlePayload(payload);
        setStatus("Loaded", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Load failed: " + err.message, "error");
        alert("Failed to load profiles: " + err.message);
      }
    }

    async function refreshDatasets() {
      if (!state.datasetMode) return;
      if (!state.loggedIn) {
        alert("Login first to view datasets.");
        return;
      }
      try {
        const res = await fetch("/api/datasets");
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        hydrateDatasets(payload);
      } catch (err) {
        console.error(err);
        setStatus("Failed to refresh datasets: " + err.message, "error");
      }
    }

    async function selectDataset(datasetId) {
      if (!datasetId) return;
      setStatus(`Claiming ${datasetId}...`, "");
      try {
        const res = await fetch("/api/select_dataset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: datasetId }),
        });
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        state.activeDataset = payload.dataset_id || datasetId;
        hydrateDatasets(payload);
        applyBundlePayload(payload);
        state.sessionStart = Date.now() - (state.reviewDurationSeconds || 0) * 1000;
        startIdleTimer();
        setStatus(`Loaded dataset ${state.activeDataset}`, "ok");
      } catch (err) {
        console.error(err);
        setStatus("Failed to claim dataset", "error");
        alert("Failed to load dataset: " + err.message);
      }
    }

    function addRow(stateType, domain) {
      const name = prompt(`New ${STATE_CONFIG[stateType].label} name?`);
      if (!name) return;
      openCellEditor("add", stateType, name, 0, domain);
    }

    dom.saveBtn.addEventListener("click", () => saveToDisk("manual"));
    dom.reloadBtn.addEventListener("click", fetchData);
    dom.applyModal.addEventListener("click", applyEditor);
    dom.closeModal.addEventListener("click", closeEditor);
    dom.modal.addEventListener("click", (e) => {
      if (e.target === dom.modal) closeEditor();
    });
    dom.reasonButtons.addEventListener("click", (e) => {
      const btn = e.target.closest(".reason-btn");
      if (!btn) return;
      setReasonSelection(btn.dataset.reason);
    });
    dom.detail.addEventListener("click", (e) => {
      const editBtn = e.target.closest("[data-edit-scope]");
      if (editBtn) {
        const scope = editBtn.dataset.editScope;
        const domain = editBtn.dataset.domain;
        openEditor(scope, domain);
        return;
      }
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const action = btn.dataset.action;
      if (action === "add-row") {
        const stateType = btn.dataset.state;
        const domain = btn.dataset.domain;
        addRow(stateType, domain);
        return;
      }
      if (action === "delete-row") {
        const stateType = btn.dataset.state;
        const domain = btn.dataset.domain;
        const name = btn.dataset.name;
        deleteRow(stateType, name, domain);
        return;
      }
      if (action === "edit-window-field") {
        const domain = btn.dataset.domain;
        const windowIdx = Number(btn.dataset.window);
        const field = btn.dataset.field;
        openWindowEditor(domain, windowIdx, field);
        return;
      }
      const cell = btn.closest("td[data-state]");
      if (!cell) return;
      const stateType = cell.dataset.state;
      const name = cell.dataset.name;
      const colIdx = Number(cell.dataset.col);
      const domain = cell.dataset.domain;
      if (action === "edit" || action === "add") {
        openCellEditor(action, stateType, name, colIdx, domain);
      } else if (action === "delete") {
        deleteCell(stateType, name, colIdx, domain);
      }
    });
    window.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        saveToDisk("keyboard");
      }
      if (e.key === "Escape") closeEditor();
    });
    if (dom.refreshDatasets) dom.refreshDatasets.addEventListener("click", refreshDatasets);
    if (dom.datasetCard) {
      dom.datasetCard.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-dataset-action]");
        if (!btn) return;
        if (!state.loggedIn) {
          alert("Please login first.");
          return;
        }
        selectDataset(btn.dataset.datasetId);
      });
    }

    function startIdleTimer() {
      if (idleTimer) clearInterval(idleTimer);
      idleTimer = setInterval(() => {
        const now = Date.now();
        const idleMs = now - state.lastActivity;
        if (state.dirty && idleMs > 5 * 60 * 1000) {
          saveToDisk("auto-idle");
        }
      }, 60000);
    }

    async function loginAndLoad() {
      const annotator = dom.annotatorInput.value.trim();
      if (!annotator) {
        alert("Please enter annotator name.");
        return;
      }
      setStatus("Logging in...", "");
      try {
        const res = await fetch("/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ annotator }),
        });
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        state.annotator = payload.annotator || annotator;
        state.loggedIn = true;
        state.lastActivity = Date.now();
        state.datasetMode = payload.dataset_mode ?? state.datasetMode;
        state.activeDataset = payload.active_dataset || payload.dataset_id || state.activeDataset;
        state.dataVerifyInstructions = payload.data_verify_instructions || state.dataVerifyInstructions;
        state.dataVerifyInstructionsBrief = payload.data_verify_instructions_brief || payload.data_verify_instructions || "";
        state.dataVerifyInstructionsDetailed = payload.data_verify_instructions_detailed || "";
        hydrateDatasets(payload);
        dom.loginOverlay.style.display = "none";
        if (payload.profiles) {
          applyBundlePayload(payload);
        } else {
          resetDataView("Select a dataset to start.");
          renderDatasets();
        }
        startIdleTimer();
        const suffix = state.activeDataset ? ` · ${shortId(state.activeDataset)}` : state.datasetMode ? " · select a dataset" : "";
        setStatus(`Logged in as ${state.annotator}${suffix}`, "ok");
      } catch (err) {
        console.error(err);
        setStatus("Login failed", "error");
        alert("Login failed: " + err.message);
      }
    }

    dom.loginBtn.addEventListener("click", loginAndLoad);
    dom.annotatorInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loginAndLoad();
    });
    updateActionButtons();
    renderDatasets();
    dom.loginOverlay.style.display = "flex";
  </script>
</body>
</html>
"""


class EditorRequestHandler(BaseHTTPRequestHandler):
    server_version = "DynamicProfileEditor/0.1"

    def __init__(self, *args, state: EditorState, **kwargs) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def _json_response(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - naming matches BaseHTTPRequestHandler
        if self.path == "/":
            server_info = {
                "target_path": str(self.state.target_path) if self.state.target_path else "",
                "base_dir": str(self.state.base_dir) if self.state.base_dir else "",
                "dataset_root": str(self.state.dataset_root) if self.state.dataset_root else "",
                "dataset_mode": bool(self.state.assignment_registry),
                "target_filename": self.state.target_filename,
            }
            html = HTML_TEMPLATE.replace("__SERVER_STATE__", json.dumps(server_info))
            return self._html_response(html)

        if self.path.startswith("/api/datasets"):
            desc = (
                self.state.assignment_registry.describe_datasets(self.state.current_user)
                if self.state.assignment_registry
                else {"datasets": []}
            )
            return self._json_response(desc)

        if self.path.startswith("/api/data"):
            try:
                bundle = self.state.load_bundle()
                profiles = bundle.get("profiles", {})
            except ValueError as exc:
                return self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - runtime guard
                return self._json_response(
                    {"error": f"Failed to load profiles: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
                )
            return self._json_response(
                {
                    "profiles": profiles,
                    "source_path": str(self.state.target_path) if self.state.target_path else "",
                    "annotator": self.state.current_user,
                    "user_basic_profile": self.state.user_basic_profile,
                    "world_background": self.state.world_background,
                    "data_verify_instructions": self.state.data_verify_instructions,
                    "data_verify_instructions_brief": self.state.data_verify_instructions_brief,
                    "data_verify_instructions_detailed": self.state.data_verify_instructions_detailed,
                    "annotations": bundle.get("annotations") or {},
                    "stats": bundle.get("stats") or {},
                    "save_path": str(self.state.save_path) if self.state.save_path else "",
                    "review_duration_seconds": bundle.get("review_duration_seconds"),
                    "dataset_id": self.state.active_dataset_id,
                }
            )

        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def _handle_save(self) -> None:
        try:
            self.state.ensure_dataset_active()
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            profiles = payload.get("profiles")
            annotations = payload.get("annotations") or {}
            stats = payload.get("stats") or {}
            review_duration_seconds = payload.get("review_duration_seconds")
            if not isinstance(profiles, dict):
                raise ValueError("`profiles` must be a JSON object")
        except Exception as exc:
            return self._json_response({"error": f"Invalid payload: {exc}"}, HTTPStatus.BAD_REQUEST)

        try:
            result = self.state.save_bundle(profiles, annotations, stats, review_duration_seconds)
        except Exception as exc:  # pragma: no cover - runtime guard
            return self._json_response(
                {"error": f"Failed to save profiles: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

        return self._json_response(result, HTTPStatus.OK)

    def do_POST(self) -> None:  # noqa: N802 - naming matches BaseHTTPRequestHandler
        if self.path == "/api/save":
            return self._handle_save()
        if self.path == "/api/login":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                annotator = (payload.get("annotator") or "").strip()
                self.state.set_user(annotator)
                datasets_desc = (
                    self.state.assignment_registry.describe_datasets(self.state.current_user)
                    if self.state.assignment_registry
                    else {"datasets": []}
                )
                bundle: Dict[str, Any] | None = None
                if self.state.assignment_registry:
                    existing = next((d for d in datasets_desc.get("datasets", []) if d.get("status") == "yours"), None)
                    if existing:
                        self.state.assignment_registry.claim(existing["id"], self.state.current_user)
                        self.state.activate_dataset(existing["id"])
                        bundle = self.state.load_bundle()
                else:
                    bundle = self.state.load_bundle()

                resp: Dict[str, Any] = {
                    "ok": True,
                    "annotator": self.state.current_user,
                    "datasets": datasets_desc,
                    "dataset_mode": bool(self.state.assignment_registry),
                    "active_dataset": self.state.active_dataset_id,
                    "data_verify_instructions": self.state.data_verify_instructions,
                    "data_verify_instructions_brief": self.state.data_verify_instructions_brief,
                    "data_verify_instructions_detailed": self.state.data_verify_instructions_detailed,
                }
                if bundle is not None:
                    resp.update(
                        {
                            "profiles": bundle.get("profiles", {}),
                            "annotations": bundle.get("annotations", {}),
                            "stats": bundle.get("stats", {}),
                            "save_path": str(self.state.save_path) if self.state.save_path else "",
                            "source_path": str(self.state.target_path) if self.state.target_path else "",
                            "review_duration_seconds": bundle.get("review_duration_seconds"),
                            "user_basic_profile": self.state.user_basic_profile,
                            "world_background": self.state.world_background,
                            "data_verify_instructions": self.state.data_verify_instructions,
                            "data_verify_instructions_brief": self.state.data_verify_instructions_brief,
                            "data_verify_instructions_detailed": self.state.data_verify_instructions_detailed,
                            "dataset_id": self.state.active_dataset_id,
                        }
                    )
                return self._json_response(resp)
            except Exception as exc:  # pragma: no cover - runtime guard
                return self._json_response({"error": f"Login failed: {exc}"}, HTTPStatus.BAD_REQUEST)

        if self.path == "/api/select_dataset":
            if not self.state.current_user:
                return self._json_response({"error": "Login required first."}, HTTPStatus.BAD_REQUEST)
            if not self.state.assignment_registry:
                return self._json_response(
                    {"error": "Dataset selection is not enabled on this server."}, HTTPStatus.BAD_REQUEST
                )
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                dataset_id = (payload.get("dataset_id") or "").strip()
                if not dataset_id:
                    raise ValueError("dataset_id is required")
                self.state.assignment_registry.claim(dataset_id, self.state.current_user)
                self.state.activate_dataset(dataset_id)
                bundle = self.state.load_bundle()
                datasets_desc = self.state.assignment_registry.describe_datasets(self.state.current_user)
                return self._json_response(
                    {
                        "ok": True,
                        "dataset_id": dataset_id,
                        "datasets": datasets_desc,
                        "profiles": bundle.get("profiles", {}),
                        "annotations": bundle.get("annotations", {}),
                        "stats": bundle.get("stats", {}),
                        "save_path": str(self.state.save_path) if self.state.save_path else "",
                        "source_path": str(self.state.target_path) if self.state.target_path else "",
                        "review_duration_seconds": bundle.get("review_duration_seconds"),
                        "user_basic_profile": self.state.user_basic_profile,
                        "world_background": self.state.world_background,
                        "data_verify_instructions": self.state.data_verify_instructions,
                        "data_verify_instructions_brief": self.state.data_verify_instructions_brief,
                        "data_verify_instructions_detailed": self.state.data_verify_instructions_detailed,
                    }
                )
            except ValueError as exc:
                return self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - runtime guard
                return self._json_response({"error": f"Failed to load dataset: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - BaseHTTPRequestHandler API
        """Route logs through print to keep them visible in the terminal."""
        msg = fmt % args
        print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive editor for dynamic profile bundles.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_outputs_debug_v3" / "gemini_3_pro_preview",
        help="Directory containing dynamic_profiles_conflict_resolved.json",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="dynamic_profiles_conflict_resolved.json",
        help="Relative or absolute path to the profile JSON to edit.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="If set, enables dataset claiming flow by scanning subfolders for the target file.",
    )
    parser.add_argument(
        "--save-as",
        type=Path,
        default=None,
        help="Where to write the corrected profiles (defaults to <file>_annotated.json).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the editor server.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a one-time <file>.bak before the first save in this session.",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open the editor in your default browser.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve() if args.dataset_root else None

    if dataset_root and not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if dataset_root:
        base_dir = None
        target_path = None
    else:
        base_dir = args.output_dir.resolve()
        target_path = _resolve_target_path(base_dir, args.file)

    state = EditorState(
        base_dir=base_dir,
        target_path=target_path,
        backup=args.backup,
        save_as=args.save_as,
        dataset_root=dataset_root,
        target_filename=args.file,
    )

    handler = partial(EditorRequestHandler, state=state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    if dataset_root:
        print(f"Dataset root: {dataset_root}")
        print(f"Target filename: {args.file}")
        if state.assignment_registry:
            print(f"Assignments recorded at: {state.assignment_registry.registry_path}")
    else:
        print(f"Editing source: {target_path}")
        print(f"Saving annotated copy to: {state.save_path}")
    print(f"Serving on http://{args.host}:{args.port} (Ctrl+C to stop)")
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    main()
