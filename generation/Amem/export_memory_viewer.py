#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryNoteStub:
    content: str
    id: str
    keywords: List[str]
    links: List[str]
    retrieval_count: int
    timestamp: str
    last_accessed: str
    context: str
    evolution_history: List[Any]
    category: str
    tags: List[str]


class _MemoryNoteUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "generation.Amem.agentic_memory.memory_system" and name == "MemoryNote":
            return MemoryNoteStub
        return super().find_class(module, name)


def _load_state(path: Path) -> Dict[str, MemoryNoteStub]:
    with path.open("rb") as handle:
        state = _MemoryNoteUnpickler(handle).load()
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected state payload type: {type(state)!r}")
    return state


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_usage_cost(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _maybe_json(value: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except Exception:
        lines = [line.strip() for line in str(value).splitlines() if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if not first_line.startswith("User:"):
            return None
        try:
            parsed = json.loads(first_line.split("User:", 1)[1].strip())
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def _short_text(value: str, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _serialize_notes(notes: Iterable[MemoryNoteStub]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ordinal, note in enumerate(notes, start=1):
        content_json = _maybe_json(getattr(note, "content", ""))
        rows.append(
            {
                "ordinal": ordinal,
                "id": getattr(note, "id", ""),
                "timestamp": getattr(note, "timestamp", ""),
                "last_accessed": getattr(note, "last_accessed", ""),
                "context": getattr(note, "context", ""),
                "category": getattr(note, "category", ""),
                "tags": list(getattr(note, "tags", []) or []),
                "keywords": list(getattr(note, "keywords", []) or []),
                "links": list(getattr(note, "links", []) or []),
                "retrieval_count": int(getattr(note, "retrieval_count", 0) or 0),
                "evolution_history": list(getattr(note, "evolution_history", []) or []),
                "content_text": getattr(note, "content", ""),
                "content_preview": _short_text(getattr(note, "content", "")),
                "content_json": content_json,
                "app_log_id": (content_json or {}).get("app_log_id", ""),
                "app_name": (content_json or {}).get("app_name", ""),
                "api_name": (content_json or {}).get("api_name", ""),
            }
        )
    return rows


def _resolve_usage_sidecar_path(state_path: Path, run_name: str, user_id: str) -> Optional[Path]:
    prediction_dir = Path("generation/Amem/results") / user_id / "prediction" / run_name
    live_path = prediction_dir / "usage_cost_live.json"
    final_path = prediction_dir / "usage_cost.json"
    if live_path.exists():
        return live_path
    if final_path.exists():
        return final_path
    checkpoint_live_path = state_path.parent / f"{state_path.stem}_usage_cost_live.json"
    checkpoint_final_path = state_path.parent / f"{state_path.stem}_usage_cost.json"
    if checkpoint_live_path.exists():
        return checkpoint_live_path
    if checkpoint_final_path.exists():
        return checkpoint_final_path
    legacy_checkpoint_live_path = state_path.parent / "usage_cost_live.json"
    legacy_checkpoint_final_path = state_path.parent / "usage_cost.json"
    if legacy_checkpoint_live_path.exists():
        return legacy_checkpoint_live_path
    if legacy_checkpoint_final_path.exists():
        return legacy_checkpoint_final_path
    return None


def _parse_run_meta(state_path: Path) -> Dict[str, str]:
    stem = state_path.stem
    prefix = "membench_amem_"
    suffix = ".pkl"
    if not stem.startswith(prefix):
        return {"run_name": state_path.parent.name, "user_id": "", "size": ""}
    raw = stem[len(prefix):]
    user_id = ""
    size = ""
    parts = raw.split("_")
    if len(parts) >= 4:
        user_id = "_".join(parts[:3])
        size = "_".join(parts[3:])
    return {"run_name": state_path.parent.name, "user_id": user_id, "size": size}


def _build_payload(state_path: Path, checkpoint_path: Path) -> Dict[str, Any]:
    notes_by_id = _load_state(state_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    notes = _serialize_notes(notes_by_id.values())
    run_meta = _parse_run_meta(state_path)
    usage_sidecar_path = _resolve_usage_sidecar_path(state_path, run_meta["run_name"], run_meta["user_id"])
    usage_cost = _load_usage_cost(usage_sidecar_path) if usage_sidecar_path else {}
    summary = {
        "notes_with_links": sum(1 for row in notes if row["links"]),
        "notes_with_tags": sum(1 for row in notes if row["tags"]),
        "notes_with_keywords": sum(1 for row in notes if row["keywords"]),
        "notes_with_nondefault_context": sum(1 for row in notes if row["context"] not in {"", "General"}),
        "notes_with_evolution_history": sum(1 for row in notes if row["evolution_history"]),
    }
    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "state_pkl_path": str(state_path),
            "checkpoint_json_path": str(checkpoint_path),
            "run_name": run_meta["run_name"],
            "user_id": run_meta["user_id"],
            "size": run_meta["size"],
            "note_count": len(notes),
            "last_event_idx": checkpoint.get("last_event_idx"),
            "events_processed": checkpoint.get("events_processed"),
            "saved_at": checkpoint.get("saved_at"),
            "usage_cost_path": str(usage_sidecar_path) if usage_sidecar_path else "",
        },
        "summary": summary,
        "usage_cost": usage_cost,
        "notes": notes,
    }


def _write_payload(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, output_path)


def _run_once(state_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    _write_payload(output_path, _build_payload(state_path, checkpoint_path))


def _discover_run_state_files(checkpoints_root: Path) -> List[Dict[str, Path]]:
    rows: List[Dict[str, Path]] = []
    if not checkpoints_root.exists():
        return rows
    for run_dir in sorted(p for p in checkpoints_root.iterdir() if p.is_dir()):
        state_candidates = sorted(run_dir.glob("membench_amem_*_*.pkl"))
        checkpoint_candidates = sorted(run_dir.glob("membench_amem_*_*.json"))
        if len(state_candidates) != 1 or len(checkpoint_candidates) != 1:
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "state_path": state_candidates[0],
                "checkpoint_path": checkpoint_candidates[0],
            }
        )
    return rows


def _export_runs_index(checkpoints_root: Path, runs_json_dir: Path, index_output: Path) -> None:
    runs_json_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for row in _discover_run_state_files(checkpoints_root):
        state_path = row["state_path"]
        checkpoint_path = row["checkpoint_path"]
        payload = _build_payload(state_path, checkpoint_path)
        run_name = str(payload.get("meta", {}).get("run_name") or row["run_dir"].name)
        run_output = runs_json_dir / f"{run_name}.json"
        _write_payload(run_output, payload)
        meta = payload.get("meta", {})
        entries.append(
            {
                "run_name": run_name,
                "data_path": f"./runs/{run_output.name}",
                "user_id": meta.get("user_id", ""),
                "size": meta.get("size", ""),
                "note_count": meta.get("note_count"),
                "events_processed": meta.get("events_processed"),
                "saved_at": meta.get("saved_at"),
                "usage_cost_path": meta.get("usage_cost_path", ""),
            }
        )
    entries.sort(
        key=lambda row: (
            int(row.get("events_processed") or 0),
            str(row.get("saved_at") or ""),
            str(row.get("run_name") or ""),
        ),
        reverse=True,
    )
    _write_payload(
        index_output,
        {
            "generated_at": datetime.now().isoformat(),
            "runs": entries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export live Amem memory state to viewer JSON.")
    parser.add_argument("--state-pkl", type=Path, required=True, help="Path to Amem state.pkl checkpoint file.")
    parser.add_argument("--checkpoint-json", type=Path, required=True, help="Path to Amem checkpoint progress json.")
    parser.add_argument("--output", type=Path, required=True, help="Viewer JSON output path.")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh the output file.")
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Watch interval.")
    parser.add_argument("--checkpoints-root", type=Path, default=None, help="Optional checkpoints root to export all run snapshots.")
    parser.add_argument("--runs-json-dir", type=Path, default=None, help="Optional output directory for per-run viewer JSON files.")
    parser.add_argument("--runs-index-output", type=Path, default=None, help="Optional viewer run index JSON output path.")
    args = parser.parse_args()

    if not args.watch:
        _run_once(args.state_pkl, args.checkpoint_json, args.output)
        if args.checkpoints_root and args.runs_json_dir and args.runs_index_output:
            _export_runs_index(args.checkpoints_root, args.runs_json_dir, args.runs_index_output)
        return

    while True:
        try:
            _run_once(args.state_pkl, args.checkpoint_json, args.output)
            if args.checkpoints_root and args.runs_json_dir and args.runs_index_output:
                _export_runs_index(args.checkpoints_root, args.runs_json_dir, args.runs_index_output)
        except FileNotFoundError:
            pass
        time.sleep(max(float(args.interval_seconds), 0.5))


if __name__ == "__main__":
    main()
