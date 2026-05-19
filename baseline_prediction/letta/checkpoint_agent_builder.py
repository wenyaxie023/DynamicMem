#!/usr/bin/env python3
"""Build and resume Letta agent snapshots (.af) at benchmark checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from baseline_prediction.letta.agent_loop import LettaAgentLoop


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_logs(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    elif isinstance(payload, list):
        logs = payload
    else:
        logs = []
    return [x for x in logs if isinstance(x, dict)]


def _load_checkpoint_targets(benchmark_path: Optional[Path]) -> List[Tuple[int, str]]:
    if benchmark_path is None or not benchmark_path.exists():
        return []
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    out: List[Tuple[int, str]] = []
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        log_index = (cp.get("as_of") or {}).get("log_index")
        if checkpoint_id and isinstance(log_index, int) and log_index >= 0:
            out.append((log_index, checkpoint_id))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_state_shape(state: Dict[str, Any], logs_path: Path, total_logs: int) -> Dict[str, Any]:
    ckpts = state.get("checkpoint_agentfiles")
    if not isinstance(ckpts, dict):
        ckpts = {}
    state["checkpoint_agentfiles"] = ckpts
    state["logs_path"] = str(logs_path)
    state["total_logs"] = int(total_logs)
    state["last_ingested_log_index"] = int(state.get("last_ingested_log_index", -1))
    if "final_agentfile" not in state:
        state["final_agentfile"] = None
    return state


def _index_from_agentfile_name(
    file_name: str,
    checkpoint_id_to_idx: Dict[str, int],
    total_logs: int,
) -> Optional[int]:
    stem = Path(file_name).stem
    if stem.startswith("final_"):
        raw = stem[len("final_") :]
        if raw.isdigit():
            return max(-1, min(int(raw) - 1, total_logs - 1))
    if stem.startswith("log_"):
        raw = stem[len("log_") :]
        if raw.isdigit():
            return max(-1, min(int(raw) - 1, total_logs - 1))
    if stem in checkpoint_id_to_idx:
        return max(-1, min(checkpoint_id_to_idx[stem], total_logs - 1))
    return None


def _resolve_latest_agentfile(
    *,
    output_dir: Path,
    state: Dict[str, Any],
    checkpoint_id_to_idx: Dict[str, int],
    total_logs: int,
) -> Optional[Tuple[Path, int]]:
    candidates: List[Path] = []

    final_af = state.get("final_agentfile")
    if isinstance(final_af, str) and final_af.strip():
        candidates.append(Path(final_af))

    ckpt_map = state.get("checkpoint_agentfiles")
    if isinstance(ckpt_map, dict):
        for v in ckpt_map.values():
            if isinstance(v, str) and v.strip():
                candidates.append(Path(v))

    if output_dir.exists():
        candidates.extend(sorted(output_dir.glob("*.af")))

    best_path: Optional[Path] = None
    best_idx = -1
    best_mtime = -1.0
    seen: Set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            continue
        idx = _index_from_agentfile_name(path.name, checkpoint_id_to_idx, total_logs)
        if idx is None:
            continue
        mtime = path.stat().st_mtime
        if (idx > best_idx) or (idx == best_idx and mtime > best_mtime):
            best_path = path
            best_idx = idx
            best_mtime = mtime

    if best_path is None:
        return None
    return best_path, best_idx


def build_checkpoint_agents(
    *,
    logs_path: Path,
    output_dir: Path,
    checkpoint_state_path: Path,
    benchmark_path: Optional[Path],
    resume: bool,
    save_every: Optional[int],
    hot_resume_latest: bool = False,
) -> Dict[str, Any]:
    logs = _load_logs(logs_path)
    if not logs:
        raise ValueError(f"No valid logs found: {logs_path}")

    checkpoint_targets = _load_checkpoint_targets(benchmark_path)
    checkpoint_idx_to_ids: Dict[int, List[str]] = {}
    checkpoint_id_to_idx: Dict[str, int] = {}
    for idx, checkpoint_id in checkpoint_targets:
        checkpoint_idx_to_ids.setdefault(idx, []).append(checkpoint_id)
        checkpoint_id_to_idx[checkpoint_id] = idx

    base_state = _read_state(checkpoint_state_path) if resume else {}
    state = _ensure_state_shape(base_state, logs_path=logs_path, total_logs=len(logs))

    agent = LettaAgentLoop()
    start_index = 0
    imported_from_state = False

    if resume:
        last_idx = int(state.get("last_ingested_log_index", -1))
        if hot_resume_latest:
            latest = _resolve_latest_agentfile(
                output_dir=output_dir,
                state=state,
                checkpoint_id_to_idx=checkpoint_id_to_idx,
                total_logs=len(logs),
            )
            if latest is not None:
                latest_path, latest_idx = latest
                try:
                    agent.load_agent_file(latest_path, activate=True)
                    imported_from_state = True
                    start_index = min(latest_idx + 1, len(logs))
                    state["last_ingested_log_index"] = max(int(state.get("last_ingested_log_index", -1)), latest_idx)
                except Exception:
                    start_index = 0

        if not imported_from_state:
            final_af = state.get("final_agentfile")
            if isinstance(final_af, str) and final_af:
                final_path = Path(final_af)
                if final_path.exists():
                    try:
                        agent.load_agent_file(final_path, activate=True)
                        imported_from_state = True
                        start_index = min(last_idx + 1, len(logs))
                    except Exception:
                        start_index = 0
            elif last_idx >= 0:
                start_index = min(last_idx + 1, len(logs))
    else:
        state["last_ingested_log_index"] = -1
        state["checkpoint_agentfiles"] = {}
        state["final_agentfile"] = None

    output_dir.mkdir(parents=True, exist_ok=True)
    already_saved: Set[str] = {
        str(k) for k, v in state.get("checkpoint_agentfiles", {}).items() if isinstance(v, str) and v
    }

    if start_index > 0 and not imported_from_state:
        for i in tqdm(range(0, start_index), desc="LETTA replay", unit="log"):
            agent.ingest_log(logs[i])

    for i in tqdm(range(start_index, len(logs)), desc="LETTA ingest", unit="log"):
        agent.ingest_log(logs[i])
        state["last_ingested_log_index"] = i

        for checkpoint_id in checkpoint_idx_to_ids.get(i, []):
            if checkpoint_id in already_saved:
                continue
            af_path = output_dir / f"{checkpoint_id}.af"
            agent.save_agent_file(af_path)
            state["checkpoint_agentfiles"][checkpoint_id] = str(af_path)
            already_saved.add(checkpoint_id)
            _write_state(checkpoint_state_path, state)

        if save_every and save_every > 0 and (i + 1) % save_every == 0:
            periodic_name = f"log_{i + 1:05d}"
            if periodic_name not in already_saved:
                af_path = output_dir / f"{periodic_name}.af"
                agent.save_agent_file(af_path)
                state["checkpoint_agentfiles"][periodic_name] = str(af_path)
                already_saved.add(periodic_name)
                _write_state(checkpoint_state_path, state)

    final_path = output_dir / f"final_{len(logs)}.af"
    agent.save_agent_file(final_path)
    state["final_agentfile"] = str(final_path)
    state["total_logs"] = len(logs)
    _write_state(checkpoint_state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Letta .af snapshots by checkpoint and support resume.")
    parser.add_argument(
        "--logs-path",
        type=Path,
        default=_repo_root() / "data" / "user1" / "app_log_large.json",
        help="Path to app logs JSON.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="Optional dynamic_state_prediction_benchmark.json for checkpoint_id -> log_index mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "generation" / "letta" / "agents",
        help="Directory to write .af files.",
    )
    parser.add_argument(
        "--checkpoint-state-path",
        type=Path,
        default=_repo_root() / "generation" / "letta" / "agents" / "checkpoint_state.json",
        help="JSON state file storing ingestion breakpoint and checkpoint->.af mapping.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--hot-resume-latest",
        action="store_true",
        help="When resuming, prefer latest on-disk .af snapshot (checkpoint/log/final) for recovery.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Optional periodic snapshot interval by number of ingested logs.",
    )
    args = parser.parse_args()

    state = build_checkpoint_agents(
        logs_path=args.logs_path,
        output_dir=args.output_dir,
        checkpoint_state_path=args.checkpoint_state_path,
        benchmark_path=args.benchmark_path,
        resume=args.resume,
        save_every=args.save_every,
        hot_resume_latest=args.hot_resume_latest,
    )
    print("Saved checkpoint state:", args.checkpoint_state_path)
    print("Total logs:", state.get("total_logs", 0))
    print("Last ingested log index:", state.get("last_ingested_log_index", -1))
    print("Final agentfile:", state.get("final_agentfile"))


if __name__ == "__main__":
    main()
