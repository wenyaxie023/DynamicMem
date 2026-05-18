#!/usr/bin/env python3
"""Unified Letta pipeline: checkpoint build + QA + DSP."""

import argparse
from pathlib import Path
from typing import List, Optional

from generation.letta.checkpoint_agent_builder import build_checkpoint_agents
from generation.letta.dynamic_state_prediction import run_generation as run_dsp_generation
from generation.letta.agent_loop import LettaAgentLoop
from generation.letta.letta import run_qa


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_user_dirs(user: str) -> List[str]:
    raw = str(user).strip()
    out: List[str] = []
    if not raw:
        return out
    out.append(raw)
    if raw.isdigit():
        n = int(raw)
        out.append(f"user{n}")
        out.append(f"{n:03d}_user_{n:03d}")
    return list(dict.fromkeys(out))


def _resolve_user_dir(data_root: Path, user: str) -> str:
    for c in _candidate_user_dirs(user):
        if (data_root / c).exists():
            return c
    raise FileNotFoundError(f"Cannot resolve user directory for '{user}' under {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Letta pipeline for checkpoint build + QA + DSP.")
    parser.add_argument(
        "--action",
        choices=["checkpoint", "qa", "dsp", "all"],
        default="all",
        help="Which stages to run.",
    )
    parser.add_argument("--user", type=str, default="user1", help="User id/dir, e.g. user1, 1, 001_user_001.")
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--data-root", type=Path, default=None, help="Default: <repo_root>/data")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Default: <repo_root>/generation/letta/results",
    )
    parser.add_argument("--agents-dir", type=Path, default=None, help="Default: <repo_root>/generation/letta/agents")

    parser.add_argument("--app-logs-filename", type=str, default="app_log_large.json")
    parser.add_argument("--qa-filename", type=str, default="qa.json")
    parser.add_argument("--benchmark-filename", type=str, default="dynamic_state_prediction_benchmark.json")
    parser.add_argument("--dsp-output-filename", type=str, default="dynamic_state_prediction_results.json")
    parser.add_argument("--qa-output-filename", type=str, default="letta_results.json")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--hot-resume-latest",
        action="store_true",
        help="Checkpoint builder resume: prefer latest on-disk .af snapshot for recovery.",
    )
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument(
        "--keep-imported-agents",
        action="store_true",
        help="DSP test mode: do not delete imported checkpoint agents.",
    )
    parser.add_argument(
        "--gc-leased-agents",
        action="store_true",
        help="If set, cleanup leased temporary agent ids before running stages.",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = (args.data_root or (repo_root / "data")).resolve()
    result_root = (args.result_root or (repo_root / "generation" / "letta" / "results")).resolve()
    agents_dir = (args.agents_dir or (repo_root / "generation" / "letta" / "agents")).resolve()

    user_dir_name = _resolve_user_dir(data_root, args.user)
    user_data_dir = data_root / user_dir_name
    user_result_pred_dir = result_root / user_dir_name / "prediction"

    logs_path = user_data_dir / args.app_logs_filename
    qa_path = user_data_dir / args.qa_filename
    benchmark_path = user_data_dir / args.benchmark_filename
    checkpoint_state_path = agents_dir / f"{user_dir_name}_checkpoint_state.json"
    lease_registry_path = agents_dir / f"{user_dir_name}_leased_agent_ids.json"
    qa_output_path = user_result_pred_dir / args.qa_output_filename
    dsp_output_path = user_result_pred_dir / args.dsp_output_filename

    run_checkpoint = args.action in {"checkpoint", "all"}
    run_qa_stage = args.action in {"qa", "all"}
    run_dsp_stage = args.action in {"dsp", "all"}

    if args.gc_leased_agents:
        gc_agent = LettaAgentLoop(create_agent=False, lease_registry_path=lease_registry_path)
        gc_result = gc_agent.cleanup_leased_agents()
        print(
            "Leased-agent GC:",
            f"attempted={gc_result.get('attempted', 0)}",
            f"deleted={gc_result.get('deleted', 0)}",
            f"remaining={gc_result.get('remaining_leases', 0)}",
        )

    if run_checkpoint:
        if not logs_path.exists():
            raise FileNotFoundError(f"Missing app logs: {logs_path}")
        build_checkpoint_agents(
            logs_path=logs_path,
            output_dir=agents_dir,
            checkpoint_state_path=checkpoint_state_path,
            benchmark_path=benchmark_path if benchmark_path.exists() else None,
            resume=args.resume,
            save_every=args.save_every,
            hot_resume_latest=args.hot_resume_latest,
        )
        print("Checkpoint state:", checkpoint_state_path)

    if run_qa_stage:
        if not logs_path.exists():
            raise FileNotFoundError(f"Missing app logs: {logs_path}")
        if not qa_path.exists():
            raise FileNotFoundError(f"Missing QA file: {qa_path}")
        qa_results = run_qa(
            input_user_dir=user_data_dir,
            qa_user_dir=user_data_dir,
            prediction_dir=user_result_pred_dir,
            output_path=qa_output_path,
            app_logs_filename=args.app_logs_filename,
            qa_filename=args.qa_filename,
            resume=args.resume,
            checkpoint_state_path=checkpoint_state_path if checkpoint_state_path.exists() else None,
            agentfile_path=None,
            lease_registry_path=lease_registry_path,
        )
        print("QA output:", qa_output_path)
        print("QA count:", len(qa_results))

    if run_dsp_stage:
        if not logs_path.exists():
            raise FileNotFoundError(f"Missing app logs: {logs_path}")
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")
        dsp_results = run_dsp_generation(
            benchmark_path=benchmark_path,
            app_logs_path=logs_path,
            output_path=dsp_output_path,
            max_visible_logs=args.max_visible_logs,
            retrieval_top_k=args.retrieval_top_k,
            llm_provider=None,
            llm_model=None,
            llm_max_workers=None,
            letta_mode="sdk",
            allow_local_fallback=True,
            checkpoint_state_path=checkpoint_state_path if checkpoint_state_path.exists() else None,
            lease_registry_path=lease_registry_path,
            resume=args.resume,
            max_checkpoints=args.max_checkpoints,
            debug=args.debug,
            debug_dir=args.debug_dir,
            save_prompt_and_raw=args.save_prompt_and_raw,
            keep_imported_agents=args.keep_imported_agents,
        )
        print("DSP output:", dsp_output_path)
        print("DSP checkpoints:", len(dsp_results.get("predictions", [])))


if __name__ == "__main__":
    main()
