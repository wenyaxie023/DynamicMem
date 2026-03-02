#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from dynamic_state_prediction_core.pipeline import run_pipeline
from generation.letta.agent_loop import LettaAgentLoop

load_dotenv()


def _build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    as_of = checkpoint.get("as_of", {})
    ts = as_of.get("timestamp", "")
    keys_hint = ", ".join(target_keys[:20])
    return (
        f"At checkpoint time {ts}, infer current values for state keys. "
        f"Target keys: {keys_hint}"
    )


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    retrieval_top_k: int,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_max_workers: Optional[int],
    letta_mode: str,
    allow_local_fallback: bool,
    checkpoint_state_path: Optional[Path],
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    lease_registry_path: Optional[Path] = None,
    keep_imported_agents: bool = False,
) -> Dict[str, Any]:
    # Keep adapter compatibility args explicitly accepted even if not used by SDK mode.
    _ = (llm_provider, llm_model, llm_max_workers, letta_mode, allow_local_fallback)

    agent = LettaAgentLoop(
        user_namespace=f"dsp::{app_logs_path.parent.name}",
        create_agent=False,
        lease_registry_path=lease_registry_path,
    )

    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(all_logs_payload, dict):
        all_logs = all_logs_payload.get("app_logs", [])
    elif isinstance(all_logs_payload, list):
        all_logs = all_logs_payload
    else:
        all_logs = []
    all_logs = [x for x in all_logs if isinstance(x, dict)]

    ingested_until = -1
    latest_request: Dict[str, Any] = {
        "checkpoint_id": "",
        "checkpoint_timestamp": "",
        "target_keys": [],
        "available_app_log_ids": [],
    }
    if checkpoint_state_path is not None and not isinstance(checkpoint_state_path, Path):
        checkpoint_state_path = Path(str(checkpoint_state_path))

    checkpoint_agentfiles: Dict[str, Path] = {}
    if checkpoint_state_path is not None and checkpoint_state_path.exists():
        state_raw = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
        if isinstance(state_raw, dict):
            raw_map = state_raw.get("checkpoint_agentfiles")
            if isinstance(raw_map, dict):
                for cp_id, af_path in raw_map.items():
                    if cp_id is None or af_path is None:
                        continue
                    p = Path(str(af_path))
                    if p.exists():
                        checkpoint_agentfiles[str(cp_id)] = p

    def ask_json(prompt: str) -> Any:
        # We intentionally ignore the pipeline-generated prompt and ask the Letta agent
        # directly using its already-ingested memory state.
        target_keys = latest_request.get("target_keys") or []
        available_ids = latest_request.get("available_app_log_ids") or []
        checkpoint_ts = latest_request.get("checkpoint_timestamp", "")
        fill_template = {
            "snapshot_state": {k: "<value>" for k in target_keys},
            "evidence": {k: ["<app_log_id>"] for k in target_keys},
        }
        agent_prompt = (
            "Using ONLY your memory built from previously ingested app logs, "
            f"predict values at checkpoint time {checkpoint_ts}.\n"
            f"Target keys: {json.dumps(target_keys, ensure_ascii=False)}\n"
            f"Valid evidence app_log_ids (historical): {json.dumps(available_ids, ensure_ascii=False)}\n"
            "Return JSON only with this exact top-level shape:\n"
            f"{json.dumps(fill_template, ensure_ascii=False)}\n"
            "Rules:\n"
            "1. Include every target key exactly once in snapshot_state and evidence.\n"
            "2. Evidence must be list of app_log_id strings.\n"
            "3. If unsure, use null value and [] evidence.\n"
        )
        def _parse_if_json_string(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            s = value.strip()
            if not s:
                return value
            if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
                return value
            try:
                return json.loads(s)
            except Exception:
                return value

        checkpoint_id = str(latest_request.get("checkpoint_id", "")).strip()
        checkpoint_af = checkpoint_agentfiles.get(checkpoint_id)
        if checkpoint_af is not None:
            temp_agent_id = agent.load_agent_file(checkpoint_af, activate=False)
            if keep_imported_agents:
                print(f"[LETTA-DSP] keep imported agent for checkpoint {checkpoint_id}: {temp_agent_id}")
                raw = agent.ask_with_agent(temp_agent_id, agent_prompt, expect_json=True)
                return _parse_if_json_string(raw)
            try:
                raw = agent.ask_with_agent(temp_agent_id, agent_prompt, expect_json=True)
                return _parse_if_json_string(raw)
            finally:
                agent.delete_agent(temp_agent_id, ignore_missing=True)

        raw = agent.ask_json(agent_prompt)
        return _parse_if_json_string(raw)

    def ask_structured(prompt: str, text_format: Any) -> Any:
        raise RuntimeError("Structured call is disabled for Letta agent-loop mode.")

    def close() -> None:
        agent.close()

    def retrieve_context(
        cp: Dict[str, Any],
        memory_pool: List[Dict[str, Any]],
        target_keys: List[str],
    ) -> Dict[str, Any]:
        nonlocal ingested_until
        query = _build_retrieval_query(cp, target_keys)
        checkpoint_id = str(cp.get("checkpoint_id", "")).strip()
        cp_idx_raw = (cp.get("as_of") or {}).get("log_index")
        cp_idx = cp_idx_raw if isinstance(cp_idx_raw, int) else len(memory_pool) - 1
        cp_idx = max(-1, min(cp_idx, len(all_logs) - 1))
        use_snapshot_agent = checkpoint_id in checkpoint_agentfiles
        if not use_snapshot_agent and cp_idx > ingested_until:
            for i in range(ingested_until + 1, cp_idx + 1):
                agent.ingest_log(all_logs[i])
            ingested_until = cp_idx

        allowed_ids = [
            str(log.get("app_log_id")).strip()
            for log in memory_pool
            if log.get("app_log_id") is not None
        ]
        latest_request["checkpoint_id"] = str(cp.get("checkpoint_id", ""))
        latest_request["checkpoint_timestamp"] = str((cp.get("as_of") or {}).get("timestamp", ""))
        latest_request["target_keys"] = list(target_keys)
        latest_request["available_app_log_ids"] = allowed_ids

        # Agent-loop mode does not require explicit retrieved logs in prompt context.
        context_logs: List[Dict[str, Any]] = []
        return {
            "context_logs": context_logs,
            "context_note": (
                "Letta agent-loop mode: logs were ingested as sequential messages; "
                "prediction is generated from agent memory state."
            ),
            "retrieval_query": query,
            "metadata": {
                "retrieval_mode": (
                    "letta_checkpoint_agentfile"
                    if use_snapshot_agent
                    else "letta_agent_loop_sdk"
                ),
                "retrieval_top_k": retrieval_top_k,
                "num_retrieved_logs": 0,
                "retrieved_app_log_ids": [],
                "ingested_until_log_index": ingested_until,
                "num_ingested_logs": ingested_until + 1,
            },
        }

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=False,
        close=close,
        retrieve_context=retrieve_context,
        baseline_name="letta",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Letta baseline generation for dynamic state prediction.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to dynamic_state_prediction_benchmark.json")
    parser.add_argument(
        "--app-logs-path",
        type=Path,
        required=True,
        help="Path to raw app logs (recommended: app_log_large.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for dynamic_state_prediction_results.json",
    )
    parser.add_argument(
        "--max-visible-logs",
        type=int,
        default=None,
        help="Optional tail truncation for debugging. Default uses full history until checkpoint.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
        help="Top-k logs retrieved by Letta from the checkpoint-visible memory pool.",
    )
    parser.add_argument(
        "--checkpoint-state-path",
        type=Path,
        default=None,
        help="Optional checkpoint_state.json generated by checkpoint_agent_builder.py.",
    )
    parser.add_argument(
        "--lease-registry-path",
        type=Path,
        default=None,
        help="Optional JSON path to track temporary imported agent ids.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug artifacts (save prompt/raw outputs per checkpoint).")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug directory path.")
    parser.add_argument(
        "--save-prompt-and-raw",
        action="store_true",
        help="Save prompt and raw model output into prediction metadata.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Only run the first N checkpoints (for quick debugging).",
    )
    parser.add_argument(
        "--keep-imported-agents",
        action="store_true",
        help="Testing only: do not delete imported checkpoint agents after each checkpoint.",
    )
    args = parser.parse_args()

    result = run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        retrieval_top_k=args.retrieval_top_k,
        llm_provider=None,
        llm_model=None,
        llm_max_workers=None,
        letta_mode="sdk",
        allow_local_fallback=True,
        checkpoint_state_path=args.checkpoint_state_path,
        lease_registry_path=args.lease_registry_path,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        keep_imported_agents=args.keep_imported_agents,
    )

    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
