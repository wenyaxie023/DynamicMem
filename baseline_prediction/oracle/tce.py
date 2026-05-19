#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from baseline_prediction.oracle.client import LLMClient
from tce_core.orchestrator_protocol import CheckpointHandle, RetrievalOptions, RetrievalResult
from tce_contracts import has_materialized_value, infer_task_contract_version, normalize_task_a_current_value
from tce_core.pipeline import flatten_snapshot, run_pipeline, to_log_text

load_dotenv()


def _flatten_observability(observability: Any) -> Dict[str, Any]:
    if not isinstance(observability, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in observability.keys()):
        return {str(k): v for k, v in observability.items()}

    flat: Dict[str, Any] = {}
    for category, content in observability.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def _extract_ids(raw_ids: Any) -> List[str]:
    if not isinstance(raw_ids, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw_ids:
        if isinstance(item, (str, int, float)):
            sid = str(item).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _current_validated_state_by_key(
    checkpoint: Dict[str, Any],
    *,
    task_contract_version: Any,
) -> Dict[str, Any]:
    raw_state = checkpoint.get("validated_snapshot_state")
    if not isinstance(raw_state, dict):
        raise ValueError(
            "oracle_state requires checkpoint {} to contain validated_snapshot_state.".format(
                checkpoint.get("checkpoint_id") or "<unknown>"
            )
        )

    out: Dict[str, Any] = {}
    for key, value in flatten_snapshot(raw_state).items():
        normalized = normalize_task_a_current_value(
            value,
            task_contract_version=task_contract_version,
        )
        if has_materialized_value(normalized):
            out[str(key)] = normalized
    return out


def _evidence_ids_for_key(checkpoint: Dict[str, Any], state_key: str) -> List[str]:
    obs = _flatten_observability(checkpoint.get("state_observability") or {}).get(state_key)
    if not isinstance(obs, dict):
        return []
    ids = _extract_ids(obs.get("evidence_app_log_ids"))
    if ids:
        return ids
    last_id = obs.get("last_app_log_id")
    if last_id is None or not str(last_id).strip():
        return []
    return [str(last_id)]


def _format_current_state_memory_block(
    *,
    state_key: str,
    state_value: Any,
    evidence_app_log_ids: List[str],
) -> str:
    payload = {
        "state_key": state_key,
        "current_validated_state": state_value,
        "supporting_app_log_ids": evidence_app_log_ids,
    }
    return (
        "[Validated Current User State]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n[/Validated Current User State]"
    )


def run_generation(
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    answer_temperature: Optional[float] = 0.0,
    answer_top_p: Optional[float] = 1.0,
    answer_top_k: Optional[int] = None,
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    oracle_context_mode: str = "evidence_logs",
    baseline_name: str = "oracle",
    task_selection: str = "all",
) -> Dict[str, Any]:
    oracle_context_mode = str(oracle_context_mode or "").strip().lower()
    if oracle_context_mode not in {"evidence_logs", "golden_state"}:
        raise ValueError(
            "Unsupported oracle_context_mode: {}. Use 'evidence_logs' or 'golden_state'.".format(
                oracle_context_mode
            )
        )
    benchmark_contract_version = infer_task_contract_version(
        json.loads(benchmark_path.read_text(encoding="utf-8"))
    )

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=answer_temperature,
        top_p=answer_top_p,
        top_k=answer_top_k,
    )

    all_logs_payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(all_logs_payload, dict):
        all_logs = all_logs_payload.get("app_logs", [])
    elif isinstance(all_logs_payload, list):
        all_logs = all_logs_payload
    else:
        all_logs = []
    all_logs = [x for x in all_logs if isinstance(x, dict)]

    def ask_json(prompt: str) -> Any:
        return client.ask(prompt, response_type="json")

    def ask_structured(prompt: str, text_format: Any) -> Any:
        return client.ask_structured(prompt, text_format=text_format)

    def close() -> None:
        client.close()

    def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
        return CheckpointHandle(
            checkpoint_id=str(cp.get("checkpoint_id") or ""),
            state_kind=(
                "oracle_validated_current_state"
                if oracle_context_mode == "golden_state"
                else "oracle_observability"
            ),
            state_ref=cp,
            metadata={
                "checkpoint_timestamp": str((cp.get("as_of") or {}).get("timestamp", "")),
                "checkpoint_app_log_id": str((cp.get("as_of") or {}).get("app_log_id") or ""),
                "memory_pool_size": len(memory_pool),
                "oracle_context_mode": oracle_context_mode,
            },
        )

    def retrieve_context_for_query(
        checkpoint_handle: CheckpointHandle,
        query_spec,
        retrieval_options: RetrievalOptions,
        memory_pool: List[Dict[str, Any]],
    ) -> RetrievalResult:
        del retrieval_options
        cp = checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {}

        if oracle_context_mode == "golden_state":
            current_state = _current_validated_state_by_key(
                cp,
                task_contract_version=benchmark_contract_version,
            )
            missing = [key for key in query_spec.target_keys if key not in current_state]
            if missing:
                raise ValueError(
                    "oracle_state could not find validated current state for checkpoint {} keys: {}".format(
                        checkpoint_handle.checkpoint_id or "<unknown>",
                        ", ".join(missing),
                    )
                )

            blocks: List[str] = []
            evidence_ids_by_key: Dict[str, List[str]] = {}
            for key in query_spec.target_keys:
                evidence_ids = _evidence_ids_for_key(cp, key)
                evidence_ids_by_key[key] = evidence_ids
                blocks.append(
                    _format_current_state_memory_block(
                        state_key=key,
                        state_value=current_state[key],
                        evidence_app_log_ids=evidence_ids,
                    )
                )

            retrieved_ids: List[str] = []
            for ids in evidence_ids_by_key.values():
                for sid in ids:
                    if sid not in retrieved_ids:
                        retrieved_ids.append(sid)
            return RetrievalResult(
                mode="inline_memory",
                inline_memory_blocks=blocks,
                debug_metadata={
                    "num_retrieved_logs": 0,
                    "retrieved_app_log_ids": retrieved_ids,
                    "evidence_app_log_ids_by_key": evidence_ids_by_key,
                    "retrieval_mode": "oracle_validated_current_state",
                    "oracle_context_mode": oracle_context_mode,
                    "retrieval_query": query_spec.retrieval_query_text,
                },
            )

        obs_flat = _flatten_observability(cp.get("state_observability") or {})

        target_ids: List[str] = []
        for key in query_spec.target_keys:
            obs = obs_flat.get(key)
            if not isinstance(obs, dict):
                continue
            ids = _extract_ids(obs.get("evidence_app_log_ids"))
            if not ids:
                last_id = obs.get("last_app_log_id")
                ids = [str(last_id)] if last_id is not None and str(last_id).strip() else []
            for sid in ids:
                if sid not in target_ids:
                    target_ids.append(sid)

        pool_by_id = {
            str(log.get("app_log_id")): log
            for log in memory_pool
            if log.get("app_log_id") is not None
        }
        all_by_id = {
            str(log.get("app_log_id")): log
            for log in all_logs
            if log.get("app_log_id") is not None
        }

        selected: List[Dict[str, Any]] = []
        for sid in target_ids:
            log = pool_by_id.get(sid)
            if log is None:
                log = all_by_id.get(sid)
            if log is not None:
                selected.append(log)

        return RetrievalResult(
            mode="inline_memory",
            inline_memory_blocks=[to_log_text(log) for log in selected],
            debug_metadata={
                "num_retrieved_logs": len(selected),
                "retrieved_app_log_ids": [x.get("app_log_id") for x in selected],
                "retrieval_mode": "oracle_evidence_from_state_observability",
                "retrieval_query": query_spec.retrieval_query_text,
            },
        )

    return run_pipeline(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
        output_path=output_path,
        max_visible_logs=max_visible_logs,
        ask_json=ask_json,
        ask_structured=ask_structured,
        use_structured_response=client.supports_structured_response(),
        close=close,
        prepare_checkpoint_state=prepare_checkpoint_state,
        retrieve_context_for_query=retrieve_context_for_query,
        baseline_name=baseline_name,
        memory_prompt_mode="inline_memory",
        resume=resume,
        max_checkpoints=max_checkpoints,
        debug=debug,
        debug_dir=debug_dir,
        save_prompt_and_raw=save_prompt_and_raw,
        enable_change_reasoning=enable_change_reasoning,
        enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=rq3_apply_save_prompt_and_raw,
        rq3_apply_retrieval_top_k=rq3_apply_retrieval_top_k,
        checkpoint_workers=checkpoint_workers,
        within_checkpoint_workers=within_checkpoint_workers,
        save_every_generation_keys=save_every_generation_keys,
        retrieval_options_backend={},
        task_selection=task_selection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle baseline generation for TCE.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to the pack-first TCE benchmark JSON (for example tce_benchmark_vnext_*task_packs*.json).",
    )
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
        help="Output path for TCE predictions JSON (for example tce_results_v14_taskabc.json).",
    )
    parser.add_argument(
        "--max-visible-logs",
        type=int,
        default=None,
        help="Optional tail truncation for debugging. Default uses full history until checkpoint.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="openai|azure|aimlapi|gemini|vllm")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--llm-max-workers", type=int, default=1, help="Max workers for LLM client")
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
    args = parser.parse_args()

    result = run_generation(
        benchmark_path=args.benchmark,
        app_logs_path=args.app_logs_path,
        output_path=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
    )

    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
