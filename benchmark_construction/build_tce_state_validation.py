#!/usr/bin/env python3
"""Standalone state-validation stage for TCE benchmark preprocessing."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from baseline_prediction.common.llm_client import LLMClient
from tce_core.state_validation import build_state_validation, write_state_validation_reports


def _resolve_app_logs_path(*, benchmark_path: Path, app_logs_path: Optional[Path]) -> Optional[Path]:
    if app_logs_path is not None:
        return app_logs_path
    candidate_large = benchmark_path.parent / "app_log_large.json"
    candidate_final = benchmark_path.parent / "app_logs_final.json"
    if candidate_large.exists():
        return candidate_large
    if candidate_final.exists():
        return candidate_final
    return None


def _load_app_logs_by_id(app_logs_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if app_logs_path is None:
        return {}
    if not app_logs_path.exists():
        raise FileNotFoundError(f"app logs file not found: {app_logs_path}")
    payload = json.loads(app_logs_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    elif isinstance(payload, list):
        logs = payload
    else:
        logs = []
    out: Dict[str, Dict[str, Any]] = {}
    for log in logs:
        if not isinstance(log, dict):
            continue
        app_log_id = str(log.get("app_log_id") or "").strip()
        if app_log_id and app_log_id not in out:
            out[app_log_id] = log
    return out


def _checkpoint_ids(payload: Dict[str, Any], max_checkpoints: Optional[int]) -> List[str]:
    checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    if not isinstance(checkpoints, list):
        return []
    if max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, int(max_checkpoints))]
    out: List[str] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        out.append(str(checkpoint.get("checkpoint_id", "")))
    return out


def _assert_resume_compatible(
    *,
    source_benchmark: Dict[str, Any],
    resume_payload: Dict[str, Any],
    max_checkpoints: Optional[int],
) -> None:
    source_ids = _checkpoint_ids(source_benchmark, max_checkpoints)
    resume_ids = _checkpoint_ids(resume_payload, max_checkpoints)
    if source_ids != resume_ids:
        raise ValueError(
            "resume artifact is incompatible with benchmark:"
            f" source_checkpoint_ids={source_ids},"
            f" resume_checkpoint_ids={resume_ids}"
        )


def _persist_state_validation_payload(
    *,
    payload: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    counts_path, eyeball_path, totals = write_state_validation_reports(
        payload=payload,
        output_path=output_path,
    )
    return {
        "counts_path": counts_path,
        "eyeball_path": eyeball_path,
        "totals": totals,
    }


def build_validated_benchmark(
    *,
    benchmark_path: Path,
    output_path: Path,
    validator_provider: str,
    validator_model: str,
    app_logs_path: Optional[Path],
    max_checkpoints: Optional[int],
    l2_evidence_top_k: int,
    show_progress: bool,
    save_every_states: int = 5,
    resume: bool = False,
) -> Dict[str, Any]:
    source_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark = source_benchmark
    if resume and output_path.exists():
        resume_payload = json.loads(output_path.read_text(encoding="utf-8"))
        _assert_resume_compatible(
            source_benchmark=source_benchmark,
            resume_payload=resume_payload,
            max_checkpoints=max_checkpoints,
        )
        benchmark = resume_payload
    resolved_app_logs_path = _resolve_app_logs_path(
        benchmark_path=benchmark_path,
        app_logs_path=app_logs_path,
    )
    app_logs_by_id = _load_app_logs_by_id(resolved_app_logs_path)
    save_counter = 0

    def _save_partial(current_payload: Dict[str, Any]) -> None:
        nonlocal save_counter
        save_counter += 1
        persisted = _persist_state_validation_payload(
            payload=current_payload,
            output_path=output_path,
        )
        if show_progress:
            totals = persisted["totals"]
            print(
                "[Stage1][save]"
                f" save_index={save_counter},"
                f" output={output_path},"
                f" pre={totals['pre_validate_count']},"
                f" after_l1={totals['after_l1_count']},"
                f" after_l2={totals['after_l2_count']},"
                f" after_l1_l2={totals['after_l1_l2_count']}"
            )

    validator_client = LLMClient(provider=validator_provider, model_name=validator_model, max_workers=1)
    try:
        payload = build_state_validation(
            benchmark=benchmark,
            validator_client=validator_client,
            app_logs_by_id=app_logs_by_id,
            max_checkpoints=max_checkpoints,
            l2_evidence_top_k=l2_evidence_top_k,
            show_progress=show_progress,
            save_every_states=save_every_states,
            save_callback=_save_partial,
        )
    except KeyboardInterrupt:
        _save_partial(benchmark)
        raise
    finally:
        validator_client.close()

    _persist_state_validation_payload(
        payload=payload,
        output_path=output_path,
    )
    return payload


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Run standalone TCE state validation.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Input raw benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output validated benchmark JSON path.")
    parser.add_argument(
        "--app-logs-path",
        type=Path,
        default=None,
        help="Optional app logs path. Defaults to sibling app_log_large.json or app_logs_final.json.",
    )
    parser.add_argument("--validator-provider", type=str, default="azure")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument(
        "--l2-evidence-top-k",
        type=int,
        default=0,
        help="0 means include all evidence logs referenced by state_observability.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during Stage 1 state validation.",
    )
    parser.add_argument(
        "--save-every-states",
        type=int,
        default=5,
        help="Incrementally save the Stage 1 artifact every N newly processed states. 0 disables periodic saves.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume Stage 1 from the existing --output artifact if it exists.",
    )
    args = parser.parse_args()

    result = build_validated_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        validator_provider=args.validator_provider,
        validator_model=args.validator_model,
        app_logs_path=args.app_logs_path,
        max_checkpoints=args.max_checkpoints,
        l2_evidence_top_k=args.l2_evidence_top_k,
        show_progress=(not args.no_progress),
        save_every_states=args.save_every_states,
        resume=args.resume,
    )
    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("checkpoints", [])))
    counts_path, eyeball_path, totals = write_state_validation_reports(
        payload=result,
        output_path=args.output,
    )
    print(
        "State validate summary:"
        f" pre={totals['pre_validate_count']},"
        f" after_l1={totals['after_l1_count']},"
        f" after_l2={totals['after_l2_count']},"
        f" after_l1_l2={totals['after_l1_l2_count']},"
        f" reused={totals['reused_count']},"
        f" computed={totals['computed_count']}"
    )
    print("State validate counts csv:", counts_path)
    print("Validated state eyeball csv:", eyeball_path)


if __name__ == "__main__":
    main()
