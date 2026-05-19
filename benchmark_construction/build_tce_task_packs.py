#!/usr/bin/env python3
"""Build precomputed TCE task packs from a validated benchmark."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from baseline_prediction.common.llm_client import LLMClient
from tce_core.task_packs import build_task_packs
from tce_contracts import (
    CANONICAL_RESEARCH_DOC_V2,
    CURRENT_TASK_CONTRACT_VERSION,
    RESEARCH_FRAME_VERSION_V2,
    infer_task_contract_version,
    normalize_stage2_tasks,
)


def _parse_tasks(raw: str) -> List[str]:
    items = [part.strip().lower() for part in str(raw or "all").split(",")]
    return [item for item in items if item]


def build_benchmark_task_packs(
    *,
    benchmark_path: Path,
    output_path: Path,
    tasks: List[str],
    provider: str,
    model: str,
    validator_provider: str,
    validator_model: str,
    item_count_per_key: int,
    reuse_scope: str,
    max_checkpoints: int | None,
    apply_workers: int,
    save_raw: bool,
    max_rewrites: int,
    save_every_apply_keys: int = 5,
    show_progress: bool = True,
    enable_state_completion_scoring_points: bool = False,
    task_contract_version: str = CURRENT_TASK_CONTRACT_VERSION,
    research_frame_version: str = RESEARCH_FRAME_VERSION_V2,
    canonical_research_doc: str = CANONICAL_RESEARCH_DOC_V2,
) -> Dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    effective_contract_version = (
        str(task_contract_version or "").strip()
        or infer_task_contract_version(benchmark)
    )
    normalized = normalize_stage2_tasks(
        tasks,
        task_contract_version=effective_contract_version,
    )
    needs_state_completion_scoring = (
        "state_completion" in normalized and bool(enable_state_completion_scoring_points)
    )
    needs_generator = bool(normalized & {"change_tracking", "apply"}) or needs_state_completion_scoring
    needs_validator = bool(normalized & {"change_tracking", "apply"}) or needs_state_completion_scoring

    generator_client = None
    validator_client = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_snapshot(payload: Dict[str, Any]) -> None:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        if needs_generator:
            generator_client = LLMClient(
                provider=provider,
                model_name=model,
                max_workers=max(1, int(apply_workers)),
            )
        if needs_validator:
            validator_client = LLMClient(
                provider=validator_provider,
                model_name=validator_model,
                max_workers=max(1, int(apply_workers)),
            )
        payload = build_task_packs(
            benchmark=benchmark,
            tasks=sorted(normalized),
            generator_client=generator_client,
            validator_client=validator_client,
            provider=provider,
            model=model,
            validator_provider=validator_provider,
            validator_model=validator_model,
            item_count_per_key=item_count_per_key,
            max_rewrites=max_rewrites,
            save_raw=save_raw,
            reuse_scope=reuse_scope,
            max_checkpoints=max_checkpoints,
            apply_workers=apply_workers,
            save_every_apply_keys=save_every_apply_keys,
            save_callback=_save_snapshot,
            show_progress=show_progress,
            enable_state_completion_scoring_points=enable_state_completion_scoring_points,
            task_contract_version=task_contract_version,
            research_frame_version=research_frame_version,
            canonical_research_doc=canonical_research_doc,
        )
    finally:
        if generator_client is not None:
            generator_client.close()
        if validator_client is not None:
            validator_client.close()

    _save_snapshot(payload)
    return payload


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Build TCE precomputed task packs.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Input validated benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output benchmark JSON path with packs.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated subset of state_completion,change_tracking,apply or all.",
    )
    parser.add_argument("--provider", type=str, default="azure")
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--validator-provider", type=str, default="azure")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini")
    parser.add_argument("--item-count-per-key", type=int, default=1)
    parser.add_argument(
        "--apply-workers",
        type=int,
        default=4,
        help="Maximum number of apply keys to process concurrently within one checkpoint.",
    )
    parser.add_argument(
        "--reuse-scope",
        type=str,
        default="key_value_signature",
        choices=["key", "key_value_signature"],
        help="Task C cross-checkpoint reuse policy. Formal protocol default is key_value_signature.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--max-rewrites", type=int, default=2)
    parser.add_argument(
        "--save-every-apply-keys",
        type=int,
        default=5,
        help="When building apply packs, incrementally save the output every N processed apply keys. 0 disables incremental saves.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Stage 2 progress bars and summary logs.",
    )
    parser.add_argument(
        "--enable-state-completion-scoring-points",
        action="store_true",
        help=(
            "Opt in to legacy Task A scoring_points generation and validation. "
            "Disabled by default for holistic LLM-judge Task A evaluation."
        ),
    )
    parser.add_argument(
        "--task-contract-version",
        type=str,
        default=CURRENT_TASK_CONTRACT_VERSION,
        help="Top-level task contract version to stamp into the benchmark metadata.",
    )
    parser.add_argument(
        "--research-frame-version",
        type=str,
        default=RESEARCH_FRAME_VERSION_V2,
        help="Top-level research frame version to stamp into the benchmark metadata.",
    )
    parser.add_argument(
        "--canonical-research-doc",
        type=str,
        default=CANONICAL_RESEARCH_DOC_V2,
        help="Canonical research-theme/RQ document path to stamp into the benchmark metadata.",
    )
    args = parser.parse_args()

    payload = build_benchmark_task_packs(
        benchmark_path=args.benchmark,
        output_path=args.output,
        tasks=_parse_tasks(args.tasks),
        provider=args.provider,
        model=args.model,
        validator_provider=args.validator_provider,
        validator_model=args.validator_model,
        item_count_per_key=args.item_count_per_key,
        apply_workers=args.apply_workers,
        reuse_scope=args.reuse_scope,
        max_checkpoints=args.max_checkpoints,
        save_raw=args.save_raw,
        max_rewrites=args.max_rewrites,
        save_every_apply_keys=args.save_every_apply_keys,
        show_progress=(not args.no_progress),
        enable_state_completion_scoring_points=args.enable_state_completion_scoring_points,
        task_contract_version=args.task_contract_version,
        research_frame_version=args.research_frame_version,
        canonical_research_doc=args.canonical_research_doc,
    )
    print("Saved:", args.output)
    print("Tasks:", ",".join(_parse_tasks(args.tasks)))
    print("Total checkpoints:", len(payload.get("checkpoints", [])))


if __name__ == "__main__":
    main()
