#!/usr/bin/env python3
"""Compatibility wrapper for TCE apply-pack build after pipeline split."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from benchmark_construction.build_tce_state_validation import build_validated_benchmark
from benchmark_construction.build_tce_task_packs import build_benchmark_task_packs
from tce_core.state_validation import write_state_validation_reports
from tce_contracts import (
    infer_canonical_research_doc,
    infer_research_frame_version,
    infer_task_contract_version,
)


def build_apply_pack(
    *,
    benchmark_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    item_count_per_key: int,
    apply_workers: int,
    reuse_scope: str,
    max_checkpoints: Optional[int],
    save_raw: bool,
    validator_provider: str,
    validator_model: str,
    max_rewrites: int,
    save_every_apply_keys: int = 5,
    questionability_mode: str,
    state_validate_only: bool,
    app_logs_path: Optional[Path],
    l2_evidence_top_k: int,
    save_every_states: int = 5,
    resume: bool = False,
) -> Dict[str, Any]:
    questionability_mode = str(questionability_mode or "filter").strip().lower()
    if questionability_mode not in {"filter", "validated_only"}:
        raise ValueError(
            "questionability_mode no longer supports raw-state generation. Use filter/validated_only."
        )

    if state_validate_only:
        return build_validated_benchmark(
            benchmark_path=benchmark_path,
            output_path=output_path,
            validator_provider=validator_provider,
            validator_model=validator_model,
            app_logs_path=app_logs_path,
            max_checkpoints=max_checkpoints,
            l2_evidence_top_k=l2_evidence_top_k,
            show_progress=True,
            save_every_states=save_every_states,
            resume=resume,
        )

    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    inherited_contract_version = infer_task_contract_version(benchmark_payload)
    inherited_research_frame_version = infer_research_frame_version(benchmark_payload)
    inherited_canonical_research_doc = infer_canonical_research_doc(benchmark_payload) or ""

    return build_benchmark_task_packs(
        benchmark_path=benchmark_path,
        output_path=output_path,
        tasks=["apply"],
        provider=provider,
        model=model,
        validator_provider=validator_provider,
        validator_model=validator_model,
        item_count_per_key=item_count_per_key,
        apply_workers=apply_workers,
        reuse_scope=reuse_scope,
        max_checkpoints=max_checkpoints,
        save_raw=save_raw,
        max_rewrites=max_rewrites,
        save_every_apply_keys=save_every_apply_keys,
        task_contract_version=inherited_contract_version,
        research_frame_version=inherited_research_frame_version,
        canonical_research_doc=inherited_canonical_research_doc,
    )


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Build TCE RQ3 apply-service QA packs into benchmark.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Input benchmark JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output benchmark JSON path.")
    parser.add_argument(
        "--app-logs-path",
        type=Path,
        default=None,
        help="Optional app logs path (app_log_large.json). Used by --state-validate-only.",
    )
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5")
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
        help="key|key_value_signature for cross-checkpoint question reuse. Formal protocol default is key_value_signature.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--validator-provider", type=str, default="azure")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--l2-evidence-top-k",
        type=int,
        default=0,
        help="How many evidence logs to include in L2 state validation prompt. 0 means all evidence.",
    )
    parser.add_argument("--max-rewrites", type=int, default=2)
    parser.add_argument(
        "--save-every-apply-keys",
        type=int,
        default=5,
        help="When building apply packs, incrementally save the output every N processed apply keys. 0 disables incremental saves.",
    )
    parser.add_argument(
        "--questionability-mode",
        type=str,
        default="filter",
        help="Deprecated. Validated pipeline always builds apply QA from validated states only.",
    )
    parser.add_argument(
        "--state-validate-only",
        action="store_true",
        help="Run only Stage 1 state validation and skip apply QA build.",
    )
    parser.add_argument(
        "--save-every-states",
        type=int,
        default=5,
        help="When using --state-validate-only, incrementally save every N newly processed states.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="When using --state-validate-only, resume from the existing --output artifact if present.",
    )
    args = parser.parse_args()

    result = build_apply_pack(
        benchmark_path=args.benchmark,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        item_count_per_key=args.item_count_per_key,
        apply_workers=args.apply_workers,
        reuse_scope=args.reuse_scope,
        max_checkpoints=args.max_checkpoints,
        save_raw=args.save_raw,
        validator_provider=args.validator_provider,
        validator_model=args.validator_model,
        max_rewrites=args.max_rewrites,
        save_every_apply_keys=args.save_every_apply_keys,
        questionability_mode=args.questionability_mode,
        state_validate_only=args.state_validate_only,
        app_logs_path=args.app_logs_path,
        l2_evidence_top_k=args.l2_evidence_top_k,
        save_every_states=args.save_every_states,
        resume=args.resume,
    )
    print("Saved:", args.output)
    print("Total checkpoints:", len(result.get("checkpoints", [])))
    if isinstance(result, dict) and isinstance(result.get("checkpoints"), list):
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
