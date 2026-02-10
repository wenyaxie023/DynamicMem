#!/usr/bin/env python3
"""Evaluate state abstraction predictions against benchmark checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from state_abstraction_core import (
    evaluate_checkpoints,
    mean_numeric_fields,
    normalize_predictions,
)

from .client import LLMClient

load_dotenv()


def _build_llm_judge_prompt(
    checkpoint_id: str,
    expected_snapshot: Dict[str, Any],
    expected_delta: Dict[str, Any],
    predicted_snapshot: Dict[str, Any],
    predicted_delta: Dict[str, Any],
) -> str:
    return """You are evaluating state abstraction quality.

Task:
Given expected state and model-predicted state for one checkpoint, score the prediction quality.

Checkpoint ID: {checkpoint_id}

[Expected Snapshot]
{expected_snapshot}

[Predicted Snapshot]
{predicted_snapshot}

[Expected Delta]
{expected_delta}

[Predicted Delta]
{predicted_delta}

Scoring rubric (0-10):
- 0-2: Almost unrelated or mostly wrong.
- 3-5: Some overlap but many key misses/errors.
- 6-8: Mostly correct with partial omissions/errors.
- 9-10: Highly accurate and complete.

Output JSON ONLY:
{{"score": <number 0-10>, "reason": "<short reason>"}}
""".format(
        checkpoint_id=checkpoint_id,
        expected_snapshot=json.dumps(expected_snapshot, ensure_ascii=False),
        predicted_snapshot=json.dumps(predicted_snapshot, ensure_ascii=False),
        expected_delta=json.dumps(expected_delta, ensure_ascii=False),
        predicted_delta=json.dumps(predicted_delta, ensure_ascii=False),
    )


def _run_llm_judge(
    rows: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
) -> None:
    if not rows:
        return

    prompts: List[str] = []
    for row in rows:
        prompts.append(
            _build_llm_judge_prompt(
                checkpoint_id=str(row.get("checkpoint_id")),
                expected_snapshot=row.get("_expected_snapshot", {}),
                expected_delta=row.get("_expected_delta", {}),
                predicted_snapshot=row.get("_pred_snapshot", {}),
                predicted_delta=row.get("_pred_delta", {}),
            )
        )

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )
    try:
        outputs = client.ask_many(prompts, response_type="json")
    finally:
        client.close()

    error_count = 0
    deployment_not_found = False
    first_error_msg = ""

    for row, out in zip(rows, outputs):
        score = 0.0
        reason = ""
        if isinstance(out, Exception):
            error_count += 1
            reason = str(out)
            if not first_error_msg:
                first_error_msg = reason
            if "DeploymentNotFound" in reason or "deployment" in reason.lower():
                deployment_not_found = True
            row["llm_judge_score"] = score
            row["llm_judge_reason"] = reason
            continue
        if isinstance(out, dict):
            reason = str(out.get("reason", ""))
            try:
                score = float(out.get("score", 0.0))
            except Exception:
                score = 0.0
        score = max(0.0, min(10.0, score))
        row["llm_judge_score"] = score
        row["llm_judge_reason"] = reason

    if error_count == len(rows) and error_count > 0:
        hint = (
            "All LLM-judge calls failed. Check provider/model/deployment mapping. "
            "If you are using Azure, --llm-model must be your Azure deployment name "
            "(not necessarily the base model id)."
        )
        if deployment_not_found:
            hint += " Received DeploymentNotFound from provider."
        raise RuntimeError(f"{hint} First error: {first_error_msg}")


def evaluate(
    benchmark: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    *,
    enable_llm_judge: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-5-mini",
    llm_max_workers: int = 4,
) -> Dict[str, Any]:
    checkpoint_rows, evaluated = evaluate_checkpoints(
        benchmark,
        predictions,
        include_internal_payload=enable_llm_judge,
    )

    if enable_llm_judge:
        _run_llm_judge(
            checkpoint_rows,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_max_workers=llm_max_workers,
        )

    for row in checkpoint_rows:
        row.pop("_expected_snapshot", None)
        row.pop("_expected_delta", None)
        row.pop("_pred_snapshot", None)
        row.pop("_pred_delta", None)

    summary = mean_numeric_fields(
        [{k: v for k, v in row.items() if k != "checkpoint_id"} for row in checkpoint_rows]
    )

    return {
        "user_id": benchmark.get("user_id"),
        "total_checkpoints": benchmark.get("total_checkpoints", len(checkpoint_rows)),
        "evaluated_checkpoints": evaluated,
        "skipped_checkpoints": max(benchmark.get("total_checkpoints", 0) - evaluated, 0),
        "summary": summary,
        "checkpoints": checkpoint_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate state abstraction predictions.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to state_abstraction_benchmark.json")
    parser.add_argument("--prediction", type=Path, required=True, help="Path to model predictions json")
    parser.add_argument("--output", type=Path, required=True, help="Output eval json path")
    parser.add_argument("--enable-llm-judge", action="store_true", help="Enable LLM-as-a-judge scoring.")
    parser.add_argument("--llm-provider", type=str, default="openai", help="LLM judge provider.")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM judge model.")
    parser.add_argument("--llm-max-workers", type=int, default=4, help="Max workers for LLM judge.")
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    raw_pred = json.loads(args.prediction.read_text(encoding="utf-8"))
    predictions = normalize_predictions(raw_pred)

    result = evaluate(
        benchmark,
        predictions,
        enable_llm_judge=args.enable_llm_judge,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    print(
        "Summary: "
        f"checkpoints={result['evaluated_checkpoints']}, "
        f"snapshot_f1={result['summary'].get('snapshot_key_f1_mean', 0.0):.4f}, "
        f"delta_f1={result['summary'].get('delta_op_f1_mean', 0.0):.4f}, "
        f"llm_judge={result['summary'].get('llm_judge_score_mean', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
