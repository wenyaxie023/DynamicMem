#!/usr/bin/env python3
"""Evaluate dynamic state prediction outputs against benchmark checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from dynamic_state_prediction_core import (
    evaluate_checkpoints,
    mean_numeric_fields,
    normalize_predictions,
)

from .client import LLMClient

load_dotenv()


def _build_llm_judge_prompt(
    value_pairs: Dict[str, Dict[str, Any]],
) -> str:
    judgment_template = {
        k: {"reason": "<very short>", "correct": False}
        for k in sorted(value_pairs.keys())
    }
    return """You are evaluating dynamic state prediction value quality.

Task:
For each key independently, decide whether predicted_value is correct vs expected_value.

[Per-key Value Pairs]
{value_pairs}

Fill this exact judgments template (do not add/drop keys):
{judgment_template}

Output JSON ONLY:
{{
  "judgments": {{
    "<key>": {{"reason": "<very short>", "correct": true or false}}
  }}
}}

Rules:
1. Keep exactly the same keys as the provided judgments template.
2. Judge only value correctness for each key.
3. If uncertain, mark correct=false.
""".format(
        value_pairs=json.dumps(value_pairs, ensure_ascii=False),
        judgment_template=json.dumps(judgment_template, ensure_ascii=False),
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return False


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
        expected_snapshot = row.get("_expected_snapshot", {}) or {}
        predicted_snapshot = row.get("_pred_snapshot", {}) or {}
        value_pairs = {
            k: {
                "expected_value": expected_snapshot.get(k),
                "predicted_value": predicted_snapshot.get(k),
            }
            for k in sorted(expected_snapshot.keys())
        }
        prompts.append(
            _build_llm_judge_prompt(
                value_pairs=value_pairs,
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
            judgments = out.get("judgments")
            expected_snapshot = row.get("_expected_snapshot", {}) or {}
            target_keys = sorted(expected_snapshot.keys())
            total = len(target_keys)
            correct = 0

            if isinstance(judgments, dict) and total > 0:
                for key in target_keys:
                    item = judgments.get(key)
                    ok = _to_bool(item.get("correct", False)) if isinstance(item, dict) else False
                    if ok:
                        correct += 1

            score = (correct / total) if total > 0 else 0.0
            row["llm_judge_correct_pairs"] = correct
            row["llm_judge_total_pairs"] = total
            reason = str(out.get("reason", ""))
        score = max(0.0, min(1.0, score))
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
        row.pop("_pred_snapshot", None)

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
    parser = argparse.ArgumentParser(description="Evaluate dynamic state prediction outputs.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to dynamic_state_prediction_benchmark.json",
    )
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    llm_judge_score = result["summary"].get("llm_judge_score_mean")
    llm_judge_text = f"{llm_judge_score:.4f}" if llm_judge_score is not None else "N/A (disabled)"
    print(
        "Summary: "
        f"checkpoints={result['evaluated_checkpoints']}, "
        f"value_f1={result['summary'].get('snapshot_value_f1_mean_on_expected_mean', 0.0):.4f}, "
        f"evidence_recall={result['summary'].get('snapshot_evidence_recall_mean_on_expected_mean', 0.0):.4f}, "
        f"llm_judge={llm_judge_text}"
    )


if __name__ == "__main__":
    main()
