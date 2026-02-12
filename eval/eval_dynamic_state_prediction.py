#!/usr/bin/env python3
"""Evaluate dynamic state prediction outputs against benchmark checkpoints."""

import argparse
import json
from concurrent.futures import as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from tqdm import tqdm

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
    judgment_template = [
        {"key": k, "reason": "<very short>", "correct": False}
        for k in sorted(value_pairs.keys())
    ]
    return """You are evaluating dynamic state prediction value quality.

Task:
For each key independently, decide whether predicted_value is correct vs expected_value.

[Per-key Value Pairs]
{value_pairs}

Fill this exact judgments template (do not add/drop keys):
{judgment_template}

Output JSON ONLY:
{{
  "judgments": [
    {{"key": "<state_key>", "reason": "<very short>", "correct": true or false}}
  ]
}}

Rules:
1. Include each key exactly once in judgments.
2. Judge only value correctness for each key.
3. If uncertain, mark correct=false.
""".format(
        value_pairs=json.dumps(value_pairs, ensure_ascii=False),
        judgment_template=json.dumps(judgment_template, ensure_ascii=False),
    )


def _build_llm_judge_text_format(target_keys: Sequence[str], model_idx: int) -> Type[BaseModel]:
    allowed = tuple(target_keys) if target_keys else ("__no_key__",)
    key_enum = Enum(
        f"JudgeKey_{model_idx}",
        {f"K_{i}": key for i, key in enumerate(allowed)},
    )
    item_model = create_model(  # type: ignore[call-overload]
        f"JudgeItem_{model_idx}",
        key=(key_enum, ...),
        correct=(bool, ...),
        reason=(str, ...),
    )
    output_model = create_model(  # type: ignore[call-overload]
        f"JudgeOutput_{model_idx}",
        judgments=(List[item_model], ...),
    )
    return output_model


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:
            dumped = value.model_dump()
        return _json_safe(dumped)
    return value


def _extract_correct_flag(item: Any) -> bool:
    if isinstance(item, dict):
        if "correct" in item:
            return _to_bool(item.get("correct"))
        # tolerate direct score-like labels
        if "is_correct" in item:
            return _to_bool(item.get("is_correct"))
        return False
    return _to_bool(item)


def _normalize_judgments(out: Any, target_keys: List[str]) -> Dict[str, bool]:
    """
    Normalize LLM judge outputs into a strict target_key -> bool mapping.
    Supports:
    - {"judgments": {"cat:state": {"correct": true}}}
    - {"judgments": {"category": {"state": {"correct": true}}}}
    - {"judgments": [{"key": "...", "correct": true}, ...]}
    - {"cat:state": {"correct": true}}  (fallback without judgments root)
    """
    result: Dict[str, bool] = {k: False for k in target_keys}

    if not isinstance(out, dict):
        return result

    data: Any = out.get("judgments")
    if data is None:
        data = out

    # list form: [{"key": "...", "correct": true}]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("state_key") or item.get("name")
            if key is None:
                continue
            key = str(key)
            if key in result:
                result[key] = _extract_correct_flag(item)
        return result

    if not isinstance(data, dict):
        return result

    # direct form: {"cat:state": {...}}
    for k in target_keys:
        if k in data:
            result[k] = _extract_correct_flag(data.get(k))

    # nested category form: {"category": {"state": {...}}}
    for cat, maybe_states in data.items():
        if not isinstance(maybe_states, dict):
            continue
        for state_name, item in maybe_states.items():
            merged = f"{cat}:{state_name}"
            if merged in result:
                result[merged] = _extract_correct_flag(item)

    return result


def _align_predictions_to_benchmark(
    benchmark: Dict[str, Any],
    raw_pred: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Align prediction checkpoint ids to current benchmark by checkpoint timestamp.
    This prevents silent mis-evaluation when checkpoint_id numbering changes
    across benchmark rebuilds.
    """
    report = {
        "total_predictions": 0,
        "aligned_by_timestamp": 0,
        "unmatched_timestamp": 0,
        "kept_original_id_no_timestamp": 0,
    }
    if not isinstance(raw_pred, dict) or "predictions" not in raw_pred:
        return raw_pred, report
    preds = raw_pred.get("predictions")
    if not isinstance(preds, list):
        return raw_pred, report

    cp_by_ts: Dict[str, str] = {}
    for cp in benchmark.get("checkpoints", []):
        if not isinstance(cp, dict):
            continue
        cid = cp.get("checkpoint_id")
        ts = (cp.get("as_of") or {}).get("timestamp")
        if cid and ts and str(ts) not in cp_by_ts:
            cp_by_ts[str(ts)] = str(cid)

    aligned: List[Dict[str, Any]] = []
    report["total_predictions"] = len(preds)
    for item in preds:
        if not isinstance(item, dict):
            aligned.append(item)
            continue
        out = dict(item)
        md = out.get("metadata") or {}
        ts = md.get("checkpoint_timestamp") if isinstance(md, dict) else None
        if ts is not None:
            old = out.get("checkpoint_id")
            md2 = dict(md) if isinstance(md, dict) else {}
            if str(ts) in cp_by_ts:
                new = cp_by_ts[str(ts)]
                out["checkpoint_id"] = new
                md2["original_checkpoint_id"] = old
                md2["alignment_status"] = "aligned_by_timestamp"
                report["aligned_by_timestamp"] += 1
            else:
                # Do not fall back to old checkpoint_id if timestamp is missing in benchmark;
                # that can silently evaluate against wrong checkpoint content.
                out["checkpoint_id"] = f"unmatched_timestamp::{old}"
                md2["original_checkpoint_id"] = old
                md2["alignment_status"] = "unmatched_timestamp"
                report["unmatched_timestamp"] += 1
            out["metadata"] = md2
        else:
            report["kept_original_id_no_timestamp"] += 1
        aligned.append(out)

    out_payload = dict(raw_pred)
    out_payload["predictions"] = aligned
    return out_payload, report


def _run_llm_judge(
    rows: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    on_row_done: Optional[Callable[[], None]] = None,
) -> None:
    if not rows:
        return

    reqs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        expected_snapshot = row.get("_expected_snapshot", {}) or {}
        predicted_snapshot = row.get("_pred_snapshot", {}) or {}
        target_keys = sorted(expected_snapshot.keys())
        value_pairs = {
            k: {
                "expected_value": expected_snapshot.get(k),
                "predicted_value": predicted_snapshot.get(k),
            }
            for k in target_keys
        }
        reqs.append(
            {
                "index": idx,
                "prompt": _build_llm_judge_prompt(value_pairs=value_pairs),
                "target_keys": target_keys,
                "text_format": _build_llm_judge_text_format(target_keys, idx),
            }
        )

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
    )
    error_count = 0
    deployment_not_found = False
    first_error_msg = ""

    def _finalize_row(row: Dict[str, Any], out: Any, prompt: str) -> None:
        nonlocal error_count, deployment_not_found, first_error_msg
        score = 0.0
        reason = ""
        row["llm_judge_prompt"] = prompt

        if isinstance(out, Exception):
            error_count += 1
            reason = str(out)
            if not first_error_msg:
                first_error_msg = reason
            if "DeploymentNotFound" in reason or "deployment" in reason.lower():
                deployment_not_found = True
            row["llm_judge_score"] = score
            row["llm_judge_reason"] = reason
            row["llm_judge_raw_output"] = {"_error": reason}
            if on_row_done is not None:
                on_row_done()
            return

        if isinstance(out, dict):
            row["llm_judge_raw_output"] = _json_safe(out)
            response_meta = out.get("__response_meta") if isinstance(out.get("__response_meta"), dict) else {}
            if response_meta:
                row["llm_judge_response_meta"] = _json_safe(response_meta)
                if response_meta.get("input_text_from_response"):
                    row["llm_judge_prompt_from_response"] = _json_safe(response_meta.get("input_text_from_response"))
                elif response_meta.get("input_from_response") is not None:
                    row["llm_judge_prompt_from_response"] = _json_safe(response_meta.get("input_from_response"))
            expected_snapshot = row.get("_expected_snapshot", {}) or {}
            target_keys = sorted(expected_snapshot.keys())
            total = len(target_keys)
            correct = 0

            if total > 0:
                judgments = _normalize_judgments(out, target_keys)
                for key in target_keys:
                    if judgments.get(key, False):
                        correct += 1

            score = (correct / total) if total > 0 else 0.0
            row["llm_judge_correct_pairs"] = correct
            row["llm_judge_total_pairs"] = total
            reason = str(out.get("reason", ""))
        else:
            row["llm_judge_raw_output"] = _json_safe(out)

        score = max(0.0, min(1.0, score))
        row["llm_judge_score"] = score
        row["llm_judge_reason"] = reason
        if on_row_done is not None:
            on_row_done()

    try:
        futures = []
        idx_by_future: Dict[Any, int] = {}
        req_by_idx: Dict[int, Dict[str, Any]] = {}
        structured = client.supports_structured_response()
        for req in reqs:
            req_by_idx[req["index"]] = req
            if structured:
                future = client.ask_structured_async(
                    req["prompt"],
                    text_format=req["text_format"],
                )
            else:
                future = client.ask_async(req["prompt"], response_type="json")
            idx_by_future[future] = req["index"]
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge", unit="cp"):
            idx = idx_by_future[future]
            req = req_by_idx[idx]
            row = rows[idx]
            try:
                out = future.result()
            except Exception as exc:
                out = exc
            _finalize_row(row, out, req["prompt"])
    finally:
        client.close()

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
    on_progress: Optional[Callable[[List[Dict[str, Any]], int], None]] = None,
) -> Dict[str, Any]:
    checkpoint_rows, evaluated = evaluate_checkpoints(
        benchmark,
        predictions,
        include_internal_payload=enable_llm_judge,
    )

    if enable_llm_judge:
        on_row_done: Optional[Callable[[], None]] = None
        if on_progress is not None:
            def on_row_done() -> None:
                on_progress(checkpoint_rows, evaluated)
        _run_llm_judge(
            checkpoint_rows,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_max_workers=llm_max_workers,
            on_row_done=on_row_done,
        )

    for row in checkpoint_rows:
        row.pop("_expected_snapshot", None)
        row.pop("_pred_snapshot", None)
        row.pop("_expected_evidence", None)
        row.pop("_pred_evidence", None)

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
    raw_pred, align_report = _align_predictions_to_benchmark(benchmark, raw_pred)
    predictions = normalize_predictions(raw_pred)

    def _write_progress(rows: List[Dict[str, Any]], evaluated: int) -> None:
        output_rows: List[Dict[str, Any]] = []
        for row in rows:
            out_row = {
                k: _json_safe(v)
                for k, v in row.items()
                if k not in {"_expected_snapshot", "_pred_snapshot", "_expected_evidence", "_pred_evidence"}
            }
            output_rows.append(out_row)
        payload = {
            "user_id": benchmark.get("user_id"),
            "total_checkpoints": benchmark.get("total_checkpoints", len(output_rows)),
            "evaluated_checkpoints": evaluated,
            "skipped_checkpoints": max(benchmark.get("total_checkpoints", 0) - evaluated, 0),
            "summary": mean_numeric_fields(
                [{k: v for k, v in row.items() if k != "checkpoint_id"} for row in output_rows]
            ),
            "checkpoints": output_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = evaluate(
        benchmark,
        predictions,
        enable_llm_judge=args.enable_llm_judge,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        on_progress=_write_progress if args.enable_llm_judge else None,
    )
    result["prediction_alignment"] = align_report
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
