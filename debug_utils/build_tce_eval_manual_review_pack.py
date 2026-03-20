#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.evaluation import value_f1


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_option_label(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    for label in ("A", "B", "C", "D", "E"):
        for prefix in (f"{label}.", f"{label})", f"OPTION {label}", f"{label}:"):
            if upper.startswith(prefix):
                return label
    return ""


def _snapshot_cases(
    eval_payload: Dict[str, Any],
    prediction_by_cp: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    judge_inputs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in eval_payload.get("snapshot_slot_judge_inputs", []) or eval_payload.get("llm_judge_inputs", []):
        if not isinstance(item, dict):
            continue
        judge_inputs[(str(item.get("checkpoint_id")), str(item.get("state_key")))] = item
    for cp in eval_payload.get("checkpoints", []):
        cid = str(cp.get("checkpoint_id"))
        slot_payloads = cp.get("snapshot_slot_judgments_by_key") or {}
        pred_cp = prediction_by_cp.get(cid) or {}
        pred_evidence = pred_cp.get("evidence") or {}
        groundtruth_snapshot = cp.get("groundtruth_snapshot") or {}
        prediction_snapshot = cp.get("prediction_snapshot") or pred_cp.get("snapshot_state") or {}
        if slot_payloads:
            for key, payload in slot_payloads.items():
                judgments = list((payload or {}).get("judgments") or [])
                expected_value = groundtruth_snapshot.get(key)
                predicted_value = prediction_snapshot.get(key)
                slot_score = float((payload or {}).get("score_0_1") or 0.0)
                vf1 = value_f1(expected_value, predicted_value)
                rows.append(
                    {
                        "checkpoint_id": cid,
                        "state_key": key,
                        "value_f1": vf1,
                        "slot_score_0_1": slot_score,
                        "judge_vs_value_gap": slot_score - vf1,
                        "expected_value": expected_value,
                        "predicted_value": predicted_value,
                        "predicted_evidence": pred_evidence.get(key, []),
                        "judge": judgments,
                    }
                )
            continue
        judgments = cp.get("llm_judge_judgments") or {}
        for key, judgment in judgments.items():
            cp_input = judge_inputs.get((cid, str(key))) or {}
            value_pairs = cp_input.get("value_pairs") or {}
            pair = value_pairs.get(key) or {}
            expected_value = pair.get("expected_value")
            predicted_value = pair.get("predicted_value")
            judge_mean = _mean(
                [
                    float(judgment.get("core_value_match", 1.0)),
                    float(judgment.get("constraint_coverage", 1.0)),
                    float(judgment.get("value_precision", 1.0)),
                    float(judgment.get("unsupported_inference_control", 1.0)),
                ]
            )
            vf1 = value_f1(expected_value, predicted_value)
            rows.append(
                {
                    "checkpoint_id": cid,
                    "state_key": key,
                    "value_f1": vf1,
                    "judge_mean_1_5": judge_mean,
                    "judge_vs_value_gap": judge_mean - (vf1 * 5.0),
                    "expected_value": expected_value,
                    "predicted_value": predicted_value,
                    "predicted_evidence": pred_evidence.get(key, []),
                    "judge": judgment,
                }
            )
    rows.sort(key=lambda x: (x["judge_vs_value_gap"], -x["judge_mean_1_5"]), reverse=True)
    return rows


def _change_pair_f1(pair: Dict[str, Any]) -> float:
    before = value_f1(pair.get("expected_before"), pair.get("predicted_before"))
    after = value_f1(pair.get("expected_after"), pair.get("predicted_after"))
    return (before + after) / 2.0


def _change_cases(eval_payload: Dict[str, Any], prediction_by_cp: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    judge_inputs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in eval_payload.get("change_slot_judge_inputs", []) or eval_payload.get("llm_change_judge_inputs", []):
        if not isinstance(item, dict):
            continue
        judge_inputs[(str(item.get("checkpoint_id")), str(item.get("state_key")))] = item
    for cp in eval_payload.get("checkpoints", []):
        cid = str(cp.get("checkpoint_id"))
        slot_payloads = cp.get("change_slot_judgments_by_key") or {}
        pred_cp = prediction_by_cp.get(cid) or {}
        pred_change = pred_cp.get("change_analysis") or {}
        groundtruth_change = cp.get("groundtruth_change") or {}
        prediction_change = cp.get("prediction_change") or pred_change
        if slot_payloads:
            for key, payload in slot_payloads.items():
                pair_f1 = _change_pair_f1(
                    {
                        "expected_before": (groundtruth_change.get(key) or {}).get("before"),
                        "expected_after": (groundtruth_change.get(key) or {}).get("after"),
                        "predicted_before": (prediction_change.get(key) or {}).get("before"),
                        "predicted_after": (prediction_change.get(key) or {}).get("after"),
                    }
                )
                slot_score = float((payload or {}).get("state_predict_score_0_1") or 0.0)
                rows.append(
                    {
                        "checkpoint_id": cid,
                        "state_key": key,
                        "before_after_pair_f1": pair_f1,
                        "slot_score_0_1": slot_score,
                        "judge_vs_pair_gap": slot_score - pair_f1,
                        "expected_before": (groundtruth_change.get(key) or {}).get("before"),
                        "expected_after": (groundtruth_change.get(key) or {}).get("after"),
                        "predicted_before": (prediction_change.get(key) or {}).get("before"),
                        "predicted_after": (prediction_change.get(key) or {}).get("after"),
                        "predicted_change_reason": (prediction_change.get(key) or {}).get("change_reason"),
                        "expected_evidence_ids": cp.get("groundtruth_change_evidence", {}).get(key, []),
                        "predicted_evidence": (pred_change.get(key) or {}).get("evidence", []),
                        "judge": payload,
                    }
                )
            continue
        judgments = cp.get("change_llm_judge_judgments") or {}
        for key, judgment in judgments.items():
            cp_input = judge_inputs.get((cid, str(key))) or {}
            change_pairs = cp_input.get("change_pairs") or {}
            pair = change_pairs.get(key) or {}
            judge_mean = _mean(
                [
                    float(judgment.get("core_value_match", 1.0)),
                    float(judgment.get("constraint_coverage", 1.0)),
                    float(judgment.get("value_precision", 1.0)),
                    float(judgment.get("unsupported_inference_control", 1.0)),
                ]
            )
            pair_f1 = _change_pair_f1(pair)
            rows.append(
                {
                    "checkpoint_id": cid,
                    "state_key": key,
                    "before_after_pair_f1": pair_f1,
                    "judge_mean_1_5": judge_mean,
                    "judge_vs_pair_gap": judge_mean - (pair_f1 * 5.0),
                    "expected_before": pair.get("expected_before"),
                    "expected_after": pair.get("expected_after"),
                    "predicted_before": pair.get("predicted_before"),
                    "predicted_after": pair.get("predicted_after"),
                    "predicted_change_reason": pair.get("predicted_change_reason"),
                    "expected_evidence_ids": pair.get("expected_evidence_ids", []),
                    "predicted_evidence": (pred_change.get(key) or {}).get("evidence", []),
                    "judge": judgment,
                }
            )
    rows.sort(key=lambda x: (x["judge_vs_pair_gap"], -x["judge_mean_1_5"]), reverse=True)
    return rows


def _apply_cases(
    eval_payload: Dict[str, Any],
    prediction_by_cp: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cp in eval_payload.get("checkpoints", []):
        cid = str(cp.get("checkpoint_id"))
        slot_payloads = cp.get("rq3_apply_answer_slot_judgments_by_item") or {}
        expected_rq3 = cp.get("groundtruth_rq3_apply") or {}
        if not isinstance(expected_rq3, dict):
            continue
        pred_cp = prediction_by_cp.get(cid) or {}
        pred_rq3 = cp.get("prediction_rq3_apply") or pred_cp.get("rq3_apply_answers") or {}
        for state_key in sorted(expected_rq3.keys()):
            pred_items = ((pred_rq3.get(state_key) or {}).get("items") or [])
            pred_by_id = {
                str(candidate.get("qa_id") or ""): candidate
                for candidate in pred_items
                if isinstance(candidate, dict)
            }
            for item in ((expected_rq3.get(state_key) or {}).get("items") or []):
                if not isinstance(item, dict):
                    continue
                qa_id = str(item.get("qa_id") or "")
                pred_item = pred_by_id.get(qa_id) or {}
                reference_answer = item.get("apply_reference_answer")
                predicted_answer = pred_item.get("answer")
                reference_option = _extract_option_label(reference_answer)
                predicted_option = _extract_option_label(predicted_answer)
                option_extractable = bool(reference_option)
                option_correct = bool(reference_option and predicted_option and reference_option == predicted_option)
                item_id = f"{state_key}::{qa_id}"
                slot_payload = slot_payloads.get(item_id) or {}
                rows.append(
                    {
                        "checkpoint_id": cid,
                        "state_key": state_key,
                        "qa_id": qa_id,
                        "slot_score_0_1": float(slot_payload.get("score_0_1") or 0.0),
                        "option_extractable": option_extractable,
                        "reference_option": reference_option,
                        "predicted_option": predicted_option,
                        "option_correct": option_correct,
                        "option_vs_slot_gap": (1.0 if option_correct else 0.0) - float(slot_payload.get("score_0_1") or 0.0),
                        "apply_scenario": item.get("apply_scenario"),
                        "question": item.get("apply_question"),
                        "reference_answer": reference_answer,
                        "predicted_answer": predicted_answer,
                        "predicted_evidence": pred_item.get("evidence", []),
                        "slot_judgments": list(slot_payload.get("judgments") or []),
                    }
                )
    rows.sort(key=lambda x: (x["option_vs_slot_gap"], int(bool(x["option_correct"]))), reverse=True)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list)):
                    out[field] = json.dumps(value, ensure_ascii=False)
                else:
                    out[field] = value
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual-review pack for TCE evaluation outputs.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    _ = _load_json(args.benchmark)
    prediction = _load_json(args.prediction)
    eval_payload = _load_json(args.eval)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_by_cp = {
        str(x.get("checkpoint_id")): x
        for x in prediction.get("predictions", [])
        if isinstance(x, dict) and x.get("checkpoint_id")
    }

    snapshot_rows = _snapshot_cases(eval_payload, prediction_by_cp)
    change_rows = _change_cases(eval_payload, prediction_by_cp)
    apply_rows = _apply_cases(eval_payload, prediction_by_cp)

    snapshot_top = snapshot_rows[: args.top_n]
    change_top = change_rows[: args.top_n]
    apply_top = apply_rows[: args.top_n]

    summary = {
        "eval_file": str(args.eval),
        "prediction_file": str(args.prediction),
        "top_n": args.top_n,
        "snapshot": {
            "total_cases": len(snapshot_rows),
            "top_suspicious_by_judge_vs_value_gap": [
                {
                    "checkpoint_id": x["checkpoint_id"],
                    "state_key": x["state_key"],
                    "value_f1": x["value_f1"],
                    "judge_mean_1_5": x["judge_mean_1_5"],
                    "judge_vs_value_gap": x["judge_vs_value_gap"],
                }
                for x in snapshot_top[:10]
            ],
        },
        "change": {
            "total_cases": len(change_rows),
            "top_suspicious_by_judge_vs_pair_gap": [
                {
                    "checkpoint_id": x["checkpoint_id"],
                    "state_key": x["state_key"],
                    "before_after_pair_f1": x["before_after_pair_f1"],
                    "judge_mean_1_5": x["judge_mean_1_5"],
                    "judge_vs_pair_gap": x["judge_vs_pair_gap"],
                }
                for x in change_top[:10]
            ],
        },
        "apply": {
            "total_cases": len(apply_rows),
            "top_suspicious_by_option_vs_snapshot_value_gap": [
                {
                    "checkpoint_id": x["checkpoint_id"],
                    "state_key": x["state_key"],
                    "qa_id": x["qa_id"],
                    "option_correct": x["option_correct"],
                    "snapshot_value_f1": x["snapshot_value_f1"],
                    "option_vs_snapshot_value_gap": x["option_vs_snapshot_value_gap"],
                }
                for x in apply_top[:10]
            ],
        },
    }

    _write_json(args.output_dir / "tce_eval_manual_review_summary.json", summary)
    _write_json(args.output_dir / "snapshot_judge_suspicious_cases.json", snapshot_top)
    _write_json(args.output_dir / "change_judge_suspicious_cases.json", change_top)
    _write_json(args.output_dir / "apply_judge_suspicious_cases.json", apply_top)
    _write_csv(
        args.output_dir / "snapshot_judge_suspicious_cases.csv",
        snapshot_top,
        ["checkpoint_id", "state_key", "value_f1", "judge_mean_1_5", "judge_vs_value_gap"],
    )
    _write_csv(
        args.output_dir / "change_judge_suspicious_cases.csv",
        change_top,
        ["checkpoint_id", "state_key", "before_after_pair_f1", "judge_mean_1_5", "judge_vs_pair_gap"],
    )
    _write_csv(
        args.output_dir / "apply_judge_suspicious_cases.csv",
        apply_top,
        ["checkpoint_id", "state_key", "qa_id", "option_extractable", "reference_option", "predicted_option", "option_correct", "snapshot_judge_mean_1_5", "snapshot_value_f1", "option_vs_snapshot_value_gap"],
    )

    review_md = f"""# TCE Eval Manual Review Pack

Files:
- `snapshot_judge_suspicious_cases.json`
- `change_judge_suspicious_cases.json`
- `apply_judge_suspicious_cases.json`

How to read:
- `snapshot`: top cases where Task A structured value quality is low but snapshot LLM judge is comparatively lenient.
- `change`: top cases where before/after reconstruction is poor but the unified 4-rubric judge remains comparatively lenient.
- `apply`: top cases where Task C gets the option answer right while the paired Task A state for the same key is weak.

Recommended review order:
1. `apply_judge_suspicious_cases.json`
2. `change_judge_suspicious_cases.json`
3. `snapshot_judge_suspicious_cases.json`
"""
    (args.output_dir / "README.md").write_text(review_md, encoding="utf-8")


if __name__ == "__main__":
    main()
