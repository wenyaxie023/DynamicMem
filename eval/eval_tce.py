#!/usr/bin/env python3
"""Evaluate TCE outputs against benchmark checkpoints."""

import argparse
import json
import os
from concurrent.futures import as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> None:
        return None

try:
    from pydantic import BaseModel, Field, create_model
except Exception:  # pragma: no cover
    BaseModel = Any  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    create_model = None  # type: ignore[assignment]
from tqdm import tqdm

from tce_core import (
    mean_numeric_fields,
)
from bench_core.tce_evaluator import (
    build_tce_result_payload,
    evaluate_tce_rows,
)
from tce_contracts import infer_task_contract_version, normalized_contract_metadata, task_contract_is_v2
from .prompts_tce import (
    build_apply_slot_judge_prompt,
    build_change_slot_judge_prompt,
    build_snapshot_slot_judge_prompt,
)

load_dotenv()


def _validate_llm_judge_env(provider: str) -> None:
    p = str(provider or "").strip().lower()
    try:
        from generation.common.provider_config import setup_provider_env
        from pathlib import Path as _Path
        setup_provider_env(
            provider=p if p in {"openai", "azure"} else "openai",
            repo_root=_Path(__file__).resolve().parents[1],
            require_api_key=(p in {"openai", "azure"}),
        )
    except Exception:
        pass
    if p in {"openai", "azure"}:
        has_openai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
        has_azure = bool((os.getenv("AZURE_OPENAI_API_KEY") or "").strip())
        if not (has_openai or has_azure):
            raise RuntimeError(
                "LLM judge is enabled but no API key found. "
                "Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY."
            )
        return
    if p == "gemini":
        if not bool((os.getenv("GOOGLE_API_KEY") or "").strip()):
            raise RuntimeError(
                "LLM judge is enabled but GOOGLE_API_KEY is not set."
            )
        return


def _build_slot_judge_text_format(point_ids: Sequence[str], model_idx: int) -> Type[BaseModel]:
    if create_model is None or Field is None:
        raise RuntimeError("LLM judge structured output requires pydantic to be installed.")
    item_model = create_model(  # type: ignore[call-overload]
        f"SlotJudgeItem_{model_idx}",
        point_id=(str, Field(..., description="One point_id from the provided slots.")),
        analysis=(str, ...),
        correct=(bool, ...),
    )
    output_model = create_model(  # type: ignore[call-overload]
        f"SlotJudgeOutput_{model_idx}",
        judgments=(List[item_model], ...),
    )
    return output_model


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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_slot_judgments(out: Any, point_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {
        point_id: {
            "analysis": "missing from judge output",
            "correct": False,
        }
        for point_id in point_ids
    }
    if not isinstance(out, dict):
        return result
    data: Any = out.get("judgments")
    if data is None:
        data = out

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            point_id = str(item.get("point_id") or "").strip()
            if point_id not in result:
                continue
            result[point_id] = {
                "analysis": str(item.get("analysis") or "").strip(),
                "correct": bool(item.get("correct")),
            }
        return result

    if isinstance(data, dict):
        for point_id in point_ids:
            item = data.get(point_id)
            if not isinstance(item, dict):
                continue
            result[point_id] = {
                "analysis": str(item.get("analysis") or "").strip(),
                "correct": bool(item.get("correct")),
            }
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


def _ordered_judgments(
    slots: Sequence[Dict[str, Any]],
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    for slot in slots:
        point_id = str(slot.get("point_id") or "")
        judgment = judgments_by_point_id.get(point_id) or {
            "analysis": "missing from judge output",
            "correct": False,
        }
        ordered.append(
            {
                "point_id": point_id,
                "analysis": str(judgment.get("analysis") or ""),
                "correct": bool(judgment.get("correct")),
            }
        )
    return ordered


def _slot_score(judgments: List[Dict[str, Any]]) -> float:
    if not judgments:
        return 0.0
    return sum(1.0 if item.get("correct") else 0.0 for item in judgments) / len(judgments)


_SNAPSHOT_RESUME_FIELDS = (
    "snapshot_slot_eval_by_key",
    "snapshot_slot_judge_reason",
    "snapshot_point_score_mean_on_expected",
)
_CHANGE_RESUME_FIELDS = (
    "change_slot_eval_by_key",
    "change_slot_judge_reason",
    "before_point_score_mean_on_changed",
    "after_point_score_mean_on_changed",
    "change_state_predict_point_score_mean_on_changed",
    "change_reason_point_score_mean_on_changed",
)
_APPLY_RESUME_FIELDS = (
    "rq3_apply_slot_eval_by_item",
    "rq3_apply_answer_slot_judge_reason",
    "rq3_apply_answer_point_score_mean",
)


def _judgment_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        return 0
    return len(judgments)


def _snapshot_eval_matches(slots: Sequence[Dict[str, Any]], payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    slot_count = len(slots)
    return (
        int(payload.get("slot_count") or 0) == slot_count
        and _judgment_count(payload) == slot_count
    )


def _change_eval_matches(groups: Dict[str, List[Dict[str, Any]]], payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    before_slots = list(groups.get("before") or [])
    after_slots = list(groups.get("after") or [])
    reason_slots = list(groups.get("change_reason") or [])
    state_predict_slots = before_slots + after_slots
    required_groups = {
        "before": before_slots,
        "after": after_slots,
        "change_reason": reason_slots,
        "state_predict": state_predict_slots,
    }
    has_any = False
    for group_name, slots in required_groups.items():
        if not slots:
            continue
        has_any = True
        group_payload = payload.get(group_name)
        if not isinstance(group_payload, dict):
            return False
        if int(group_payload.get("slot_count") or 0) != len(slots):
            return False
        if _judgment_count(group_payload) != len(slots):
            return False
    return has_any


def _apply_eval_matches(slots: Sequence[Dict[str, Any]], payload: Any) -> bool:
    return _snapshot_eval_matches(slots, payload)


def _completed_snapshot_keys(row: Dict[str, Any]) -> Set[str]:
    slots_by_key = row.get("_snapshot_slots_by_key", {}) or {}
    existing = row.get("snapshot_slot_eval_by_key", {}) or {}
    completed: Set[str] = set()
    for key in sorted(slots_by_key.keys()):
        slots = list(slots_by_key.get(key) or [])
        if slots and _snapshot_eval_matches(slots, existing.get(key)):
            completed.add(str(key))
    return completed


def _completed_change_keys(row: Dict[str, Any]) -> Set[str]:
    slots_by_key = row.get("_change_slots_by_key", {}) or {}
    existing = row.get("change_slot_eval_by_key", {}) or {}
    completed: Set[str] = set()
    for key in sorted(slots_by_key.keys()):
        groups = slots_by_key.get(key) or {}
        if _change_eval_matches(groups, existing.get(key)):
            completed.add(str(key))
    return completed


def _completed_apply_items(row: Dict[str, Any]) -> Set[str]:
    slots_by_item = row.get("_rq3_apply_slots_by_item", {}) or {}
    existing = row.get("rq3_apply_slot_eval_by_item", {}) or {}
    completed: Set[str] = set()
    for item_id in sorted(slots_by_item.keys()):
        item = slots_by_item.get(item_id) or {}
        slots = list(item.get("slots") or [])
        if slots and _apply_eval_matches(slots, existing.get(item_id)):
            completed.add(str(item_id))
    return completed


def _merge_reason(existing_reason: Any, new_errors: Sequence[str]) -> str:
    parts: List[str] = []
    if str(existing_reason or "").strip():
        parts.append(str(existing_reason).strip())
    if new_errors:
        parts.append("; ".join(new_errors))
    return "; ".join(parts)


def _copy_resume_fields(
    row: Dict[str, Any],
    existing_row: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(existing_row, dict):
        return
    for field in _SNAPSHOT_RESUME_FIELDS + _CHANGE_RESUME_FIELDS + _APPLY_RESUME_FIELDS:
        if field in existing_row:
            row[field] = _json_safe(existing_row.get(field))


def _merge_existing_progress(
    checkpoint_rows: List[Dict[str, Any]],
    existing_result: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(existing_result, dict):
        return
    existing_by_checkpoint_id: Dict[str, Dict[str, Any]] = {}
    for row in existing_result.get("checkpoints", []):
        if not isinstance(row, dict):
            continue
        checkpoint_id = str(row.get("checkpoint_id") or "").strip()
        if checkpoint_id:
            existing_by_checkpoint_id[checkpoint_id] = row
    for row in checkpoint_rows:
        checkpoint_id = str(row.get("checkpoint_id") or "").strip()
        if checkpoint_id:
            _copy_resume_fields(row, existing_by_checkpoint_id.get(checkpoint_id))


def _snapshot_audit_record(
    *,
    checkpoint_id: str,
    state_key: str,
    slots: Sequence[Dict[str, Any]],
    prompt: str,
    raw_output: Any,
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "state_key": state_key,
        "slot_context": list(slots),
        "prompt": prompt,
        "raw_output": _json_safe(raw_output),
        "judgments": _ordered_judgments(slots, judgments_by_point_id),
    }


def _snapshot_eval_record(
    slots: Sequence[Dict[str, Any]],
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    judgments = _ordered_judgments(slots, judgments_by_point_id)
    return {
        "score_0_1": _slot_score(judgments),
        "slot_count": len(slots),
        "slot_context": list(slots),
        "judgments": judgments,
    }


def _change_audit_record(
    *,
    checkpoint_id: str,
    state_key: str,
    groups: Dict[str, List[Dict[str, Any]]],
    prompt: str,
    raw_output: Any,
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "state_key": state_key,
        "prompt": prompt,
        "raw_output": _json_safe(raw_output),
        "groups": {
            group_name: {
                "slot_context": list(group_slots),
                "judgments": _ordered_judgments(group_slots, judgments_by_point_id),
            }
            for group_name, group_slots in groups.items()
            if group_slots
        },
    }


def _change_eval_record(
    groups: Dict[str, List[Dict[str, Any]]],
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    before_slots = list(groups.get("before") or [])
    after_slots = list(groups.get("after") or [])
    reason_slots = list(groups.get("change_reason") or [])
    before = _ordered_judgments(before_slots, judgments_by_point_id)
    after = _ordered_judgments(after_slots, judgments_by_point_id)
    reason = _ordered_judgments(reason_slots, judgments_by_point_id)
    state_predict_slots = before_slots + after_slots
    state_predict = before + after
    record: Dict[str, Any] = {}
    if before:
        record["before"] = {
            "score_0_1": _slot_score(before),
            "slot_count": len(before_slots),
            "slot_context": before_slots,
            "judgments": before,
        }
    if after:
        record["after"] = {
            "score_0_1": _slot_score(after),
            "slot_count": len(after_slots),
            "slot_context": after_slots,
            "judgments": after,
        }
    if state_predict_slots:
        record["state_predict"] = {
            "score_0_1": _slot_score(state_predict),
            "slot_count": len(state_predict_slots),
            "slot_context": state_predict_slots,
            "judgments": state_predict,
        }
    if reason:
        record["change_reason"] = {
            "score_0_1": _slot_score(reason),
            "slot_count": len(reason_slots),
            "slot_context": reason_slots,
            "judgments": reason,
        }
    return record


def _apply_audit_record(
    *,
    checkpoint_id: str,
    item_id: str,
    state_key: str,
    qa_id: str,
    slots: Sequence[Dict[str, Any]],
    prompt: str,
    raw_output: Any,
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "item_id": item_id,
        "state_key": state_key,
        "qa_id": qa_id,
        "slot_context": list(slots),
        "prompt": prompt,
        "raw_output": _json_safe(raw_output),
        "judgments": _ordered_judgments(slots, judgments_by_point_id),
    }


def _apply_eval_record(
    item: Dict[str, Any],
    slots: Sequence[Dict[str, Any]],
    judgments_by_point_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    judgments = _ordered_judgments(slots, judgments_by_point_id)
    return {
        "state_key": str(item.get("state_key") or ""),
        "qa_id": str(item.get("qa_id") or ""),
        "score_0_1": _slot_score(judgments),
        "slot_count": len(slots),
        "slot_context": list(slots),
        "judgments": judgments,
    }


def _execute_slot_judge_requests(
    reqs: List[Dict[str, Any]],
    *,
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    desc: str,
    on_row_done: Optional[Callable[[], None]] = None,
    on_unit_done: Optional[Callable[[Dict[str, Any], Any, Dict[str, Dict[str, Any]], Optional[str]], None]] = None,
) -> Tuple[
    Dict[int, Dict[str, str]],
    Dict[int, Dict[str, Any]],
    Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    Dict[int, List[str]],
]:
    if not reqs:
        return {}, {}, {}, {}
    from .client import LLMClient

    client = LLMClient(
        provider=llm_provider,
        model_name=llm_model,
        max_workers=llm_max_workers,
        temperature=None,
    )
    error_count = 0
    deployment_not_found = False
    first_error_msg = ""
    prompts_by_row: Dict[int, Dict[str, str]] = {}
    raw_by_row: Dict[int, Dict[str, Any]] = {}
    judgments_by_row: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    errors_by_row: Dict[int, List[str]] = {}

    try:
        futures = []
        req_by_future: Dict[Any, Dict[str, Any]] = {}
        structured = client.supports_structured_response()
        for req in reqs:
            if structured:
                future = client.ask_structured_async(req["prompt"], text_format=req["text_format"])
            else:
                future = client.ask_async(req["prompt"], response_type="json")
            req_by_future[future] = req
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="unit"):
            req = req_by_future[future]
            row_index = int(req["row_index"])
            unit_id = str(req["unit_id"])
            point_ids = list(req.get("point_ids") or [])
            prompts_by_row.setdefault(row_index, {})[unit_id] = req["prompt"]
            raw_by_row.setdefault(row_index, {})
            judgments_by_row.setdefault(row_index, {})
            errors_by_row.setdefault(row_index, [])
            try:
                out = future.result()
                normalized = _normalize_slot_judgments(out, point_ids)
                raw_by_row[row_index][unit_id] = _json_safe(out)
                judgments_by_row[row_index][unit_id] = normalized
                if on_unit_done is not None:
                    on_unit_done(req, out, normalized, None)
            except Exception as exc:
                reason = str(exc)
                error_count += 1
                if not first_error_msg:
                    first_error_msg = reason
                if "DeploymentNotFound" in reason or "deployment" in reason.lower():
                    deployment_not_found = True
                raw_by_row[row_index][unit_id] = {"_error": reason}
                errors_by_row[row_index].append(f"{unit_id}: {reason}")
                normalized = {
                    point_id: {
                        "correct": False,
                        "analysis": f"judge_error: {reason}",
                    }
                    for point_id in point_ids
                }
                judgments_by_row[row_index][unit_id] = normalized
                if on_unit_done is not None:
                    on_unit_done(req, {"_error": reason}, normalized, reason)
            if on_row_done is not None:
                on_row_done()
    finally:
        client.close()

    if error_count == len(reqs) and error_count > 0:
        hint = (
            "All slot-level LLM judge calls failed. Check provider/model/deployment mapping. "
            "If you are using Azure, --llm-model must be your Azure deployment name "
            "(not necessarily the base model id)."
        )
        if deployment_not_found:
            hint += " Received DeploymentNotFound from provider."
        raise RuntimeError(f"{hint} First error: {first_error_msg}")
    return prompts_by_row, raw_by_row, judgments_by_row, errors_by_row


def _run_snapshot_slot_judge(
    rows: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    on_row_done: Optional[Callable[[], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reqs: List[Dict[str, Any]] = []
    completed_keys_by_row: Dict[int, Set[str]] = {}
    for row_idx, row in enumerate(rows):
        slots_by_key = row.get("_snapshot_slots_by_key", {}) or {}
        completed_keys = _completed_snapshot_keys(row)
        completed_keys_by_row[row_idx] = set(completed_keys)
        for req_idx, key in enumerate(sorted(slots_by_key.keys())):
            slots = list(slots_by_key.get(key) or [])
            if not slots or str(key) in completed_keys:
                continue
            reqs.append(
                {
                    "row_index": row_idx,
                    "unit_id": str(key),
                    "state_key": str(key),
                    "slots": slots,
                    "point_ids": [str(slot.get("point_id") or "") for slot in slots],
                    "prompt": build_snapshot_slot_judge_prompt(state_key=str(key), slots=slots),
                    "text_format": _build_slot_judge_text_format(
                        [str(slot.get("point_id") or "") for slot in slots],
                        row_idx * 10000 + req_idx,
                    ),
                }
            )
    def _on_unit_done(req: Dict[str, Any], _raw_output: Any, normalized: Dict[str, Dict[str, Any]], error: Optional[str]) -> None:
        row = rows[int(req["row_index"])]
        per_key = row.setdefault("snapshot_slot_eval_by_key", {})
        per_key[str(req["unit_id"])] = _snapshot_eval_record(req["slots"], normalized)
        row["snapshot_slot_judge_reason"] = _merge_reason(
            row.get("snapshot_slot_judge_reason"),
            [f"{req['unit_id']}: {error}"] if error else [],
        )
        scores = [
            float((payload or {}).get("score_0_1") or 0.0)
            for payload in (row.get("snapshot_slot_eval_by_key", {}) or {}).values()
            if isinstance(payload, dict)
        ]
        if scores:
            row["snapshot_point_score_mean_on_expected"] = _mean(scores)
    prompts_by_row, raw_by_row, judgments_by_row, errors_by_row = _execute_slot_judge_requests(
        reqs,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        desc="Task A Slot Judge",
        on_row_done=on_row_done,
        on_unit_done=_on_unit_done,
    )
    audit_records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        slots_by_key = row.get("_snapshot_slots_by_key", {}) or {}
        existing_per_key = row.get("snapshot_slot_eval_by_key", {}) or {}
        completed_keys = completed_keys_by_row.get(row_index, set())
        per_key: Dict[str, Dict[str, Any]] = {
            str(key): _json_safe(existing_per_key.get(key))
            for key in sorted(completed_keys)
        }
        scores: List[float] = [
            float((existing_per_key.get(key) or {}).get("score_0_1") or 0.0)
            for key in sorted(completed_keys)
        ]
        for key in sorted(slots_by_key.keys()):
            slots = list(slots_by_key.get(key) or [])
            if not slots or str(key) in completed_keys:
                continue
            normalized = ((judgments_by_row.get(row_index) or {}).get(key) or {})
            record = _snapshot_eval_record(slots, normalized)
            per_key[key] = record
            scores.append(float(record["score_0_1"]))
            audit_records.append(
                _snapshot_audit_record(
                    checkpoint_id=str(row.get("checkpoint_id") or ""),
                    state_key=str(key),
                    slots=slots,
                    prompt=str(((prompts_by_row.get(row_index) or {}).get(key) or "")),
                    raw_output=((raw_by_row.get(row_index) or {}).get(key)),
                    judgments_by_point_id=normalized,
                )
            )
        if per_key:
            row["snapshot_slot_eval_by_key"] = per_key
            row["snapshot_slot_judge_reason"] = _merge_reason(
                row.get("snapshot_slot_judge_reason"),
                errors_by_row.get(row_index, []),
            )
            row["snapshot_point_score_mean_on_expected"] = _mean(scores)
    return reqs, audit_records


def _run_change_slot_judge(
    rows: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    on_row_done: Optional[Callable[[], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reqs: List[Dict[str, Any]] = []
    completed_keys_by_row: Dict[int, Set[str]] = {}
    for row_idx, row in enumerate(rows):
        slots_by_key = row.get("_change_slots_by_key", {}) or {}
        completed_keys = _completed_change_keys(row)
        completed_keys_by_row[row_idx] = set(completed_keys)
        for req_idx, key in enumerate(sorted(slots_by_key.keys())):
            groups = slots_by_key.get(key) or {}
            slots = list(groups.get("before") or []) + list(groups.get("after") or []) + list(groups.get("change_reason") or [])
            if not slots or str(key) in completed_keys:
                continue
            reqs.append(
                {
                    "row_index": row_idx,
                    "unit_id": str(key),
                    "state_key": str(key),
                    "slots": slots,
                    "point_ids": [str(slot.get("point_id") or "") for slot in slots],
                    "prompt": build_change_slot_judge_prompt(state_key=str(key), slots=slots),
                    "text_format": _build_slot_judge_text_format(
                        [str(slot.get("point_id") or "") for slot in slots],
                        row_idx * 10000 + req_idx,
                    ),
                }
            )
    def _on_unit_done(req: Dict[str, Any], _raw_output: Any, normalized: Dict[str, Dict[str, Any]], error: Optional[str]) -> None:
        row = rows[int(req["row_index"])]
        state_key = str(req["unit_id"])
        groups = (row.get("_change_slots_by_key", {}) or {}).get(state_key) or {}
        record = _change_eval_record(groups, normalized)
        per_key = row.setdefault("change_slot_eval_by_key", {})
        per_key[state_key] = record
        row["change_slot_judge_reason"] = _merge_reason(
            row.get("change_slot_judge_reason"),
            [f"{state_key}: {error}"] if error else [],
        )
        before_scores: List[float] = []
        after_scores: List[float] = []
        state_predict_scores: List[float] = []
        reason_scores: List[float] = []
        for payload in (row.get("change_slot_eval_by_key", {}) or {}).values():
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("before"), dict):
                before_scores.append(float((payload.get("before") or {}).get("score_0_1") or 0.0))
            if isinstance(payload.get("after"), dict):
                after_scores.append(float((payload.get("after") or {}).get("score_0_1") or 0.0))
            if isinstance(payload.get("state_predict"), dict):
                state_predict_scores.append(float((payload.get("state_predict") or {}).get("score_0_1") or 0.0))
            if isinstance(payload.get("change_reason"), dict):
                reason_scores.append(float((payload.get("change_reason") or {}).get("score_0_1") or 0.0))
        if before_scores:
            row["before_point_score_mean_on_changed"] = _mean(before_scores)
        if after_scores:
            row["after_point_score_mean_on_changed"] = _mean(after_scores)
        if state_predict_scores:
            row["change_state_predict_point_score_mean_on_changed"] = _mean(state_predict_scores)
        if reason_scores:
            row["change_reason_point_score_mean_on_changed"] = _mean(reason_scores)
    prompts_by_row, raw_by_row, judgments_by_row, errors_by_row = _execute_slot_judge_requests(
        reqs,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        desc="Task B Slot Judge",
        on_row_done=on_row_done,
        on_unit_done=_on_unit_done,
    )
    audit_records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        slots_by_key = row.get("_change_slots_by_key", {}) or {}
        existing_per_key = row.get("change_slot_eval_by_key", {}) or {}
        completed_keys = completed_keys_by_row.get(row_index, set())
        per_key: Dict[str, Dict[str, Any]] = {
            str(key): _json_safe(existing_per_key.get(key))
            for key in sorted(completed_keys)
        }
        before_scores: List[float] = []
        after_scores: List[float] = []
        state_predict_scores: List[float] = []
        reason_scores: List[float] = []
        for key in sorted(completed_keys):
            record = existing_per_key.get(key) or {}
            if isinstance(record.get("before"), dict):
                before_scores.append(float((record.get("before") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("after"), dict):
                after_scores.append(float((record.get("after") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("state_predict"), dict):
                state_predict_scores.append(float((record.get("state_predict") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("change_reason"), dict):
                reason_scores.append(float((record.get("change_reason") or {}).get("score_0_1") or 0.0))
        for key in sorted(slots_by_key.keys()):
            groups = slots_by_key.get(key) or {}
            if str(key) in completed_keys:
                continue
            normalized = ((judgments_by_row.get(row_index) or {}).get(key) or {})
            before_slots = list(groups.get("before") or [])
            after_slots = list(groups.get("after") or [])
            reason_slots = list(groups.get("change_reason") or [])
            state_predict_slots = before_slots + after_slots
            record = _change_eval_record(groups, normalized)
            if isinstance(record.get("before"), dict):
                before_scores.append(float((record.get("before") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("after"), dict):
                after_scores.append(float((record.get("after") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("state_predict"), dict):
                state_predict_scores.append(float((record.get("state_predict") or {}).get("score_0_1") or 0.0))
            if isinstance(record.get("change_reason"), dict):
                reason_scores.append(float((record.get("change_reason") or {}).get("score_0_1") or 0.0))
            if record:
                per_key[key] = record
                audit_records.append(
                    _change_audit_record(
                        checkpoint_id=str(row.get("checkpoint_id") or ""),
                        state_key=str(key),
                        groups={
                            "before": before_slots,
                            "after": after_slots,
                            "state_predict": state_predict_slots,
                            "change_reason": reason_slots,
                        },
                        prompt=str(((prompts_by_row.get(row_index) or {}).get(key) or "")),
                        raw_output=((raw_by_row.get(row_index) or {}).get(key)),
                        judgments_by_point_id=normalized,
                    )
                )
        if per_key:
            row["change_slot_eval_by_key"] = per_key
            row["change_slot_judge_reason"] = _merge_reason(
                row.get("change_slot_judge_reason"),
                errors_by_row.get(row_index, []),
            )
            if before_scores:
                row["before_point_score_mean_on_changed"] = _mean(before_scores)
            if after_scores:
                row["after_point_score_mean_on_changed"] = _mean(after_scores)
            if state_predict_scores:
                row["change_state_predict_point_score_mean_on_changed"] = _mean(state_predict_scores)
            if reason_scores:
                row["change_reason_point_score_mean_on_changed"] = _mean(reason_scores)
    return reqs, audit_records


def _run_apply_slot_judge(
    rows: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    llm_max_workers: int,
    on_row_done: Optional[Callable[[], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reqs: List[Dict[str, Any]] = []
    completed_items_by_row: Dict[int, Set[str]] = {}
    for row_idx, row in enumerate(rows):
        slots_by_item = row.get("_rq3_apply_slots_by_item", {}) or {}
        completed_items = _completed_apply_items(row)
        completed_items_by_row[row_idx] = set(completed_items)
        for req_idx, item_id in enumerate(sorted(slots_by_item.keys())):
            item = slots_by_item.get(item_id) or {}
            slots = list(item.get("slots") or [])
            if not slots or str(item_id) in completed_items:
                continue
            reqs.append(
                {
                    "row_index": row_idx,
                    "unit_id": str(item_id),
                    "item_id": str(item_id),
                    "state_key": str(item.get("state_key") or ""),
                    "qa_id": str(item.get("qa_id") or ""),
                    "slots": slots,
                    "point_ids": [str(slot.get("point_id") or "") for slot in slots],
                    "prompt": build_apply_slot_judge_prompt(
                        state_key=str(item.get("state_key") or ""),
                        qa_id=str(item.get("qa_id") or ""),
                        question=str(item.get("question") or item.get("task_instruction") or ""),
                        reference_answer=str(item.get("reference_answer") or ""),
                        predicted_answer=str(item.get("predicted_answer") or ""),
                        service_family=str(item.get("service_family") or ""),
                        scenario=str(item.get("scenario") or ""),
                        task_instruction=str(item.get("task_instruction") or ""),
                        reference_output=item.get("reference_output"),
                        predicted_output=item.get("predicted_output"),
                        slots=slots,
                    ),
                    "text_format": _build_slot_judge_text_format(
                        [str(slot.get("point_id") or "") for slot in slots],
                        row_idx * 10000 + req_idx,
                    ),
                }
            )
    def _on_unit_done(req: Dict[str, Any], _raw_output: Any, normalized: Dict[str, Dict[str, Any]], error: Optional[str]) -> None:
        row = rows[int(req["row_index"])]
        item_id = str(req["unit_id"])
        item = (row.get("_rq3_apply_slots_by_item", {}) or {}).get(item_id) or {}
        record = _apply_eval_record(item, req["slots"], normalized)
        per_item = row.setdefault("rq3_apply_slot_eval_by_item", {})
        per_item[item_id] = record
        row["rq3_apply_answer_slot_judge_reason"] = _merge_reason(
            row.get("rq3_apply_answer_slot_judge_reason"),
            [f"{item_id}: {error}"] if error else [],
        )
        scores = [
            float((payload or {}).get("score_0_1") or 0.0)
            for payload in (row.get("rq3_apply_slot_eval_by_item", {}) or {}).values()
            if isinstance(payload, dict)
        ]
        if scores:
            row["rq3_apply_answer_point_score_mean"] = _mean(scores)
    prompts_by_row, raw_by_row, judgments_by_row, errors_by_row = _execute_slot_judge_requests(
        reqs,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        desc="Task C Slot Judge",
        on_row_done=on_row_done,
        on_unit_done=_on_unit_done,
    )
    audit_records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        slots_by_item = row.get("_rq3_apply_slots_by_item", {}) or {}
        existing_per_item = row.get("rq3_apply_slot_eval_by_item", {}) or {}
        completed_items = completed_items_by_row.get(row_index, set())
        per_item: Dict[str, Dict[str, Any]] = {
            str(item_id): _json_safe(existing_per_item.get(item_id))
            for item_id in sorted(completed_items)
        }
        scores: List[float] = [
            float((existing_per_item.get(item_id) or {}).get("score_0_1") or 0.0)
            for item_id in sorted(completed_items)
        ]
        for item_id in sorted(slots_by_item.keys()):
            item = slots_by_item.get(item_id) or {}
            slots = list(item.get("slots") or [])
            if not slots or str(item_id) in completed_items:
                continue
            normalized = ((judgments_by_row.get(row_index) or {}).get(item_id) or {})
            record = _apply_eval_record(item, slots, normalized)
            per_item[item_id] = record
            scores.append(float(record["score_0_1"]))
            audit_records.append(
                _apply_audit_record(
                    checkpoint_id=str(row.get("checkpoint_id") or ""),
                    item_id=str(item_id),
                    state_key=str(item.get("state_key") or ""),
                    qa_id=str(item.get("qa_id") or ""),
                    slots=slots,
                    prompt=str(((prompts_by_row.get(row_index) or {}).get(item_id) or "")),
                    raw_output=((raw_by_row.get(row_index) or {}).get(item_id)),
                    judgments_by_point_id=normalized,
                )
            )
        if per_item:
            row["rq3_apply_slot_eval_by_item"] = per_item
            row["rq3_apply_answer_slot_judge_reason"] = _merge_reason(
                row.get("rq3_apply_answer_slot_judge_reason"),
                errors_by_row.get(row_index, []),
            )
            row["rq3_apply_answer_point_score_mean"] = _mean(scores)
    return reqs, audit_records


def evaluate(
    benchmark: Dict[str, Any],
    raw_prediction: Any,
    *,
    enable_llm_judge: bool = False,
    save_eyeball: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-5-mini",
    llm_max_workers: int = 4,
    return_audit: bool = False,
    on_progress: Optional[Callable[[List[Dict[str, Any]], int], None]] = None,
    existing_result: Optional[Dict[str, Any]] = None,
) -> Any:
    checkpoint_rows, evaluated, align_report = evaluate_tce_rows(
        benchmark,
        raw_prediction,
        align_by_timestamp=True,
        include_internal_payload=(enable_llm_judge or save_eyeball),
    )
    _merge_existing_progress(checkpoint_rows, existing_result)

    snapshot_slot_judge_inputs: List[Dict[str, Any]] = []
    change_slot_judge_inputs: List[Dict[str, Any]] = []
    apply_slot_judge_inputs: List[Dict[str, Any]] = []
    slot_judge_audit: Dict[str, List[Dict[str, Any]]] = {"snapshot": [], "change": [], "apply": []}
    if enable_llm_judge:
        on_row_done: Optional[Callable[[], None]] = None
        if on_progress is not None:
            def on_row_done() -> None:
                on_progress(checkpoint_rows, evaluated)
        snapshot_slot_judge_inputs, slot_judge_audit["snapshot"] = _run_snapshot_slot_judge(
            checkpoint_rows,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_max_workers=llm_max_workers,
            on_row_done=on_row_done,
        )
        change_slot_judge_inputs, slot_judge_audit["change"] = _run_change_slot_judge(
            checkpoint_rows,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_max_workers=llm_max_workers,
            on_row_done=on_row_done,
        )
        apply_slot_judge_inputs, slot_judge_audit["apply"] = _run_apply_slot_judge(
            checkpoint_rows,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_max_workers=llm_max_workers,
            on_row_done=on_row_done,
        )
    result = build_tce_result_payload(
        benchmark,
        checkpoint_rows,
        evaluated,
        save_eyeball=save_eyeball,
        strip_internal_payload=True,
    )
    if return_audit:
        slot_judge_audit["snapshot_inputs"] = [
            {
                "checkpoint_id": checkpoint_rows[int(req.get("row_index", -1))].get("checkpoint_id")
                if isinstance(req.get("row_index"), int) and 0 <= int(req.get("row_index")) < len(checkpoint_rows)
                else None,
                "state_key": req.get("state_key"),
                "slot_context": req.get("slots", []),
            }
            for req in snapshot_slot_judge_inputs
        ]
        slot_judge_audit["change_inputs"] = [
            {
                "checkpoint_id": checkpoint_rows[int(req.get("row_index", -1))].get("checkpoint_id")
                if isinstance(req.get("row_index"), int) and 0 <= int(req.get("row_index")) < len(checkpoint_rows)
                else None,
                "state_key": req.get("state_key"),
                "slot_context": req.get("slots", []),
            }
            for req in change_slot_judge_inputs
        ]
        slot_judge_audit["apply_inputs"] = [
            {
                "checkpoint_id": checkpoint_rows[int(req.get("row_index", -1))].get("checkpoint_id")
                if isinstance(req.get("row_index"), int) and 0 <= int(req.get("row_index")) < len(checkpoint_rows)
                else None,
                "state_key": req.get("state_key"),
                "qa_id": req.get("qa_id"),
                "item_id": req.get("item_id"),
                "slot_context": req.get("slots", []),
            }
            for req in apply_slot_judge_inputs
        ]
        return result, align_report, slot_judge_audit
    return result, align_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TCE outputs.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to the pack-first TCE benchmark JSON (for example tce_benchmark_vnext_*task_packs*.json).",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Path to model predictions JSON (for example prediction/<run_name>/tce_results_v14_taskabc.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output eval JSON path (for example eval/<run_name>/tce_eval_v14_taskabc.json).",
    )
    parser.add_argument("--enable-llm-judge", action="store_true", help="Enable slot-level LLM judge scoring.")
    parser.add_argument(
        "--save-eyeball",
        action="store_true",
        help="Save per-checkpoint groundtruth/prediction payloads for manual eyeballing.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="Slot-level LLM judge provider.")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="Slot-level LLM judge model.")
    parser.add_argument("--llm-max-workers", type=int, default=4, help="Max workers for slot-level LLM judge.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial eval JSON at --output by skipping completed slot-judge units.",
    )
    parser.add_argument(
        "--llm-judge-input-output",
        type=Path,
        default=None,
        help="Optional JSON path to save exact slot-level LLM judge inputs for auditing.",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    raw_pred = json.loads(args.prediction.read_text(encoding="utf-8"))
    existing_result: Optional[Dict[str, Any]] = None
    if args.resume and args.output.exists():
        existing_result = json.loads(args.output.read_text(encoding="utf-8"))
    if args.enable_llm_judge:
        _validate_llm_judge_env(args.llm_provider)

    def _write_progress(rows: List[Dict[str, Any]], evaluated: int) -> None:
        output_rows: List[Dict[str, Any]] = []
        for row in rows:
            out_row = {
                k: _json_safe(v)
                for k, v in row.items()
                if k
                not in {
                    "_expected_snapshot",
                    "_pred_snapshot",
                    "_expected_evidence",
                    "_pred_evidence",
                    "_expected_change",
                    "_pred_change",
                    "_expected_change_evidence",
                    "_pred_change_evidence",
                    "_changed_keys",
                    "_expected_rq3",
                    "_pred_rq3",
                    "_snapshot_slots_by_key",
                    "_change_slots_by_key",
                    "_rq3_apply_slots_by_item",
                }
            }
            if args.save_eyeball:
                out_row["groundtruth_snapshot"] = _json_safe(row.get("_expected_snapshot", {}))
                out_row["prediction_snapshot"] = _json_safe(row.get("_pred_snapshot", {}))
                out_row["groundtruth_evidence"] = _json_safe(row.get("_expected_evidence", {}))
                out_row["prediction_evidence"] = _json_safe(row.get("_pred_evidence", {}))
                out_row["groundtruth_change"] = _json_safe(row.get("_expected_change", {}))
                out_row["prediction_change"] = _json_safe(row.get("_pred_change", {}))
                out_row["groundtruth_change_evidence"] = _json_safe(row.get("_expected_change_evidence", {}))
                out_row["prediction_change_evidence"] = _json_safe(row.get("_pred_change_evidence", {}))
                out_row["groundtruth_rq3_apply"] = _json_safe(row.get("_expected_rq3", {}))
                out_row["prediction_rq3_apply"] = _json_safe(row.get("_pred_rq3", {}))
            output_rows.append(out_row)
        payload = {
            "user_id": benchmark.get("user_id"),
            "benchmark_path": str(args.benchmark),
            "prediction_path": str(args.prediction),
            "total_checkpoints": benchmark.get("total_checkpoints", len(output_rows)),
            "evaluated_checkpoints": evaluated,
            "skipped_checkpoints": max(benchmark.get("total_checkpoints", 0) - evaluated, 0),
            "summary": mean_numeric_fields(
                [{k: v for k, v in row.items() if k != "checkpoint_id"} for row in output_rows]
            ),
            "checkpoints": output_rows,
        }
        payload.update(normalized_contract_metadata(benchmark))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_payload = None
    if args.llm_judge_input_output is not None:
        result, align_report, audit_payload = evaluate(
            benchmark,
            raw_pred,
            enable_llm_judge=args.enable_llm_judge,
            save_eyeball=args.save_eyeball,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_max_workers=args.llm_max_workers,
            return_audit=True,
            on_progress=_write_progress if args.enable_llm_judge else None,
            existing_result=existing_result,
        )
    else:
        result, align_report = evaluate(
            benchmark,
            raw_pred,
            enable_llm_judge=args.enable_llm_judge,
            save_eyeball=args.save_eyeball,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_max_workers=args.llm_max_workers,
            on_progress=_write_progress if args.enable_llm_judge else None,
            existing_result=existing_result,
        )
    result["prediction_alignment"] = align_report
    result["benchmark_path"] = str(args.benchmark)
    result["prediction_path"] = str(args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.llm_judge_input_output is not None:
        args.llm_judge_input_output.parent.mkdir(parents=True, exist_ok=True)
        args.llm_judge_input_output.write_text(
            json.dumps(audit_payload or {"snapshot": [], "change": [], "apply": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Saved: {args.output}")
    benchmark_contract_version = infer_task_contract_version(benchmark)
    summary_prefix = (
        "Summary: "
        f"checkpoints={result['evaluated_checkpoints']}, "
        f"snapshot_point={result['summary'].get('snapshot_point_score_mean_on_expected_mean', 0.0):.4f}, "
        f"value_f1={result['summary'].get('snapshot_value_f1_mean_on_expected_mean', 0.0):.4f}, "
        f"evidence_recall={result['summary'].get('snapshot_evidence_recall_mean_on_expected_mean', 0.0):.4f}, "
    )
    if task_contract_is_v2(benchmark_contract_version):
        print(
            summary_prefix
            + f"rq3_apply={result['summary'].get('rq3_apply_answer_point_score_mean_mean', 0.0):.4f}"
        )
    else:
        print(
            summary_prefix
            + f"change_state={result['summary'].get('change_state_predict_point_score_mean_on_changed_mean', 0.0):.4f}, "
            + f"change_reason={result['summary'].get('change_reason_point_score_mean_on_changed_mean', 0.0):.4f}, "
            + f"rq3_apply={result['summary'].get('rq3_apply_answer_point_score_mean_mean', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
