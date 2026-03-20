"""Manual review sampling helpers for TCE Stage1/Stage2 audits."""

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


FIXED_SEED = 20260308


def _flat_validated_snapshot(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for category, items in (checkpoint.get("validated_snapshot_state") or {}).items():
        if not isinstance(items, dict):
            continue
        for state_name, value in items.items():
            out[f"{category}:{state_name}"] = value
    return out


def build_stage1_pool(validated_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checkpoints = validated_payload.get("checkpoints") or []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        expected_snapshot = checkpoint.get("expected_snapshot_state") or {}
        observability = checkpoint.get("state_observability") or {}
        state_questionability = checkpoint.get("state_questionability") or {}
        validated_flat = _flat_validated_snapshot(checkpoint)
        for state_key, qmeta in sorted(state_questionability.items()):
            if not isinstance(qmeta, dict):
                continue
            category, state_name = state_key.split(":", 1)
            if not bool(qmeta.get("l1_is_questionable")):
                bucket = "l1_fail"
            elif not bool(qmeta.get("l2_is_questionable")):
                bucket = "l2_fail"
            elif bool(qmeta.get("is_questionable")):
                bucket = "pass"
            else:
                continue
            row = {
                "sample_unit_id": f"stage1::{checkpoint_id}::{state_key}",
                "stage": "stage1",
                "bucket": bucket,
                "checkpoint_id": checkpoint_id,
                "state_key": state_key,
                "state_category": category,
                "changed_status": "na",
                "source_reason": str((qmeta.get("reason_codes") or [""])[0] or ""),
                "source_reasons": list(qmeta.get("reason_codes") or []),
                "expected_state_value": copy.deepcopy(
                    (expected_snapshot.get(category) or {}).get(state_name)
                ),
                "state_observability": copy.deepcopy((observability.get(category) or {}).get(state_name)),
                "state_questionability": copy.deepcopy(qmeta),
                "validated_state_value": copy.deepcopy(validated_flat.get(state_key)),
                "review_checklist_ref": f"stage1_{bucket}",
            }
            rows.append(row)
    return rows


def build_state_completion_pool(task_pack_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checkpoints = task_pack_payload.get("checkpoints") or []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        validated_flat = _flat_validated_snapshot(checkpoint)
        keys = ((checkpoint.get("state_completion_pack") or {}).get("keys") or {})
        for state_key, item in sorted(keys.items()):
            if not isinstance(item, dict):
                continue
            row = {
                "sample_unit_id": f"stage2_state_completion::{checkpoint_id}::{state_key}",
                "stage": "stage2_state_completion",
                "bucket": str(item.get("pack_source") or "unknown"),
                "checkpoint_id": checkpoint_id,
                "state_key": state_key,
                "state_category": state_key.split(":", 1)[0],
                "changed_status": "unchanged",
                "source_reason": str(item.get("pack_source") or ""),
                "source_reasons": [str(item.get("pack_source") or "")],
                "validated_state_value": copy.deepcopy(validated_flat.get(state_key)),
                "state_completion_item": copy.deepcopy(item),
                "review_checklist_ref": "stage2_state_completion",
            }
            rows.append(row)
    return rows


def build_change_tracking_pool(task_pack_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checkpoints = task_pack_payload.get("checkpoints") or []
    prev_flat: Optional[Dict[str, Any]] = None
    prev_checkpoint_id = ""
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        validated_flat = _flat_validated_snapshot(checkpoint)
        keys = ((checkpoint.get("change_tracking_pack") or {}).get("keys") or {})
        for state_key, item in sorted(keys.items()):
            if not isinstance(item, dict):
                continue
            row = {
                "sample_unit_id": f"stage2_change_tracking::{checkpoint_id}::{state_key}",
                "stage": "stage2_change_tracking",
                "bucket": "changed",
                "checkpoint_id": checkpoint_id,
                "state_key": state_key,
                "state_category": state_key.split(":", 1)[0],
                "changed_status": "changed",
                "source_reason": "changed",
                "source_reasons": ["changed"],
                "previous_checkpoint_id": prev_checkpoint_id,
                "before_validated_state_value": copy.deepcopy((prev_flat or {}).get(state_key)),
                "after_validated_state_value": copy.deepcopy(validated_flat.get(state_key)),
                "change_tracking_item": copy.deepcopy(item),
                "review_checklist_ref": "stage2_change_tracking",
            }
            rows.append(row)
        prev_flat = validated_flat
        prev_checkpoint_id = checkpoint_id
    return rows


def build_apply_pool(task_pack_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checkpoints = task_pack_payload.get("checkpoints") or []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        validated_flat = _flat_validated_snapshot(checkpoint)
        keys = ((checkpoint.get("rq3_apply_service_qa") or {}).get("keys") or {})
        for state_key, node in sorted(keys.items()):
            if not isinstance(node, dict):
                continue
            for item in node.get("items") or []:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "sample_unit_id": f"stage2_apply::{checkpoint_id}::{state_key}::{item.get('qa_id') or ''}::accepted",
                        "stage": "stage2_apply",
                        "bucket": "accepted",
                        "checkpoint_id": checkpoint_id,
                        "state_key": state_key,
                        "state_category": state_key.split(":", 1)[0],
                        "changed_status": "unchanged",
                        "source_reason": "accepted",
                        "source_reasons": ["accepted"],
                        "validated_state_value": copy.deepcopy(validated_flat.get(state_key)),
                        "apply_item": copy.deepcopy(item),
                        "risk_tag": "lookup_risk"
                        if _is_lookup_risk_state(validated_flat.get(state_key), state_key)
                        else "",
                        "review_checklist_ref": "stage2_apply_accepted",
                    }
                )
            for item in node.get("discarded_items") or []:
                if not isinstance(item, dict):
                    continue
                qa_validation = item.get("qa_validation") or {}
                atomic_fact_validation = item.get("atomic_fact_validation") or {}
                failed_rules = list(qa_validation.get("failed_rules") or atomic_fact_validation.get("failed_rules") or [])
                rows.append(
                    {
                        "sample_unit_id": f"stage2_apply::{checkpoint_id}::{state_key}::{item.get('qa_id') or ''}::discarded",
                        "stage": "stage2_apply",
                        "bucket": "discarded",
                        "checkpoint_id": checkpoint_id,
                        "state_key": state_key,
                        "state_category": state_key.split(":", 1)[0],
                        "changed_status": "unchanged",
                        "source_reason": str(failed_rules[0] if failed_rules else ""),
                        "source_reasons": failed_rules,
                        "validated_state_value": copy.deepcopy(validated_flat.get(state_key)),
                        "apply_item": copy.deepcopy(item),
                        "risk_tag": "lookup_risk"
                        if _is_lookup_risk_state(validated_flat.get(state_key), state_key)
                        else "",
                        "review_checklist_ref": "stage2_apply_discarded",
                    }
                )
    return rows


def _is_lookup_risk_state(state_value: Any, state_key: str) -> bool:
    if state_key.startswith("habits_state:"):
        return True
    if isinstance(state_value, dict) and "schedule_dates" in state_value:
        return True
    return False


def _sorted_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(rows),
        key=lambda row: (
            str(row.get("checkpoint_id") or ""),
            str(row.get("state_key") or ""),
            str(row.get("sample_unit_id") or ""),
        ),
    )


def _pick_one(
    candidates: Sequence[Dict[str, Any]],
    used: Set[str],
    rng: random.Random,
    *,
    predicate=None,
) -> Dict[str, Any]:
    available = [
        row
        for row in _sorted_candidates(candidates)
        if row.get("sample_unit_id") not in used and (predicate(row) if predicate is not None else True)
    ]
    if not available:
        raise ValueError("No candidate available for requested stratum")
    picked = rng.choice(available)
    used.add(str(picked.get("sample_unit_id")))
    return picked


def _pick_many(
    candidates: Sequence[Dict[str, Any]],
    count: int,
    used: Set[str],
    rng: random.Random,
    *,
    predicate=None,
) -> List[Dict[str, Any]]:
    available = [
        row
        for row in _sorted_candidates(candidates)
        if row.get("sample_unit_id") not in used and (predicate(row) if predicate is not None else True)
    ]
    if len(available) <= count:
        for row in available:
            used.add(str(row.get("sample_unit_id")))
        return available
    picked = rng.sample(available, count)
    for row in picked:
        used.add(str(row.get("sample_unit_id")))
    return _sorted_candidates(picked)


def _fill_count(
    base: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    target_count: int,
    used: Set[str],
    rng: random.Random,
    *,
    predicate=None,
) -> List[Dict[str, Any]]:
    if len(base) >= target_count:
        return base[:target_count]
    need = target_count - len(base)
    base.extend(_pick_many(candidates, need, used, rng, predicate=predicate))
    return base


def select_manual_review_samples(
    *,
    stage1_pool: Sequence[Dict[str, Any]],
    state_completion_pool: Sequence[Dict[str, Any]],
    change_tracking_pool: Sequence[Dict[str, Any]],
    apply_pool: Sequence[Dict[str, Any]],
    seed: int = FIXED_SEED,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    used: Set[str] = set()
    selected: List[Dict[str, Any]] = []

    # A1: Stage1 L1 fail (take all 5).
    l1_fail = _sorted_candidates([row for row in stage1_pool if row.get("bucket") == "l1_fail"])
    for row in l1_fail:
        used.add(str(row.get("sample_unit_id")))
    selected.extend(l1_fail)

    # A2: Stage1 L2 fail, one per category then fill to 5 with priority reasons and new checkpoints.
    l2_fail = [row for row in stage1_pool if row.get("bucket") == "l2_fail"]
    priority_reasons = {"insufficient_evidence", "partial_field_support"}
    l2_selected: List[Dict[str, Any]] = []
    available_categories = sorted({str(row.get("state_category") or "") for row in l2_fail})
    for category in available_categories:
        subset = [
            row
            for row in l2_fail
            if row.get("state_category") == category
            and (
                str(row.get("source_reason") or "") in priority_reasons
                or any(reason in priority_reasons for reason in row.get("source_reasons") or [])
            )
        ]
        if not subset:
            subset = [row for row in l2_fail if row.get("state_category") == category]
        l2_selected.append(_pick_one(subset, used, rng))
    while len(l2_selected) < 5 and len({row.get("checkpoint_id") for row in l2_selected}) < 3:
        extra = _pick_one(
            l2_fail,
            used,
            rng,
            predicate=lambda row: row.get("checkpoint_id")
            not in {picked.get("checkpoint_id") for picked in l2_selected},
        )
        l2_selected.append(extra)
    l2_selected = _fill_count(
        l2_selected,
        [
            row
            for row in l2_fail
            if str(row.get("source_reason") or "") in priority_reasons
            or any(reason in priority_reasons for reason in row.get("source_reasons") or [])
        ]
        or l2_fail,
        5,
        used,
        rng,
    )
    selected.extend(_sorted_candidates(l2_selected))

    # A3: Stage1 pass exact slots.
    stage1_pass = [row for row in stage1_pool if row.get("bucket") == "pass"]
    pass_slots = [
        ("cal_quarterly_001", "user_attributes_state", "scalar"),
        ("cal_quarterly_001", "habits_state", None),
        ("cal_quarterly_002", "user_attributes_state", None),
        ("cal_quarterly_002", "preferences_state", "statement"),
        ("cal_quarterly_003", "user_attributes_state", None),
        ("cal_quarterly_004", "habits_state", "schedule_dates"),
        ("cal_quarterly_004", "preferences_state", None),
        ("cal_quarterly_005", "user_attributes_state", None),
    ]
    for checkpoint_id, category, special in pass_slots:
        def _predicate(row: Dict[str, Any], cp=checkpoint_id, cat=category, tag=special) -> bool:
            if row.get("checkpoint_id") != cp or row.get("state_category") != cat:
                return False
            value = row.get("validated_state_value")
            if tag == "scalar":
                return not isinstance(value, dict)
            if tag == "schedule_dates":
                return isinstance(value, dict) and "schedule_dates" in value
            if tag == "statement":
                return isinstance(value, dict) and "statement" in value
            return True

        subset = [row for row in stage1_pass if _predicate(row)]
        if not subset:
            subset = [
                row
                for row in stage1_pass
                if row.get("checkpoint_id") == checkpoint_id and row.get("state_category") == category
            ]
        selected.append(_pick_one(subset, used, rng))

    # B: Stage2 state_completion
    state_completion_computed = [row for row in state_completion_pool if row.get("bucket") == "computed"]
    state_completion_reused = [row for row in state_completion_pool if row.get("bucket") == "reused"]
    sc_slots = [
        ("computed", "cal_quarterly_001", "user_attributes_state"),
        ("computed", "cal_quarterly_002", "habits_state"),
        ("computed", "cal_quarterly_004", "preferences_state"),
        ("computed", "cal_quarterly_005", "user_attributes_state"),
        ("reused", "cal_quarterly_002", "user_attributes_state"),
        ("reused", "cal_quarterly_003", "preferences_state"),
        ("reused", "cal_quarterly_004", "habits_state"),
        ("reused", "cal_quarterly_005", "user_attributes_state"),
    ]
    for bucket, checkpoint_id, category in sc_slots:
        source = state_completion_computed if bucket == "computed" else state_completion_reused
        subset = [
            row
            for row in source
            if row.get("checkpoint_id") == checkpoint_id and row.get("state_category") == category
        ]
        if not subset:
            subset = [
                row for row in source if row.get("checkpoint_id") == checkpoint_id
            ] or [row for row in source if row.get("state_category") == category]
        selected.append(_pick_one(subset, used, rng))

    # C: Stage2 change_tracking, 2 per cp2-cp5.
    change_slots = [
        ("cal_quarterly_002", "habits_state"),
        ("cal_quarterly_002", "preferences_state"),
        ("cal_quarterly_003", "user_attributes_state"),
        ("cal_quarterly_003", "habits_state"),
        ("cal_quarterly_004", "user_attributes_state"),
        ("cal_quarterly_004", "preferences_state"),
        ("cal_quarterly_005", "user_attributes_state"),
        ("cal_quarterly_005", "preferences_state"),
    ]
    for checkpoint_id, category in change_slots:
        subset = [
            row
            for row in change_tracking_pool
            if row.get("checkpoint_id") == checkpoint_id and row.get("state_category") == category
        ]
        if not subset:
            subset = [row for row in change_tracking_pool if row.get("checkpoint_id") == checkpoint_id]
        selected.append(_pick_one(subset, used, rng))

    # D1: Stage2 apply accepted.
    apply_accepted = [row for row in apply_pool if row.get("bucket") == "accepted"]
    accepted_slots = [
        ("cal_quarterly_001", "habits_state", "lookup_risk"),
        ("cal_quarterly_001", "user_attributes_state", None),
        ("cal_quarterly_002", "preferences_state", None),
        ("cal_quarterly_002", "user_attributes_state", None),
        ("cal_quarterly_003", "habits_state", "lookup_risk"),
        ("cal_quarterly_004", "preferences_state", None),
        ("cal_quarterly_005", "user_attributes_state", None),
        ("cal_quarterly_005", "user_attributes_state", None),
    ]
    for checkpoint_id, category, risk_tag in accepted_slots:
        subset = [
            row
            for row in apply_accepted
            if row.get("checkpoint_id") == checkpoint_id and row.get("state_category") == category
        ]
        if risk_tag:
            risk_subset = [row for row in subset if row.get("risk_tag") == risk_tag]
            if risk_subset:
                subset = risk_subset
        selected.append(_pick_one(subset, used, rng))

    # D2: Stage2 apply discarded.
    apply_discarded = [row for row in apply_pool if row.get("bucket") == "discarded"]
    discarded_slots = [
        ("service_decision_quality", "habits_state"),
        ("service_decision_quality", "habits_state"),
        ("service_decision_quality", "user_attributes_state"),
        ("personalization_necessity", "preferences_state"),
        ("answer_groundedness", "user_attributes_state"),
        ("question_too_long", None),
    ]
    for reason, category in discarded_slots:
        subset = [
            row
            for row in apply_discarded
            if reason in (row.get("source_reasons") or [])
            and (category is None or row.get("state_category") == category)
        ]
        if not subset and reason == "question_too_long":
            subset = [
                row
                for row in apply_discarded
                if "answer_groundedness" in (row.get("source_reasons") or [])
                or "answer_determinacy" in (row.get("source_reasons") or [])
            ]
        if not subset:
            subset = [
                row for row in apply_discarded if reason in (row.get("source_reasons") or [])
                or (
                    reason == "service_decision_quality"
                    and any(
                        legacy in (row.get("source_reasons") or [])
                        for legacy in ("service_actionability", "reasoning_required")
                    )
                )
                or (
                    reason == "answer_groundedness"
                    and "answer_determinacy" in (row.get("source_reasons") or [])
                )
            ] or list(apply_discarded)
        selected.append(_pick_one(subset, used, rng))

    if len(selected) != 48:
        raise ValueError(f"Expected 48 selected samples, got {len(selected)}")
    return _sorted_candidates(selected)


def build_review_sheet_rows(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        row = {
            "sample_id": f"S{idx:03d}",
            "stage": sample.get("stage"),
            "bucket": sample.get("bucket"),
            "checkpoint_id": sample.get("checkpoint_id"),
            "state_key": sample.get("state_key"),
            "state_category": sample.get("state_category"),
            "changed_status": sample.get("changed_status"),
            "source_reason": sample.get("source_reason"),
            "human_verdict": "",
            "issue_type": "",
            "severity": "",
            "notes": "",
            "action_recommendation": "",
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def review_plan_markdown() -> str:
    return """# TCE Stage1 + Stage2 Manual Review Plan

Seed: `20260308`

## Coverage
- Stage 1: `l1_fail / l2_fail / pass`
- Stage 2: `state_completion / change_tracking / apply`
- changed vs unchanged
- `user_attributes_state / habits_state / preferences_state`
- apply `accepted / discarded`

## Sample Counts
- Stage1 `l1_fail`: 5
- Stage1 `l2_fail`: 5
- Stage1 `pass`: 8
- Stage2 `state_completion`: 8
- Stage2 `change_tracking`: 8
- Stage2 `apply accepted`: 8
- Stage2 `apply discarded`: 6
- Total: 48

## Review Checklist Mapping
- `stage1_l1_fail`
  - Check deterministic reason code, evidence coverage, and whether L2 was correctly skipped.
- `stage1_l2_fail`
  - Check askable fields, field verdicts, evidence sufficiency, and false negatives.
- `stage1_pass`
  - Check validated value is neither overvalidated nor undervalidated.
- `stage2_state_completion`
  - Check question/template alignment and reuse correctness.
- `stage2_change_tracking`
  - Check changed-key selection and before/after correctness.
- `stage2_apply_accepted`
  - Check service actionability, personalization necessity, reasoning requirement, answer determinacy, and rubric/evidence contract health.
- `stage2_apply_discarded`
  - Check whether discard was justified and whether the item was recoverable.

## Recording Fields
- `sample_id`
- `stage`
- `bucket`
- `checkpoint_id`
- `state_key`
- `state_category`
- `changed_status`
- `source_reason`
- `human_verdict`
- `issue_type`
- `severity`
- `notes`
- `action_recommendation`
"""
