"""Shared TCE point-schema helpers for pack build and evaluation."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompts import (
    build_apply_rubric_validation_prompt,
    build_apply_rubric_rewrite_prompt,
    build_apply_answer_scoring_points_prompt,
    build_change_reason_rubric_rewrite_prompt,
    build_change_reason_rubric_validation_prompt,
    build_change_reason_scoring_points_prompt,
    build_value_micro_points_prompt,
    build_value_rubric_rewrite_prompt,
    build_value_rubric_validation_prompt,
)
from .task_spec import humanize_key

POINT_TYPE_FIELD = "field"
POINT_TYPE_LIST_ITEM = "list_item"
POINT_TYPE_MICRO = "micro"
POINT_ROLE_IDENTITY_GATE = "identity_gate"
POINT_POLARITY_POSITIVE = "positive"
POINT_POLARITY_NEGATIVE = "negative"
SCORING_POINTS_VERSION = "spv4"
MAX_MICRO_POINTS = 3
MAX_VALUE_RUBRIC_REWRITES = 2
_NEGATIVE_POINT_PREFIXES = ("do not ", "don't ", "avoid ", "must not ", "should not ", "never ")
_ALLOWED_POINT_FAIL_REASONS = {
    "unsupported",
    "drift",
    "not_atomic",
    "redundant",
    "over_specific",
}
_ALLOWED_SET_FAIL_REASONS = {"coverage_gap"}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_DATE_PATTERNS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d")
_TIME_PATTERNS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p")
_WEEKDAY_NAME_BY_INDEX = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}
_WEEKDAY_ALIASES = {
    "mon": "monday",
    "monday": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "tuesday": "tuesday",
    "wed": "wednesday",
    "wednesday": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "thursday": "thursday",
    "fri": "friday",
    "friday": "friday",
    "sat": "saturday",
    "saturday": "saturday",
    "sun": "sunday",
    "sunday": "sunday",
}
_COMPLEX_FIELD_NAMES = {
    "statement",
    "reason",
    "change_reason",
    "summary",
    "description",
    "details",
}


def _tokenize_text(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _collect_leaf_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for child in value.values():
            out.extend(_collect_leaf_texts(child))
        return out
    if isinstance(value, list):
        out: List[str] = []
        for child in value:
            out.extend(_collect_leaf_texts(child))
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    try:
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    except Exception:
        return [str(value)]


def _value_to_text(value: Any) -> str:
    leaves = [leaf.strip() for leaf in _collect_leaf_texts(value) if str(leaf).strip()]
    return " | ".join(leaves)


def _value_signature(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _text_f1(reference: Any, predicted: Any) -> float:
    ref_tokens = _tokenize_text(_value_to_text(reference))
    pred_tokens = _tokenize_text(_value_to_text(predicted))
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = 0
    pred_remaining: Dict[str, int] = {}
    for token in pred_tokens:
        pred_remaining[token] = pred_remaining.get(token, 0) + 1
    for token in ref_tokens:
        remain = pred_remaining.get(token, 0)
        if remain > 0:
            overlap += 1
            pred_remaining[token] = remain - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _map_similarity_to_binary_hit(similarity: float, point_type: str) -> int:
    normalized_type = str(point_type or "").strip().lower()
    threshold = 0.5 if normalized_type == POINT_TYPE_MICRO else 0.8
    return 1 if similarity >= threshold else 0


def _path_parts(target_path: str) -> List[str]:
    return [part.strip().lower() for part in str(target_path or "").split(".") if part.strip()]


def _is_date_path(target_path: str) -> bool:
    parts = _path_parts(target_path)
    return any(part in {"date", "dates", "schedule_dates"} for part in parts)


def _is_time_path(target_path: str) -> bool:
    parts = _path_parts(target_path)
    return any(part in {"time", "start_time", "end_time"} for part in parts)


def _is_weekday_path(target_path: str) -> bool:
    parts = _path_parts(target_path)
    return any(part in {"day_of_week", "days_of_week", "weekday", "weekdays"} for part in parts)


def _normalize_numeric_scalar(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return f"num:{int(numeric)}"
        return f"num:{numeric:.12g}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in {"true", "false"}:
            return f"bool:{text.lower()}"
        if re.fullmatch(r"[+-]?\d+", text):
            return f"num:{int(text)}"
        if re.fullmatch(r"[+-]?\d+\.\d+", text):
            numeric = float(text)
            if numeric.is_integer():
                return f"num:{int(numeric)}"
            return f"num:{numeric:.12g}"
    return None


def _normalize_date_scalar(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(text, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_time_scalar(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in _TIME_PATTERNS:
        try:
            return datetime.strptime(text.upper(), pattern).strftime("%H:%M")
        except ValueError:
            continue
    return None


def _normalize_weekday_scalar(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if float(value).is_integer() and numeric in _WEEKDAY_NAME_BY_INDEX:
            return _WEEKDAY_NAME_BY_INDEX[numeric]
        return None
    text = str(value or "").strip().lower()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        numeric = int(text)
        if numeric in _WEEKDAY_NAME_BY_INDEX:
            return _WEEKDAY_NAME_BY_INDEX[numeric]
        return None
    return _WEEKDAY_ALIASES.get(text)


def _typed_scalar_similarity(reference_value: Any, predicted_value: Any, target_path: str) -> Optional[float]:
    if _is_weekday_path(target_path):
        ref = _normalize_weekday_scalar(reference_value)
        pred = _normalize_weekday_scalar(predicted_value)
        if ref is None and pred is None:
            return None
        return 1.0 if ref is not None and ref == pred else 0.0
    if _is_date_path(target_path):
        ref = _normalize_date_scalar(reference_value)
        pred = _normalize_date_scalar(predicted_value)
        if ref is None and pred is None:
            return None
        return 1.0 if ref is not None and ref == pred else 0.0
    if _is_time_path(target_path):
        ref = _normalize_time_scalar(reference_value)
        pred = _normalize_time_scalar(predicted_value)
        if ref is None and pred is None:
            return None
        return 1.0 if ref is not None and ref == pred else 0.0
    ref = _normalize_numeric_scalar(reference_value)
    pred = _normalize_numeric_scalar(predicted_value)
    if ref is None and pred is None:
        return None
    return 1.0 if ref is not None and ref == pred else 0.0


def _scalar_similarity(reference_value: Any, predicted_value: Any, target_path: str) -> float:
    typed_similarity = _typed_scalar_similarity(reference_value, predicted_value, target_path)
    if typed_similarity is not None:
        return typed_similarity
    return _text_f1(reference_value, predicted_value)


def _ordered_list_item_match(
    reference_value: Any,
    predicted_value: Any,
    target_path: str,
    target_index: int,
) -> Tuple[float, Any]:
    if not isinstance(predicted_value, list):
        predicted_items = [] if predicted_value is None else [predicted_value]
    else:
        predicted_items = list(predicted_value)
    matched_item = predicted_items[target_index] if 0 <= target_index < len(predicted_items) else None
    similarity = _scalar_similarity(reference_value, matched_item, target_path)
    return similarity, matched_item


def _point_reason(*, similarity: float, polarity: str, reference_value: Any, predicted_value: Any) -> str:
    return (
        f"polarity={polarity}; similarity={similarity:.2f}; "
        f"reference={_value_to_text(reference_value)[:140]}; "
        f"predicted={_value_to_text(predicted_value)[:140]}"
    )


def _is_complex_string(path: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    tokens = _tokenize_text(value)
    if len(tokens) >= 10:
        return True
    if len(tokens) >= 4 and any(marker in value for marker in ("(", ")", ";", "—", " - ")):
        return True
    last = str(path or "").split(".")[-1].strip().lower()
    return last in _COMPLEX_FIELD_NAMES


def _normalize_generated_points(
    raw: Any,
    *,
    prefix: str,
    allow_negative: bool,
    max_points: int = MAX_MICRO_POINTS,
) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    points = raw.get("points")
    if not isinstance(points, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(points[: max(1, int(max_points))]):
        if not isinstance(item, dict):
            continue
        polarity = str(item.get("polarity") or POINT_POLARITY_POSITIVE).strip().lower()
        if polarity not in {POINT_POLARITY_POSITIVE, POINT_POLARITY_NEGATIVE}:
            polarity = POINT_POLARITY_POSITIVE
        if polarity == POINT_POLARITY_NEGATIVE and not allow_negative:
            polarity = POINT_POLARITY_POSITIVE
        point_text = str(item.get("point_text") or item.get("description") or "").strip()
        reference_value = item.get("reference_value")
        if not point_text:
            continue
        out.append(
            {
                "point_id": f"{prefix}_p{idx + 1}",
                "point_type": POINT_TYPE_MICRO,
                "polarity": polarity,
                "point_text": point_text,
                **({"reference_value": reference_value} if reference_value is not None else {}),
            }
        )
    return out


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _infer_micro_point_polarity(text: str, allow_negative: bool) -> str:
    normalized = str(text or "").strip().lower()
    if allow_negative and any(normalized.startswith(prefix) for prefix in _NEGATIVE_POINT_PREFIXES):
        return POINT_POLARITY_NEGATIVE
    return POINT_POLARITY_POSITIVE


def _normalize_atomic_rubric_points(
    raw: Any,
    *,
    prefix: str,
    allow_negative: bool,
    max_points: int = MAX_MICRO_POINTS,
) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rubric = raw.get("rubric")
    if not isinstance(rubric, list):
        rubric = raw.get("points")
    if not isinstance(rubric, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(rubric[: max(1, int(max_points))]):
        if isinstance(item, dict):
            point_text = str(item.get("point_text") or item.get("text") or item.get("description") or "").strip()
            polarity = str(item.get("polarity") or "").strip().lower() or _infer_micro_point_polarity(point_text, allow_negative)
        else:
            point_text = str(item or "").strip()
            polarity = _infer_micro_point_polarity(point_text, allow_negative)
        if not point_text:
            continue
        if polarity not in {POINT_POLARITY_POSITIVE, POINT_POLARITY_NEGATIVE}:
            polarity = POINT_POLARITY_POSITIVE
        if polarity == POINT_POLARITY_NEGATIVE and not allow_negative:
            polarity = POINT_POLARITY_POSITIVE
        out.append(
            {
                "point_id": f"{prefix}_p{idx + 1}",
                "point_type": POINT_TYPE_MICRO,
                "polarity": polarity,
                "point_text": point_text,
            }
        )
    return out


def _fallback_atomic_fact_from_text(*, prefix: str, text: str) -> List[Dict[str, Any]]:
    original = str(text or "").strip()
    if not original:
        return []
    return [
        {
            "point_id": f"{prefix}_p1",
            "point_type": POINT_TYPE_MICRO,
            "polarity": POINT_POLARITY_POSITIVE,
            "point_text": original,
        }
    ]


def build_atomic_rubric_scoring_points(
    *,
    rubric: Sequence[Any],
    prefix: str,
    allow_negative: bool,
    max_points: int = MAX_MICRO_POINTS,
) -> List[Dict[str, Any]]:
    return _normalize_atomic_rubric_points(
        {"rubric": list(rubric or [])},
        prefix=prefix,
        allow_negative=allow_negative,
        max_points=max_points,
    )


def _normalize_value_rubric_validation(
    raw: Any,
    *,
    expected_point_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], bool, List[str], List[str]]:
    schema_failures: List[str] = []
    if not isinstance(raw, dict):
        return [], False, [], ["llm_invalid"]
    raw_points = raw.get("points")
    raw_set_pass = raw.get("set_pass")
    raw_set_failures = raw.get("set_failures")
    if not isinstance(raw_points, list):
        return [], False, [], ["llm_invalid"]
    if not isinstance(raw_set_pass, bool):
        schema_failures.append("llm_invalid")
        raw_set_pass = False
    if not isinstance(raw_set_failures, list):
        schema_failures.append("llm_invalid")
        raw_set_failures = []

    expected_ids = [str(point_id or "").strip() for point_id in expected_point_ids if str(point_id or "").strip()]
    seen_ids = set()
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw_points:
        if not isinstance(item, dict):
            schema_failures.append("llm_invalid")
            continue
        point_id = str(item.get("point_id") or "").strip()
        if not point_id or point_id not in expected_ids or point_id in seen_ids:
            schema_failures.append("llm_invalid")
            continue
        raw_by_id[point_id] = item
        seen_ids.add(point_id)

    normalized_points: List[Dict[str, Any]] = []
    for point_id in expected_ids:
        item = raw_by_id.get(point_id)
        if not isinstance(item, dict):
            schema_failures.append("llm_invalid")
            normalized_points.append(
                {
                    "point_id": point_id,
                    "pass": False,
                    "analysis": "",
                    "fail_reasons": ["llm_invalid"],
                }
            )
            continue
        passed = item.get("pass")
        analysis = str(item.get("analysis") or "").strip()
        fail_reasons_raw = item.get("fail_reasons")
        if not isinstance(passed, bool):
            schema_failures.append("llm_invalid")
            passed = False
        if not analysis:
            schema_failures.append("llm_invalid")
        if not isinstance(fail_reasons_raw, list):
            schema_failures.append("llm_invalid")
            fail_reasons_raw = []
        normalized_fail_reasons: List[str] = []
        for reason in fail_reasons_raw:
            normalized = str(reason or "").strip()
            if not normalized:
                continue
            if normalized not in _ALLOWED_POINT_FAIL_REASONS:
                schema_failures.append("llm_invalid")
                continue
            if normalized not in normalized_fail_reasons:
                normalized_fail_reasons.append(normalized)
        normalized_points.append(
            {
                "point_id": point_id,
                "pass": bool(passed),
                "analysis": analysis,
                "fail_reasons": normalized_fail_reasons,
            }
        )

    set_failures: List[str] = []
    for reason in raw_set_failures:
        normalized = str(reason or "").strip()
        if not normalized:
            continue
        if normalized not in _ALLOWED_SET_FAIL_REASONS:
            schema_failures.append("llm_invalid")
            continue
        if normalized not in set_failures:
            set_failures.append(normalized)
    return normalized_points, bool(raw_set_pass), set_failures, _dedupe_preserve_order(schema_failures)


def _validate_value_rubric_points(
    *,
    state_key: str,
    text_value: str,
    points: Sequence[Dict[str, Any]],
    validator_client: Optional[Any],
) -> Dict[str, Any]:
    schema_valid, schema_failures = validate_scoring_points(points)
    if not schema_valid:
        return {
            "is_valid": False,
            "point_results": [],
            "set_pass": False,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": _dedupe_preserve_order(["rubric_invalid", *schema_failures]),
        }
    if validator_client is None:
        return {
            "is_valid": True,
            "point_results": [],
            "set_pass": True,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": [],
        }
    prompt = build_value_rubric_validation_prompt(
        state_key=state_key,
        text_value=text_value,
        points=list(points),
    )
    raw = None
    try:
        raw = validator_client.ask(prompt, response_type="json")
    except Exception:
        raw = None
    expected_point_ids = [str(point.get("point_id") or "").strip() for point in points if isinstance(point, dict)]
    point_results, set_pass, set_failures, normalize_failures = _normalize_value_rubric_validation(
        raw,
        expected_point_ids=expected_point_ids,
    )
    failed_point_ids = [
        str(result.get("point_id") or "").strip()
        for result in point_results
        if isinstance(result, dict) and not bool(result.get("pass"))
    ]
    failed_rules: List[str] = []
    if normalize_failures:
        failed_rules.extend(["rubric_invalid", *normalize_failures])
    for result in point_results:
        if not isinstance(result, dict) or bool(result.get("pass")):
            continue
        fail_reasons = list(result.get("fail_reasons") or [])
        if fail_reasons:
            failed_rules.extend(str(reason or "").strip() for reason in fail_reasons)
        else:
            failed_rules.append("rubric_invalid")
    failed_rules.extend(set_failures)
    failed_rules = _dedupe_preserve_order(failed_rules)
    is_valid = (
        not normalize_failures
        and bool(set_pass)
        and not set_failures
        and not failed_point_ids
    )
    return {
        "is_valid": is_valid,
        "point_results": point_results,
        "set_pass": bool(set_pass),
        "set_failures": set_failures,
        "failed_point_ids": failed_point_ids,
        "failed_rules": failed_rules,
    }


def _validate_change_reason_rubric_points(
    *,
    state_key: str,
    before_value: Any,
    after_value: Any,
    reference_change_reason: str,
    points: Sequence[Dict[str, Any]],
    validator_client: Optional[Any],
) -> Dict[str, Any]:
    schema_valid, schema_failures = validate_scoring_points(points)
    if not schema_valid:
        return {
            "is_valid": False,
            "point_results": [],
            "set_pass": False,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": _dedupe_preserve_order(["rubric_invalid", *schema_failures]),
        }
    if validator_client is None:
        return {
            "is_valid": True,
            "point_results": [],
            "set_pass": True,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": [],
        }
    prompt = build_change_reason_rubric_validation_prompt(
        state_key=state_key,
        before_value=before_value,
        after_value=after_value,
        reference_change_reason=reference_change_reason,
        points=list(points),
    )
    raw = None
    try:
        raw = validator_client.ask(prompt, response_type="json")
    except Exception:
        raw = None
    expected_point_ids = [str(point.get("point_id") or "").strip() for point in points if isinstance(point, dict)]
    point_results, set_pass, set_failures, normalize_failures = _normalize_value_rubric_validation(
        raw,
        expected_point_ids=expected_point_ids,
    )
    failed_point_ids = [
        str(result.get("point_id") or "").strip()
        for result in point_results
        if isinstance(result, dict) and not bool(result.get("pass"))
    ]
    failed_rules: List[str] = []
    if normalize_failures:
        failed_rules.extend(["rubric_invalid", *normalize_failures])
    for result in point_results:
        if not isinstance(result, dict) or bool(result.get("pass")):
            continue
        fail_reasons = list(result.get("fail_reasons") or [])
        if fail_reasons:
            failed_rules.extend(str(reason or "").strip() for reason in fail_reasons)
        else:
            failed_rules.append("rubric_invalid")
    failed_rules.extend(set_failures)
    failed_rules = _dedupe_preserve_order(failed_rules)
    is_valid = (
        not normalize_failures
        and bool(set_pass)
        and not set_failures
        and not failed_point_ids
    )
    return {
        "is_valid": is_valid,
        "point_results": point_results,
        "set_pass": bool(set_pass),
        "set_failures": set_failures,
        "failed_point_ids": failed_point_ids,
        "failed_rules": failed_rules,
    }


def validate_apply_answer_rubric_points(
    *,
    state_key: str,
    state_value: Any,
    service_category: str,
    question: str,
    reference_answer: str,
    points: Sequence[Dict[str, Any]],
    validator_client: Optional[Any],
) -> Dict[str, Any]:
    schema_valid, schema_failures = validate_scoring_points(points)
    if not schema_valid:
        return {
            "is_valid": False,
            "point_results": [],
            "set_pass": False,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": _dedupe_preserve_order(["rubric_invalid", *schema_failures]),
        }
    if validator_client is None:
        return {
            "is_valid": True,
            "point_results": [],
            "set_pass": True,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": [],
        }
    prompt = build_apply_rubric_validation_prompt(
        state_key=state_key,
        state_value=state_value,
        service_category=service_category,
        question=question,
        reference_answer=reference_answer,
        points=list(points),
    )
    raw = None
    try:
        raw = validator_client.ask(prompt, response_type="json")
    except Exception:
        raw = None
    expected_point_ids = [str(point.get("point_id") or "").strip() for point in points if isinstance(point, dict)]
    point_results, set_pass, set_failures, normalize_failures = _normalize_value_rubric_validation(
        raw,
        expected_point_ids=expected_point_ids,
    )
    failed_point_ids = [
        str(result.get("point_id") or "").strip()
        for result in point_results
        if isinstance(result, dict) and not bool(result.get("pass"))
    ]
    failed_rules: List[str] = []
    if normalize_failures:
        failed_rules.extend(["rubric_invalid", *normalize_failures])
    for result in point_results:
        if not isinstance(result, dict) or bool(result.get("pass")):
            continue
        fail_reasons = list(result.get("fail_reasons") or [])
        if fail_reasons:
            failed_rules.extend(str(reason or "").strip() for reason in fail_reasons)
        else:
            failed_rules.append("rubric_invalid")
    failed_rules.extend(set_failures)
    failed_rules = _dedupe_preserve_order(failed_rules)
    is_valid = (
        not normalize_failures
        and bool(set_pass)
        and not set_failures
        and not failed_point_ids
    )
    return {
        "is_valid": is_valid,
        "point_results": point_results,
        "set_pass": bool(set_pass),
        "set_failures": set_failures,
        "failed_point_ids": failed_point_ids,
        "failed_rules": failed_rules,
    }


def validate_scoring_points(points: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if not isinstance(points, Sequence) or not list(points):
        return False, ["points_empty"]
    seen_ids = set()
    normalized_points = [point for point in points if isinstance(point, dict)]
    if len(normalized_points) != len(list(points)):
        failures.append("points_non_object")
    for point in normalized_points:
        point_id = str(point.get("point_id") or "").strip()
        point_type = str(point.get("point_type") or "").strip().lower()
        polarity = str(point.get("polarity") or "").strip().lower()
        point_text = str(point.get("point_text") or "").strip()
        if not point_id:
            failures.append("missing_point_id")
        elif point_id in seen_ids:
            failures.append("duplicate_point_id")
        else:
            seen_ids.add(point_id)
        if point_type not in {POINT_TYPE_FIELD, POINT_TYPE_LIST_ITEM, POINT_TYPE_MICRO}:
            failures.append("invalid_point_type")
        if polarity and polarity not in {POINT_POLARITY_POSITIVE, POINT_POLARITY_NEGATIVE}:
            failures.append("invalid_polarity")
        if not point_text:
            failures.append("missing_point_text")
        if point_type in {POINT_TYPE_FIELD, POINT_TYPE_LIST_ITEM}:
            if not str(point.get("target_path") or "").strip():
                failures.append("missing_target_path")
            if "reference_value" not in point:
                failures.append("missing_reference_value")
    deduped: List[str] = []
    for failure in failures:
        if failure not in deduped:
            deduped.append(failure)
    return not deduped, deduped


def _make_field_point(*, prefix: str, path: str, value: Any, idx: int) -> Dict[str, Any]:
    return {
        "point_id": f"{prefix}_p{idx}",
        "point_type": POINT_TYPE_FIELD,
        "polarity": POINT_POLARITY_POSITIVE,
        "point_text": f"{path} should match the validated value",
        "target_path": path,
        "reference_value": value,
    }


def _make_list_item_point(*, prefix: str, path: str, value: Any, idx: int) -> Dict[str, Any]:
    return {
        "point_id": f"{prefix}_p{idx}",
        "point_type": POINT_TYPE_LIST_ITEM,
        "polarity": POINT_POLARITY_POSITIVE,
        "point_text": f"{path} should include this item",
        "target_path": path,
        "reference_value": value,
    }


def _attach_target_path_to_points(points: Sequence[Dict[str, Any]], target_path: str) -> List[Dict[str, Any]]:
    normalized_path = str(target_path or "").strip()
    out: List[Dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        copied = dict(point)
        if normalized_path:
            copied["target_path"] = normalized_path
        out.append(copied)
    return out


def build_value_scoring_points(
    *,
    state_key: str,
    value: Any,
    generator_client: Optional[Any] = None,
    validator_client: Optional[Any] = None,
    prefix: str = "scp",
) -> List[Dict[str, Any]]:
    counter = [0]

    def next_id() -> int:
        counter[0] += 1
        return counter[0]

    def walk(node: Any, path: str) -> List[Dict[str, Any]]:
        if _is_complex_string(path or state_key.split(":")[-1], node):
            return build_micro_points_for_value(
                state_key=state_key,
                text_value=str(node),
                generator_client=generator_client,
                validator_client=validator_client,
                prefix=f"{prefix}_{(path or 'current_value').replace('.', '_')}",
                target_path=path or "current_value",
            )
        if isinstance(node, dict):
            out: List[Dict[str, Any]] = []
            for raw_key, child in node.items():
                key = str(raw_key).strip().lower()
                if not key:
                    continue
                child_path = key if not path else f"{path}.{key}"
                if _is_complex_string(child_path, child):
                    out.extend(
                        build_micro_points_for_value(
                            state_key=state_key,
                            text_value=str(child),
                            generator_client=generator_client,
                            validator_client=validator_client,
                            prefix=f"{prefix}_{child_path.replace('.', '_')}",
                            target_path=child_path,
                        )
                    )
                    continue
                out.extend(walk(child, child_path))
            return out
        if isinstance(node, list):
            out: List[Dict[str, Any]] = []
            base_path = path or "current_value"
            for list_index, item in enumerate(node):
                child_path = f"{base_path}.{list_index}"
                if _is_complex_string(child_path, item):
                    out.extend(
                        build_micro_points_for_value(
                            state_key=state_key,
                            text_value=str(item),
                            generator_client=generator_client,
                            validator_client=validator_client,
                            prefix=f"{prefix}_{child_path.replace('.', '_')}",
                            target_path=child_path,
                        )
                    )
                    continue
                out.extend(walk(item, child_path))
            return out
        return [_make_field_point(prefix=prefix, path=path or "current_value", value=node, idx=next_id())]

    out = walk(value, "")
    valid, _ = validate_scoring_points(out)
    return out if valid else []


def build_micro_points_for_value(
    *,
    state_key: str,
    text_value: str,
    generator_client: Optional[Any],
    validator_client: Optional[Any],
    prefix: str,
    target_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prompt = build_value_micro_points_prompt(
        state_key=state_key,
        text_value=text_value,
        max_points=MAX_MICRO_POINTS,
    )
    candidate_points: List[Dict[str, Any]] = []
    raw = None
    if generator_client is not None:
        try:
            raw = generator_client.ask(prompt, response_type="json")
        except Exception:
            raw = None
        candidate_points = _normalize_atomic_rubric_points(raw, prefix=prefix, allow_negative=False)
        candidate_points = _attach_target_path_to_points(candidate_points, target_path or "")
    if not candidate_points:
        return _attach_target_path_to_points(
            _fallback_atomic_fact_from_text(prefix=prefix, text=text_value),
            target_path or "",
        )

    validation = _validate_value_rubric_points(
        state_key=state_key,
        text_value=text_value,
        points=candidate_points,
        validator_client=validator_client,
    )
    if validation.get("is_valid"):
        return candidate_points

    rewritten_points = candidate_points
    for _ in range(MAX_VALUE_RUBRIC_REWRITES):
        if generator_client is None:
            break
        rewrite_prompt = build_value_rubric_rewrite_prompt(
            state_key=state_key,
            text_value=text_value,
            points=rewritten_points,
            validation_points=list(validation.get("point_results") or []),
            set_failures=list(validation.get("set_failures") or []),
            max_points=MAX_MICRO_POINTS,
        )
        rewrite_raw = None
        try:
            rewrite_raw = generator_client.ask(rewrite_prompt, response_type="json")
        except Exception:
            rewrite_raw = None
        rewritten_candidate = _normalize_generated_points(
            rewrite_raw,
            prefix=prefix,
            allow_negative=False,
        )
        if not rewritten_candidate:
            rewritten_candidate = _normalize_atomic_rubric_points(
                rewrite_raw,
                prefix=prefix,
                allow_negative=False,
            )
        rewritten_candidate = _attach_target_path_to_points(rewritten_candidate, target_path or "")
        if not rewritten_candidate:
            break
        rewritten_points = rewritten_candidate
        validation = _validate_value_rubric_points(
            state_key=state_key,
            text_value=text_value,
            points=rewritten_points,
            validator_client=validator_client,
        )
        if validation.get("is_valid"):
            return rewritten_points

    return _attach_target_path_to_points(
        _fallback_atomic_fact_from_text(prefix=prefix, text=text_value),
        target_path or "",
    )


def build_change_reason_scoring_points(
    *,
    state_key: str,
    before_value: Any,
    after_value: Any,
    reference_change_reason: str,
    generator_client: Optional[Any],
    validator_client: Optional[Any] = None,
    prefix: str = "crp",
) -> List[Dict[str, Any]]:
    fallback_text = str(reference_change_reason or "").strip() or (
        f"{state_key} changed from {_value_signature(before_value)} to {_value_signature(after_value)}"
    )
    prompt = build_change_reason_scoring_points_prompt(
        state_key=state_key,
        before_value=before_value,
        after_value=after_value,
        reference_change_reason=reference_change_reason,
        max_points=MAX_MICRO_POINTS,
    )
    raw = None
    if generator_client is not None:
        try:
            raw = generator_client.ask(prompt, response_type="json")
        except Exception:
            raw = None
    points = _normalize_generated_points(raw, prefix=prefix, allow_negative=False)
    if not points:
        points = _normalize_atomic_rubric_points(raw, prefix=prefix, allow_negative=False)
    if not points:
        return _fallback_atomic_fact_from_text(prefix=prefix, text=fallback_text)
    validation = _validate_change_reason_rubric_points(
        state_key=state_key,
        before_value=before_value,
        after_value=after_value,
        reference_change_reason=reference_change_reason,
        points=points,
        validator_client=validator_client,
    )
    if validation.get("is_valid"):
        return points
    rewritten_points = points
    for _ in range(MAX_VALUE_RUBRIC_REWRITES):
        if generator_client is None:
            break
        rewrite_prompt = build_change_reason_rubric_rewrite_prompt(
            state_key=state_key,
            before_value=before_value,
            after_value=after_value,
            reference_change_reason=reference_change_reason,
            points=rewritten_points,
            validation_points=list(validation.get("point_results") or []),
            set_failures=list(validation.get("set_failures") or []),
            max_points=MAX_MICRO_POINTS,
        )
        rewrite_raw = None
        try:
            rewrite_raw = generator_client.ask(rewrite_prompt, response_type="json")
        except Exception:
            rewrite_raw = None
        rewritten_candidate = _normalize_generated_points(
            rewrite_raw,
            prefix=prefix,
            allow_negative=False,
        )
        if not rewritten_candidate:
            rewritten_candidate = _normalize_atomic_rubric_points(
                rewrite_raw,
                prefix=prefix,
                allow_negative=False,
            )
        if not rewritten_candidate:
            break
        rewritten_points = rewritten_candidate
        validation = _validate_change_reason_rubric_points(
            state_key=state_key,
            before_value=before_value,
            after_value=after_value,
            reference_change_reason=reference_change_reason,
            points=rewritten_points,
            validator_client=validator_client,
        )
        if validation.get("is_valid"):
            return rewritten_points
    return _fallback_atomic_fact_from_text(prefix=prefix, text=fallback_text)


def build_apply_answer_scoring_points(
    *,
    state_key: str,
    state_value: Any,
    apply_scenario: str,
    apply_question: str,
    apply_reference_answer: str,
    generator_client: Optional[Any],
    rubric: Optional[Sequence[Any]] = None,
    prefix: str = "aqp",
) -> List[Dict[str, Any]]:
    points = _build_apply_answer_scoring_points_candidate(
        state_key=state_key,
        state_value=state_value,
        apply_scenario=apply_scenario,
        apply_question=apply_question,
        apply_reference_answer=apply_reference_answer,
        generator_client=generator_client,
        rubric=rubric,
        prefix=prefix,
    )
    if points:
        return points
    return _fallback_atomic_fact_from_text(prefix=prefix, text=apply_reference_answer)


def _build_apply_answer_scoring_points_candidate(
    *,
    state_key: str,
    state_value: Any,
    apply_scenario: str,
    apply_question: str,
    apply_reference_answer: str,
    generator_client: Optional[Any],
    rubric: Optional[Sequence[Any]] = None,
    prefix: str = "aqp",
) -> List[Dict[str, Any]]:
    if rubric:
        points = build_atomic_rubric_scoring_points(
            rubric=rubric,
            prefix=prefix,
            allow_negative=True,
            max_points=MAX_MICRO_POINTS,
        )
        if points:
            return points
    prompt = build_apply_answer_scoring_points_prompt(
        state_key=state_key,
        state_value=state_value,
        apply_scenario=apply_scenario,
        apply_question=apply_question,
        apply_reference_answer=apply_reference_answer,
        max_points=MAX_MICRO_POINTS,
    )
    raw = None
    if generator_client is not None:
        try:
            raw = generator_client.ask(prompt, response_type="json")
        except Exception:
            raw = None
    points = _normalize_generated_points(raw, prefix=prefix, allow_negative=True)
    if not points:
        points = _normalize_atomic_rubric_points(raw, prefix=prefix, allow_negative=True)
    return points


def build_validated_apply_answer_scoring_points(
    *,
    state_key: str,
    state_value: Any,
    service_category: str,
    apply_scenario: str,
    apply_question: str,
    apply_reference_answer: str,
    generator_client: Optional[Any],
    validator_client: Optional[Any],
    rubric: Optional[Sequence[Any]] = None,
    prefix: str = "aqp",
    max_rewrites: int = MAX_VALUE_RUBRIC_REWRITES,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    points = _build_apply_answer_scoring_points_candidate(
        state_key=state_key,
        state_value=state_value,
        apply_scenario=apply_scenario,
        apply_question=apply_question,
        apply_reference_answer=apply_reference_answer,
        generator_client=generator_client,
        rubric=rubric,
        prefix=prefix,
    )
    validation: Dict[str, Any]
    if not points:
        validation = {
            "is_valid": False,
            "point_results": [],
            "set_pass": False,
            "set_failures": [],
            "failed_point_ids": [],
            "failed_rules": ["rubric_invalid", "points_empty"],
        }
    else:
        validation = validate_apply_answer_rubric_points(
            state_key=state_key,
            state_value=state_value,
            service_category=service_category,
            question=apply_question,
            reference_answer=apply_reference_answer,
            points=points,
            validator_client=validator_client,
        )
        if validation.get("is_valid"):
            return points, {
                **validation,
                "rewrite_attempts": 0,
                "used_safe_fallback": False,
            }

    rewritten_points = list(points)
    max_rewrites = max(0, int(max_rewrites))
    for attempt in range(max_rewrites):
        if generator_client is None:
            break
        rewrite_prompt = build_apply_rubric_rewrite_prompt(
            state_key=state_key,
            state_value=state_value,
            service_category=service_category,
            question=apply_question,
            reference_answer=apply_reference_answer,
            points=rewritten_points,
            validation_points=list(validation.get("point_results") or []),
            set_failures=list(validation.get("set_failures") or []),
            max_points=MAX_MICRO_POINTS,
        )
        rewrite_raw = None
        try:
            rewrite_raw = generator_client.ask(rewrite_prompt, response_type="json")
        except Exception:
            rewrite_raw = None
        rewritten_candidate = _normalize_generated_points(
            rewrite_raw,
            prefix=prefix,
            allow_negative=True,
        )
        if not rewritten_candidate:
            rewritten_candidate = _normalize_atomic_rubric_points(
                rewrite_raw,
                prefix=prefix,
                allow_negative=True,
            )
        if not rewritten_candidate:
            break
        rewritten_points = rewritten_candidate
        validation = validate_apply_answer_rubric_points(
            state_key=state_key,
            state_value=state_value,
            service_category=service_category,
            question=apply_question,
            reference_answer=apply_reference_answer,
            points=rewritten_points,
            validator_client=validator_client,
        )
        if validation.get("is_valid"):
            return rewritten_points, {
                **validation,
                "rewrite_attempts": attempt + 1,
                "used_safe_fallback": False,
            }

    fallback_points = _fallback_atomic_fact_from_text(prefix=prefix, text=apply_reference_answer)
    return fallback_points, {
        **validation,
        "rewrite_attempts": max_rewrites,
        "used_safe_fallback": True,
    }


def _task_c_v2_fill_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _task_c_v2_fill_template(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_task_c_v2_fill_template(child) for child in value]
    return "<fill>"


def _task_c_v2_leaf_paths(value: Any, path: str = "") -> List[Tuple[str, Any]]:
    if isinstance(value, dict):
        out: List[Tuple[str, Any]] = []
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            child_path = key if not path else f"{path}.{key}"
            out.extend(_task_c_v2_leaf_paths(child, child_path))
        return out
    if isinstance(value, list):
        out: List[Tuple[str, Any]] = []
        base_path = path or "current_value"
        for index, child in enumerate(value):
            out.extend(_task_c_v2_leaf_paths(child, f"{base_path}.{index}"))
        return out
    return [(path or "current_value", value)]


def _task_c_v2_state_field_paths(value: Any, path: str = "") -> List[Tuple[str, Any]]:
    if isinstance(value, dict):
        out: List[Tuple[str, Any]] = []
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if not key:
                continue
            child_path = key if not path else f"{path}.{key}"
            out.extend(_task_c_v2_state_field_paths(child, child_path))
        return out
    return [(path or "current_value", value)]


def _task_c_v2_normalize_rubric_path(path: Any) -> str:
    normalized = str(path or "").strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"\[(\d+)\]", r".\1", normalized)
    normalized = re.sub(r"\.+", ".", normalized).strip(".")
    return normalized


def _task_c_v2_render_rubric_path(path: str) -> str:
    parts = [part.strip() for part in str(path or "").split(".") if part.strip()]
    rendered = ""
    for part in parts:
        if part.isdigit():
            rendered += f"[{part}]"
        else:
            rendered = part if not rendered else f"{rendered}.{part}"
    return rendered


def _task_c_v2_match_source_field_path(
    source_leaf_paths: Sequence[Tuple[str, Any]],
    reference_value: Any,
) -> Optional[str]:
    matches = [path for path, value in source_leaf_paths if value == reference_value]
    if len(matches) == 1:
        return matches[0]
    return None


def _task_c_v2_field_point(
    *,
    prefix: str,
    idx: int,
    target_path: str,
    reference_value: Any,
    description: str,
    source_field_path: Optional[str] = None,
) -> Dict[str, Any]:
    point = {
        "point_id": f"{prefix}_p{idx}",
        "point_type": POINT_TYPE_FIELD,
        "point_text": description or f"The structured service output correctly fills {target_path}.",
        "output_field_path": target_path,
        "target_path": target_path,
        "reference_value": reference_value,
    }
    if source_field_path:
        point["source_field_path"] = source_field_path
    return point


def _task_c_v2_user_communication_point(
    *,
    prefix: str,
    idx: int,
    source_field_path: str,
    reference_value: Any,
) -> Dict[str, Any]:
    return {
        "point_id": f"{prefix}_p{idx}",
        "point_type": POINT_TYPE_MICRO,
        "point_text": _task_c_v2_user_communication_point_text(source_field_path, reference_value),
        "source_field_path": source_field_path,
        "reference_value": reference_value,
    }


def _task_c_v2_identity_gate_point(
    *,
    prefix: str,
    state_key: str,
) -> Dict[str, Any]:
    key_text = str(state_key or "").strip()
    if ":" in key_text:
        _category, state_name = key_text.split(":", 1)
    else:
        state_name = key_text
    state_label = state_name.replace("_", " ").strip() or humanize_key(key_text)
    return {
        "point_id": f"{prefix}_identity",
        "point_type": POINT_TYPE_MICRO,
        "point_role": POINT_ROLE_IDENTITY_GATE,
        "point_text": (
            f"The message is clearly about the {state_label} routine itself, "
            "not a different routine or unrelated task."
        ),
    }


def _task_c_v2_format_user_communication_value(path: str, value: Any) -> str:
    normalized_path = str(path or "").strip().lower()

    def _weekday_label(raw: Any) -> Optional[str]:
        if isinstance(raw, int) and raw in _WEEKDAY_NAME_BY_INDEX:
            return f"{raw} ({_WEEKDAY_NAME_BY_INDEX[raw].title()})"
        return None

    if normalized_path.endswith("days_of_week") and isinstance(value, list):
        rendered_days = [_weekday_label(item) or json.dumps(item, ensure_ascii=False) for item in value]
        return "[" + ", ".join(rendered_days) + "]"
    if normalized_path.endswith("day_of_week"):
        rendered_day = _weekday_label(value)
        if rendered_day:
            return rendered_day
    return json.dumps(value, ensure_ascii=False)


def _task_c_v2_user_communication_point_text(source_field_path: str, reference_value: Any) -> str:
    return (
        "The message correctly uses the state field "
        f"{source_field_path} with value "
        f"{_task_c_v2_format_user_communication_value(source_field_path, reference_value)}."
    )


def build_task_c_v2_user_communication_answer_scoring_points(
    *,
    state_key: str,
    state_value: Any,
    reference_answer: str,
    prefix: str = "aqp",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    del reference_answer
    original_state_key = state_key
    source_field_paths = _task_c_v2_state_field_paths(state_value)
    expected_ids = [path for path, _value in source_field_paths]
    source_value_by_path = {path: value for path, value in source_field_paths}
    failed_rules: List[str] = []
    points: List[Dict[str, Any]] = []
    uses_identity_gate = str(original_state_key or "").strip().startswith("habits_state:")

    if uses_identity_gate:
        points.append(
            _task_c_v2_identity_gate_point(
                prefix=prefix,
                state_key=original_state_key,
            )
        )

    for idx, source_field_path in enumerate(expected_ids, start=1):
        points.append(
            _task_c_v2_user_communication_point(
                prefix=prefix,
                idx=idx,
                source_field_path=source_field_path,
                reference_value=source_value_by_path.get(source_field_path),
            )
        )

    valid, point_failures = validate_scoring_points(points)
    for failure in point_failures:
        if failure not in failed_rules:
            failed_rules.append(failure)

    validation = {
        "is_valid": (not failed_rules) and valid and bool(points),
        "service_family": "user_communication",
        "failed_rules": failed_rules,
        "rewrite_attempts": 0,
        "uses_identity_gate": uses_identity_gate,
        "expected_source_field_paths": expected_ids,
    }
    return points, validation


def build_task_c_v2_structured_answer_scoring_points(
    *,
    state_key: str,
    state_value: Any,
    service_family: str,
    output_template: Any,
    reference_output: Any,
    prefix: str = "aqp",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    del state_key
    source_leaf_paths = _task_c_v2_leaf_paths(state_value)
    failed_rules: List[str] = []
    if not isinstance(output_template, dict) or not isinstance(reference_output, dict):
        failed_rules.append("not_structured_service_object")
        output_leaf_paths: List[Tuple[str, Any]] = []
    else:
        if output_template != _task_c_v2_fill_template(reference_output):
            failed_rules.append("output_template_mismatch")
        output_leaf_paths = _task_c_v2_leaf_paths(reference_output)
        if reference_output == state_value:
            failed_rules.append("raw_state_mirror")

    points: List[Dict[str, Any]] = []
    for idx, (output_path, output_value) in enumerate(output_leaf_paths, start=1):
        points.append(
            _task_c_v2_field_point(
                prefix=prefix,
                idx=idx,
                target_path=output_path,
                reference_value=output_value,
                description=f"The structured service output correctly fills {output_path}.",
                source_field_path=_task_c_v2_match_source_field_path(source_leaf_paths, output_value),
            )
        )

    valid, point_failures = validate_scoring_points(points)
    for failure in point_failures:
        if failure not in failed_rules:
            failed_rules.append(failure)

    validation = {
        "is_valid": (not failed_rules) and valid and bool(points),
        "service_family": str(service_family or ""),
        "failed_rules": failed_rules,
        "rewrite_attempts": 0,
        "expected_source_field_paths": [path for path, _ in source_leaf_paths],
        "expected_output_field_paths": [path for path, _ in output_leaf_paths],
    }
    return points, validation


def extract_value_at_path(value: Any, target_path: str) -> Any:
    path = str(target_path or "").strip().lower()
    if not path:
        return value
    parts = [part for part in path.split(".") if part]
    # Root-level values may still use the canonical alias `current_value`.
    if parts and parts[0] == "current_value" and not isinstance(value, dict):
        if len(parts) == 1:
            return value
        current = value
        parts = parts[1:]
    else:
        current = value
    for part in parts:
        if isinstance(current, dict):
            matched = None
            for key, child in current.items():
                if str(key).strip().lower() == part:
                    matched = child
                    break
            current = matched
            continue
        if isinstance(current, list):
            if not part.isdigit():
                return None
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
            return None
        return None
    return current


def score_points(points: Sequence[Dict[str, Any]], predicted_blob: Any) -> Tuple[float, List[Dict[str, Any]]]:
    judgments: List[Dict[str, Any]] = []
    score_sum = 0.0
    count = 0
    list_item_occurrence_by_path: Dict[str, int] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        point_id = str(point.get("point_id") or "").strip()
        point_type = str(point.get("point_type") or "").strip().lower()
        polarity = str(point.get("polarity") or POINT_POLARITY_POSITIVE).strip().lower()
        reference_value = point.get("reference_value", point.get("point_text"))
        target_path = str(point.get("target_path") or "")
        if point_type == POINT_TYPE_FIELD:
            predicted_value = extract_value_at_path(predicted_blob, target_path)
            similarity = _scalar_similarity(reference_value, predicted_value, target_path)
            predicted_for_reason = predicted_value
        elif point_type == POINT_TYPE_LIST_ITEM:
            predicted_value = extract_value_at_path(predicted_blob, target_path)
            target_index = list_item_occurrence_by_path.get(target_path, 0)
            list_item_occurrence_by_path[target_path] = target_index + 1
            similarity, predicted_for_reason = _ordered_list_item_match(
                reference_value,
                predicted_value,
                target_path,
                target_index,
            )
        else:
            predicted_value = predicted_blob
            similarity = _text_f1(reference_value, predicted_blob)
            predicted_for_reason = predicted_value
        raw_score = _map_similarity_to_binary_hit(similarity, point_type)
        if polarity == POINT_POLARITY_NEGATIVE:
            score = 1 - raw_score
        else:
            score = raw_score
        score_sum += float(score)
        count += 1
        judgments.append(
            {
                "point_id": point_id,
                "score_0_1": float(score),
                "reason": _point_reason(
                    similarity=similarity,
                    polarity=polarity,
                    reference_value=reference_value,
                    predicted_value=predicted_for_reason,
                ),
            }
        )
    return (score_sum / count) if count else 0.0, judgments
