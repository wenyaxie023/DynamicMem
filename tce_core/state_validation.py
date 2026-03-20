"""Shared state-validation utilities for TCE benchmark preprocessing."""

import copy
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from .prompts import (
    build_change_reason_validation_prompt,
    build_state_questionability_validation_prompt,
)
from .questionability import evaluate_state_questionability
from .task_spec import flatten_snapshot

STATE_VALIDATOR_VERSION = "qv2_l1_l2"
STATE_VALIDATE_PROMPT_VERSION = "state_validate_prompt_v2"
CHANGE_REASON_VALIDATE_PROMPT_VERSION = "change_reason_validate_prompt_v1"


def bool_like(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        return default
    return bool(value)


def value_signature(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def flatten_observability(observability: Any) -> Dict[str, Any]:
    if not isinstance(observability, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in observability.keys()):
        return {str(k): v for k, v in observability.items()}
    flat: Dict[str, Any] = {}
    for category, content in observability.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def expand_snapshot(flat_snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    expanded: Dict[str, Dict[str, Any]] = {}
    for key, value in flat_snapshot.items():
        category, state_name = str(key).split(":", 1)
        expanded.setdefault(category, {})[state_name] = value
    return expanded


def extract_observable_valid_keys(checkpoint: Dict[str, Any]) -> List[str]:
    expected_flat = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
    obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
    out: List[str] = []
    for key in sorted(expected_flat.keys()):
        obs = obs_flat.get(key)
        if isinstance(obs, dict) and bool(obs.get("is_valid")):
            out.append(key)
    return out


def collect_leaf_paths(value: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(value, dict):
        for key, child_value in value.items():
            normalized_key = str(key).strip().lower()
            if not normalized_key:
                continue
            child_prefix = normalized_key if not prefix else f"{prefix}.{normalized_key}"
            out.extend(collect_leaf_paths(child_value, child_prefix))
        return out
    if isinstance(value, list):
        if prefix:
            out.append(prefix)
        elif value:
            out.append("current_value")
        return out
    if prefix:
        out.append(prefix)
    else:
        out.append("current_value")
    return out


def prune_value_by_paths(value: Any, keep_paths: Set[str], prefix: str = "") -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child_value in value.items():
            normalized_key = str(key).strip().lower()
            if not normalized_key:
                continue
            child_prefix = normalized_key if not prefix else f"{prefix}.{normalized_key}"
            has_any = any(path == child_prefix or path.startswith(child_prefix + ".") for path in keep_paths)
            if not has_any:
                continue
            pruned_child = prune_value_by_paths(child_value, keep_paths, child_prefix)
            if pruned_child is not None:
                out[key] = pruned_child
        return out if out else None
    if isinstance(value, list):
        return value if prefix in keep_paths else None
    if not prefix:
        return value
    return value if prefix in keep_paths else None


def collect_evidence_ids(obs: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    raw_ids = obs.get("evidence_app_log_ids")
    if isinstance(raw_ids, list):
        for item in raw_ids:
            sid = str(item).strip()
            if sid and sid not in out:
                out.append(sid)
    last_id = obs.get("last_app_log_id")
    if last_id is not None:
        sid = str(last_id).strip()
        if sid and sid not in out:
            out.append(sid)
    return out


def compute_evidence_signature(
    *,
    obs: Dict[str, Any],
    validator_version: str = STATE_VALIDATOR_VERSION,
    prompt_version: str = STATE_VALIDATE_PROMPT_VERSION,
) -> str:
    payload = {
        "evidence_app_log_ids": sorted(collect_evidence_ids(obs)),
        "last_app_log_id": str(obs.get("last_app_log_id", "") or ""),
        "is_valid": bool(obs.get("is_valid")),
        "validator_version": validator_version,
        "prompt_version": prompt_version,
    }
    return value_signature(payload)


def extract_last_change_reason(obs: Dict[str, Any]) -> str:
    return str((obs or {}).get("last_change_reason", "") or "").strip()


def compute_change_reason_signature(obs: Dict[str, Any]) -> str:
    return value_signature(extract_last_change_reason(obs))


def to_compact_json(value: Any, *, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def build_evidence_logs(
    *,
    obs: Dict[str, Any],
    app_logs_by_id: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    ids = collect_evidence_ids(obs)
    selected_ids = ids if int(top_k) <= 0 else ids[: int(top_k)]
    out: List[Dict[str, Any]] = []
    for sid in selected_ids:
        log = app_logs_by_id.get(sid)
        if not isinstance(log, dict):
            continue
        out.append(
            {
                "app_log_id": sid,
                "timestamp": str(log.get("timestamp", "")),
                "app_name": str(log.get("app_name", "")),
                "api_name": str(log.get("api_name", "")),
                "request": to_compact_json(log.get("request"), limit=700),
                "response": to_compact_json(log.get("response"), limit=1200),
            }
        )
    return out


def _call_json(client: Any, prompt: str) -> Any:
    try:
        return client.ask(prompt, response_type="json")
    except Exception as exc:
        return {"_error": str(exc)}


def validate_state_questionability_with_llm(
    *,
    validator_client: Any,
    state_key: str,
    state_value: Any,
    askable_fields: List[str],
    state_observability: Dict[str, Any],
    evidence_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = build_state_questionability_validation_prompt(
        state_key=state_key,
        state_value=state_value,
        askable_fields=askable_fields,
        state_observability=state_observability,
        evidence_logs=evidence_logs,
    )
    raw = _call_json(validator_client, prompt)
    if not isinstance(raw, dict):
        return {
            "is_questionable": False,
            "reason_codes": ["llm_validate_error"],
            "field_verdicts": [],
        }

    out_is_q = bool_like(raw.get("is_questionable"), default=False)
    out_reason_codes: List[str] = []
    for code in (raw.get("reason_codes") or []):
        scode = str(code).strip()
        if scode:
            out_reason_codes.append(scode)
    out_field_verdicts: List[Dict[str, Any]] = []
    raw_field_verdicts = raw.get("field_verdicts")
    if not isinstance(raw_field_verdicts, list):
        raw_field_verdicts = raw.get("field_validations")
    if isinstance(raw_field_verdicts, list):
        for item in raw_field_verdicts:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("field_name") or item.get("field_path") or "").strip().lower()
            if not field_name:
                continue
            out_field_verdicts.append(
                {
                    "field_name": field_name,
                    "reason_analysis": str(item.get("reason_analysis") or item.get("reason") or "").strip(),
                    "is_valid": bool_like(
                        item.get("is_valid"),
                        default=bool_like(item.get("is_questionable"), default=False),
                    ),
                }
            )
    if not out_is_q and not out_reason_codes:
        out_reason_codes.append("llm_not_questionable")
    return {
        "is_questionable": out_is_q,
        "reason_codes": out_reason_codes,
        "field_verdicts": out_field_verdicts,
    }


def validate_change_reason_with_llm(
    *,
    validator_client: Any,
    state_key: str,
    state_value: Any,
    change_reason: str,
    state_observability: Dict[str, Any],
    evidence_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    exists = bool(str(change_reason or "").strip())
    if not exists:
        return {
            "exists": False,
            "reason_analysis": "",
            "is_valid": False,
            "reason_codes": [],
        }
    prompt = build_change_reason_validation_prompt(
        state_key=state_key,
        state_value=state_value,
        change_reason=change_reason,
        state_observability=state_observability,
        evidence_logs=evidence_logs,
    )
    raw = _call_json(validator_client, prompt)
    if not isinstance(raw, dict):
        return {
            "exists": True,
            "reason_analysis": "",
            "is_valid": False,
            "reason_codes": ["llm_validate_error"],
        }
    out_exists = bool_like(raw.get("exists"), default=True)
    out_reason_analysis = str(raw.get("reason_analysis") or raw.get("analysis") or "").strip()
    out_is_valid = bool_like(raw.get("is_valid"), default=False)
    out_reason_codes: List[str] = []
    for code in list(raw.get("reason_codes") or []):
        scode = str(code).strip()
        if scode:
            out_reason_codes.append(scode)
    if out_exists and not out_is_valid and not out_reason_codes:
        out_reason_codes.append("change_reason_not_supported")
    return {
        "exists": out_exists,
        "reason_analysis": out_reason_analysis,
        "is_valid": out_exists and out_is_valid,
        "reason_codes": out_reason_codes,
    }


def _build_l1_state_payload(
    *,
    state_key: str,
    state_value: Any,
    obs: Optional[Dict[str, Any]] = None,
    app_logs_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    qmeta_l1 = evaluate_state_questionability(
        state_key=state_key,
        state_value=state_value,
        state_observability=obs,
        app_logs_by_id=app_logs_by_id,
        validator_version="qv1",
    )
    candidate_field_paths: List[str] = []
    seen_field_paths = set()
    for path in list(qmeta_l1.get("askable_fields") or []) + collect_leaf_paths(state_value):
        spath = str(path).strip().lower()
        if not spath or spath in seen_field_paths:
            continue
        seen_field_paths.add(spath)
        candidate_field_paths.append(spath)
    return qmeta_l1, candidate_field_paths


def _build_change_reason_validation_payload(
    *,
    state_key: str,
    state_value: Any,
    obs: Dict[str, Any],
    validator_client: Any,
    app_logs_by_id: Dict[str, Dict[str, Any]],
    l2_evidence_top_k: int,
) -> Optional[Dict[str, Any]]:
    change_reason = extract_last_change_reason(obs)
    if not change_reason:
        return None
    evidence_logs = build_evidence_logs(
        obs=obs,
        app_logs_by_id=app_logs_by_id,
        top_k=l2_evidence_top_k,
    )
    if validator_client is None:
        return {
            "exists": True,
            "reason_analysis": "",
            "is_valid": True,
            "reason_codes": [],
        }
    return validate_change_reason_with_llm(
        validator_client=validator_client,
        state_key=state_key,
        state_value=state_value,
        change_reason=change_reason,
        state_observability=obs,
        evidence_logs=evidence_logs,
    )


def _build_final_questionability_payload(
    *,
    state_key: str,
    state_value: Any,
    obs: Dict[str, Any],
    validator_client: Any,
    app_logs_by_id: Dict[str, Dict[str, Any]],
    l2_evidence_top_k: int,
    qmeta_l1: Optional[Dict[str, Any]] = None,
    candidate_field_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if qmeta_l1 is None or candidate_field_paths is None:
        qmeta_l1, candidate_field_paths = _build_l1_state_payload(
            state_key=state_key,
            state_value=state_value,
            obs=obs,
            app_logs_by_id=app_logs_by_id,
        )

    l1_is_questionable = bool(qmeta_l1.get("is_questionable"))
    evidence_logs = build_evidence_logs(
        obs=obs,
        app_logs_by_id=app_logs_by_id,
        top_k=l2_evidence_top_k,
    )
    if l1_is_questionable:
        qmeta_l2 = validate_state_questionability_with_llm(
            validator_client=validator_client,
            state_key=state_key,
            state_value=state_value,
            askable_fields=candidate_field_paths,
            state_observability=obs,
            evidence_logs=evidence_logs,
        )
    else:
        qmeta_l2 = {
            "is_questionable": False,
            "reason_codes": ["l2_skipped_due_to_l1_fail"],
            "field_verdicts": [],
        }

    merged_reason_codes: List[str] = []
    for code in list(qmeta_l1.get("reason_codes") or []) + list(qmeta_l2.get("reason_codes") or []):
        scode = str(code).strip()
        if scode and scode not in merged_reason_codes:
            merged_reason_codes.append(scode)

    l2_is_questionable = bool_like(qmeta_l2.get("is_questionable"), default=False)
    field_verdicts = qmeta_l2.get("field_verdicts") if isinstance(qmeta_l2.get("field_verdicts"), list) else []
    allowed_fields = {
        str(item.get("field_name") or item.get("field_path") or "").strip().lower()
        for item in field_verdicts
        if isinstance(item, dict)
        and bool_like(item.get("is_valid"), default=bool_like(item.get("is_questionable"), default=False))
    }
    if allowed_fields:
        validated_field_paths = [path for path in candidate_field_paths if path in allowed_fields]
    else:
        validated_field_paths = list(candidate_field_paths) if l2_is_questionable else []
    dropped_field_paths = [path for path in candidate_field_paths if path not in set(validated_field_paths)]
    validated_state_value = prune_value_by_paths(state_value, set(validated_field_paths))
    if validated_state_value is None:
        validated_state_value = {}
    change_reason_validation = _build_change_reason_validation_payload(
        state_key=state_key,
        state_value=state_value,
        obs=obs,
        validator_client=validator_client,
        app_logs_by_id=app_logs_by_id,
        l2_evidence_top_k=l2_evidence_top_k,
    )
    is_questionable_final = l1_is_questionable and l2_is_questionable and bool(validated_field_paths)
    out = {
        "is_questionable": is_questionable_final,
        "l1_is_questionable": l1_is_questionable,
        "l2_is_questionable": l2_is_questionable,
        "reason_codes": merged_reason_codes,
        "askable_fields": candidate_field_paths,
        "validated_field_paths": validated_field_paths,
        "dropped_field_paths": dropped_field_paths,
        "validated_state_value": validated_state_value,
        "field_verdicts": field_verdicts,
        "validator_version": STATE_VALIDATOR_VERSION,
        "validation_source": "computed",
    }
    if change_reason_validation is not None:
        out["change_reason_validation"] = change_reason_validation
    return out


def _normalize_saved_questionability_entry(
    *,
    state_key: str,
    state_value: Any,
    obs: Dict[str, Any],
    qmeta: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(qmeta)
    askable_fields = [str(path).strip().lower() for path in list(out.get("askable_fields") or []) if str(path).strip()]
    if not askable_fields:
        askable_fields = collect_leaf_paths(state_value)
    validated_field_paths = [
        str(path).strip().lower() for path in list(out.get("validated_field_paths") or []) if str(path).strip()
    ]
    if not validated_field_paths:
        validated_field_paths = collect_leaf_paths(out.get("validated_state_value"))
    dropped_field_paths = [
        str(path).strip().lower() for path in list(out.get("dropped_field_paths") or []) if str(path).strip()
    ]
    if not dropped_field_paths:
        dropped_field_paths = [path for path in askable_fields if path not in set(validated_field_paths)]

    out["askable_fields"] = askable_fields
    out["validated_field_paths"] = validated_field_paths
    out["dropped_field_paths"] = dropped_field_paths
    out["validated_state_value"] = out.get("validated_state_value") or {}
    out["reason_codes"] = [str(code).strip() for code in list(out.get("reason_codes") or []) if str(code).strip()]
    out["field_verdicts"] = list(out.get("field_verdicts") or [])
    change_reason = extract_last_change_reason(obs)
    raw_change_reason_validation = (
        out.get("change_reason_validation")
        if isinstance(out.get("change_reason_validation"), dict)
        else None
    )
    if change_reason or raw_change_reason_validation is not None:
        if isinstance(raw_change_reason_validation, dict):
            out["change_reason_validation"] = {
                "exists": bool_like(raw_change_reason_validation.get("exists"), default=bool(change_reason)),
                "reason_analysis": str(
                    raw_change_reason_validation.get("reason_analysis")
                    or raw_change_reason_validation.get("analysis")
                    or ""
                ).strip(),
                "is_valid": bool_like(raw_change_reason_validation.get("is_valid"), default=False),
                "reason_codes": [
                    str(code).strip()
                    for code in list(raw_change_reason_validation.get("reason_codes") or [])
                    if str(code).strip()
                ],
            }
        elif change_reason:
            out["change_reason_validation"] = {
                "exists": True,
                "reason_analysis": "",
                "is_valid": False,
                "reason_codes": ["change_reason_validation_missing"],
            }
    out.pop("llm_reason", None)
    out["validator_version"] = str(out.get("validator_version") or STATE_VALIDATOR_VERSION)
    validation_source = str(out.get("validation_source") or "computed").strip().lower()
    out["validation_source"] = validation_source if validation_source in {"computed", "reused"} else "computed"
    out["l1_is_questionable"] = bool_like(
        out.get("l1_is_questionable"),
        default=bool_like(out.get("is_questionable"), default=False),
    )
    out["l2_is_questionable"] = bool_like(
        out.get("l2_is_questionable"),
        default=bool_like(out.get("is_questionable"), default=False),
    )
    out["is_questionable"] = bool_like(out.get("is_questionable"), default=False) and bool(validated_field_paths)

    validation_identity = out.get("validation_identity") if isinstance(out.get("validation_identity"), dict) else {}
    if not validation_identity:
        validation_identity = {
            "state_key": state_key,
            "validated_state_value_signature": value_signature(out.get("validated_state_value")),
            "evidence_signature": compute_evidence_signature(obs=obs),
            "validator_version": out["validator_version"],
            "prompt_version": STATE_VALIDATE_PROMPT_VERSION,
            "change_reason_signature": compute_change_reason_signature(obs),
            "change_reason_prompt_version": CHANGE_REASON_VALIDATE_PROMPT_VERSION,
        }
    else:
        validation_identity.setdefault("change_reason_signature", compute_change_reason_signature(obs))
        validation_identity.setdefault("change_reason_prompt_version", CHANGE_REASON_VALIDATE_PROMPT_VERSION)
    out["validation_identity"] = validation_identity
    return out


def _build_checkpoint_summary(
    *,
    keys: List[str],
    questionability_map: Dict[str, Any],
    after_l1_count: int,
) -> Tuple[int, Dict[str, Any], Dict[str, int]]:
    validated_snapshot_flat: Dict[str, Any] = {}
    processed_count = 0
    cp_l2_pass = 0
    cp_final_pass = 0
    cp_reused = 0
    cp_computed = 0

    for key in keys:
        qmeta = questionability_map.get(key)
        if not isinstance(qmeta, dict):
            continue
        processed_count += 1
        l2_is_questionable = bool_like(qmeta.get("l2_is_questionable"), default=False)
        is_questionable_final = bool_like(qmeta.get("is_questionable"), default=False)
        if l2_is_questionable:
            cp_l2_pass += 1
        if is_questionable_final:
            cp_final_pass += 1
            validated_snapshot_flat[key] = qmeta.get("validated_state_value")
        if str(qmeta.get("validation_source") or "").strip().lower() == "reused":
            cp_reused += 1
        else:
            cp_computed += 1

    summary = {
        "pre_validate_count": int(len(keys)),
        "after_l1_count": int(after_l1_count),
        "after_l2_count": int(cp_l2_pass),
        "after_l1_l2_count": int(cp_final_pass),
        "reused_count": int(cp_reused),
        "computed_count": int(cp_computed),
    }
    return processed_count, validated_snapshot_flat, summary


def _needs_state_l2_revalidation(*, qmeta: Dict[str, Any]) -> bool:
    validation_identity = qmeta.get("validation_identity") if isinstance(qmeta.get("validation_identity"), dict) else {}
    return (
        str(qmeta.get("validator_version") or STATE_VALIDATOR_VERSION) != STATE_VALIDATOR_VERSION
        or str(validation_identity.get("prompt_version") or "") != STATE_VALIDATE_PROMPT_VERSION
    )


def _needs_change_reason_revalidation(*, qmeta: Dict[str, Any], obs: Dict[str, Any]) -> bool:
    change_reason = extract_last_change_reason(obs)
    if not change_reason:
        return False
    validation_identity = qmeta.get("validation_identity") if isinstance(qmeta.get("validation_identity"), dict) else {}
    change_reason_validation = (
        qmeta.get("change_reason_validation")
        if isinstance(qmeta.get("change_reason_validation"), dict)
        else None
    )
    if change_reason_validation is None:
        return True
    reason_codes = {
        str(code).strip()
        for code in list(change_reason_validation.get("reason_codes") or [])
        if str(code).strip()
    }
    if "change_reason_validation_missing" in reason_codes:
        return True
    if not bool_like(change_reason_validation.get("exists"), default=False):
        return True
    if str(validation_identity.get("change_reason_signature") or "") != compute_change_reason_signature(obs):
        return True
    if str(validation_identity.get("change_reason_prompt_version") or "") != CHANGE_REASON_VALIDATE_PROMPT_VERSION:
        return True
    return False


def build_state_validation(
    *,
    benchmark: Dict[str, Any],
    validator_client: Any,
    app_logs_by_id: Dict[str, Dict[str, Any]],
    max_checkpoints: Optional[int],
    l2_evidence_top_k: int,
    show_progress: bool = False,
    save_every_states: int = 0,
    save_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    all_checkpoints = benchmark.get("checkpoints", [])
    if not isinstance(all_checkpoints, list):
        all_checkpoints = []
    checkpoints = all_checkpoints
    if max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, int(max_checkpoints))]
        benchmark["checkpoints"] = checkpoints
        benchmark["total_checkpoints"] = len(checkpoints)

    reuse_cache: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_keys: List[Tuple[Dict[str, Any], List[str], int, Dict[str, Any]]] = []
    pending_change_reason_keys: Set[Tuple[str, str]] = set()
    total_states = 0
    total_existing_processed = 0
    pre_l1_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    checkpoint_l1_summary: List[Dict[str, Any]] = []
    total_l1_pass = 0
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        keys = extract_observable_valid_keys(checkpoint)
        expected_flat = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
        obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
        checkpoint_l1_pass = 0
        for key in keys:
            state_value = expected_flat.get(key)
            obs = obs_flat.get(key) if isinstance(obs_flat.get(key), dict) else {}
            qmeta_l1, candidate_field_paths = _build_l1_state_payload(
                state_key=key,
                state_value=state_value,
                obs=obs,
                app_logs_by_id=app_logs_by_id,
            )
            pre_l1_records[(str(checkpoint.get("checkpoint_id", "")), key)] = {
                "qmeta_l1": qmeta_l1,
                "candidate_field_paths": candidate_field_paths,
            }
            if bool(qmeta_l1.get("is_questionable")):
                checkpoint_l1_pass += 1
                total_l1_pass += 1
        existing_qmap_raw = checkpoint.get("state_questionability") if isinstance(checkpoint.get("state_questionability"), dict) else {}
        existing_qmap: Dict[str, Any] = {}
        for key in keys:
            qmeta_existing = existing_qmap_raw.get(key)
            if not isinstance(qmeta_existing, dict):
                continue
            obs = obs_flat.get(key) if isinstance(obs_flat.get(key), dict) else {}
            normalized_qmeta = _normalize_saved_questionability_entry(
                state_key=key,
                state_value=expected_flat.get(key),
                obs=obs,
                qmeta=qmeta_existing,
            )
            if _needs_state_l2_revalidation(qmeta=normalized_qmeta):
                continue
            if _needs_change_reason_revalidation(qmeta=normalized_qmeta, obs=obs):
                pending_change_reason_keys.add((str(checkpoint.get("checkpoint_id", "")), key))
            existing_qmap[key] = normalized_qmeta
        checkpoint_keys.append((checkpoint, keys, checkpoint_l1_pass, existing_qmap))
        total_states += len(keys)
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        total_existing_processed += sum(
            1
            for key in existing_qmap.keys()
            if (checkpoint_id, key) not in pending_change_reason_keys
        )
        checkpoint_l1_summary.append(
            {
                "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
                "pre_validate_count": len(keys),
                "after_l1_count": checkpoint_l1_pass,
            }
        )

    if show_progress:
        print(
            "[Stage1][pre-L2]"
            f" checkpoints={len(checkpoint_keys)},"
            f" pre_validate_count={total_states},"
            f" after_l1_count={total_l1_pass},"
            f" l1_filtered_count={max(0, total_states - total_l1_pass)}"
        )
        if total_existing_processed > 0:
            print(
                "[Stage1][resume]"
                f" already_processed_count={total_existing_processed},"
                f" remaining_count={max(0, total_states - total_existing_processed)}"
            )
        for row in checkpoint_l1_summary:
            print(
                "[Stage1][pre-L2][checkpoint]"
                f" checkpoint_id={row['checkpoint_id']},"
                f" pre_validate_count={row['pre_validate_count']},"
                f" after_l1_count={row['after_l1_count']}"
            )

    state_progress = tqdm(
        total=total_states,
        desc="TCE Stage1 states",
        unit="state",
        initial=min(total_existing_processed, total_states),
        disable=(not show_progress),
    )

    for checkpoint, keys, checkpoint_l1_pass, existing_qmap in checkpoint_keys:
        if not isinstance(checkpoint, dict):
            continue
        questionability_map: Dict[str, Any] = dict(existing_qmap)
        _, validated_snapshot_flat, checkpoint_summary = _build_checkpoint_summary(
            keys=keys,
            questionability_map=questionability_map,
            after_l1_count=checkpoint_l1_pass,
        )
        checkpoint["state_questionability"] = questionability_map
        checkpoint["validated_snapshot_state"] = expand_snapshot(validated_snapshot_flat)
        checkpoint["state_validation_summary"] = checkpoint_summary

    new_states_since_save = 0
    try:
        for checkpoint, keys, checkpoint_l1_pass, existing_qmap in checkpoint_keys:
            expected_flat = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
            obs_flat = flatten_observability(checkpoint.get("state_observability") or {})
            questionability_map: Dict[str, Any] = dict(existing_qmap)
            _, validated_snapshot_flat, checkpoint_summary = _build_checkpoint_summary(
                keys=keys,
                questionability_map=questionability_map,
                after_l1_count=checkpoint_l1_pass,
            )
            checkpoint["state_questionability"] = questionability_map
            checkpoint["validated_snapshot_state"] = expand_snapshot(validated_snapshot_flat)
            checkpoint["state_validation_summary"] = checkpoint_summary

            for key in keys:
                existing_qmeta = questionability_map.get(key)
                if isinstance(existing_qmeta, dict):
                    obs = obs_flat.get(key) if isinstance(obs_flat.get(key), dict) else {}
                    checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
                    if (checkpoint_id, key) in pending_change_reason_keys:
                        qmeta_final = copy.deepcopy(existing_qmeta)
                        qmeta_final["change_reason_validation"] = _build_change_reason_validation_payload(
                            state_key=key,
                            state_value=expected_flat.get(key),
                            obs=obs,
                            validator_client=validator_client,
                            app_logs_by_id=app_logs_by_id,
                            l2_evidence_top_k=l2_evidence_top_k,
                        )
                        validation_identity = (
                            qmeta_final.get("validation_identity")
                            if isinstance(qmeta_final.get("validation_identity"), dict)
                            else {}
                        )
                        validation_identity["change_reason_signature"] = compute_change_reason_signature(obs)
                        validation_identity["change_reason_prompt_version"] = CHANGE_REASON_VALIDATE_PROMPT_VERSION
                        qmeta_final["validation_identity"] = validation_identity
                        qmeta_final["validation_source"] = "computed"
                        questionability_map[key] = qmeta_final
                        reuse_cache.setdefault(key, []).append(copy.deepcopy(qmeta_final))
                        _, validated_snapshot_flat, checkpoint_summary = _build_checkpoint_summary(
                            keys=keys,
                            questionability_map=questionability_map,
                            after_l1_count=checkpoint_l1_pass,
                        )
                        checkpoint["state_questionability"] = questionability_map
                        checkpoint["validated_snapshot_state"] = expand_snapshot(validated_snapshot_flat)
                        checkpoint["state_validation_summary"] = checkpoint_summary
                        state_progress.update(1)
                        new_states_since_save += 1
                        if (
                            save_callback is not None
                            and int(save_every_states) > 0
                            and new_states_since_save >= int(save_every_states)
                        ):
                            save_callback(benchmark)
                            new_states_since_save = 0
                        continue
                    if isinstance(existing_qmeta.get("validation_identity"), dict):
                        reuse_cache.setdefault(key, []).append(copy.deepcopy(existing_qmeta))
                    continue

                if show_progress:
                    state_progress.set_postfix(
                        checkpoint_id=str(checkpoint.get("checkpoint_id", "")),
                        state_key=str(key),
                    )
                state_value = expected_flat.get(key)
                obs = obs_flat.get(key) if isinstance(obs_flat.get(key), dict) else {}
                l1_precomputed = pre_l1_records.get((str(checkpoint.get("checkpoint_id", "")), key)) or {}
                evidence_signature = compute_evidence_signature(obs=obs)
                qmeta_final: Optional[Dict[str, Any]] = None
                for cached in reuse_cache.get(key, []):
                    cached_identity = (
                        cached.get("validation_identity")
                        if isinstance(cached.get("validation_identity"), dict)
                        else {}
                    )
                    if str(cached_identity.get("evidence_signature", "")) != evidence_signature:
                        continue
                    projected_value = prune_value_by_paths(state_value, set(cached.get("validated_field_paths") or []))
                    if projected_value is None:
                        projected_value = {}
                    projected_signature = value_signature(projected_value)
                    cached_signature = str(cached_identity.get("validated_state_value_signature", ""))
                    if projected_signature != cached_signature:
                        raise ValueError(
                            "state_validation_invariant_violation_same_evidence_changed_validated_projection:"
                            f" state_key={key},"
                            f" checkpoint_id={checkpoint.get('checkpoint_id', '')}"
                        )
                    qmeta_final = copy.deepcopy(cached)
                    qmeta_final["validation_source"] = "reused"
                    break

                if qmeta_final is None:
                    qmeta_final = _build_final_questionability_payload(
                        state_key=key,
                        state_value=state_value,
                        obs=obs,
                        validator_client=validator_client,
                        app_logs_by_id=app_logs_by_id,
                        l2_evidence_top_k=l2_evidence_top_k,
                        qmeta_l1=l1_precomputed.get("qmeta_l1"),
                        candidate_field_paths=l1_precomputed.get("candidate_field_paths"),
                    )
                    qmeta_final["validation_identity"] = {
                        "state_key": key,
                        "validated_state_value_signature": value_signature(qmeta_final.get("validated_state_value")),
                        "evidence_signature": evidence_signature,
                        "validator_version": STATE_VALIDATOR_VERSION,
                        "prompt_version": STATE_VALIDATE_PROMPT_VERSION,
                        "change_reason_signature": compute_change_reason_signature(obs),
                        "change_reason_prompt_version": CHANGE_REASON_VALIDATE_PROMPT_VERSION,
                    }
                reuse_cache.setdefault(key, []).append(copy.deepcopy(qmeta_final))
                questionability_map[key] = qmeta_final
                _, validated_snapshot_flat, checkpoint_summary = _build_checkpoint_summary(
                    keys=keys,
                    questionability_map=questionability_map,
                    after_l1_count=checkpoint_l1_pass,
                )
                checkpoint["state_questionability"] = questionability_map
                checkpoint["validated_snapshot_state"] = expand_snapshot(validated_snapshot_flat)
                checkpoint["state_validation_summary"] = checkpoint_summary
                state_progress.update(1)
                new_states_since_save += 1
                if save_callback is not None and int(save_every_states) > 0 and new_states_since_save >= int(save_every_states):
                    save_callback(benchmark)
                    new_states_since_save = 0
    finally:
        state_progress.close()

    for checkpoint, keys, checkpoint_l1_pass, existing_qmap in checkpoint_keys:
        if not isinstance(checkpoint, dict):
            continue
        questionability_map = checkpoint.get("state_questionability")
        if not isinstance(questionability_map, dict):
            questionability_map = dict(existing_qmap)
        _, validated_snapshot_flat, checkpoint_summary = _build_checkpoint_summary(
            keys=keys,
            questionability_map=questionability_map,
            after_l1_count=checkpoint_l1_pass,
        )
        checkpoint["state_questionability"] = questionability_map
        checkpoint["validated_snapshot_state"] = expand_snapshot(validated_snapshot_flat)
        checkpoint["state_validation_summary"] = checkpoint_summary

    if save_callback is not None and new_states_since_save > 0:
        save_callback(benchmark)

    return benchmark


def write_state_validation_reports(*, payload: Dict[str, Any], output_path: Path) -> Tuple[Path, Path, Dict[str, int]]:
    checkpoints = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    count_rows: List[Dict[str, Any]] = []
    eyeball_rows: List[Dict[str, Any]] = []
    g_pre = 0
    g_l1 = 0
    g_l2 = 0
    g_final = 0
    g_reused = 0
    g_computed = 0

    for checkpoint in checkpoints if isinstance(checkpoints, list) else []:
        if not isinstance(checkpoint, dict):
            continue
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        summary = checkpoint.get("state_validation_summary") if isinstance(checkpoint.get("state_validation_summary"), dict) else {}
        pre = int(summary.get("pre_validate_count", 0))
        l1 = int(summary.get("after_l1_count", 0))
        l2 = int(summary.get("after_l2_count", 0))
        final = int(summary.get("after_l1_l2_count", 0))
        reused = int(summary.get("reused_count", 0))
        computed = int(summary.get("computed_count", 0))
        count_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "pre_validate_count": pre,
                "after_l1_count": l1,
                "after_l2_count": l2,
                "after_l1_l2_count": final,
                "reused_count": reused,
                "computed_count": computed,
            }
        )
        g_pre += pre
        g_l1 += l1
        g_l2 += l2
        g_final += final
        g_reused += reused
        g_computed += computed

        qmap = checkpoint.get("state_questionability") if isinstance(checkpoint.get("state_questionability"), dict) else {}
        for state_key, qmeta in qmap.items():
            if not isinstance(qmeta, dict) or not bool(qmeta.get("is_questionable")):
                continue
            try:
                state_value_text = json.dumps(qmeta.get("validated_state_value"), ensure_ascii=False, sort_keys=True)
            except Exception:
                state_value_text = str(qmeta.get("validated_state_value"))
            eyeball_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "state_key": str(state_key),
                    "validated_state_value": state_value_text,
                    "validated_field_paths": json.dumps(list(qmeta.get("validated_field_paths") or []), ensure_ascii=False),
                    "validation_source": str(qmeta.get("validation_source") or ""),
                }
            )

    counts_path = output_path.parent / f"{output_path.stem}_state_validate_counts.csv"
    eyeball_path = output_path.parent / f"{output_path.stem}_state_validated_eyeball.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint_id",
                "pre_validate_count",
                "after_l1_count",
                "after_l2_count",
                "after_l1_l2_count",
                "reused_count",
                "computed_count",
            ],
        )
        writer.writeheader()
        for row in count_rows:
            writer.writerow(row)

    with eyeball_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint_id",
                "state_key",
                "validated_state_value",
                "validated_field_paths",
                "validation_source",
            ],
        )
        writer.writeheader()
        for row in eyeball_rows:
            writer.writerow(row)

    totals = {
        "pre_validate_count": g_pre,
        "after_l1_count": g_l1,
        "after_l2_count": g_l2,
        "after_l1_l2_count": g_final,
        "reused_count": g_reused,
        "computed_count": g_computed,
    }
    return counts_path, eyeball_path, totals
