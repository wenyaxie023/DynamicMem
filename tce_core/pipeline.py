"""Shared generation pipeline for TCE baselines."""

import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, create_model
except Exception:  # pragma: no cover
    BaseModel = Any  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    create_model = None  # type: ignore[assignment]

from tqdm import tqdm
from generation.adapters.base import get_baseline_concurrency_policy
from .orchestrator_protocol import (
    AnswerExecutionResult,
    CheckpointHandle,
    QuerySpec,
    RetrievalOptions,
    RetrievalResult,
    ensure_answer_execution_result,
    ensure_checkpoint_handle,
    ensure_retrieval_result,
)
from .prompts import (
    build_task_c_prompt_with_agent_memory,
    build_task_c_prompt_with_inline_memory,
    build_change_reasoning_prompt_with_agent_memory,
    build_change_reasoning_prompt_with_inline_memory,
    build_state_completion_prompt_with_agent_memory,
    build_state_completion_prompt_with_inline_memory,
)
from .task_packs import (
    build_change_targets_from_pack,
    build_state_completion_targets_from_pack,
    extract_pack_keys,
)
from tce_contracts import (
    infer_task_contract_version,
    normalized_contract_metadata,
    task_contract_is_v2,
    task_contract_supports_change_tracking,
)
from .task_spec import (
    build_prediction_task_from_checkpoint,
)
from .exposure_checkpoint_builder import (
    build_calendar_anchor_checkpoints,
    build_token_exposure_checkpoints,
)

EXCLUDED_VALUE_FIELDS = {"priority", "schedule_date", "schedule_dates"}
_BASIC_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return datetime.max


def to_log_text(log: Dict[str, Any]) -> str:
    return json.dumps(log, ensure_ascii=False)


def _inline_memory_token_stats(
    inline_memory_blocks: Sequence[str],
    tokenizer_model: str,
) -> Dict[str, Any]:
    text = "\n<->\n".join(str(block) for block in inline_memory_blocks if str(block).strip())
    if not text:
        return {
            "inline_memory_total_tokens": 0,
            "inline_memory_tokenizer_model": str(tokenizer_model or ""),
            "inline_memory_tokenizer_backend": "empty",
        }
    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            enc = tiktoken.encoding_for_model(str(tokenizer_model or "gpt-4o-mini"))
            tokenizer_model_used = str(tokenizer_model or "gpt-4o-mini")
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
            tokenizer_model_used = "o200k_base"
        return {
            "inline_memory_total_tokens": len(enc.encode(text)),
            "inline_memory_tokenizer_model": tokenizer_model_used,
            "inline_memory_tokenizer_backend": "tiktoken",
        }
    except ModuleNotFoundError:
        tokens = _BASIC_TOKEN_RE.findall(text)
        return {
            "inline_memory_total_tokens": len(tokens),
            "inline_memory_tokenizer_model": "basic_regex_v1",
            "inline_memory_tokenizer_backend": "basic_regex",
            "inline_memory_tokenizer_warning": "tiktoken_unavailable",
        }


def _attach_inline_memory_stats(
    retrieval_meta: Dict[str, Any],
    retrieval_result: RetrievalResult,
    tokenizer_model: str,
) -> None:
    inline_memory_blocks = [str(block) for block in retrieval_result.inline_memory_blocks]
    retrieval_meta.setdefault("inline_memory_blocks", inline_memory_blocks)
    retrieval_meta.setdefault("inline_memory_block_count", len(inline_memory_blocks))
    retrieval_meta.setdefault(
        "inline_memory_total_chars",
        sum(len(block) for block in inline_memory_blocks),
    )
    for key, value in _inline_memory_token_stats(inline_memory_blocks, tokenizer_model).items():
        retrieval_meta.setdefault(key, value)


def flatten_snapshot(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in snapshot.keys()):
        return {str(k): v for k, v in snapshot.items()}
    flat: Dict[str, Any] = {}
    for category, content in snapshot.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def drop_excluded_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in EXCLUDED_VALUE_FIELDS:
                continue
            out[k] = drop_excluded_fields(v)
        return out
    if isinstance(value, list):
        return [drop_excluded_fields(v) for v in value]
    return value


ScalarValue = Union[str, int, float, bool, None]


def _normalize_task_selection(value: Any) -> str:
    raw = str(value or "all").strip().lower() or "all"
    if raw in {"all", "task_c_only"}:
        return raw
    raise ValueError(
        "Unsupported task_selection: {}. Use 'all' or 'task_c_only'.".format(value)
    )


def fill_blank_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: fill_blank_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_blank_template(v) for v in value]
    return "<fill the blank>"


def _build_change_templates(before_value: Any, after_value: Any) -> Dict[str, Any]:
    return {
        "before": fill_blank_template(before_value),
        "after": fill_blank_template(after_value),
        "change_reason": "<fill the blank>",
        "evidence": [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}],
    }


def _compute_changed_items_by_checkpoint(checkpoints: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    prev_snapshot: Optional[Dict[str, Any]] = None
    prev_ts = ""
    for cp in checkpoints:
        cid = str(cp.get("checkpoint_id"))
        cp_ts = str((cp.get("as_of") or {}).get("timestamp", ""))
        cur = flatten_snapshot(cp.get("expected_snapshot_state") or {})
        cur = {k: drop_excluded_fields(v) for k, v in cur.items()}
        changed_keys: List[str] = []
        templates: Dict[str, Any] = {}
        if prev_snapshot is not None:
            all_keys = sorted(set(prev_snapshot.keys()) | set(cur.keys()))
            for k in all_keys:
                before_v = prev_snapshot.get(k)
                after_v = cur.get(k)
                if before_v != after_v:
                    changed_keys.append(k)
                    templates[k] = _build_change_templates(before_v, after_v)
        out[cid] = {
            "previous_cutoff_ts": prev_ts,
            "changed_keys": changed_keys,
            "changed_templates": templates,
        }
        prev_snapshot = cur
        prev_ts = cp_ts
    return out


def normalize_change_reasoning_output(raw_out: Any, changed_keys: List[str]) -> Dict[str, Any]:
    if not isinstance(raw_out, dict):
        return {}
    payload = raw_out.get("change_analysis")
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in changed_keys:
        item = payload.get(key)
        if not isinstance(item, dict):
            out[key] = {"before": None, "after": None, "change_reason": "", "evidence": []}
            continue
        evidence = item.get("evidence")
        evidence_records = normalize_evidence_prediction({"_": evidence}, ["_"]).get("_", [])
        change_reason = item.get("change_reason")
        if change_reason is None:
            change_reason = item.get("reason", "")
        out[key] = {
            "before": item.get("before"),
            "after": item.get("after"),
            "change_reason": str(change_reason or ""),
            "evidence": evidence_records,
        }
    return out


def _checkpoint_handle_metadata(
    checkpoint: Dict[str, Any],
    *,
    memory_pool: List[Dict[str, Any]],
    state_kind: str,
    state_ref: Any = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> CheckpointHandle:
    metadata = {
        "checkpoint_timestamp": str(((checkpoint.get("as_of") or {}).get("timestamp") or "")),
        "checkpoint_app_log_id": str(((checkpoint.get("as_of") or {}).get("app_log_id") or "")),
        "memory_pool_size": len(memory_pool),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return CheckpointHandle(
        checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
        state_kind=state_kind,
        state_ref=state_ref,
        metadata=metadata,
    )


def _build_task_query_spec(
    *,
    task_name: str,
    checkpoint: Dict[str, Any],
    item_key: str,
    target_keys: List[str],
    task_query_text: str,
    retrieval_query_text: str,
    answer_query_text: str,
    task_payload: Optional[Dict[str, Any]] = None,
) -> QuerySpec:
    return QuerySpec(
        task_name=str(task_name),
        item_key=str(item_key),
        target_keys=[str(x) for x in target_keys],
        checkpoint_timestamp=str(((checkpoint.get("as_of") or {}).get("timestamp") or "")),
        task_query_text=str(task_query_text or ""),
        retrieval_query_text=str(retrieval_query_text or ""),
        answer_query_text=str(answer_query_text or ""),
        task_payload=dict(task_payload or {}),
    )


def align_prediction_to_template(pred_value: Any, template_value: Any) -> Any:
    if isinstance(template_value, dict):
        pred_dict = pred_value if isinstance(pred_value, dict) else {}
        return {k: align_prediction_to_template(pred_dict.get(k), v) for k, v in template_value.items()}
    if isinstance(template_value, list):
        pred_list = pred_value if isinstance(pred_value, list) else []
        if not template_value:
            return pred_list
        item_tmpl = template_value[0]
        return [align_prediction_to_template(v, item_tmpl) for v in pred_list]
    return pred_value


def _build_value_model(value: Any, model_idx: int, counter: List[int]) -> Any:
    if isinstance(value, dict):
        node_idx = counter[0]
        counter[0] += 1
        fields: Dict[str, Tuple[Any, Any]] = {}
        for i, (k, v) in enumerate(value.items()):
            field_name = f"f_{i}"
            fields[field_name] = (
                _build_value_model(v, model_idx, counter),
                Field(..., alias=str(k)),
            )
        return create_model(  # type: ignore[call-overload]
            f"TceValueNode_{model_idx}_{node_idx}",
            __config__=ConfigDict(extra="forbid", populate_by_name=True),
            **fields,
        )
    if isinstance(value, list):
        if not value:
            return List[ScalarValue]  # type: ignore[valid-type]
        return List[_build_value_model(value[0], model_idx, counter)]  # type: ignore[valid-type]
    return ScalarValue


def _require_nonempty_pack_query(
    *,
    task_name: str,
    checkpoint_id: str,
    item_key: str,
    query_text: str,
) -> str:
    query = str(query_text or "").strip()
    if query:
        return query
    raise ValueError(
        "{} requires a non-empty prebuilt retrieval_query for checkpoint {} item {}.".format(
            task_name,
            checkpoint_id or "<unknown>",
            item_key or "<unknown>",
        )
    )


def _select_state_completion_prompt_builder(memory_prompt_mode: str):
    if memory_prompt_mode == "inline_memory":
        return build_state_completion_prompt_with_inline_memory
    if memory_prompt_mode == "agent_memory":
        return build_state_completion_prompt_with_agent_memory
    raise ValueError("Unsupported memory_prompt_mode: {}".format(memory_prompt_mode))


def _select_change_reasoning_prompt_builder(memory_prompt_mode: str):
    if memory_prompt_mode == "inline_memory":
        return build_change_reasoning_prompt_with_inline_memory
    if memory_prompt_mode == "agent_memory":
        return build_change_reasoning_prompt_with_agent_memory
    raise ValueError("Unsupported memory_prompt_mode: {}".format(memory_prompt_mode))


def _select_task_c_prompt_builder(memory_prompt_mode: str):
    if memory_prompt_mode == "inline_memory":
        return build_task_c_prompt_with_inline_memory
    if memory_prompt_mode == "agent_memory":
        return build_task_c_prompt_with_agent_memory
    raise ValueError("Unsupported memory_prompt_mode: {}".format(memory_prompt_mode))


def build_state_completion_prompt(
    *,
    checkpoint: Dict[str, Any],
    context_logs: Optional[List[Dict[str, Any]]],
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    memory_prompt_mode: str,
    inline_memory_blocks: Optional[List[str]] = None,
    task_text_override: Optional[str] = None,
) -> str:
    task_query = task_text_override or build_prediction_task_from_checkpoint(checkpoint, target_keys)
    prompt_builder = _select_state_completion_prompt_builder(memory_prompt_mode)
    return prompt_builder(
        context_logs=context_logs,
        task_query=task_query,
        target_keys=target_keys,
        target_value_templates=target_value_templates,
        log_to_text=to_log_text,
        inline_memory_blocks=inline_memory_blocks,
    )


def _build_change_prompt_from_queryspec(
    *,
    memory_prompt_mode: str,
    query_spec: QuerySpec,
    changed_value_templates: Dict[str, Any],
    retrieval_result: RetrievalResult,
) -> str:
    return _select_change_reasoning_prompt_builder(memory_prompt_mode)(
        context_logs=None,
        task_query=query_spec.answer_query_text,
        changed_keys=query_spec.target_keys,
        changed_value_templates=changed_value_templates,
        log_to_text=to_log_text,
        inline_memory_blocks=list(retrieval_result.inline_memory_blocks),
    )


def _build_apply_prompt_from_queryspec(
    *,
    memory_prompt_mode: str,
    query_spec: QuerySpec,
    retrieval_result: RetrievalResult,
) -> str:
    structured_v2 = bool((query_spec.task_payload or {}).get("structured_task_c_v2"))
    service_family = str((query_spec.task_payload or {}).get("service_family") or "")
    response_mode = "text"
    if structured_v2 and service_family != "user_communication":
        response_mode = "structured"
    prompt_builder = _select_task_c_prompt_builder(memory_prompt_mode)
    return prompt_builder(
        response_mode=response_mode,
        task_body=query_spec.retrieval_query_text,
        output_template=(query_spec.task_payload or {}).get("output_template"),
        context_logs=None,
        log_to_text=to_log_text,
        inline_memory_blocks=list(retrieval_result.inline_memory_blocks),
    )


def normalize_evidence_prediction(raw_evidence: Any, target_keys: List[str]) -> Dict[str, List[Dict[str, str]]]:
    def to_records(value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, str]] = []
        seen = set()
        for item in value:
            log_id = ""
            evidence_content = ""
            if isinstance(item, dict):
                candidate = item.get("app_log_id")
                if candidate is not None:
                    log_id = str(candidate).strip()
                evidence_content = str(item.get("evidence_content") or "").strip()
            elif isinstance(item, (str, int, float)):
                log_id = str(item).strip()
            dedupe_key = (log_id, evidence_content)
            if dedupe_key in seen:
                continue
            if log_id or evidence_content:
                seen.add(dedupe_key)
                out.append(
                    {
                        "app_log_id": log_id,
                        "evidence_content": evidence_content,
                    }
                )
        return out

    evidence: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(raw_evidence, dict):
        for key in target_keys:
            evidence[key] = to_records(raw_evidence.get(key))
    else:
        for key in target_keys:
            evidence[key] = []
    return evidence


def _extract_rq3_apply_pack_for_checkpoint(
    checkpoint: Dict[str, Any],
    target_keys: List[str],
    *,
    task_contract_version: str,
) -> Dict[str, List[Dict[str, str]]]:
    if "rq3_know_apply" in checkpoint:
        raise ValueError("Legacy field rq3_know_apply is no longer supported. Use rq3_apply_service_qa.")
    payload = checkpoint.get("rq3_apply_service_qa")
    if not isinstance(payload, dict):
        return {}
    keys_obj = payload.get("keys")
    if not isinstance(keys_obj, dict):
        return {}

    out: Dict[str, List[Dict[str, str]]] = {}
    for key in target_keys:
        key_obj = keys_obj.get(key)
        if not isinstance(key_obj, dict):
            continue
        items = key_obj.get("items")
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            qa_id = str(item.get("qa_id") or f"q{idx+1}")
            if task_contract_is_v2(task_contract_version):
                service_family = str(item.get("service_family") or "").strip()
                scenario = str(item.get("scenario") or "").strip()
                task_instruction = str(item.get("task_instruction") or "").strip()
                output_template = item.get("output_template")
                reference_output = item.get("reference_output")
                if not scenario or not task_instruction:
                    continue
                normalized.append(
                    {
                        "qa_id": qa_id,
                        "service_family": service_family,
                        "scenario": scenario,
                        "task_instruction": task_instruction,
                        "output_template": output_template,
                        "reference_output": reference_output,
                        "retrieval_query": str(item.get("retrieval_query") or "").strip(),
                    }
                )
            else:
                service_category = str(item.get("service_category") or "").strip()
                apply_scenario = str(item.get("apply_scenario") or "").strip()
                apply_q = str(item.get("question") or item.get("apply_question") or "").strip()
                apply_a = str(item.get("reference_answer") or item.get("apply_reference_answer") or "").strip()
                if not apply_q:
                    continue
                normalized.append(
                    {
                        "qa_id": qa_id,
                        "service_category": service_category,
                        "apply_scenario": apply_scenario,
                        "apply_question": apply_q,
                        "retrieval_query": str(item.get("retrieval_query") or "").strip(),
                        "apply_reference_answer": apply_a,
                    }
                )
        if normalized:
            out[key] = normalized[:1]
    return out


def _normalize_rq3_apply_answer_output(
    raw_out: Any,
    *,
    task_contract_version: str,
    service_family: str = "",
) -> Dict[str, Any]:
    if not isinstance(raw_out, dict):
        return {"answer": "", "evidence": []}
    if task_contract_is_v2(task_contract_version):
        if str(service_family or "").strip() == "user_communication":
            answer = raw_out.get("answer")
            if answer is None:
                answer = raw_out.get("final_answer", "")
            return {
                "answer": str(answer or "").strip(),
                "evidence": normalize_evidence_prediction({"_": raw_out.get("evidence")}, ["_"]).get("_", []),
            }
        return {
            "answer": raw_out.get("answer"),
            "evidence": normalize_evidence_prediction({"_": raw_out.get("evidence")}, ["_"]).get("_", []),
        }
    answer = raw_out.get("answer")
    if answer is None:
        answer = raw_out.get("final_answer", "")
    text = str(answer or "").strip()
    evidence_records = normalize_evidence_prediction({"_": raw_out.get("evidence")}, ["_"]).get("_", [])
    return {"answer": text, "evidence": evidence_records}


def build_generation_text_format(
    target_keys: Sequence[str],
    target_value_templates: Dict[str, Any],
    model_idx: int,
) -> Type[BaseModel]:
    if create_model is None or Field is None or ConfigDict is None:
        raise RuntimeError(
            "Structured response requires pydantic. Install pydantic to use this path."
        )
    if not target_keys:
        target_keys = ["__no_key__"]

    snapshot_fields: Dict[str, Tuple[Any, Any]] = {}
    evidence_fields: Dict[str, Tuple[Any, Any]] = {}
    counter = [0]
    for i, key in enumerate(target_keys):
        field_name = f"k_{i}"
        evidence_model = create_model(  # type: ignore[call-overload]
            f"TceEvidenceItem_{model_idx}_{i}",
            __config__=ConfigDict(extra="forbid", populate_by_name=True),
            app_log_id=(str, ...),
            evidence_content=(str, ...),
        )
        value_template = target_value_templates.get(key, "<fill the blank>")
        snapshot_fields[field_name] = (
            _build_value_model(value_template, model_idx, counter),
            Field(
                ...,
                alias=key,
                description="Predicted value with the same nested structure as template.",
            ),
        )
        evidence_fields[field_name] = (
            List[evidence_model],  # type: ignore[valid-type]
            Field(
                ...,
                alias=key,
                description="Supporting evidence objects for this key.",
            ),
        )

    snapshot_model = create_model(  # type: ignore[call-overload]
        f"TceSnapshotState_{model_idx}",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **snapshot_fields,
    )
    evidence_model = create_model(  # type: ignore[call-overload]
        f"TceEvidence_{model_idx}",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **evidence_fields,
    )
    return create_model(  # type: ignore[call-overload]
        f"TcePredOutput_{model_idx}",
        __config__=ConfigDict(extra="forbid"),
        user_state=(snapshot_model, ...),
        evidence=(evidence_model, ...),
    )


def normalize_generation_output(raw_out: Any, target_keys: List[str]) -> Dict[str, Any]:
    if isinstance(raw_out, dict):
        # Native schema path.
        if "user_state" in raw_out or "snapshot_state" in raw_out or "evidence" in raw_out:
            state_payload = raw_out.get("user_state") if "user_state" in raw_out else raw_out.get("snapshot_state", {})
            return {
                "snapshot_state": state_payload,
                "evidence": raw_out.get("evidence", {}),
            }

        # Backward-compatible structured list schema path.
        items = raw_out.get("predictions")
        if isinstance(items, list):
            snapshot_state: Dict[str, Any] = {}
            evidence: Dict[str, Any] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                if key is None:
                    continue
                key = str(key)
                if key not in target_keys:
                    continue
                value = item.get("snapshot_value")
                if value is None and "snapshot_value_json" in item:
                    value = item.get("snapshot_value_json")
                snapshot_state[key] = value
                evidence[key] = item.get("evidence", [])
            return {"snapshot_state": snapshot_state, "evidence": evidence}

    return {"snapshot_state": {}, "evidence": {}}


def write_debug_artifact(debug_dir: Path, checkpoint_id: str, payload: Dict[str, Any]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{checkpoint_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_checkpoint_prediction_complete(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("_checkpoint_complete"))


def _is_effectively_empty_prediction_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _rq3_apply_item_id(state_key: str, qa_id: str) -> str:
    return "{}::{}".format(str(state_key or "").strip(), str(qa_id or "").strip())


def _extract_task_a_resume_state(
    prediction_item: Any,
    target_keys: Sequence[str],
) -> Dict[str, Any]:
    valid_snapshot_state: Dict[str, Any] = {}
    valid_evidence: Dict[str, Any] = {}
    valid_records_by_key: Dict[str, Dict[str, Any]] = {}
    valid_retrieval_by_key: Dict[str, Dict[str, Any]] = {}
    pending_keys: List[str] = []

    if not isinstance(prediction_item, dict):
        return {
            "valid_snapshot_state": valid_snapshot_state,
            "valid_evidence": valid_evidence,
            "valid_records_by_key": valid_records_by_key,
            "valid_retrieval_by_key": valid_retrieval_by_key,
            "pending_keys": list(target_keys),
        }

    snapshot_state = prediction_item.get("snapshot_state")
    if not isinstance(snapshot_state, dict):
        snapshot_state = {}
    evidence = prediction_item.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    metadata = prediction_item.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    raw_model_output = metadata.get("raw_model_output")
    records = raw_model_output.get("records") if isinstance(raw_model_output, dict) else None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            key = str(record.get("key") or "").strip()
            if key:
                valid_records_by_key[key] = copy.deepcopy(record)

    retrieval_records = metadata.get("per_key_retrieval")
    if isinstance(retrieval_records, list):
        for record in retrieval_records:
            if not isinstance(record, dict):
                continue
            key = str(record.get("key") or "").strip()
            if key:
                valid_retrieval_by_key[key] = copy.deepcopy(record)

    for key in target_keys:
        value = snapshot_state.get(key)
        if _is_effectively_empty_prediction_value(value):
            pending_keys.append(key)
            valid_records_by_key.pop(key, None)
            valid_retrieval_by_key.pop(key, None)
            continue
        valid_snapshot_state[key] = copy.deepcopy(value)
        valid_evidence[key] = copy.deepcopy(evidence.get(key, []))

    valid_records_by_key = {
        key: record
        for key, record in valid_records_by_key.items()
        if key in valid_snapshot_state
    }
    valid_retrieval_by_key = {
        key: record
        for key, record in valid_retrieval_by_key.items()
        if key in valid_snapshot_state
    }
    return {
        "valid_snapshot_state": valid_snapshot_state,
        "valid_evidence": valid_evidence,
        "valid_records_by_key": valid_records_by_key,
        "valid_retrieval_by_key": valid_retrieval_by_key,
        "pending_keys": pending_keys,
    }


def _is_rq3_apply_answer_valid(
    answer_item: Any,
    *,
    task_contract_version: str,
    service_family: str,
) -> bool:
    if not isinstance(answer_item, dict):
        return False
    status = str(answer_item.get("status") or "").strip().lower()
    if status == "invalid":
        return False
    return not _is_effectively_empty_prediction_value(answer_item.get("answer"))


def _extract_rq3_apply_resume_state(
    prediction_item: Any,
    rq3_pack: Dict[str, List[Dict[str, Any]]],
    *,
    task_contract_version: str,
) -> Dict[str, Any]:
    valid_answers_by_key: Dict[str, List[Dict[str, Any]]] = {}
    valid_raw_records_by_key: Dict[str, List[Dict[str, Any]]] = {}
    pending_items_by_key: Dict[str, List[Dict[str, Any]]] = {}

    if not isinstance(prediction_item, dict):
        for key, items in rq3_pack.items():
            pending_items_by_key[key] = [copy.deepcopy(item) for item in items]
        return {
            "valid_answers_by_key": valid_answers_by_key,
            "valid_raw_records_by_key": valid_raw_records_by_key,
            "pending_items_by_key": pending_items_by_key,
        }

    predicted_answers = prediction_item.get("rq3_apply_answers")
    if not isinstance(predicted_answers, dict):
        predicted_answers = {}

    raw_record_by_item_id: Dict[str, Dict[str, Any]] = {}
    metadata = prediction_item.get("metadata")
    rq3_meta = metadata.get("rq3_apply") if isinstance(metadata, dict) else None
    records = rq3_meta.get("records") if isinstance(rq3_meta, dict) else None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            item_id = _rq3_apply_item_id(record.get("key") or "", record.get("qa_id") or "")
            if item_id != "::":
                raw_record_by_item_id[item_id] = copy.deepcopy(record)

    for key, expected_items in rq3_pack.items():
        key_payload = predicted_answers.get(key)
        predicted_items = key_payload.get("items") if isinstance(key_payload, dict) else None
        predicted_by_qa_id: Dict[str, Dict[str, Any]] = {}
        if isinstance(predicted_items, list):
            for pred_item in predicted_items:
                if not isinstance(pred_item, dict):
                    continue
                qa_id = str(pred_item.get("qa_id") or "").strip()
                if qa_id:
                    predicted_by_qa_id[qa_id] = pred_item

        for expected_item in expected_items:
            qa_id = str(expected_item.get("qa_id") or "").strip()
            item_id = _rq3_apply_item_id(key, qa_id)
            pred_item = predicted_by_qa_id.get(qa_id)
            service_family = str(expected_item.get("service_family") or "").strip()
            if _is_rq3_apply_answer_valid(
                pred_item,
                task_contract_version=task_contract_version,
                service_family=service_family,
            ):
                valid_answers_by_key.setdefault(key, []).append(copy.deepcopy(pred_item))
                raw_record = raw_record_by_item_id.get(item_id)
                if isinstance(raw_record, dict):
                    valid_raw_records_by_key.setdefault(key, []).append(copy.deepcopy(raw_record))
            else:
                pending_items_by_key.setdefault(key, []).append(copy.deepcopy(expected_item))

    return {
        "valid_answers_by_key": valid_answers_by_key,
        "valid_raw_records_by_key": valid_raw_records_by_key,
        "pending_items_by_key": pending_items_by_key,
    }


def _has_complete_rq3_apply_answers(
    checkpoint: Dict[str, Any],
    prediction_item: Dict[str, Any],
    *,
    task_contract_version: str,
) -> bool:
    target_keys = extract_pack_keys(checkpoint.get("rq3_apply_service_qa"))
    expected_pack = _extract_rq3_apply_pack_for_checkpoint(
        checkpoint,
        target_keys=target_keys,
        task_contract_version=task_contract_version,
    )
    if not expected_pack:
        return True
    resume_state = _extract_rq3_apply_resume_state(
        prediction_item,
        expected_pack,
        task_contract_version=task_contract_version,
    )
    return not any(resume_state.get("pending_items_by_key", {}).values())


def _is_checkpoint_prediction_reusable_for_resume(
    checkpoint: Dict[str, Any],
    prediction_item: Any,
    *,
    task_contract_version: str,
    enable_rq3_apply_service_qa: bool,
) -> bool:
    if not _is_checkpoint_prediction_complete(prediction_item):
        return False
    if not isinstance(prediction_item, dict):
        return False
    if enable_rq3_apply_service_qa and not _has_complete_rq3_apply_answers(
        checkpoint,
        prediction_item,
        task_contract_version=task_contract_version,
    ):
        return False
    return True


def normalize_app_logs(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        logs = payload
    elif isinstance(payload, dict):
        logs = payload.get("app_logs", [])
    else:
        logs = []
    logs = [x for x in logs if isinstance(x, dict)]
    logs.sort(
        key=lambda x: (
            parse_ts(str(x.get("timestamp", "9999-12-31 23:59:59"))),
            str(x.get("app_log_id", "")),
        )
    )
    return logs


def observed_logs_for_checkpoint(
    checkpoint: Dict[str, Any],
    app_logs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], datetime, str]:
    cp_ts = str((checkpoint.get("as_of") or {}).get("timestamp", ""))
    cp_idx_raw = (checkpoint.get("as_of") or {}).get("log_index")
    if isinstance(cp_idx_raw, int) and 0 <= cp_idx_raw < len(app_logs):
        observed = app_logs[: cp_idx_raw + 1]
        cp_dt = parse_ts(cp_ts)
    else:
        cp_dt = parse_ts(cp_ts)
        observed = [
            log
            for log in app_logs
            if parse_ts(str(log.get("timestamp", "9999-12-31 23:59:59"))) <= cp_dt
        ]
    return observed, cp_dt, cp_ts


def build_target_templates(checkpoint: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    expected_snapshot_flat = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
    expected_snapshot_flat = {k: drop_excluded_fields(v) for k, v in expected_snapshot_flat.items()}
    target_keys = sorted(expected_snapshot_flat.keys())
    target_value_templates = {k: fill_blank_template(expected_snapshot_flat.get(k)) for k in target_keys}
    return target_keys, target_value_templates


def build_observable_target_templates(checkpoint: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Select per-checkpoint keys that are observable+valid at this anchor.
    Fallback to expected snapshot keys when observability payload is missing.
    """
    expected_snapshot_flat = flatten_snapshot(checkpoint.get("expected_snapshot_state") or {})
    expected_snapshot_flat = {k: drop_excluded_fields(v) for k, v in expected_snapshot_flat.items()}
    observability_flat = flatten_snapshot(checkpoint.get("state_observability") or {})

    key_status: Dict[str, Any] = {}
    candidate_keys: List[str] = []
    for key, value in expected_snapshot_flat.items():
        obs = observability_flat.get(key)
        is_observable = isinstance(obs, dict)
        is_valid = bool(isinstance(obs, dict) and obs.get("is_valid"))
        if not observability_flat:
            # Backward compatibility for older checkpoints without state_observability.
            is_observable = True
            is_valid = True
        key_status[key] = {
            "observable": is_observable,
            "valid": is_valid,
            "selected": bool(is_observable and is_valid),
            "template_source": "expected_snapshot_state",
            "value": value,
        }
        if is_observable and is_valid:
            candidate_keys.append(key)

    target_keys = sorted(set(candidate_keys))
    if not target_keys:
        # Last-resort fallback so generation never crashes on malformed checkpoints.
        target_keys = sorted(expected_snapshot_flat.keys())
        for key in target_keys:
            status = key_status.setdefault(key, {})
            status["selected"] = True
            status["fallback_selected"] = True

    target_value_templates = {k: fill_blank_template(expected_snapshot_flat.get(k)) for k in target_keys}
    for key in key_status:
        key_status[key].pop("value", None)
    return target_keys, target_value_templates, key_status


def _extract_prebuilt_sampling_meta(checkpoints: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cp in checkpoints:
        if not isinstance(cp, dict):
            continue
        cid = str(cp.get("checkpoint_id", "")).strip()
        if not cid:
            continue
        sampling = cp.get("sampling")
        if not isinstance(sampling, dict):
            continue
        params = sampling.get("params")
        mode = str(sampling.get("mode", "")).strip()
        meta: Dict[str, Any] = {}
        if mode:
            meta["sampling_mode"] = mode
        if isinstance(params, dict):
            meta["sampling_params"] = dict(params)
        if meta:
            out[cid] = meta
    return out


def run_pipeline(
    *,
    benchmark_path: Path,
    app_logs_path: Path,
    output_path: Path,
    max_visible_logs: Optional[int],
    ask_json: Callable[[str], Any],
    ask_structured: Optional[Callable[[str, Type[BaseModel]], Any]],
    use_structured_response: bool,
    close: Callable[[], None],
    prepare_checkpoint_state: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Any]] = None,
    retrieve_context_for_query: Optional[
        Callable[[CheckpointHandle, QuerySpec, RetrievalOptions, List[Dict[str, Any]]], Any]
    ] = None,
    answer_query: Optional[Callable[[CheckpointHandle, QuerySpec, RetrievalResult], Any]] = None,
    finalize_checkpoint_state: Optional[Callable[[CheckpointHandle], None]] = None,
    baseline_name: str,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    memory_prompt_mode: str = "inline_memory",
    predict_per_key: bool = True,
    exposure_anchors: Optional[Sequence[int]] = None,
    calendar_anchor_freq: Optional[str] = None,
    exposure_tokenizer_model: str = "gpt-4o-mini",
    enable_change_reasoning: bool = False,
    enable_rq3_apply_service_qa: bool = False,
    rq3_apply_save_prompt_and_raw: bool = True,
    rq3_apply_retrieval_top_k: Optional[int] = None,
    retrieval_options_backend: Optional[Dict[str, Any]] = None,
    checkpoint_workers: int = 1,
    within_checkpoint_workers: int = 1,
    save_every_generation_keys: int = 1,
    enable_final_qa: bool = False,
    final_qa_path: Optional[str] = None,
    final_qa_output_path: Optional[str] = None,
    final_qa_retrieval_top_k: Optional[int] = None,
    final_qa_save_prompt_and_raw: bool = False,
    task_selection: str = "all",
) -> Dict[str, Any]:
    task_selection = _normalize_task_selection(task_selection)
    if task_selection == "task_c_only" and not enable_rq3_apply_service_qa:
        raise ValueError(
            "task_selection=task_c_only requires runtime.enable_rq3_apply_service_qa=true."
        )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_contract_version = infer_task_contract_version(benchmark)
    task_b_supported_by_contract = task_contract_supports_change_tracking(benchmark_contract_version)
    app_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))
    raw_checkpoints = benchmark.get("checkpoints", [])
    checkpoints = [cp for cp in raw_checkpoints if isinstance(cp, dict)]
    checkpoint_sampling_meta = _extract_prebuilt_sampling_meta(checkpoints)
    sampling_strategy = benchmark.get("sampling_strategy")
    has_prebuilt_sampling = bool(
        isinstance(sampling_strategy, dict)
        and str(sampling_strategy.get("stage", "")).strip().lower() == "benchmark_build"
    )

    # Sampling strategy resolution:
    # - prebuilt benchmark sampling takes precedence
    # - runtime sampling is used only for legacy/non-prebuilt benchmark payloads
    
    ## legacy sampling code
    if has_prebuilt_sampling:
        if exposure_anchors or calendar_anchor_freq:
            print("[TCE] benchmark contains prebuilt sampled checkpoints; runtime sampling params are ignored.")
    elif exposure_anchors:
        checkpoints, checkpoint_sampling_meta = build_token_exposure_checkpoints(
            benchmark_path=benchmark_path,
            app_logs_large=app_logs,
            exposure_anchors=list(exposure_anchors or []),
            tokenizer_model=exposure_tokenizer_model,
        )
    elif calendar_anchor_freq:
        checkpoints, checkpoint_sampling_meta = build_calendar_anchor_checkpoints(
            benchmark_path=benchmark_path,
            app_logs_large=app_logs,
            calendar_anchor_freq=calendar_anchor_freq,
            tokenizer_model=exposure_tokenizer_model,
        )
    if max_checkpoints is not None:
        checkpoints = checkpoints[:max_checkpoints]
    existing: Dict[str, Dict[str, Any]] = {}
    if resume and output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            for item in raw.get("predictions", []):
                cid = item.get("checkpoint_id")
                if cid and isinstance(item, dict):
                    existing[str(cid)] = item
        except Exception:
            pass

    policy = get_baseline_concurrency_policy(baseline_name)
    requested_checkpoint_workers = max(1, int(checkpoint_workers))
    requested_within_checkpoint_workers = max(1, int(within_checkpoint_workers))
    checkpoint_workers = requested_checkpoint_workers if policy.checkpoint_parallelism == "allowed" else 1
    within_checkpoint_workers = (
        requested_within_checkpoint_workers if policy.within_checkpoint_parallelism == "allowed" else 1
    )
    save_every_generation_keys = max(1, int(save_every_generation_keys))
    checkpoint_order = [str(cp.get("checkpoint_id")) for cp in checkpoints if str(cp.get("checkpoint_id"))]
    predictions_by_cid: Dict[str, Dict[str, Any]] = {}
    save_lock = threading.Lock()

    if debug:
        if debug_dir is None:
            debug_dir = output_path.parent / "debug_tce"
        debug_dir.mkdir(parents=True, exist_ok=True)

    if not predict_per_key:
        raise ValueError(
            "Current TCE protocol requires per-key Task A state-completion generation. "
            "Checkpoint-level combined retrieval/prompting is no longer supported."
        )

    if prepare_checkpoint_state is None:
        def prepare_checkpoint_state(cp: Dict[str, Any], memory_pool: List[Dict[str, Any]]) -> CheckpointHandle:
            return _checkpoint_handle_metadata(
                cp,
                memory_pool=memory_pool,
                state_kind="prepared_memory",
                state_ref=cp,
                extra_metadata={"baseline": baseline_name},
            )

    if retrieve_context_for_query is None:
        raise ValueError("run_pipeline requires retrieve_context_for_query.")

    if finalize_checkpoint_state is None:
        def finalize_checkpoint_state(_checkpoint_handle: CheckpointHandle) -> None:
            return None

    if answer_query is None:
        def answer_query(
            checkpoint_handle: CheckpointHandle,
            query_spec: QuerySpec,
            retrieval_result: RetrievalResult,
        ) -> AnswerExecutionResult:
            prompt = ""
            raw_out: Any = {}
            if query_spec.task_name == "Task A":
                target_value_templates = dict((query_spec.task_payload or {}).get("target_value_templates") or {})
                prompt = build_state_completion_prompt(
                    checkpoint=checkpoint_handle.state_ref if isinstance(checkpoint_handle.state_ref, dict) else {},
                    context_logs=None,
                    target_keys=query_spec.target_keys,
                    target_value_templates=target_value_templates,
                    memory_prompt_mode=memory_prompt_mode,
                    inline_memory_blocks=list(retrieval_result.inline_memory_blocks),
                    task_text_override=query_spec.answer_query_text,
                )
                if use_structured_response and ask_structured is not None:
                    model_idx = int((query_spec.task_payload or {}).get("model_idx") or 0)
                    text_format = build_generation_text_format(
                        query_spec.target_keys,
                        target_value_templates,
                        model_idx,
                    )
                    raw_out = ask_structured(prompt, text_format)
                else:
                    raw_out = ask_json(prompt)
            elif query_spec.task_name == "Task B":
                prompt = _build_change_prompt_from_queryspec(
                    memory_prompt_mode=memory_prompt_mode,
                    query_spec=query_spec,
                    changed_value_templates=dict((query_spec.task_payload or {}).get("changed_value_templates") or {}),
                    retrieval_result=retrieval_result,
                )
                raw_out = ask_json(prompt)
            elif query_spec.task_name == "Task C":
                prompt = _build_apply_prompt_from_queryspec(
                    memory_prompt_mode=memory_prompt_mode,
                    query_spec=query_spec,
                    retrieval_result=retrieval_result,
                )
                raw_out = ask_json(prompt)
            elif query_spec.task_name == "Final QA":
                from .final_checkpoint_qa import (
                    build_final_qa_prompt_with_agent_memory,
                    build_final_qa_prompt_with_inline_memory,
                )

                if memory_prompt_mode == "inline_memory":
                    prompt = build_final_qa_prompt_with_inline_memory(
                        question=query_spec.answer_query_text,
                        context_text="\n<->\n".join(list(retrieval_result.inline_memory_blocks)),
                    )
                elif memory_prompt_mode == "agent_memory":
                    prompt = build_final_qa_prompt_with_agent_memory(question=query_spec.answer_query_text)
                else:
                    raise ValueError(
                        "Unsupported memory_prompt_mode for default answer_query: {}".format(memory_prompt_mode)
                    )
                raw_out = ask_json(prompt)
            else:
                raise ValueError("Unsupported task_name for default answer_query: {}".format(query_spec.task_name))
            return AnswerExecutionResult(
                raw_output=raw_out,
                prompt=prompt,
                debug_metadata={"retrieval_metadata": dict(retrieval_result.debug_metadata or {})},
            )

    def _ordered_predictions() -> List[Dict[str, Any]]:
        return [
            copy.deepcopy(predictions_by_cid[cid])
            for cid in checkpoint_order
            if cid in predictions_by_cid
        ]

    def _save_predictions_snapshot() -> None:
        payload = {
            "user_id": benchmark.get("user_id"),
            "benchmark_path": str(benchmark_path),
            "predictions": _ordered_predictions(),
        }
        payload.update(normalized_contract_metadata(benchmark))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_checkpoint_item(item: Dict[str, Any]) -> None:
        cid = str(item.get("checkpoint_id") or "")
        if not cid:
            return
        with save_lock:
            predictions_by_cid[cid] = copy.deepcopy(item)
            _save_predictions_snapshot()

    def _process_checkpoint(cp: Dict[str, Any]) -> Dict[str, Any]:
        cid = str(cp.get("checkpoint_id"))
        existing_item = existing.get(cid) if resume else None
        observed, cp_dt, cp_ts = observed_logs_for_checkpoint(cp, app_logs)

        memory_pool = observed if not max_visible_logs or max_visible_logs <= 0 else observed[-max_visible_logs:]
        checkpoint_handle = ensure_checkpoint_handle(
            prepare_checkpoint_state(cp, memory_pool),
            checkpoint_id=cid,
        )
        checkpoint_handle.metadata.setdefault("checkpoint_timestamp", cp_ts)

        state_completion_pack_used = False
        prebuilt_state_completion_records: Dict[str, Any] = {}
        target_keys: List[str] = []
        target_value_templates: Dict[str, Any] = {}
        target_key_status: Dict[str, Any] = {}
        if task_selection == "task_c_only":
            target_keys = extract_pack_keys(cp.get("rq3_apply_service_qa"))
        else:
            pack_targets = build_state_completion_targets_from_pack(cp)
            if pack_targets is not None:
                (
                    target_keys,
                    target_value_templates,
                    target_key_status,
                    prebuilt_state_completion_records,
                ) = pack_targets
                state_completion_pack_used = True
            else:
                raise ValueError(
                    "Task A requires state_completion_pack for checkpoint {}.".format(cid or "<unknown>")
                )

            for key in target_keys:
                _require_nonempty_pack_query(
                    task_name="Task A",
                    checkpoint_id=cid,
                    item_key=key,
                    query_text=(prebuilt_state_completion_records.get(key) or {}).get("retrieval_query") or "",
                )

        item: Dict[str, Any] = {
            "checkpoint_id": cid,
            "snapshot_state": {},
            "evidence": {},
            "metadata": {
                "task_selection": task_selection,
                "checkpoint_timestamp": cp_ts,
                "checkpoint_date": cp_dt.strftime("%Y-%m-%d") if cp_dt != datetime.max else "",
                "history_mode": "full_until_checkpoint" if not max_visible_logs or max_visible_logs <= 0 else "truncated_tail",
                "max_visible_logs": max_visible_logs,
                "num_observed_logs": len(observed),
                "num_memory_pool_logs": len(memory_pool),
                "target_keys": target_keys,
                "target_key_count": len(target_keys),
                "target_value_templates": target_value_templates,
                "target_key_status": target_key_status,
                "state_completion_pack_used": state_completion_pack_used,
                "baseline": baseline_name,
                "concurrency_policy": {
                    "checkpoint_parallelism": policy.checkpoint_parallelism,
                    "within_checkpoint_parallelism": policy.within_checkpoint_parallelism,
                },
                "requested_checkpoint_workers": requested_checkpoint_workers,
                "requested_within_checkpoint_workers": requested_within_checkpoint_workers,
                "effective_checkpoint_workers": checkpoint_workers,
                "effective_within_checkpoint_workers": within_checkpoint_workers,
                "sampled_checkpoint_id": cid,
                "predict_per_key": predict_per_key,
                "_checkpoint_complete": False,
            },
        }
        if task_selection == "task_c_only":
            item["metadata"]["task_a"] = {
                "enabled": False,
                "skipped_by_task_selection": True,
            }
        source_cid = cp.get("_source_checkpoint_id")
        if source_cid:
            item["metadata"]["source_checkpoint_id"] = source_cid
        if cid in checkpoint_sampling_meta:
            sampling_meta = checkpoint_sampling_meta[cid]
            item["metadata"]["sampling_mode"] = str(sampling_meta.get("sampling_mode", ""))
            if isinstance(sampling_meta.get("sampling_params"), dict):
                item["metadata"]["sampling_params"] = dict(sampling_meta.get("sampling_params") or {})
            item["metadata"]["sampling_strategy"] = {
                "mode": item["metadata"].get("sampling_mode", ""),
                "params": item["metadata"].get("sampling_params", {}),
            }
        if save_prompt_and_raw:
            item["metadata"]["prompt"] = []
            item["metadata"]["raw_model_output"] = {"mode": "per_key", "records": []}

        save_counter = 0

        def _maybe_persist() -> None:
            nonlocal save_counter
            save_counter += 1
            if save_counter % save_every_generation_keys == 0:
                _persist_checkpoint_item(item)

        try:
            if task_selection == "task_c_only":
                item["snapshot_state"] = {}
                item["evidence"] = {}
                item["metadata"]["retrieval_mode"] = "task_c_only"
                if save_prompt_and_raw:
                    item["metadata"]["prompt"] = []
                    item["metadata"]["raw_model_output"] = {"mode": "task_c_only", "records": []}
            elif state_completion_pack_used and not target_keys:
                item["snapshot_state"] = {}
                item["evidence"] = {}
                item["metadata"]["retrieval_mode"] = "pack_empty_scope"
                if save_prompt_and_raw:
                    item["metadata"]["prompt"] = []
                    item["metadata"]["raw_model_output"] = {"mode": "pack_empty_scope", "records": []}
            else:
                per_key_records_by_key: Dict[str, Dict[str, Any]] = {}
                per_key_retrieval_by_key: Dict[str, Dict[str, Any]] = {}
                item["snapshot_state"] = {}
                item["evidence"] = {}
                task_a_resume_state = _extract_task_a_resume_state(
                    existing_item if resume else None,
                    target_keys,
                )
                item["snapshot_state"].update(task_a_resume_state.get("valid_snapshot_state") or {})
                item["evidence"].update(task_a_resume_state.get("valid_evidence") or {})
                per_key_records_by_key.update(task_a_resume_state.get("valid_records_by_key") or {})
                per_key_retrieval_by_key.update(task_a_resume_state.get("valid_retrieval_by_key") or {})
                pending_target_keys = list(task_a_resume_state.get("pending_keys") or [])
                reused_task_a_keys = len(target_keys) - len(pending_target_keys)
                if resume and existing_item is not None and reused_task_a_keys > 0:
                    print(
                        "[TCE][resume] checkpoint {} reusing {} Task A keys and rerunning {} pending keys.".format(
                            cid,
                            reused_task_a_keys,
                            len(pending_target_keys),
                        )
                    )

                def _run_state_completion_key(key_idx: int, key: str) -> Dict[str, Any]:
                    single_keys = [key]
                    single_template = {key: target_value_templates.get(key)}
                    pack_record = prebuilt_state_completion_records.get(key) or {}
                    retrieval_query_text = _require_nonempty_pack_query(
                        task_name="Task A",
                        checkpoint_id=cid,
                        item_key=key,
                        query_text=pack_record.get("retrieval_query") or "",
                    )
                    answer_query_text = str(pack_record.get("question_text") or retrieval_query_text or "").strip()
                    query_spec = _build_task_query_spec(
                        task_name="Task A",
                        checkpoint=cp,
                        item_key=key,
                        target_keys=single_keys,
                        task_query_text=answer_query_text,
                        retrieval_query_text=retrieval_query_text,
                        answer_query_text=answer_query_text,
                        task_payload={
                            "target_value_templates": single_template,
                            "model_idx": checkpoint_order.index(cid) * 1000 + target_keys.index(key),
                        },
                    )
                    retrieval_options = RetrievalOptions(
                        common={},
                        backend=dict(retrieval_options_backend or {}),
                    )
                    single_error_messages: List[str] = []
                    single_raw_out: Any = {}
                    single_prompt = ""
                    answer_result = AnswerExecutionResult(raw_output={}, prompt="", debug_metadata={})
                    retrieval_result = RetrievalResult(mode="", inline_memory_blocks=[], debug_metadata={})
                    try:
                        retrieval_result = ensure_retrieval_result(
                            retrieve_context_for_query(checkpoint_handle, query_spec, retrieval_options, memory_pool)
                        )
                        answer_result = ensure_answer_execution_result(
                            answer_query(checkpoint_handle, query_spec, retrieval_result)
                        )
                        single_raw_out = answer_result.raw_output
                        single_prompt = answer_result.prompt
                        single_out = normalize_generation_output(single_raw_out, single_keys)
                    except Exception as exc:
                        single_error_messages.append(f"task_a_failed: {exc}")
                        single_raw_out = {
                            "_error": "; ".join(single_error_messages)
                            if single_error_messages
                            else f"task_a_failed: {exc}"
                        }
                        single_out = {"snapshot_state": {}, "evidence": {}}

                    if single_error_messages and isinstance(single_raw_out, dict):
                        single_raw_out = dict(single_raw_out)
                        single_raw_out["_warnings"] = single_error_messages

                    single_snapshot = flatten_snapshot(single_out.get("snapshot_state"))
                    if state_completion_pack_used:
                        single_snapshot = {key: single_snapshot.get(key)}
                    else:
                        single_snapshot = {key: drop_excluded_fields(single_snapshot.get(key))}
                    single_evidence = normalize_evidence_prediction(single_out.get("evidence"), single_keys)
                    retrieval_meta = dict(retrieval_result.debug_metadata or {})
                    retrieval_meta.update(dict((answer_result.debug_metadata or {}).get("retrieval_metadata") or {}))
                    retrieval_meta.setdefault("checkpoint_state_kind", checkpoint_handle.state_kind)
                    retrieval_meta.setdefault("retrieval_query", query_spec.retrieval_query_text)
                    _attach_inline_memory_stats(retrieval_meta, retrieval_result, exposure_tokenizer_model)
                    return {
                        "key": key,
                        "snapshot_value": align_prediction_to_template(
                            single_snapshot.get(key),
                            target_value_templates.get(key),
                        ),
                        "evidence": single_evidence.get(key, []),
                        "record": {
                            "key": key,
                            "prompt": single_prompt,
                            "retrieval_query": query_spec.retrieval_query_text,
                            "retrieval_metadata": retrieval_meta,
                            "raw_model_output": single_raw_out,
                        },
                        "retrieval": {
                            "key": key,
                            "retrieval_query": query_spec.retrieval_query_text,
                            "retrieval_metadata": retrieval_meta,
                            "context_log_ids": list(retrieval_meta.get("retrieved_app_log_ids") or []),
                        },
                    }

                with tqdm(
                    total=len(pending_target_keys),
                    desc=f"{baseline_name.upper()} {cid} Task A",
                    unit="key",
                    position=1,
                    leave=False,
                    disable=checkpoint_workers != 1 or not pending_target_keys,
                ) as task_a_progress:
                    with ThreadPoolExecutor(max_workers=within_checkpoint_workers) as executor:
                        futures = {
                            executor.submit(_run_state_completion_key, key_idx, key): key
                            for key_idx, key in enumerate(pending_target_keys)
                        }
                        for future in as_completed(futures):
                            result = future.result()
                            key = result["key"]
                            item["snapshot_state"][key] = result["snapshot_value"]
                            item["evidence"][key] = result["evidence"]
                            per_key_records_by_key[key] = result["record"]
                            per_key_retrieval_by_key[key] = result["retrieval"]
                            ordered_records = [
                                per_key_records_by_key[state_key]
                                for state_key in target_keys
                                if state_key in per_key_records_by_key
                            ]
                            ordered_retrieval = [
                                per_key_retrieval_by_key[state_key]
                                for state_key in target_keys
                                if state_key in per_key_retrieval_by_key
                            ]
                            item["metadata"]["retrieval_mode"] = "per_key_isolated"
                            item["metadata"]["per_key_retrieval"] = ordered_retrieval
                            if save_prompt_and_raw:
                                item["metadata"]["prompt"] = [x["prompt"] for x in ordered_records]
                                item["metadata"]["raw_model_output"] = {"mode": "per_key", "records": ordered_records}
                            task_a_progress.set_postfix({"key": key})
                            task_a_progress.update(1)
                            _maybe_persist()

                ordered_records = [
                    per_key_records_by_key[state_key]
                    for state_key in target_keys
                    if state_key in per_key_records_by_key
                ]
                ordered_retrieval = [
                    per_key_retrieval_by_key[state_key]
                    for state_key in target_keys
                    if state_key in per_key_retrieval_by_key
                ]
                item["metadata"]["retrieval_mode"] = "per_key_isolated"
                item["metadata"]["per_key_retrieval"] = ordered_retrieval
                if save_prompt_and_raw:
                    item["metadata"]["prompt"] = [x["prompt"] for x in ordered_records]
                    item["metadata"]["raw_model_output"] = {"mode": "per_key", "records": ordered_records}
                if reused_task_a_keys > 0 and not pending_target_keys:
                    _maybe_persist()

            if task_selection == "task_c_only" and enable_change_reasoning:
                item["change_analysis"] = {}
                item["metadata"]["change_reasoning"] = {
                    "enabled": False,
                    "requested": True,
                    "skipped_by_task_selection": True,
                }
            elif enable_change_reasoning and task_b_supported_by_contract:
                change_pack_used = False
                prebuilt_change_records: Dict[str, Any] = {}
                pack_change_info = build_change_targets_from_pack(cp)
                if pack_change_info is not None:
                    changed_keys, changed_templates, prev_cutoff_ts, prebuilt_change_records = pack_change_info
                    change_pack_used = True
                else:
                    raise ValueError(
                        "Task B requires change_tracking_pack for checkpoint {} when enable_change_reasoning=true.".format(
                            cid or "<unknown>"
                        )
                    )
                if changed_keys and prev_cutoff_ts:
                    change_analysis: Dict[str, Any] = {}
                    per_key_records: List[Dict[str, Any]] = []

                    def _run_change_key(key: str) -> Dict[str, Any]:
                        single_keys = [key]
                        single_templates = {key: changed_templates.get(key, {})}
                        pack_record = prebuilt_change_records.get(key) or {}
                        change_retrieval_query = _require_nonempty_pack_query(
                            task_name="Task B",
                            checkpoint_id=cid,
                            item_key=key,
                            query_text=pack_record.get("retrieval_query") or "",
                        )
                        change_answer_query = str(pack_record.get("question_text") or change_retrieval_query or "").strip()
                        query_spec = _build_task_query_spec(
                            task_name="Task B",
                            checkpoint=cp,
                            item_key=key,
                            target_keys=single_keys,
                            task_query_text=change_answer_query,
                            retrieval_query_text=change_retrieval_query,
                            answer_query_text=change_answer_query,
                            task_payload={"changed_value_templates": single_templates},
                        )
                        retrieval_result = RetrievalResult(mode="", inline_memory_blocks=[], debug_metadata={})
                        change_prompt = ""
                        change_raw: Any = {}
                        try:
                            retrieval_result = ensure_retrieval_result(
                                retrieve_context_for_query(
                                    checkpoint_handle,
                                    query_spec,
                                    RetrievalOptions(common={}, backend=dict(retrieval_options_backend or {})),
                                    memory_pool,
                                )
                            )
                            answer_result = ensure_answer_execution_result(
                                answer_query(checkpoint_handle, query_spec, retrieval_result)
                            )
                            change_prompt = answer_result.prompt
                            change_raw = answer_result.raw_output
                            parsed_single = normalize_change_reasoning_output(change_raw, single_keys)
                        except Exception as exc:
                            answer_result = AnswerExecutionResult(raw_output={}, prompt="", debug_metadata={})
                            change_raw = {"_error": f"change_reasoning_failed: {exc}"}
                            parsed_single = {
                                key: {"before": None, "after": None, "change_reason": "", "evidence": []}
                            }
                        retrieval_meta = dict(retrieval_result.debug_metadata or {})
                        retrieval_meta.update(dict((answer_result.debug_metadata or {}).get("retrieval_metadata") or {}))
                        retrieval_meta.setdefault("checkpoint_state_kind", checkpoint_handle.state_kind)
                        retrieval_meta.setdefault("retrieval_query", query_spec.retrieval_query_text)
                        _attach_inline_memory_stats(retrieval_meta, retrieval_result, exposure_tokenizer_model)
                        return {
                            "key": key,
                            "change_value": parsed_single.get(
                                key,
                                {"before": None, "after": None, "change_reason": "", "evidence": []},
                            ),
                            "record": {
                                "key": key,
                                "prompt": change_prompt,
                                "retrieval_query": query_spec.retrieval_query_text,
                                "retrieval_metadata": retrieval_meta,
                                "raw_model_output": change_raw,
                            },
                        }

                    with ThreadPoolExecutor(max_workers=within_checkpoint_workers) as executor:
                        futures = {executor.submit(_run_change_key, key): key for key in changed_keys}
                        for future in as_completed(futures):
                            result = future.result()
                            key = result["key"]
                            change_analysis[key] = result["change_value"]
                            per_key_records.append(result["record"])
                            item["change_analysis"] = change_analysis
                            item["metadata"]["change_reasoning"] = {
                                "enabled": True,
                                "mode": "per_key",
                                "change_tracking_pack_used": change_pack_used,
                                "changed_keys_groundtruth": changed_keys,
                                "previous_cutoff_ts": prev_cutoff_ts,
                                "per_key_records": per_key_records,
                            }
                            _maybe_persist()
                else:
                    item["change_analysis"] = {}
                    item["metadata"]["change_reasoning"] = {
                        "enabled": True,
                        "mode": "per_key",
                        "change_tracking_pack_used": change_pack_used,
                        "changed_keys_groundtruth": changed_keys,
                        "previous_cutoff_ts": prev_cutoff_ts,
                    }
            elif enable_change_reasoning:
                item["metadata"]["change_reasoning"] = {
                    "enabled": False,
                    "requested": True,
                    "skipped_by_task_contract": benchmark_contract_version,
                }

            task_c_invalid_item_count = 0
            if enable_rq3_apply_service_qa:
                rq3_target_keys = list(target_keys)
                if task_selection == "task_c_only":
                    rq3_target_keys = extract_pack_keys(cp.get("rq3_apply_service_qa"))
                rq3_pack = _extract_rq3_apply_pack_for_checkpoint(
                    cp,
                    target_keys=rq3_target_keys,
                    task_contract_version=benchmark_contract_version,
                )
                if not rq3_pack:
                    item["rq3_apply_answers"] = {}
                    item["metadata"]["rq3_apply"] = {
                        "enabled": True,
                    }
                else:
                    resume_state = _extract_rq3_apply_resume_state(
                        existing_item,
                        rq3_pack,
                        task_contract_version=benchmark_contract_version,
                    )
                    rq3_answers: Dict[str, Any] = {
                        key: {"items": [copy.deepcopy(x) for x in value]}
                        for key, value in (resume_state.get("valid_answers_by_key") or {}).items()
                    }
                    rq3_raw_records: List[Dict[str, Any]] = []
                    discard_counts: Dict[str, int] = {}
                    invalid_counts: Dict[str, int] = {}
                    pending_items_by_key = dict(resume_state.get("pending_items_by_key") or {})

                    reused_valid_items = sum(
                        len(items)
                        for items in (resume_state.get("valid_answers_by_key") or {}).values()
                    )
                    pending_items = sum(len(items) for items in pending_items_by_key.values())
                    if resume and existing_item is not None and reused_valid_items > 0:
                        print(
                            "[TCE][resume] checkpoint {} reusing {} valid Task C items and rerunning {} pending items.".format(
                                cid,
                                reused_valid_items,
                                pending_items,
                            )
                        )

                    task_c_progress = tqdm(
                        total=pending_items,
                        desc=f"{baseline_name.upper()} {cid} Task C",
                        unit="item",
                        position=1,
                        leave=False,
                        disable=checkpoint_workers != 1 or pending_items <= 0,
                    )

                    def _run_apply_key(key: str) -> Dict[str, Any]:
                        item_answers: List[Dict[str, Any]] = [
                            copy.deepcopy(x)
                            for x in (resume_state.get("valid_answers_by_key") or {}).get(key, [])
                        ]
                        raw_records: List[Dict[str, Any]] = [
                            copy.deepcopy(x)
                            for x in (resume_state.get("valid_raw_records_by_key") or {}).get(key, [])
                        ]
                        key_invalid_count = 0
                        pending_items = pending_items_by_key.get(key, [])
                        for qa_item in pending_items:
                            qa_id = str(qa_item.get("qa_id") or "")
                            service_family = str(qa_item.get("service_family") or "")
                            structured_task_c_v2 = task_contract_is_v2(benchmark_contract_version)
                            service_category = str(qa_item.get("service_category") or "")
                            apply_scenario = str(qa_item.get("apply_scenario") or qa_item.get("scenario") or "")
                            apply_q = str(qa_item.get("apply_question") or qa_item.get("task_instruction") or "")
                            output_template = qa_item.get("output_template")
                            apply_retrieval_query = _require_nonempty_pack_query(
                                task_name="Task C",
                                checkpoint_id=cid,
                                item_key="{}::{}".format(key, qa_id or "<unknown>"),
                                query_text=qa_item.get("retrieval_query") or "",
                            )
                            query_spec = _build_task_query_spec(
                                task_name="Task C",
                                checkpoint=cp,
                                item_key="{}::{}".format(key, qa_id or "<unknown>"),
                                target_keys=[key],
                                task_query_text=apply_retrieval_query,
                                retrieval_query_text=apply_retrieval_query,
                                answer_query_text="",
                                task_payload={
                                    "service_category": service_category,
                                    "apply_scenario": apply_scenario,
                                    "service_family": service_family,
                                    "scenario": apply_scenario,
                                    "task_instruction": apply_q,
                                    "output_template": output_template,
                                    "structured_task_c_v2": structured_task_c_v2,
                                },
                            )
                            retrieval_result = RetrievalResult(mode="", inline_memory_blocks=[], debug_metadata={})
                            apply_prompt = ""
                            apply_raw: Any = {}
                            invalid_reason = ""
                            try:
                                retrieval_result = ensure_retrieval_result(
                                    retrieve_context_for_query(
                                        checkpoint_handle,
                                        query_spec,
                                        RetrievalOptions(
                                            common=(
                                                {"top_k": int(rq3_apply_retrieval_top_k)}
                                                if rq3_apply_retrieval_top_k is not None
                                                else {}
                                            ),
                                            backend=dict(retrieval_options_backend or {}),
                                        ),
                                        memory_pool,
                                    )
                                )
                                answer_result = ensure_answer_execution_result(
                                    answer_query(checkpoint_handle, query_spec, retrieval_result)
                                )
                                apply_prompt = answer_result.prompt
                                apply_raw = answer_result.raw_output
                            except Exception as exc:
                                answer_result = AnswerExecutionResult(raw_output={}, prompt="", debug_metadata={})
                                invalid_reason = f"rq3_apply_failed: {exc}"
                                apply_raw = {"_error": invalid_reason}
                            apply_norm = _normalize_rq3_apply_answer_output(
                                apply_raw,
                                task_contract_version=benchmark_contract_version,
                                service_family=service_family,
                            )
                            answer_payload = {
                                "qa_id": qa_id,
                                "evidence": apply_norm.get("evidence", []),
                            }
                            is_valid = False
                            if structured_task_c_v2 and service_family != "user_communication":
                                answer_payload["service_family"] = service_family
                                normalized_answer = apply_norm.get("answer")
                                is_valid = not _is_effectively_empty_prediction_value(normalized_answer)
                                if is_valid:
                                    answer_payload["answer"] = normalized_answer
                            elif structured_task_c_v2:
                                answer_payload["service_family"] = service_family
                                normalized_answer = apply_norm.get("answer", "")
                                is_valid = not _is_effectively_empty_prediction_value(normalized_answer)
                                if is_valid:
                                    answer_payload["answer"] = normalized_answer
                            else:
                                answer_payload["service_category"] = service_category
                                normalized_answer = apply_norm.get("answer", "")
                                is_valid = not _is_effectively_empty_prediction_value(normalized_answer)
                                if is_valid:
                                    answer_payload["answer"] = normalized_answer
                            if is_valid:
                                answer_payload["status"] = "valid"
                            else:
                                if not invalid_reason:
                                    invalid_reason = "empty_model_output"
                                answer_payload["status"] = "invalid"
                                answer_payload["invalid_reason"] = invalid_reason
                                key_invalid_count += 1
                            item_answers.append(answer_payload)
                            retrieval_meta = dict(retrieval_result.debug_metadata or {})
                            retrieval_meta.update(dict((answer_result.debug_metadata or {}).get("retrieval_metadata") or {}))
                            retrieval_meta.setdefault("checkpoint_state_kind", checkpoint_handle.state_kind)
                            retrieval_meta.setdefault("retrieval_query", query_spec.retrieval_query_text)
                            _attach_inline_memory_stats(retrieval_meta, retrieval_result, exposure_tokenizer_model)
                            raw_records.append(
                                {
                                    "key": key,
                                    "qa_id": qa_id,
                                    "scenario": apply_scenario,
                                    "question": apply_q,
                                    "service_family": service_family,
                                    "output_template": output_template,
                                    "retrieval_query": apply_retrieval_query,
                                    "prompt": apply_prompt,
                                    "retrieval_metadata": retrieval_meta,
                                    "raw_model_output": apply_raw,
                                    "status": "valid" if is_valid else "invalid",
                                    "invalid_reason": invalid_reason if not is_valid else "",
                                }
                            )
                            task_c_progress.set_postfix({"key": key, "qa_id": qa_id})
                            task_c_progress.update(1)
                        expected_items = (((cp.get("rq3_apply_service_qa") or {}).get("keys", {}).get(key, {}) or {}).get("items", []))
                        return {
                            "key": key,
                            "answers": {"items": item_answers},
                            "discard_count": max(0, int(len(expected_items)) - int(len(item_answers))),
                            "invalid_count": int(key_invalid_count),
                            "raw_records": raw_records,
                        }

                    try:
                        with ThreadPoolExecutor(max_workers=within_checkpoint_workers) as executor:
                            futures = {executor.submit(_run_apply_key, key): key for key in sorted(rq3_pack.keys())}
                            for future in as_completed(futures):
                                result = future.result()
                                key = result["key"]
                                rq3_answers[key] = result["answers"]
                                discard_counts[key] = result["discard_count"]
                                invalid_counts[key] = result["invalid_count"]
                                task_c_invalid_item_count += int(result["invalid_count"])
                                rq3_raw_records.extend(result["raw_records"])
                                item["rq3_apply_answers"] = rq3_answers
                                item["metadata"]["rq3_apply"] = {
                                    "enabled": True,
                                    "rq3_apply_retrieval_top_k": rq3_apply_retrieval_top_k,
                                    "validator_model": str(((cp.get("rq3_apply_service_qa") or {}).get("validator") or {}).get("model", "")),
                                    "discard_counts": discard_counts,
                                    "invalid_counts": invalid_counts,
                                }
                                if rq3_apply_save_prompt_and_raw:
                                    item["metadata"]["rq3_apply"]["records"] = rq3_raw_records
                                _maybe_persist()
                    finally:
                        task_c_progress.close()
            else:
                item["metadata"]["rq3_apply"] = {"enabled": False}

            if debug and debug_dir is not None:
                write_debug_artifact(
                    debug_dir=debug_dir,
                    checkpoint_id=cid,
                    payload={
                        "checkpoint_id": cid,
                        "checkpoint_state_kind": checkpoint_handle.state_kind,
                        "target_keys": target_keys,
                        "target_value_templates": target_value_templates,
                        "per_key_retrieval": item.get("metadata", {}).get("per_key_retrieval"),
                        "prompt": item.get("metadata", {}).get("prompt"),
                        "raw_model_output": item.get("metadata", {}).get("raw_model_output"),
                        "normalized_prediction": {
                            "snapshot_state": item.get("snapshot_state"),
                            "evidence": item.get("evidence"),
                        },
                    },
                )

            item["metadata"]["_checkpoint_complete"] = (task_c_invalid_item_count == 0)
            _persist_checkpoint_item(item)
            return item
        finally:
            finalize_checkpoint_state(checkpoint_handle)

    progress = tqdm(checkpoints, desc=f"{baseline_name.upper()}-TCE checkpoints", unit="cp")
    try:
        pending = []
        for cp in checkpoints:
            cid = str(cp.get("checkpoint_id"))
            existing_item = existing.get(cid)
            if existing_item is not None and _is_checkpoint_prediction_reusable_for_resume(
                cp,
                existing_item,
                task_contract_version=benchmark_contract_version,
                enable_rq3_apply_service_qa=enable_rq3_apply_service_qa,
            ):
                predictions_by_cid[cid] = copy.deepcopy(existing[cid])
                progress.update(1)
                _save_predictions_snapshot()
                continue
            if existing_item is not None and _is_checkpoint_prediction_complete(existing_item):
                print(
                    "[TCE][resume] checkpoint {} is marked complete in the saved prediction, "
                    "but Task C outputs are missing or empty; regenerating this checkpoint.".format(cid)
                )
            pending.append(cp)
        progress.set_postfix({"pending": len(pending)})

        with ThreadPoolExecutor(max_workers=checkpoint_workers) as executor:
            futures = {
                executor.submit(_process_checkpoint, cp): str(cp.get("checkpoint_id"))
                for cp in pending
            }
            for future in as_completed(futures):
                cid = futures[future]
                future.result()
                progress.set_postfix({"completed": cid})
                progress.update(1)

        final_qa_summary: Optional[Dict[str, Any]] = None
        if task_selection == "task_c_only" and enable_final_qa:
            final_qa_summary = {
                "enabled": False,
                "requested": True,
                "skipped_by_task_selection": True,
            }
        elif enable_final_qa:
            from .final_checkpoint_qa import (
                _final_qa_output_path,
                _item_key,
                _load_qa_list,
                _normalize_answer_output,
                _reference_app_logs,
                _select_qa_path,
            )

            if checkpoints:
                final_checkpoint = checkpoints[-1]
                final_checkpoint_id = str(final_checkpoint.get("checkpoint_id", "")).strip()
                observed_logs, _cp_dt, final_cp_ts = observed_logs_for_checkpoint(final_checkpoint, app_logs)
                memory_pool = (
                    observed_logs if not max_visible_logs or max_visible_logs <= 0 else observed_logs[-max_visible_logs:]
                )
                app_log_by_id: Dict[str, Dict[str, Any]] = {}
                for log in app_logs:
                    app_log_id = log.get("app_log_id")
                    if app_log_id is not None:
                        app_log_by_id[str(app_log_id).strip()] = log

                qa_path = _select_qa_path(benchmark_path, str(benchmark.get("user_id") or ""), final_qa_path)
                qa_items = _load_qa_list(qa_path)
                qa_output = _final_qa_output_path(output_path, final_qa_output_path)

                existing_results: List[Dict[str, Any]] = []
                done_keys = set()
                if resume and qa_output.exists():
                    try:
                        existing_raw = json.loads(qa_output.read_text(encoding="utf-8"))
                        if isinstance(existing_raw, list):
                            existing_results = [item for item in existing_raw if isinstance(item, dict)]
                            for qa_item in existing_results:
                                key = _item_key(qa_item)
                                if key is not None:
                                    done_keys.add(key)
                    except Exception:
                        existing_results = []
                        done_keys = set()

                final_checkpoint_handle = ensure_checkpoint_handle(
                    prepare_checkpoint_state(final_checkpoint, memory_pool),
                    checkpoint_id=final_checkpoint_id,
                )
                final_checkpoint_handle.metadata.setdefault("checkpoint_timestamp", final_cp_ts)
                results = list(existing_results)
                try:
                    for qa_item in qa_items:
                        key = _item_key(qa_item)
                        if resume and key is not None and key in done_keys:
                            continue

                        query = str(qa_item.get("query", "") or "")
                        query_spec = _build_task_query_spec(
                            task_name="Final QA",
                            checkpoint=final_checkpoint,
                            item_key=str(qa_item.get("id") or query or ""),
                            target_keys=[],
                            task_query_text=query,
                            retrieval_query_text=query,
                            answer_query_text=query,
                            task_payload={},
                        )
                        retrieval_result = ensure_retrieval_result(
                            retrieve_context_for_query(
                                final_checkpoint_handle,
                                query_spec,
                                RetrievalOptions(
                                    common=(
                                        {"top_k": int(final_qa_retrieval_top_k)}
                                        if final_qa_retrieval_top_k is not None
                                        else {}
                                    ),
                                    backend=dict(retrieval_options_backend or {}),
                                ),
                                memory_pool,
                            )
                        )
                        retrieval_meta = dict(retrieval_result.debug_metadata or {})
                        retrieval_meta.setdefault("checkpoint_state_kind", final_checkpoint_handle.state_kind)
                        retrieval_meta.setdefault("retrieval_query", query_spec.retrieval_query_text)
                        _attach_inline_memory_stats(retrieval_meta, retrieval_result, exposure_tokenizer_model)
                        t1 = datetime.now().timestamp()
                        raw = None
                        answer_result = AnswerExecutionResult(raw_output={}, prompt="", debug_metadata={})
                        try:
                            answer_result = ensure_answer_execution_result(
                                answer_query(final_checkpoint_handle, query_spec, retrieval_result)
                            )
                            raw = answer_result.raw_output
                            parsed, parse_error = _normalize_answer_output(raw)
                            prompt = answer_result.prompt
                        except Exception as exc:
                            parsed = {"answer": "", "evidence": []}
                            parse_error = "LLM request failed: {}".format(exc)
                            prompt = ""
                        retrieval_meta.update(dict((answer_result.debug_metadata or {}).get("retrieval_metadata") or {}))
                        t2 = datetime.now().timestamp()

                        result_item = {
                            "id": qa_item.get("id"),
                            "query": query,
                            "reference": qa_item.get("reference", ""),
                            "prediction": parsed.get("answer", ""),
                            "reference_app_logs": _reference_app_logs(qa_item, app_log_by_id),
                            "metadata": {
                                **(qa_item.get("metadata") or {}),
                                "evidence_prediction": parsed.get("evidence", []),
                                "response_time": t2 - t1,
                                "context": list(retrieval_result.inline_memory_blocks),
                                "num_context_logs": len(retrieval_result.inline_memory_blocks),
                                "retrieval_query": query_spec.retrieval_query_text,
                                "retrieval_metadata": retrieval_meta,
                                "tce_final_checkpoint_id": final_checkpoint_id,
                                "tce_final_checkpoint_app_log_id": str(
                                    ((final_checkpoint.get("as_of") or {}).get("app_log_id") or "")
                                ),
                            },
                        }
                        if parse_error:
                            result_item["metadata"]["llm_parse_error"] = parse_error
                        if final_qa_save_prompt_and_raw:
                            result_item["metadata"]["prompt"] = prompt
                            result_item["metadata"]["raw_model_output"] = raw

                        results.append(result_item)
                        qa_output.parent.mkdir(parents=True, exist_ok=True)
                        qa_output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                finally:
                    finalize_checkpoint_state(final_checkpoint_handle)

                final_qa_summary = {
                    "enabled": True,
                    "qa_path": str(qa_path),
                    "output_path": str(qa_output),
                    "final_checkpoint_id": final_checkpoint_id,
                    "qa_count": len(results),
                }
            else:
                final_qa_summary = {"enabled": True, "qa_count": 0, "reason": "no_checkpoints"}

        persisted_result = {
            "user_id": benchmark.get("user_id"),
            "benchmark_path": str(benchmark_path),
            "predictions": _ordered_predictions(),
        }
        persisted_result.update(normalized_contract_metadata(benchmark))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(persisted_result, ensure_ascii=False, indent=2), encoding="utf-8")

        result = dict(persisted_result)
        if final_qa_summary is not None:
            result["final_qa"] = final_qa_summary
        return result
    finally:
        close()
