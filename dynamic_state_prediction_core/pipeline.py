"""Shared generation pipeline for dynamic state prediction baselines."""

import json
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
from .prompts import build_generation_prompt

EXCLUDED_VALUE_FIELDS = {"priority", "schedule_date", "schedule_dates"}


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
    return json.dumps(
        {
            "app_log_id": log.get("app_log_id"),
            "timestamp": log.get("timestamp"),
            "app_name": log.get("app_name"),
            "api_name": log.get("api_name"),
            "request": log.get("request"),
            "response": log.get("response"),
        },
        ensure_ascii=False,
    )


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


def fill_blank_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: fill_blank_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_blank_template(v) for v in value]
    return "<fill the blank>"


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
            f"DspValueNode_{model_idx}_{node_idx}",
            __config__=ConfigDict(extra="forbid", populate_by_name=True),
            **fields,
        )
    if isinstance(value, list):
        if not value:
            return List[ScalarValue]  # type: ignore[valid-type]
        return List[_build_value_model(value[0], model_idx, counter)]  # type: ignore[valid-type]
    return ScalarValue


def build_prompt(
    *,
    checkpoint: Dict[str, Any],
    context_logs: List[Dict[str, Any]],
    context_note: str,
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    retrieval_query: Optional[str] = None,
) -> str:
    return build_generation_prompt(
        context_logs=context_logs,
        target_keys=target_keys,
        target_value_templates=target_value_templates,
        log_to_text=to_log_text,
        retrieval_query=retrieval_query,
        context_note=context_note,
    )


def normalize_evidence_prediction(raw_evidence: Any, target_keys: List[str]) -> Dict[str, List[str]]:
    def to_ids(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        seen = set()
        for item in value:
            log_id = None
            if isinstance(item, dict):
                candidate = item.get("app_log_id")
                if candidate is not None:
                    log_id = str(candidate).strip()
            elif isinstance(item, (str, int, float)):
                log_id = str(item).strip()
            if log_id and log_id not in seen:
                seen.add(log_id)
                out.append(log_id)
        return out

    evidence: Dict[str, List[str]] = {}
    if isinstance(raw_evidence, dict):
        for key in target_keys:
            evidence[key] = to_ids(raw_evidence.get(key))
    else:
        for key in target_keys:
            evidence[key] = []
    return evidence


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
            List[str],
            Field(
                ...,
                alias=key,
                description="Supporting app_log_id list for this key.",
            ),
        )

    snapshot_model = create_model(  # type: ignore[call-overload]
        f"DspSnapshotState_{model_idx}",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **snapshot_fields,
    )
    evidence_model = create_model(  # type: ignore[call-overload]
        f"DspEvidence_{model_idx}",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **evidence_fields,
    )
    return create_model(  # type: ignore[call-overload]
        f"DspPredOutput_{model_idx}",
        __config__=ConfigDict(extra="forbid"),
        snapshot_state=(snapshot_model, ...),
        evidence=(evidence_model, ...),
    )


def normalize_generation_output(raw_out: Any, target_keys: List[str]) -> Dict[str, Any]:
    if isinstance(raw_out, dict):
        # Native schema path.
        if "snapshot_state" in raw_out or "evidence" in raw_out:
            return {
                "snapshot_state": raw_out.get("snapshot_state", {}),
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
    retrieve_context: Callable[[Dict[str, Any], List[Dict[str, Any]], List[str]], Dict[str, Any]],
    baseline_name: str,
    resume: bool,
    max_checkpoints: Optional[int],
    debug: bool,
    debug_dir: Optional[Path],
    save_prompt_and_raw: bool,
    predict_per_key: bool = False,
) -> Dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    app_logs = normalize_app_logs(json.loads(app_logs_path.read_text(encoding="utf-8")))

    checkpoints = benchmark.get("checkpoints", [])
    if max_checkpoints is not None:
        checkpoints = checkpoints[:max_checkpoints]

    existing: Dict[str, Dict[str, Any]] = {}
    if resume and output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            for item in raw.get("predictions", []):
                cid = item.get("checkpoint_id")
                if cid:
                    existing[str(cid)] = item
        except Exception:
            pass

    predictions: List[Dict[str, Any]] = []

    if debug:
        if debug_dir is None:
            debug_dir = output_path.parent / "debug_dynamic_state_prediction"
        debug_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(checkpoints, desc=f"{baseline_name.upper()}-DSP checkpoints", unit="cp")
    try:
        for cp in progress:
            cid = str(cp.get("checkpoint_id"))
            progress.set_postfix({"checkpoint_id": cid})

            if cid in existing:
                predictions.append(existing[cid])
                continue

            observed, cp_dt, cp_ts = observed_logs_for_checkpoint(cp, app_logs)

            memory_pool = observed if not max_visible_logs or max_visible_logs <= 0 else observed[-max_visible_logs:]

            target_keys, target_value_templates = build_target_templates(cp)

            retrieval_meta: Dict[str, Any] = {}
            context_logs: List[Dict[str, Any]] = []
            context_note = "Context app logs"
            retrieval_query = None

            raw_out: Any = {}
            prompt: Any = ""
            if not predict_per_key:
                ctx_info = retrieve_context(cp, memory_pool, target_keys)
                context_logs = ctx_info.get("context_logs") or []
                retrieval_query = ctx_info.get("retrieval_query")
                context_note = ctx_info.get("context_note") or "Context app logs"
                retrieval_meta = ctx_info.get("metadata") or {}
                prompt = build_prompt(
                    checkpoint=cp,
                    context_logs=context_logs,
                    context_note=context_note,
                    target_keys=target_keys,
                    target_value_templates=target_value_templates,
                    retrieval_query=retrieval_query,
                )

                error_messages: List[str] = []
                try:
                    if use_structured_response and ask_structured is not None:
                        try:
                            text_format = build_generation_text_format(
                                target_keys,
                                target_value_templates,
                                len(predictions),
                            )
                            raw_out = ask_structured(prompt, text_format)
                            out = normalize_generation_output(raw_out, target_keys)
                        except Exception as exc:
                            error_messages.append(f"structured_call_failed: {exc}")
                            raw_out = ask_json(prompt)
                            out = normalize_generation_output(raw_out, target_keys)
                    else:
                        raw_out = ask_json(prompt)
                        out = normalize_generation_output(raw_out, target_keys)
                except Exception as exc:
                    error_messages.append(f"json_call_failed: {exc}")
                    raw_out = {"_error": "; ".join(error_messages) if error_messages else f"llm_call_failed: {exc}"}
                    out = {"snapshot_state": {}, "evidence": {}}

                if error_messages and isinstance(raw_out, dict):
                    raw_out = dict(raw_out)
                    raw_out["_warnings"] = error_messages

                pred_snapshot = flatten_snapshot(out.get("snapshot_state"))
                pred_snapshot = {k: drop_excluded_fields(pred_snapshot.get(k)) for k in target_keys}
                snapshot = {
                    k: align_prediction_to_template(pred_snapshot.get(k), target_value_templates.get(k))
                    for k in target_keys
                }
                evidence = normalize_evidence_prediction(out.get("evidence"), target_keys)
            else:
                per_key_records: List[Dict[str, Any]] = []
                snapshot: Dict[str, Any] = {}
                evidence: Dict[str, List[str]] = {}
                per_key_retrieval: List[Dict[str, Any]] = []
                for key_idx, key in enumerate(target_keys):
                    single_keys = [key]
                    single_template = {key: target_value_templates.get(key)}
                    single_ctx_info = retrieve_context(cp, memory_pool, single_keys)
                    single_context_logs = single_ctx_info.get("context_logs") or []
                    single_retrieval_query = single_ctx_info.get("retrieval_query")
                    single_context_note = single_ctx_info.get("context_note") or "Context app logs"
                    single_retrieval_meta = single_ctx_info.get("metadata") or {}
                    single_prompt = build_prompt(
                        checkpoint=cp,
                        context_logs=single_context_logs,
                        context_note=single_context_note,
                        target_keys=single_keys,
                        target_value_templates=single_template,
                        retrieval_query=single_retrieval_query,
                    )
                    single_error_messages: List[str] = []
                    single_raw_out: Any = {}
                    try:
                        if use_structured_response and ask_structured is not None:
                            try:
                                text_format = build_generation_text_format(
                                    single_keys,
                                    single_template,
                                    len(predictions) * 1000 + key_idx,
                                )
                                single_raw_out = ask_structured(single_prompt, text_format)
                                single_out = normalize_generation_output(single_raw_out, single_keys)
                            except Exception as exc:
                                single_error_messages.append(f"structured_call_failed: {exc}")
                                single_raw_out = ask_json(single_prompt)
                                single_out = normalize_generation_output(single_raw_out, single_keys)
                        else:
                            single_raw_out = ask_json(single_prompt)
                            single_out = normalize_generation_output(single_raw_out, single_keys)
                    except Exception as exc:
                        single_error_messages.append(f"json_call_failed: {exc}")
                        single_raw_out = {
                            "_error": "; ".join(single_error_messages)
                            if single_error_messages
                            else f"llm_call_failed: {exc}"
                        }
                        single_out = {"snapshot_state": {}, "evidence": {}}

                    if single_error_messages and isinstance(single_raw_out, dict):
                        single_raw_out = dict(single_raw_out)
                        single_raw_out["_warnings"] = single_error_messages

                    single_snapshot = flatten_snapshot(single_out.get("snapshot_state"))
                    single_snapshot = {key: drop_excluded_fields(single_snapshot.get(key))}
                    snapshot[key] = align_prediction_to_template(
                        single_snapshot.get(key),
                        target_value_templates.get(key),
                    )
                    single_evidence = normalize_evidence_prediction(single_out.get("evidence"), single_keys)
                    evidence[key] = single_evidence.get(key, [])
                    per_key_records.append(
                        {
                            "key": key,
                            "prompt": single_prompt,
                            "retrieval_query": single_retrieval_query,
                            "retrieval_metadata": single_retrieval_meta,
                            "raw_model_output": single_raw_out,
                        }
                    )
                    per_key_retrieval.append(
                        {
                            "key": key,
                            "retrieval_query": single_retrieval_query,
                            "retrieval_metadata": single_retrieval_meta,
                            "context_log_ids": [x.get("app_log_id") for x in single_context_logs],
                        }
                    )

                prompt = [x["prompt"] for x in per_key_records]
                raw_out = {"mode": "per_key", "records": per_key_records}
                retrieval_meta = {"retrieval_mode": "per_key_isolated", "per_key_retrieval": per_key_retrieval}
                context_logs = []
                context_note = "Per-key isolated retrieval contexts"

            item = {
                "checkpoint_id": cid,
                "snapshot_state": snapshot,
                "evidence": evidence,
                "metadata": {
                    "checkpoint_timestamp": cp_ts,
                    "checkpoint_date": cp_dt.strftime("%Y-%m-%d") if cp_dt != datetime.max else "",
                    "history_mode": "full_until_checkpoint" if not max_visible_logs or max_visible_logs <= 0 else "truncated_tail",
                    "max_visible_logs": max_visible_logs,
                    "num_observed_logs": len(observed),
                    "num_memory_pool_logs": len(memory_pool),
                    "target_keys": target_keys,
                    "target_key_count": len(target_keys),
                    "target_value_templates": target_value_templates,
                    "baseline": baseline_name,
                    "predict_per_key": predict_per_key,
                    **retrieval_meta,
                },
            }
            if save_prompt_and_raw:
                item["metadata"]["prompt"] = prompt
                item["metadata"]["raw_model_output"] = raw_out

            if debug and debug_dir is not None:
                write_debug_artifact(
                    debug_dir=debug_dir,
                    checkpoint_id=cid,
                    payload={
                        "checkpoint_id": cid,
                        "prompt": prompt,
                        "target_keys": target_keys,
                        "target_value_templates": target_value_templates,
                        "context_note": context_note,
                        "context_log_ids": [x.get("app_log_id") for x in context_logs],
                        "raw_model_output": raw_out,
                        "normalized_prediction": {"snapshot_state": snapshot, "evidence": evidence},
                    },
                )

            predictions.append(item)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        close()

    result = {"predictions": predictions}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
