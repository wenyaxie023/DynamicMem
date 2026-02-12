"""Shared generation pipeline for dynamic state prediction baselines."""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, create_model

from tqdm import tqdm

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


def null_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: null_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [null_template(v) for v in value]
    return None


def fill_blank_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: fill_blank_template(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_blank_template(v) for v in value]
    return "<fill the blank>"


def align_prediction_to_template(pred_value: Any, template_value: Any) -> Any:
    if isinstance(template_value, dict):
        pred_dict = pred_value if isinstance(pred_value, dict) else {}
        return {
            k: align_prediction_to_template(pred_dict.get(k), v)
            for k, v in template_value.items()
        }
    if isinstance(template_value, list):
        pred_list = pred_value if isinstance(pred_value, list) else []
        out: List[Any] = []
        for i, tmpl_item in enumerate(template_value):
            src = pred_list[i] if i < len(pred_list) else None
            out.append(align_prediction_to_template(src, tmpl_item))
        return out
    return pred_value


def build_prompt(
    *,
    checkpoint: Dict[str, Any],
    context_logs: List[Dict[str, Any]],
    context_note: str,
    target_keys: List[str],
    target_value_templates: Dict[str, Any],
    retrieval_query: Optional[str] = None,
) -> str:
    context = "\n<->\n".join(to_log_text(log) for log in context_logs)
    fill_dict = {k: fill_blank_template(v) for k, v in target_value_templates.items()}
    fill_block = json.dumps(fill_dict, ensure_ascii=False, indent=2)
    evidence_template = {k: ["<app_log_id>"] for k in target_keys}
    evidence_block = json.dumps(evidence_template, ensure_ascii=False, indent=2)

    return f"""Based on user memory, predict the values for the following state keys:

User memory:
{context}

Fill this dict by replacing each "<fill the blank>" with predicted values:
{fill_block}

For each key, also predict supporting evidence app log ids:
{evidence_block}

Output JSON ONLY with this schema:
{{
  "predictions": [
    {{
      "key": "<one key from the fill dict>",
      "snapshot_value": <filled value object matching template>,
      "evidence": ["<app_log_id>", "..."]
    }}
  ]
}}

Rules:
1. Use only evidence implied by logs.
2. MUST include every key from the fill dict exactly once in predictions.
3. For each key, keep exactly the same nested field structure in snapshot_value and fill values only.
4. If a field is unresolvable from logs, set it to null.
5. evidence for each key must be a list of app_log_id strings from provided user memory.
6. If evidence is unknown, use empty list.
7. No markdown. No extra keys.
"""


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


def build_generation_text_format(target_keys: Sequence[str], model_idx: int) -> Type[BaseModel]:
    allowed = tuple(target_keys) if target_keys else ("__no_key__",)
    key_enum = Enum(
        f"DspPredKey_{model_idx}",
        {f"K_{i}": key for i, key in enumerate(allowed)},
    )
    item_model = create_model(  # type: ignore[call-overload]
        f"DspPredItem_{model_idx}",
        key=(key_enum, ...),
        snapshot_value=(Any, ...),
        evidence=(List[str], ...),
    )
    output_model = create_model(  # type: ignore[call-overload]
        f"DspPredOutput_{model_idx}",
        predictions=(List[item_model], ...),
    )
    return output_model


def normalize_generation_output(raw_out: Any, target_keys: List[str]) -> Dict[str, Any]:
    if isinstance(raw_out, dict):
        # Native schema path.
        if "snapshot_state" in raw_out or "evidence" in raw_out:
            return {
                "snapshot_state": raw_out.get("snapshot_state", {}),
                "evidence": raw_out.get("evidence", {}),
            }

        # Structured schema path.
        items = raw_out.get("predictions")
        if isinstance(items, list):
            snapshot_state: Dict[str, Any] = {}
            evidence: Dict[str, Any] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                if isinstance(key, dict):
                    key = key.get("value") or key.get("name")
                if key is None:
                    continue
                key = str(key)
                if key not in target_keys:
                    continue
                snapshot_state[key] = item.get("snapshot_value")
                evidence[key] = item.get("evidence", [])
            return {
                "snapshot_state": snapshot_state,
                "evidence": evidence,
            }

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
    target_value_templates = {k: null_template(expected_snapshot_flat.get(k)) for k in target_keys}
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

            raw_out: Any = {}
            try:
                if use_structured_response and ask_structured is not None:
                    text_format = build_generation_text_format(target_keys, len(predictions))
                    raw_out = ask_structured(prompt, text_format)
                else:
                    raw_out = ask_json(prompt)
                out = normalize_generation_output(raw_out, target_keys)
            except Exception:
                raw_out = {"_error": "llm_call_failed"}
                out = {"snapshot_state": {}, "evidence": {}}

            pred_snapshot = flatten_snapshot(out.get("snapshot_state"))
            pred_snapshot = {k: drop_excluded_fields(pred_snapshot.get(k)) for k in target_keys}
            snapshot = {
                k: align_prediction_to_template(pred_snapshot.get(k), target_value_templates[k])
                for k in target_keys
            }
            evidence = normalize_evidence_prediction(out.get("evidence"), target_keys)

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
