#!/usr/bin/env python3
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from generation.adapters.base import TceAdapterArgs


def derive_run_name(experiment_name: Any, run_id: Any) -> str:
    exp = str(experiment_name or "").strip()
    rid = str(run_id or "").strip() or "main"
    if rid == "main":
        return exp
    if not exp:
        return rid
    return "{}__{}".format(exp, rid)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Config not found: {}".format(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object at top level")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def coerce_optional_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def coerce_optional_int(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def build_from_config(cfg: Dict[str, Any]) -> TceAdapterArgs:
    runtime = cfg.get("runtime", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}
    llm = cfg.get("llm", {}) or {}
    retriever = cfg.get("retriever", {}) or {}
    retrieval = cfg.get("retrieval", {}) or {}
    final_qa = cfg.get("final_qa", {}) or {}
    baseline_params = cfg.get("baseline_params", {}) or {}

    legacy_runtime_keys = {"enable_rq3_know_apply", "rq3_fail_on_missing_pack", "rq3_save_prompt_and_raw"}
    found_legacy_runtime = sorted(k for k in legacy_runtime_keys if k in runtime)
    if found_legacy_runtime:
        raise ValueError(
            "Legacy runtime RQ3 config keys are no longer supported: {}. "
            "Use enable_rq3_apply_service_qa / rq3_apply_save_prompt_and_raw.".format(
                ", ".join(found_legacy_runtime)
            )
        )
    if "rq3_apply_fail_on_missing_pack" in runtime:
        raise ValueError(
            "runtime.rq3_apply_fail_on_missing_pack is no longer supported. "
            "Task C pack validity must be guaranteed during task-pack build, not controlled by generation runtime."
        )
    legacy_bp_keys = {
        "user_id",
        "rq3_pair_count_per_key",
        "rq3_retrieval_top_k",
        "enable_change_reasoning",
        "retrieval_top_k",
        "retriever_provider",
        "retriever_model",
        "retriever_batch_size",
        "checkpoint_workers",
        "within_checkpoint_workers",
        "save_every_generation_keys",
        "embedding_backend",
        "embedding_model",
        "embedding_model_name",
        "rq3_apply_retrieval_top_k",
        "enable_final_qa",
        "final_qa_path",
        "final_qa_output_path",
        "final_qa_retrieval_top_k",
        "final_qa_save_prompt_and_raw",
    }
    found_legacy_bp = sorted(k for k in legacy_bp_keys if k in baseline_params)
    if found_legacy_bp:
        raise ValueError(
            "Legacy baseline_params keys are no longer supported: {}. "
            "Use shared runtime / retriever / retrieval / final_qa sections instead.".format(
                ", ".join(found_legacy_bp)
            )
        )

    baseline = _get(runtime, "baseline")
    if not baseline:
        raise ValueError("Config runtime.baseline is required")

    benchmark = _get(data, "benchmark")
    app_logs_path = _get(data, "app_logs_path")
    prediction_output = _get(output, "prediction_path")
    if not benchmark or not app_logs_path or not prediction_output:
        raise ValueError("Config requires data.benchmark, data.app_logs_path, output.prediction_path")

    extras = {str(k): str(v) for k, v in baseline_params.items()}
    experiment_name = str(_get(runtime, "experiment_name", "") or "").strip()
    run_id = str(_get(runtime, "run_id", "") or "").strip() or "main"
    run_name = derive_run_name(experiment_name, run_id)
    user_id = _get(runtime, "user_id", None)
    if user_id is not None:
        user_id = str(user_id)
    if experiment_name:
        extras["__experiment_name__"] = experiment_name
    extras["__run_id__"] = run_id
    extras["__run_name__"] = run_name

    return TceAdapterArgs(
        baseline=str(baseline),
        user_id=user_id,
        benchmark=Path(str(benchmark)),
        app_logs_path=Path(str(app_logs_path)),
        output=Path(str(prediction_output)),
        max_visible_logs=_get(runtime, "max_visible_logs", None),
        llm_provider=str(_get(llm, "provider", "openai")),
        llm_model=str(_get(llm, "model", "gpt-5-mini")),
        llm_max_workers=int(_get(llm, "max_workers", 1)),
        llm_temperature=coerce_optional_float(llm.get("temperature", 0.0)),
        llm_top_p=coerce_optional_float(llm.get("top_p", 1.0)),
        llm_top_k=coerce_optional_int(llm.get("top_k", None)),
        resume=coerce_bool(_get(runtime, "resume", False)),
        max_checkpoints=_get(runtime, "max_checkpoints", None),
        debug=coerce_bool(_get(runtime, "debug", False)),
        debug_dir=Path(str(runtime["debug_dir"])) if _get(runtime, "debug_dir", None) else None,
        save_prompt_and_raw=coerce_bool(_get(runtime, "save_prompt_and_raw", False)),
        enable_change_reasoning=coerce_bool(_get(runtime, "enable_change_reasoning", False)),
        enable_rq3_apply_service_qa=coerce_bool(_get(runtime, "enable_rq3_apply_service_qa", False)),
        rq3_apply_save_prompt_and_raw=coerce_bool(_get(runtime, "rq3_apply_save_prompt_and_raw", True)),
        checkpoint_workers=int(_get(runtime, "checkpoint_workers", 1)),
        within_checkpoint_workers=int(_get(runtime, "within_checkpoint_workers", 1)),
        save_every_generation_keys=int(_get(runtime, "save_every_generation_keys", 1)),
        retriever_provider=str(_get(retriever, "provider", "openai")),
        retriever_model=str(_get(retriever, "model", "text-embedding-3-large")),
        retriever_batch_size=int(_get(retriever, "batch_size", 64)),
        retrieval_top_k=int(_get(retrieval, "top_k", 5)),
        rq3_apply_retrieval_top_k=coerce_optional_int(retrieval.get("rq3_apply_top_k", None)),
        enable_final_qa=coerce_bool(_get(final_qa, "enabled", False)),
        final_qa_path=(
            str(_get(final_qa, "path"))
            if _get(final_qa, "path", None) is not None
            else None
        ),
        final_qa_output_path=(
            str(_get(final_qa, "output_path"))
            if _get(final_qa, "output_path", None) is not None
            else None
        ),
        final_qa_retrieval_top_k=coerce_optional_int(retrieval.get("final_qa_top_k", None)),
        final_qa_save_prompt_and_raw=coerce_bool(_get(final_qa, "save_prompt_and_raw", False)),
        extras=extras,
    )


def resolved_payload(adapter_args: TceAdapterArgs, config_path: Path = None) -> Dict[str, Any]:
    payload = {
        "runtime": {
            "baseline": adapter_args.baseline,
            "user_id": adapter_args.user_id,
            "experiment_name": adapter_args.extras.get("__experiment_name__"),
            "run_id": adapter_args.extras.get("__run_id__"),
            "run_name": adapter_args.extras.get("__run_name__"),
            "max_visible_logs": adapter_args.max_visible_logs,
            "resume": adapter_args.resume,
            "debug": adapter_args.debug,
            "debug_dir": str(adapter_args.debug_dir) if adapter_args.debug_dir else None,
            "save_prompt_and_raw": adapter_args.save_prompt_and_raw,
            "enable_change_reasoning": adapter_args.enable_change_reasoning,
            "enable_rq3_apply_service_qa": adapter_args.enable_rq3_apply_service_qa,
            "rq3_apply_save_prompt_and_raw": adapter_args.rq3_apply_save_prompt_and_raw,
            "max_checkpoints": adapter_args.max_checkpoints,
            "checkpoint_workers": adapter_args.checkpoint_workers,
            "within_checkpoint_workers": adapter_args.within_checkpoint_workers,
            "save_every_generation_keys": adapter_args.save_every_generation_keys,
        },
        "data": {
            "benchmark": str(adapter_args.benchmark),
            "app_logs_path": str(adapter_args.app_logs_path),
        },
        "output": {
            "prediction_path": str(adapter_args.output),
        },
        "llm": {
            "provider": adapter_args.llm_provider,
            "model": adapter_args.llm_model,
            "max_workers": adapter_args.llm_max_workers,
            "temperature": adapter_args.llm_temperature,
            "top_p": adapter_args.llm_top_p,
            "top_k": adapter_args.llm_top_k,
        },
        "retriever": {
            "provider": adapter_args.retriever_provider,
            "model": adapter_args.retriever_model,
            "batch_size": adapter_args.retriever_batch_size,
        },
        "retrieval": {
            "top_k": adapter_args.retrieval_top_k,
            "rq3_apply_top_k": adapter_args.rq3_apply_retrieval_top_k,
            "final_qa_top_k": adapter_args.final_qa_retrieval_top_k,
        },
        "final_qa": {
            "enabled": adapter_args.enable_final_qa,
            "path": adapter_args.final_qa_path,
            "output_path": adapter_args.final_qa_output_path,
            "save_prompt_and_raw": adapter_args.final_qa_save_prompt_and_raw,
        },
        "baseline_params": {
            k: v
            for k, v in adapter_args.extras.items()
            if k not in {"__experiment_name__", "__run_id__", "__run_name__"}
        },
    }
    if config_path is not None:
        payload["source_config"] = str(config_path)
    return payload


def write_run_settings(resolved: Dict[str, Any], output_path: Path) -> Path:
    metadata = {
        "saved_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    payload = {
        "metadata": metadata,
        "resolved": resolved,
    }
    out = output_path.parent / "{}_run_settings.yaml".format(output_path.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return out


def normalize_users(raw: Any):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out = []
        for x in raw:
            if x is None:
                continue
            out.append(str(x))
        return out
    raise ValueError("users must be string or list of strings")


def render_value(value: Any, **kwargs: Any) -> Any:
    if isinstance(value, str):
        return value.format(**kwargs)
    if isinstance(value, dict):
        return {k: render_value(v, **kwargs) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, **kwargs) for v in value]
    return value
