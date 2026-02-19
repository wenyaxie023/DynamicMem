#!/usr/bin/env python3
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from generation.adapters.base import DspAdapterArgs


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


def build_from_config(cfg: Dict[str, Any]) -> DspAdapterArgs:
    runtime = cfg.get("runtime", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}
    llm = cfg.get("llm", {}) or {}
    baseline_params = cfg.get("baseline_params", {}) or {}

    baseline = _get(runtime, "baseline")
    if not baseline:
        raise ValueError("Config runtime.baseline is required")

    benchmark = _get(data, "benchmark")
    app_logs_path = _get(data, "app_logs_path")
    prediction_output = _get(output, "prediction_path")
    if not benchmark or not app_logs_path or not prediction_output:
        raise ValueError("Config requires data.benchmark, data.app_logs_path, output.prediction_path")

    return DspAdapterArgs(
        baseline=str(baseline),
        benchmark=Path(str(benchmark)),
        app_logs_path=Path(str(app_logs_path)),
        output=Path(str(prediction_output)),
        max_visible_logs=_get(runtime, "max_visible_logs", None),
        llm_provider=str(_get(llm, "provider", "openai")),
        llm_model=str(_get(llm, "model", "gpt-5-mini")),
        llm_max_workers=int(_get(llm, "max_workers", 1)),
        resume=coerce_bool(_get(runtime, "resume", False)),
        max_checkpoints=_get(runtime, "max_checkpoints", None),
        debug=coerce_bool(_get(runtime, "debug", False)),
        debug_dir=Path(str(runtime["debug_dir"])) if _get(runtime, "debug_dir", None) else None,
        save_prompt_and_raw=coerce_bool(_get(runtime, "save_prompt_and_raw", False)),
        extras={str(k): str(v) for k, v in baseline_params.items()},
    )


def resolved_payload(adapter_args: DspAdapterArgs, config_path: Path = None) -> Dict[str, Any]:
    payload = {
        "runtime": {
            "baseline": adapter_args.baseline,
            "max_visible_logs": adapter_args.max_visible_logs,
            "resume": adapter_args.resume,
            "debug": adapter_args.debug,
            "debug_dir": str(adapter_args.debug_dir) if adapter_args.debug_dir else None,
            "save_prompt_and_raw": adapter_args.save_prompt_and_raw,
            "max_checkpoints": adapter_args.max_checkpoints,
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
        },
        "baseline_params": adapter_args.extras,
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


def render_value(value: Any, user_id: str) -> Any:
    if isinstance(value, str):
        return value.format(user_id=user_id)
    if isinstance(value, dict):
        return {k: render_value(v, user_id) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, user_id) for v in value]
    return value
