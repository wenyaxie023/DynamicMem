#!/usr/bin/env python3
import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from generation.qa_adapters.base import QaAdapterArgs


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


def sanitize_experiment_name(name: Any) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    safe = re.sub(r"\s+", "-", raw)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", safe):
        raise ValueError(
            "Invalid runtime.experiment_name '{}'. Allowed: letters, digits, '.', '_', '-'".format(raw)
        )
    return safe


def resolve_output_path(run_record_path: Any, experiment_name: Any) -> Path:
    output_path = Path(str(run_record_path))
    exp = sanitize_experiment_name(experiment_name)
    if not exp:
        return output_path
    return output_path.parent / exp / output_path.name


def build_from_config(cfg: Dict[str, Any]) -> QaAdapterArgs:
    runtime = cfg.get("runtime", {}) or {}
    data = cfg.get("data", {}) or {}
    llm = cfg.get("llm", {}) or {}
    sampling = cfg.get("sampling", {}) or {}
    baseline_params = cfg.get("baseline_params", {}) or {}
    output = cfg.get("output", {}) or {}

    baseline = str(_get(runtime, "baseline", "qa_pipeline"))
    command = str(_get(runtime, "command", "pipeline"))
    subcommand = _get(runtime, "subcommand", None)
    if subcommand is not None:
        subcommand = str(subcommand)

    user_id = str(_get(runtime, "user_id", "10"))
    experiment_name = sanitize_experiment_name(_get(runtime, "experiment_name", ""))
    input_root_dir = _get(data, "input_root_dir", None)
    output_root_dir = _get(data, "output_root_dir", None)
    qa_dir = _get(data, "qa_dir", None)
    output_path = _get(output, "prediction_path", None)

    if isinstance(input_root_dir, str):
        input_root_dir = input_root_dir.format(user_id=user_id, baseline=baseline)
    if isinstance(output_root_dir, str):
        output_root_dir = output_root_dir.format(user_id=user_id, baseline=baseline)
    if isinstance(qa_dir, str):
        qa_dir = qa_dir.format(user_id=user_id, baseline=baseline)
    if isinstance(output_path, str):
        output_path = output_path.format(user_id=user_id, baseline=baseline)

    run_record_path = _get(output, "run_record_path", "data/user{user_id}/qa_run.json")
    if isinstance(run_record_path, str):
        run_record_path = run_record_path.format(user_id=user_id, baseline=baseline)

    return QaAdapterArgs(
        baseline=baseline,
        command=command,
        subcommand=subcommand,
        user_id=user_id,
        input_root_dir=Path(str(input_root_dir)) if input_root_dir else None,
        output_root_dir=Path(str(output_root_dir)) if output_root_dir else None,
        qa_dir=Path(str(qa_dir)) if qa_dir else None,
        output_path=Path(str(output_path)) if output_path else None,
        llm_provider=str(_get(llm, "provider", "")).strip() or None,
        llm_model=str(_get(llm, "model", "")).strip() or None,
        llm_max_workers=_get(llm, "max_workers", None),
        retry_times=_get(llm, "retry_times", None),
        flush_every=_get(llm, "flush_every", None),
        sample_per_group=_get(sampling, "sample_per_group", None),
        sample_seed=_get(sampling, "sample_seed", None),
        qtypes=str(_get(sampling, "qtypes", "")).strip() or None,
        categories=str(_get(sampling, "categories", "")).strip() or None,
        fix_links_inplace=coerce_bool(_get(runtime, "fix_links_inplace", False)),
        run_record_path=resolve_output_path(run_record_path, experiment_name),
        experiment_name=experiment_name or None,
        extras={str(k): str(v) for k, v in baseline_params.items()},
    )


def resolved_payload(adapter_args: QaAdapterArgs, config_path: Path = None) -> Dict[str, Any]:
    payload = {
        "runtime": {
            "baseline": adapter_args.baseline,
            "command": adapter_args.command,
            "subcommand": adapter_args.subcommand,
            "user_id": adapter_args.user_id,
            "experiment_name": adapter_args.experiment_name,
            "fix_links_inplace": adapter_args.fix_links_inplace,
        },
        "data": {
            "input_root_dir": str(adapter_args.input_root_dir) if adapter_args.input_root_dir else None,
            "output_root_dir": str(adapter_args.output_root_dir) if adapter_args.output_root_dir else None,
            "qa_dir": str(adapter_args.qa_dir) if adapter_args.qa_dir else None,
            "prediction_path": str(adapter_args.output_path) if adapter_args.output_path else None,
        },
        "llm": {
            "provider": adapter_args.llm_provider,
            "model": adapter_args.llm_model,
            "max_workers": adapter_args.llm_max_workers,
            "retry_times": adapter_args.retry_times,
            "flush_every": adapter_args.flush_every,
        },
        "sampling": {
            "sample_per_group": adapter_args.sample_per_group,
            "sample_seed": adapter_args.sample_seed,
            "qtypes": adapter_args.qtypes,
            "categories": adapter_args.categories,
        },
        "output": {
            "run_record_path": str(adapter_args.run_record_path),
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
