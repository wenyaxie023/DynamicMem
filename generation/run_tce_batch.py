#!/usr/bin/env python3
import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from generation.adapters.registry import run_adapter
from generation.tce_config import (
    build_from_config,
    deep_merge,
    load_yaml,
    normalize_users,
    render_value,
    resolved_payload,
    write_run_settings,
    coerce_bool,
)


def _load_benchmark_builder_api():
    module_name = "data_construction.build_tce_benchmark"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        module = importlib.import_module(module_name)
    return module.build_chain_state_validity, module.build_checkpoints


def _build_user_cfg(cfg: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    rendered = render_value(copy.deepcopy(cfg), user_id)
    return rendered


def _ensure_benchmark_exists(user_cfg: Dict[str, Any]) -> bool:
    runtime = user_cfg.get("runtime", {}) or {}
    data = user_cfg.get("data", {}) or {}
    auto_build = coerce_bool(runtime.get("auto_build_benchmark", False))

    benchmark = Path(str(data.get("benchmark", "")))
    if not benchmark:
        raise ValueError("Missing data.benchmark in config")
    if benchmark.exists():
        return False
    if not auto_build:
        raise FileNotFoundError(
            "Benchmark not found: {}. "
            "Set runtime.auto_build_benchmark=true to auto-generate.".format(benchmark)
        )

    app_logs_final_path = Path(str(data.get("app_logs_final_path") or (benchmark.parent / "app_logs_final.json")))
    if not app_logs_final_path.exists():
        raise FileNotFoundError(
            "Cannot auto-build benchmark because app_logs_final.json is missing: {}".format(app_logs_final_path)
        )

    all_events_chains_path = Path(str(data.get("all_events_chains_path") or (benchmark.parent / "all_events_chains.json")))
    if not all_events_chains_path.exists():
        raise FileNotFoundError(
            "Cannot auto-build benchmark because all_events_chains.json is missing: {}".format(
                all_events_chains_path
            )
        )

    build_chain_state_validity, build_checkpoints = _load_benchmark_builder_api()
    app_logs_final = json.loads(app_logs_final_path.read_text(encoding="utf-8"))
    all_events_chains = json.loads(all_events_chains_path.read_text(encoding="utf-8"))
    chain_state_validity = build_chain_state_validity(all_events_chains)
    benchmark_payload = build_checkpoints(app_logs_final, chain_state_validity)
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(json.dumps(benchmark_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not benchmark.exists():
        raise FileNotFoundError("Auto-build finished but benchmark still missing: {}".format(benchmark))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TCE runner with user templating")
    parser.add_argument("--config", type=Path, required=True, help="Batch YAML config")
    parser.add_argument("--defaults", type=Path, default=Path("configs/tce.default.yaml"), help="YAML defaults")
    parser.add_argument("--users", nargs="*", default=None, help="Optional user override list")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    defaults = load_yaml(args.defaults) if args.defaults and args.defaults.exists() else {}
    cfg = load_yaml(args.config)
    merged = deep_merge(defaults, cfg)

    users = [str(x) for x in (args.users or [])] or normalize_users(merged.get("users"))
    if not users:
        raise ValueError("No users provided. Set top-level 'users' in config or pass --users")

    summaries = []
    for user_id in users:
        user_cfg = _build_user_cfg(merged, user_id)
        adapter_args = build_from_config(user_cfg)
        resolved = resolved_payload(adapter_args, args.config)
        resolved["runtime"]["user_id"] = user_id

        if args.dry_run:
            summaries.append(resolved)
            continue

        built_now = _ensure_benchmark_exists(user_cfg)
        resolved["runtime"]["benchmark_built_now"] = built_now

        settings_path = write_run_settings(resolved, adapter_args.output)
        result = run_adapter(adapter_args)
        summaries.append(
            {
                "user_id": user_id,
                "prediction": str(adapter_args.output),
                "settings": str(settings_path),
                "benchmark_built_now": built_now,
                "total_checkpoints": len(result.get("predictions", [])),
            }
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
