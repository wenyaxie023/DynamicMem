#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from generation.qa_adapters import QaAdapterArgs
from generation.qa_adapters.registry import list_adapters, run_adapter
from generation.qa_config import (
    build_from_config,
    deep_merge,
    load_yaml,
    resolved_payload,
    write_run_settings,
)


def _build_from_legacy_args(args: argparse.Namespace) -> QaAdapterArgs:
    user_id = args.user_id or "10"
    baseline = args.baseline or "qa_pipeline"
    return QaAdapterArgs(
        baseline=baseline,
        command=args.command,
        subcommand=args.subcommand,
        user_id=user_id,
        input_root_dir=Path(args.input_root_dir) if args.input_root_dir else None,
        output_root_dir=Path(args.output_root_dir) if args.output_root_dir else None,
        qa_dir=Path(args.qa_dir) if args.qa_dir else None,
        output_path=Path(args.prediction_path.format(user_id=user_id, baseline=baseline))
        if args.prediction_path
        else None,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_max_workers=args.max_workers,
        retry_times=args.retry_times,
        flush_every=args.flush_every,
        sample_per_group=args.sample_per_group,
        sample_seed=args.sample_seed,
        qtypes=args.qtypes,
        categories=args.categories,
        fix_links_inplace=args.fix_links_inplace,
        run_record_path=Path(args.run_record_path.format(user_id=user_id, baseline=baseline)),
        experiment_name=args.experiment_name,
        extras={},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified QA runner")

    parser.add_argument("--config", type=Path, default=None, help="YAML config path (recommended)")
    parser.add_argument("--defaults", type=Path, default=Path("configs/qa.default.yaml"), help="YAML defaults")

    parser.add_argument("--baseline", type=str, default=None, help="one of: {}".format(", ".join(list_adapters())))
    parser.add_argument("--command", type=str, default="pipeline", choices=["pipeline", "context", "generation", "fix-links", "export-csv"])
    parser.add_argument("--subcommand", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--input-root-dir", type=str, default=None)
    parser.add_argument("--output-root-dir", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--prediction-path", type=str, default=None)

    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--retry-times", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--sample-per-group", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--qtypes", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)

    parser.add_argument("--fix-links-inplace", action="store_true")
    parser.add_argument("--run-record-path", type=str, default="generation/{baseline}/results/{user_id}/prediction/qa_run.json")
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = None
    if args.config is not None:
        config_path = args.config
        defaults = load_yaml(args.defaults) if args.defaults and args.defaults.exists() else {}
        cfg = load_yaml(args.config)
        merged = deep_merge(defaults, cfg)
        if args.experiment_name:
            merged.setdefault("runtime", {})
            merged["runtime"]["experiment_name"] = args.experiment_name
        if args.baseline:
            merged.setdefault("runtime", {})
            merged["runtime"]["baseline"] = args.baseline
        if args.user_id:
            merged.setdefault("runtime", {})
            merged["runtime"]["user_id"] = args.user_id
        adapter_args = build_from_config(merged)
    else:
        adapter_args = _build_from_legacy_args(args)

    resolved = resolved_payload(adapter_args, config_path)
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return

    settings_path = write_run_settings(resolved, adapter_args.run_record_path)
    result = run_adapter(adapter_args)
    print("Run settings:", settings_path)
    print("QA command:", " ".join(result.get("command", [])))


if __name__ == "__main__":
    main()
