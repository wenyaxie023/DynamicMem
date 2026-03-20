#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from generation.adapters.registry import list_adapters, run_adapter
from generation.tce_config import (
    build_from_config,
    load_yaml,
    deep_merge,
    resolved_payload,
    write_run_settings,
)
from generation.adapters.base import TceAdapterArgs


def _build_from_legacy_args(args: argparse.Namespace) -> TceAdapterArgs:
    defaults = {}
    if args.defaults and Path(args.defaults).exists():
        defaults = load_yaml(Path(args.defaults))
    llm_defaults = defaults.get("llm", {}) if isinstance(defaults, dict) else {}
    base_defaults = defaults.get("baseline_params", {}) if isinstance(defaults, dict) else {}

    llm_provider = args.llm_provider or llm_defaults.get("provider", "openai")
    llm_model = args.llm_model or llm_defaults.get("model", "gpt-5-mini")
    llm_max_workers = args.llm_max_workers or int(llm_defaults.get("max_workers", 1))

    extras = {
        "retrieval_top_k": str(args.retrieval_top_k or base_defaults.get("retrieval_top_k", 5)),
        "retriever_provider": str(args.retriever_provider or base_defaults.get("retriever_provider", "openai")),
        "retriever_model": str(args.retriever_model or base_defaults.get("retriever_model", "text-embedding-3-large")),
        "retriever_batch_size": str(args.retriever_batch_size or base_defaults.get("retriever_batch_size", 64)),
    }
    if args.hipporag_dir:
        extras["hipporag_dir"] = args.hipporag_dir

    return TceAdapterArgs(
        baseline=args.baseline,
        benchmark=args.benchmark,
        app_logs_path=args.app_logs_path,
        output=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_fail_on_missing_pack=args.rq3_apply_fail_on_missing_pack,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        extras=extras,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified TCE runner")

    parser.add_argument("--config", type=Path, default=None, help="YAML config path (preferred)")
    parser.add_argument("--defaults", type=Path, default=Path("configs/tce.default.yaml"), help="YAML defaults")

    parser.add_argument("--baseline", type=str, default=None, help="one of: {}".format(", ".join(list_adapters())))
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--app-logs-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-max-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--enable-rq3-apply-service-qa", action="store_true")
    parser.add_argument("--rq3-apply-fail-on-missing-pack", action="store_true")
    parser.add_argument(
        "--rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_true",
        help="Save rq3 apply prompt/raw records in prediction metadata.",
    )
    parser.add_argument(
        "--no-rq3-apply-save-prompt-and-raw",
        dest="rq3_apply_save_prompt_and_raw",
        action="store_false",
        help="Disable saving rq3 apply prompt/raw records.",
    )
    parser.set_defaults(rq3_apply_save_prompt_and_raw=True)
    parser.add_argument("--max-checkpoints", type=int, default=None)

    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--retriever-provider", type=str, default=None)
    parser.add_argument("--retriever-model", type=str, default=None)
    parser.add_argument("--retriever-batch-size", type=int, default=None)
    parser.add_argument("--hipporag-dir", type=str, default=None)

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = None
    if args.config is not None:
        config_path = args.config
        defaults = load_yaml(args.defaults) if args.defaults and args.defaults.exists() else {}
        cfg = load_yaml(args.config)
        merged = deep_merge(defaults, cfg)
        adapter_args = build_from_config(merged)
    else:
        adapter_args = _build_from_legacy_args(args)

    resolved = resolved_payload(adapter_args, config_path)
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return

    settings_path = write_run_settings(resolved, adapter_args.output)
    result = run_adapter(adapter_args)
    print("Saved:", adapter_args.output)
    print("Run settings:", settings_path)
    print("Total checkpoints:", len(result.get("predictions", [])))


if __name__ == "__main__":
    main()
