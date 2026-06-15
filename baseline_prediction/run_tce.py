#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from baseline_prediction.adapters.registry import list_adapters, run_adapter
from baseline_prediction.tce_config import (
    build_from_config,
    derive_run_name,
    load_yaml,
    deep_merge,
    render_value,
    resolved_payload,
    write_run_settings,
    normalize_users,
)
from baseline_prediction.adapters.base import TceAdapterArgs


def _build_from_legacy_args(args: argparse.Namespace) -> TceAdapterArgs:
    defaults = {}
    if args.defaults and Path(args.defaults).exists():
        defaults = load_yaml(Path(args.defaults))
    llm_defaults = defaults.get("llm", {}) if isinstance(defaults, dict) else {}
    retriever_defaults = defaults.get("retriever", {}) if isinstance(defaults, dict) else {}
    retrieval_defaults = defaults.get("retrieval", {}) if isinstance(defaults, dict) else {}
    runtime_defaults = defaults.get("runtime", {}) if isinstance(defaults, dict) else {}
    final_qa_defaults = defaults.get("final_qa", {}) if isinstance(defaults, dict) else {}

    llm_provider = args.llm_provider or llm_defaults.get("provider", "openai")
    llm_model = args.llm_model or llm_defaults.get("model", "gpt-5-mini")
    llm_max_workers = args.llm_max_workers or int(llm_defaults.get("max_workers", 1))
    llm_temperature = args.llm_temperature
    if llm_temperature is None:
        llm_temperature = llm_defaults.get("temperature", 0.0)
    llm_top_p = args.llm_top_p
    if llm_top_p is None:
        llm_top_p = llm_defaults.get("top_p", 1.0)
    llm_top_k = args.llm_top_k
    if llm_top_k is None:
        llm_top_k = llm_defaults.get("top_k", None)

    extras = {}
    if args.hipporag_dir:
        extras["hipporag_dir"] = args.hipporag_dir

    return TceAdapterArgs(
        baseline=args.baseline,
        user_id=args.user_id,
        benchmark=args.benchmark,
        app_logs_path=args.app_logs_path,
        output=args.output,
        max_visible_logs=args.max_visible_logs,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_workers=llm_max_workers,
        llm_temperature=float(llm_temperature) if llm_temperature is not None else None,
        llm_top_p=float(llm_top_p) if llm_top_p is not None else None,
        llm_top_k=int(llm_top_k) if llm_top_k is not None else None,
        resume=args.resume,
        max_checkpoints=args.max_checkpoints,
        debug=args.debug,
        debug_dir=args.debug_dir,
        save_prompt_and_raw=args.save_prompt_and_raw,
        enable_rq3_apply_service_qa=args.enable_rq3_apply_service_qa,
        rq3_apply_save_prompt_and_raw=args.rq3_apply_save_prompt_and_raw,
        checkpoint_workers=int(runtime_defaults.get("checkpoint_workers", 1)),
        within_checkpoint_workers=int(runtime_defaults.get("within_checkpoint_workers", 1)),
        save_every_generation_keys=int(runtime_defaults.get("save_every_generation_keys", 1)),
        retriever_provider=str(args.retriever_provider or retriever_defaults.get("provider", "openai")),
        retriever_model=str(args.retriever_model or retriever_defaults.get("model", "text-embedding-3-large")),
        retriever_batch_size=int(args.retriever_batch_size or retriever_defaults.get("batch_size", 64)),
        retrieval_top_k=int(args.retrieval_top_k or retrieval_defaults.get("top_k", 5)),
        rq3_apply_retrieval_top_k=None,
        enable_final_qa=bool(final_qa_defaults.get("enabled", False)),
        final_qa_path=final_qa_defaults.get("path"),
        final_qa_output_path=final_qa_defaults.get("output_path"),
        final_qa_retrieval_top_k=(
            int(retrieval_defaults["final_qa_top_k"])
            if retrieval_defaults.get("final_qa_top_k") not in {None, ""}
            else None
        ),
        final_qa_save_prompt_and_raw=bool(final_qa_defaults.get("save_prompt_and_raw", False)),
        extras=extras,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified TCE runner")

    parser.add_argument("--config", type=Path, default=None, help="YAML config path (preferred)")
    parser.add_argument("--defaults", type=Path, default=Path("configs/tce.default.yaml"), help="YAML defaults")

    parser.add_argument("--baseline", type=str, default=None, help="one of: {}".format(", ".join(list_adapters())))
    parser.add_argument("--user-id", type=str, default=None, help="Shared runtime user_id for single-user baselines")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--app-logs-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-visible-logs", type=int, default=None)
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-max-workers", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-top-p", type=float, default=None)
    parser.add_argument("--llm-top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--save-prompt-and-raw", action="store_true")
    parser.add_argument("--enable-rq3-apply-service-qa", action="store_true")
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
        runtime = merged.get("runtime", {}) or {}
        resolved_user_id = runtime.get("user_id")
        if not runtime.get("user_id"):
            users = normalize_users(merged.get("users"))
            if len(users) == 1:
                runtime = dict(runtime)
                runtime["user_id"] = users[0]
                merged["runtime"] = runtime
                resolved_user_id = users[0]
            elif len(users) > 1:
                raise ValueError(
                    "baseline_prediction.run_tce requires runtime.user_id when config contains multiple users. "
                    "Use baseline_prediction.run_tce_batch for multi-user configs."
                )
        experiment_name = str(runtime.get("experiment_name", "") or "").strip()
        run_id = str(runtime.get("run_id", "") or "").strip() or "main"
        run_name = derive_run_name(experiment_name, run_id)
        if resolved_user_id:
            merged = render_value(
                merged,
                user_id=str(resolved_user_id),
                experiment_name=experiment_name,
                run_id=run_id,
                run_name=run_name,
            )
        adapter_args = build_from_config(merged)
        if args.resume:
            adapter_args.resume = True
        if args.max_checkpoints is not None:
            adapter_args.max_checkpoints = args.max_checkpoints
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
