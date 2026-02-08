from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    if str(Path(__file__).resolve().parents[2]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from . import context_builder, extract_main_store, task_builder, task_store
    from shared.config import ContextConfig
except ImportError:
    if str(Path(__file__).resolve().parents[2]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import context_builder
    import extract_main_store
    import task_builder
    import task_store
    from shared.config import ContextConfig


def _build_all_tasks(config: ContextConfig) -> None:
    raw_items = task_builder._load_raw(config.raw_path)
    for idx in range(1, 7):
        tag = f"t{idx}"
        task_items, task_final = task_builder.build_by_tag(tag, raw_items)
        task_builder._write_json(config.task_path_for(tag), task_items)
        task_builder._write_json(config.task_final_path_for(tag), task_final)
        print(
            f"[{tag}] task={len(task_items)} task_final={len(task_final)} "
            f"-> {config.task_path_for(tag).name}, {config.task_final_path_for(tag).name}"
        )


def _run_pipeline(config: ContextConfig, only: str) -> int:
    steps: Iterable[str]
    if only:
        steps = [only]
    else:
        steps = ["extract", "raw", "tasks", "store"]

    for step in steps:
        if step == "extract":
            schema = extract_main_store.load_schema(config.schema_path)
            states = extract_main_store.extract_states(schema)
            events = extract_main_store.extract_events(schema)
            logs = extract_main_store.load_logs(config.data_dir / "app_log_large.json")
            extract_main_store.write_json(config.states_path, states)
            extract_main_store.write_json(config.events_path, events)
            extract_main_store.write_json(config.logs_path, logs)
            print(f"[extract] states={len(states)} events={len(events)} logs={len(logs)}")
        elif step == "raw":
            count = context_builder.build_context_store(
                states_path=config.states_path,
                events_path=config.events_path,
                logs_path=config.logs_path,
                schema_path=config.schema_path,
                app_logs_final_path=config.app_logs_final_path,
                output_path=config.raw_path,
            )
            print(f"[raw] items={count} -> {config.raw_path}")
        elif step == "tasks":
            _build_all_tasks(config)
        elif step == "store":
            tasks = task_store.build_tasks(config)
            task_store._write_json(config.tasks_path, tasks)
            print(f"[store] tasks={len(tasks)} -> {config.tasks_path}")
        else:
            raise ValueError(f"Unknown pipeline step: {step}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="qa_context CLI")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="Extract states/events/logs")
    p_extract.add_argument("--user-id", type=int, default=None)

    p_raw = sub.add_parser("raw", help="Build raw.json")
    p_raw.add_argument("--user-id", type=int, default=None)

    p_tasks = sub.add_parser("tasks", help="Build task_t1..t6 and task_final_t1..t6")
    p_tasks.add_argument("--user-id", type=int, default=None)

    p_store = sub.add_parser("store", help="Build tasks.json from task_final_t1..t6")
    p_store.add_argument("--user-id", type=int, default=None)

    p_pipeline = sub.add_parser("pipeline", help="Run extract->raw->tasks->store")
    p_pipeline.add_argument("--user-id", type=int, default=None)
    p_pipeline.add_argument(
        "--only",
        type=str,
        default="",
        choices=["", "extract", "raw", "tasks", "store"],
        help="Run only one step",
    )

    args = parser.parse_args()
    config = (
        ContextConfig(user_id=args.user_id)
        if getattr(args, "user_id", None) is not None
        else ContextConfig()
    )

    if args.command == "extract":
        return _run_pipeline(config, "extract")
    if args.command == "raw":
        return _run_pipeline(config, "raw")
    if args.command == "tasks":
        return _run_pipeline(config, "tasks")
    if args.command == "store":
        return _run_pipeline(config, "store")
    return _run_pipeline(config, "")


if __name__ == "__main__":
    raise SystemExit(main())
