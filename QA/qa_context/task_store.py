from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.task import Task
from shared.config import ContextConfig


def _load_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task_uid(task_item: Dict[str, Any]) -> str:
    payload = _stable_json(task_item)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_tasks(config: ContextConfig) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for qtype in range(1, 7):
        tag = f"t{qtype}"
        task_final_items = _load_list(config.task_final_path_for(tag))
        for item in task_final_items:
            task_item = item.get("task")
            context_items = item.get("context")
            if not isinstance(task_item, dict) or not isinstance(context_items, list):
                continue
            payload = {
                "task": task_item,
                "context": context_items,
                "qtype": qtype,
                "category": task_item.get("category"),
            }
            tasks.append(
                asdict(
                    Task(
                        uid=_task_uid(task_item),
                        qtype=qtype,
                        category=task_item.get("category"),
                        payload=payload,
                    )
                )
            )
    return tasks


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build tasks.json from task_final_t1..t6")
    parser.add_argument("--user-id", type=int, default=None)
    args = parser.parse_args()

    config = (
        ContextConfig(user_id=args.user_id)
        if args.user_id is not None
        else ContextConfig()
    )
    tasks = build_tasks(config)
    _write_json(config.tasks_path, tasks)
    print(f"Saved tasks to: {config.tasks_path} (count={len(tasks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
