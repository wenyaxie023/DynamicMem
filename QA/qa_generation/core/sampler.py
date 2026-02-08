from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from shared.config import GenerationConfig


TaskItem = Dict[str, Any]


def _read_list(path: Path) -> List[TaskItem]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_tasks(
    tasks: Iterable[TaskItem],
    *,
    qtypes: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
) -> List[TaskItem]:
    qtype_set = set(qtypes) if qtypes else None
    category_set = set(categories) if categories else None

    filtered: List[TaskItem] = []
    for item in tasks:
        qtype = item.get("qtype")
        if not isinstance(qtype, int):
            payload = item.get("payload")
            if isinstance(payload, dict):
                qtype = payload.get("qtype")
        if not isinstance(qtype, int):
            continue

        if qtype_set is not None and qtype not in qtype_set:
            continue

        category = item.get("category")
        if category is None:
            payload = item.get("payload")
            if isinstance(payload, dict):
                category = payload.get("category")

        if category_set is not None:
            selected = False
            if isinstance(category, str):
                selected = category in category_set
            elif isinstance(category, list):
                selected = any(isinstance(x, str) and x in category_set for x in category)
            if not selected:
                continue

        filtered.append(item)
    return filtered


def _normalize_category(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                return item
    return "Unknown"


def sample_tasks(
    tasks: List[TaskItem],
    *,
    sample_per_group: int,
    sample_seed: int,
) -> List[TaskItem]:
    if sample_per_group <= 0:
        return []

    grouped: DefaultDict[Tuple[int, str], List[TaskItem]] = defaultdict(list)
    for item in tasks:
        qtype = item.get("qtype")
        if not isinstance(qtype, int):
            payload = item.get("payload")
            if isinstance(payload, dict):
                qtype = payload.get("qtype")
        if not isinstance(qtype, int):
            continue

        category = item.get("category")
        if category is None:
            payload = item.get("payload")
            if isinstance(payload, dict):
                category = payload.get("category")

        grouped[(qtype, _normalize_category(category))].append(item)

    rng = random.Random(sample_seed)
    sampled: List[TaskItem] = []
    for key in sorted(grouped.keys()):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        sampled.extend(bucket[:sample_per_group])
    return sampled


def run_sampling(
    *,
    config: GenerationConfig,
    qtypes: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
) -> int:
    tasks = _read_list(config.tasks_path)
    tasks = _filter_tasks(tasks, qtypes=qtypes, categories=categories)
    tasks = sample_tasks(
        tasks,
        sample_per_group=config.sample_per_group,
        sample_seed=config.sample_seed,
    )

    out_path = config.sampled_tasks_path
    _write_json(out_path, tasks)
    print(f"Saved sampled tasks: {out_path} (count={len(tasks)})")
    return 0
