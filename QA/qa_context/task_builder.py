from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import ContextConfig


WINDOW_RE = re.compile(r"_w(\d+)_")
ALLOWED_STATE_CATEGORIES = {"habits_state", "preferences_state"}
ALLOWED_CHANGE_TYPES = {"unchanged", "acquire", "aquire"}


def _load_raw(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _window_rank(value: str) -> Optional[int]:
    if value == "init":
        return 0
    if value.startswith("w") and value[1:].isdigit():
        return int(value[1:])
    return None


def _event_id_to_window(event_id: str) -> Optional[str]:
    match = WINDOW_RE.search(event_id)
    if not match:
        return None
    return f"w{match.group(1)}"


def _event_id_rank(event_id: str) -> Optional[int]:
    window = _event_id_to_window(event_id)
    if window is None:
        return None
    return _window_rank(window)


def _sorted_event_ids(event_ids: Sequence[str]) -> List[str]:
    def key_fn(event_id: str) -> Tuple[int, str]:
        rank = _event_id_rank(event_id)
        return (rank if rank is not None else 10**9, event_id)

    uniq = sorted({eid for eid in event_ids if isinstance(eid, str)}, key=key_fn)
    return uniq


def _state_name(item: Dict[str, Any]) -> Optional[str]:
    context = item.get("context")
    if not isinstance(context, dict):
        return None
    name = context.get("state_name")
    return name if isinstance(name, str) and name else None


def _state_event_ids(item: Dict[str, Any]) -> List[str]:
    event_ids = item.get("event_ids", [])
    if not isinstance(event_ids, list):
        return []
    return [eid for eid in event_ids if isinstance(eid, str)]


def _state_window_min_rank(item: Dict[str, Any]) -> Optional[int]:
    ranks: List[int] = []
    for event_id in _state_event_ids(item):
        rank = _event_id_rank(event_id)
        if rank is not None:
            ranks.append(rank)
    if not ranks:
        return None
    return min(ranks)


def filter_only_state(item: Dict[str, Any]) -> bool:
    return item.get("from") == "state"


def filter_state_category(item: Dict[str, Any], *, allowed: Set[str]) -> bool:
    context = item.get("context")
    if not isinstance(context, dict):
        return False
    category = context.get("state_category")
    return isinstance(category, str) and category in allowed


def filter_state_change_type(item: Dict[str, Any], *, allowed: Set[str]) -> bool:
    context = item.get("context")
    if not isinstance(context, dict):
        return False
    resolved_items = context.get("resolved_items", [])
    if not isinstance(resolved_items, list) or not resolved_items:
        return False
    for entry in resolved_items:
        if not isinstance(entry, dict):
            continue
        change_type = entry.get("change_type")
        if isinstance(change_type, str) and change_type in allowed:
            return True
    return False


def filter_state_has_change_reason(item: Dict[str, Any]) -> bool:
    context = item.get("context")
    if not isinstance(context, dict):
        return False
    if item.get("from") != "state":
        return False
    resolved_items = context.get("resolved_items", [])
    if not isinstance(resolved_items, list):
        return False
    for entry in resolved_items:
        if not isinstance(entry, dict):
            continue
        reason = entry.get("change_reason")
        if isinstance(reason, str) and reason.strip():
            return True
    return False


def filter_state_time_windows(
    item: Dict[str, Any],
    *,
    start_window: str,
    end_window: str,
) -> bool:
    if item.get("from") != "state":
        return False

    start_rank = _window_rank(start_window)
    end_rank = _window_rank(end_window)
    if start_rank is None or end_rank is None:
        raise ValueError(f"Invalid window range: {start_window} -> {end_window}")
    if start_rank > end_rank:
        raise ValueError(f"start_window must be <= end_window: {start_window} -> {end_window}")

    for event_id in _state_event_ids(item):
        rank = _event_id_rank(event_id)
        if rank is None:
            continue
        if start_rank <= rank <= end_rank:
            return True
    return False


def build_task1_candidates(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_items = [x for x in raw_items if filter_only_state(x)]
    task_items = [x for x in task_items if filter_state_category(x, allowed=ALLOWED_STATE_CATEGORIES)]
    task_items = [x for x in task_items if filter_state_change_type(x, allowed=ALLOWED_CHANGE_TYPES)]
    return task_items


def build_task2_candidates(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_items = [x for x in raw_items if filter_only_state(x)]
    task_items = [x for x in task_items if filter_state_has_change_reason(x)]
    return task_items


def _build_event_index(raw_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    event_map: Dict[str, Dict[str, Any]] = {}
    for item in raw_items:
        if item.get("from") != "event":
            continue
        context = item.get("context")
        if not isinstance(context, dict):
            continue
        event_id = context.get("event_id")
        if isinstance(event_id, str) and event_id:
            event_map[event_id] = item
    return event_map


def _build_state_history_index(states: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for item in states:
        name = _state_name(item)
        if not name:
            continue
        by_name.setdefault(name, []).append(item)

    def state_sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
        rank = _state_window_min_rank(item)
        item_id = item.get("id")
        return (rank if rank is not None else 10**9, item_id if isinstance(item_id, int) else 10**9)

    for name in by_name:
        by_name[name].sort(key=state_sort_key)
    return by_name


def build_state_accum_context(
    *,
    task: Dict[str, Any],
    history_index: Dict[str, List[Dict[str, Any]]],
    event_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build cumulative same-name state context for one task:
    include self and all same-name states whose window <= current window.
    Output is a flat list of raw elements ordered as:
    [state, related events..., state, related events..., ...]
    """
    name = _state_name(task)
    cur_rank = _state_window_min_rank(task)
    task_id = task.get("id")

    context_items: List[Dict[str, Any]] = []
    if not name:
        return context_items

    candidates = history_index.get(name, [])
    for state_item in candidates:
        state_rank = _state_window_min_rank(state_item)
        state_id = state_item.get("id")

        # Always include self.
        if isinstance(task_id, int) and isinstance(state_id, int) and state_id == task_id:
            pass
        else:
            # If current rank is unknown, fall back to keeping only self.
            if cur_rank is None:
                continue
            # Keep same-name states up to current window (inclusive).
            if state_rank is None or state_rank > cur_rank:
                continue

        event_ids = _sorted_event_ids(_state_event_ids(state_item))
        context_items.append(state_item)
        for event_id in event_ids:
            event_item = event_map.get(event_id)
            if event_item is not None:
                context_items.append(event_item)
    return context_items


def build_task_final_with_state_accum_context(
    *,
    tasks: List[Dict[str, Any]],
    all_raw_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    event_map = _build_event_index(all_raw_items)
    all_states = [x for x in all_raw_items if x.get("from") == "state"]
    history_index = _build_state_history_index(all_states)

    finals: List[Dict[str, Any]] = []
    for task in tasks:
        context_items = build_state_accum_context(
            task=task,
            history_index=history_index,
            event_map=event_map,
        )
        finals.append(
            {
                "task": task,
                "context": context_items,
            }
        )

    return finals


def build_state_tasks_final(
    *,
    task_items: List[Dict[str, Any]],
    all_raw_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return build_task_final_with_state_accum_context(
        tasks=task_items,
        all_raw_items=all_raw_items,
    )


def normalize_tag(tag: str) -> str:
    if tag in {"1", "t1", "task1"}:
        return "t1"
    if tag in {"2", "t2", "task2"}:
        return "t2"
    if tag in {"3", "t3", "task3"}:
        return "t3"
    if tag in {"4", "t4", "task4"}:
        return "t4"
    if tag in {"5", "t5", "task5"}:
        return "t5"
    if tag in {"6", "t6", "task6"}:
        return "t6"
    raise ValueError(f"Unsupported tag: {tag}. Supported: t1..t6")


def build_task_t1(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_items = build_task1_candidates(raw_items)
    task_final = build_state_tasks_final(task_items=task_items, all_raw_items=raw_items)
    return task_items, task_final


def build_task_t2(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_items = build_task2_candidates(raw_items)
    task_final = build_state_tasks_final(task_items=task_items, all_raw_items=raw_items)
    return task_items, task_final


def build_task_t3(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_items = [x for x in raw_items if filter_only_state(x)]
    task_final = build_state_tasks_final(task_items=task_items, all_raw_items=raw_items)
    return task_items, task_final


def build_task_t4(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_states = [x for x in raw_items if filter_only_state(x)]
    by_name = _build_state_history_index(all_states)
    task_items: List[Dict[str, Any]] = []
    for states in by_name.values():
        if states:
            task_items.append(states[-1])
    task_final = build_state_tasks_final(task_items=task_items, all_raw_items=raw_items)
    return task_items, task_final


def build_task_t5(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    log_items = [x for x in raw_items if x.get("from") == "log" and isinstance(x.get("context"), dict)]
    log_items = sorted(
        log_items,
        key=lambda x: x.get("id") if isinstance(x.get("id"), int) else 10**9,
    )
    task_items = list(log_items)

    n = len(log_items)
    task_final: List[Dict[str, Any]] = []
    for idx, task_item in enumerate(log_items):
        if n <= 5:
            start = 0
            end = n
        else:
            start = idx - 2
            end = idx + 3
            if start < 0:
                start = 0
                end = 5
            if end > n:
                end = n
                start = n - 5
        context = log_items[start:end]
        task_final.append(
            {
                "task": task_item,
                "context": context,
            }
        )
    return task_items, task_final


def build_task_t6(raw_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_task_t5(raw_items)


def build_by_tag(
    tag: str,
    raw_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized = normalize_tag(tag)
    builders = {
        "t1": build_task_t1,
        "t2": build_task_t2,
        "t3": build_task_t3,
        "t4": build_task_t4,
        "t5": build_task_t5,
        "t6": build_task_t6,
    }
    return builders[normalized](raw_items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build task and task_final from raw.json")
    parser.add_argument("--tag", type=str, default="1")
    parser.add_argument("--user-id", type=int, default=None)
    args = parser.parse_args()

    config = (
        ContextConfig(user_id=args.user_id)
        if args.user_id is not None
        else ContextConfig()
    )
    normalized_tag = normalize_tag(args.tag)
    raw_items = _load_raw(config.raw_path)
    task_items, task_final = build_by_tag(args.tag, raw_items)
    task_path = config.task_path_for(normalized_tag)
    _write_json(task_path, task_items)

    task_final_path = config.task_final_path_for(normalized_tag)
    _write_json(task_final_path, task_final)

    print(f"Saved task to: {task_path} (count={len(task_items)})")
    print(f"Saved task_final to: {task_final_path} (count={len(task_final)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
