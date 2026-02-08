from __future__ import annotations

from typing import Any, Dict, List, Set


class EvidenceResolver:
    def __init__(self, raw_items: List[Dict[str, Any]]) -> None:
        self.raw_items = [item for item in raw_items if isinstance(item, dict)]
        self.event_to_logs, self.log_to_events = self._build_edges_from_raw(self.raw_items)

    @staticmethod
    def _normalize_str_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []
        return [x for x in value if isinstance(x, str) and x]

    def _build_edges_from_raw(self, raw_items: List[Dict[str, Any]]) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        event_to_logs: Dict[str, Set[str]] = {}
        log_to_events: Dict[str, Set[str]] = {}

        for item in raw_items:
            app_log_ids = self._normalize_str_list(item.get("app_log_ids"))
            event_ids = self._normalize_str_list(item.get("event_ids"))
            source = item.get("from")
            ctx = item.get("context")
            if not isinstance(ctx, dict):
                continue

            if source == "event":
                event_id = ctx.get("event_id")
                if isinstance(event_id, str) and event_id:
                    event_to_logs.setdefault(event_id, set()).update(app_log_ids)
            elif source == "log":
                log_id = ctx.get("app_log_id")
                if isinstance(log_id, str) and log_id:
                    log_to_events.setdefault(log_id, set()).update(event_ids)

        return event_to_logs, log_to_events

    def map_event_ids_to_log_ids(self, event_ids: List[str]) -> List[str]:
        mapped: Set[str] = set()
        for event_id in event_ids:
            mapped.update(self.event_to_logs.get(event_id, set()))
        return sorted(mapped)

    def map_log_ids_to_event_ids(self, log_ids: List[str]) -> List[str]:
        mapped: Set[str] = set()
        for log_id in log_ids:
            mapped.update(self.log_to_events.get(log_id, set()))
        return sorted(mapped)

    def context_logs_from_log_ids(self, log_ids: List[str]) -> List[Dict[str, Any]]:
        """
        For each log_id:
        1) filter raw items with from == "log"
        2) keep items whose app_log_ids equals [log_id] exactly
        3) lock unique and append its context
        """
        context_logs: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for log_id in log_ids:
            if not isinstance(log_id, str) or not log_id or log_id in seen:
                continue
            seen.add(log_id)

            candidates: List[Dict[str, Any]] = []
            for item in self.raw_items:
                if item.get("from") != "log":
                    continue
                app_log_ids = self._normalize_str_list(item.get("app_log_ids"))
                if app_log_ids == [log_id]:
                    ctx = item.get("context")
                    if isinstance(ctx, dict):
                        candidates.append(ctx)

            if len(candidates) == 1:
                context_logs.append(candidates[0])
        return context_logs

