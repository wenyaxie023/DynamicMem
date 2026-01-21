"""
Stage 3: App Logs Generation

Responsibilities:
- Merge events from all domains
- Sort events chronologically
- Convert each event to structured app log via LLM

This stage converts abstract behavioral events into concrete app API calls
with realistic request/response payloads.
"""

from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from uuid import UUID
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import GeminiJSONClient, LLMResult
from app_system import APP_API_SCHEMAS, AppRegistry, get_api_input_output_models
from prompt_templates import app_log_prompt


def _slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "domain"


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    """Write JSON to file with formatting."""
    safe_payload = _json_safe(payload)
    path.write_text(json.dumps(safe_payload, indent=2, ensure_ascii=False))


def _write_text(path: Path, text: str) -> None:
    """Write text to file."""
    path.write_text(text)


def _json_safe(value: Any) -> Any:
    """Convert non-JSON types (e.g., datetime) into JSON-serializable values."""
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, UUID):
        return str(value)
    return value


@dataclass
class Stage3Result:
    """Result from Stage 3: App Logs Generation."""
    app_logs: List[Dict[str, Any]]
    total_events: int
    total_logs_generated: int
    usage: Dict[str, Any]


@dataclass
class Stage3Checkpoint:
    """Checkpoint for resuming Stage 3 processing."""
    last_processed_event_id: str
    last_processed_index: int
    app_log_counter: int
    chain_logs_history: Dict[str, List[Dict[str, Any]]]
    app_states: Dict[str, Dict[str, Any]]  # app_name -> state
    aggregate_usage: Dict[str, Any]
    processed_event_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_processed_event_id": self.last_processed_event_id,
            "last_processed_index": self.last_processed_index,
            "app_log_counter": self.app_log_counter,
            "chain_logs_history": _json_safe(self.chain_logs_history),
            "app_states": _json_safe(self.app_states),
            "aggregate_usage": _json_safe(self.aggregate_usage),
            "processed_event_ids": self.processed_event_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage3Checkpoint":
        return cls(
            last_processed_event_id=data.get("last_processed_event_id", ""),
            last_processed_index=data.get("last_processed_index", -1),
            app_log_counter=data.get("app_log_counter", 0),
            chain_logs_history=data.get("chain_logs_history", {}),
            app_states=data.get("app_states", {}),
            aggregate_usage=data.get("aggregate_usage", {}),
            processed_event_ids=data.get("processed_event_ids", []),
        )


class AppLogsStage:
    """
    Stage 3: App Logs Generation

    This stage converts behavioral events from Stage 2 into concrete app API
    calls with realistic request/response payloads, using an LLM to generate
    contextually appropriate data.
    """

    def __init__(
        self,
        llm_client: GeminiJSONClient,
        output_dir: Path,
        debug_mode: bool = True,
        checkpoint_interval: int = 10,  # Save checkpoint every N logs
        logger: Optional[logging.Logger] = None,
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.debug_mode = debug_mode
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger or logging.getLogger(__name__)
        _ensure_dir(self.output_dir)

        if self.debug_mode:
            self.debug_dir = self.output_dir / "debug" / "stage3_app_logs"
            _ensure_dir(self.debug_dir)

        # Checkpoint file path
        self.checkpoint_path = self.output_dir / "stage3_checkpoint.json"

        # Pydantic models for validation
        self._setup_pydantic()

    def clear_checkpoint(self) -> None:
        """Clear checkpoint files to start fresh."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            self.logger.info(f"[Checkpoint] Cleared: {self.checkpoint_path}")

        intermediate_path = self.output_dir / "app_logs_intermediate.json"
        if intermediate_path.exists():
            intermediate_path.unlink()
            self.logger.info(f"[Checkpoint] Cleared: {intermediate_path}")

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_path.exists()

    def get_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current checkpoint without loading full data."""
        if not self.checkpoint_path.exists():
            return None

        try:
            data = json.loads(self.checkpoint_path.read_text())
            return {
                "last_processed_event_id": data.get("last_processed_event_id"),
                "last_processed_index": data.get("last_processed_index"),
                "app_log_counter": data.get("app_log_counter"),
                "num_processed_events": len(data.get("processed_event_ids", [])),
                "num_chains": len(data.get("chain_logs_history", {})),
                "num_app_states": len(data.get("app_states", {})),
            }
        except Exception as e:
            return {"error": str(e)}

    def _setup_pydantic(self):
        """Setup Pydantic models for validation."""
        try:
            from pydantic import BaseModel, Field, ValidationError
            self.pydantic_available = True
            self.ValidationError = ValidationError

            class AppCallPayload(BaseModel):
                input: Dict[str, Any] = Field(description="API input payload.")
                output: Dict[str, Any] = Field(description="API output payload.")

            self.AppCallPayload = AppCallPayload
        except ImportError:
            self.pydantic_available = False
            self.ValidationError = Exception
            self.AppCallPayload = None

    def run(
        self,
        events_chains_by_domain: Dict[str, List[Dict]],
        user_basic_profile: Dict[str, Any],
        user_id: str = "user_001",
        cutoff_date: Optional[str] = None,
        resume_from_existing: bool = False,
        resume_from_event_id: Optional[str] = None,
        clear_checkpoint_on_start: bool = False,
    ) -> Stage3Result:
        """
        Run the complete Stage 3 pipeline.

        Args:
            events_chains_by_domain: Events chains from Stage 2
            user_basic_profile: User's basic profile
            user_id: User identifier
            cutoff_date: Optional cutoff date (YYYY-MM-DD format)
            resume_from_existing: If True, reuse checkpoint/existing logs to resume
            resume_from_event_id: If set, continue generation starting from this event_id (inclusive)
            clear_checkpoint_on_start: If True, clear any existing checkpoint before starting

        Returns:
            Stage3Result with generated app logs

        Resume Behavior:
            1. If resume_from_existing=True and checkpoint exists:
               - Loads checkpoint with full state (chain_logs_history, app_states)
               - Continues from last_processed_index + 1
            2. If resume_from_existing=True but no checkpoint:
               - Falls back to loading existing logs
               - Rebuilds chain_logs_history and app_states by replaying logs
            3. If resume_from_event_id is specified:
               - Overrides start position to that specific event
            4. Checkpoints are saved every checkpoint_interval logs (default: 10)
            5. On error, checkpoint is saved for recovery
        """
        self.logger.info("=== Stage 3: App Logs Generation ===")

        # Handle checkpoint clearing
        if clear_checkpoint_on_start:
            self.clear_checkpoint()
        elif resume_from_existing and self.has_checkpoint():
            info = self.get_checkpoint_info()
            self.logger.info(f"[Checkpoint Found] {info}")

        # Step 1: Collect all events from all domains
        self.logger.info("Step 3.1: Collecting events from all domains...")
        all_events = self._collect_all_events(events_chains_by_domain)

        # Step 2: Sort events chronologically
        self.logger.info("Step 3.2: Sorting events chronologically...")
        all_events = self._sort_events(all_events)

        # Step 3: Apply cutoff filter if provided
        if cutoff_date:
            original_count = len(all_events)
            all_events = self._filter_by_cutoff(all_events, cutoff_date)
            self.logger.info(f"Step 3.3: Applied cutoff {cutoff_date}: {original_count} -> {len(all_events)} events")

        # Print event statistics
        self._print_event_stats(all_events)

        # Step 4: Generate app logs
        self.logger.info("Step 3.4: Generating app logs via LLM...")
        app_registry = AppRegistry()
        app_logs, usage = self._generate_app_logs(
            events=all_events,
            app_registry=app_registry,
            user_profile=user_basic_profile,
            user_id=user_id,
            resume_from_existing=resume_from_existing,
            resume_from_event_id=resume_from_event_id,
        )

        # Save results
        self._save_results(app_logs, user_id, all_events)

        # Clean up checkpoint on successful completion
        self.logger.info("[Checkpoint] Completed successfully, checkpoint preserved for reference")

        return Stage3Result(
            app_logs=app_logs,
            total_events=len(all_events),
            total_logs_generated=len(app_logs),
            usage=usage,
        )

    # =========================================================================
    # Step 3.1: Collect Events
    # =========================================================================

    def _collect_all_events(
        self, events_chains_by_domain: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Collect and flatten all events from all domains.

        Preserves event_id from Stage 2 and extracts state evidence information
        for downstream traceability (app_log -> event -> state_item).
        """
        all_events: List[Dict[str, Any]] = []

        for domain_name, windows in events_chains_by_domain.items():
            for window_data in windows:
                if not isinstance(window_data, dict):
                    continue

                window_id = window_data.get("window_id", "")
                time_range = window_data.get("time_range", [])
                event_chains = window_data.get("event_chains", [])

                for chain_idx, chain in enumerate(event_chains):
                    if not isinstance(chain, dict):
                        continue

                    # Use chain_id from Stage 2 if available, otherwise derive from event_id
                    chain_id = chain.get("chain_id") or ""

                    # Extract state_refs with resolved_items for traceability
                    state_refs = chain.get("state_refs", [])

                    for event_idx, event in enumerate(chain.get("events", [])):
                        if not isinstance(event, dict):
                            continue
                        if not chain_id:
                            chain_id = self._extract_chain_id_from_event_id(
                                event.get("event_id", "")
                            ) or f"{window_id}_chain_{chain_idx + 1:02d}"

                        # Expand events with multiple schedule_dates
                        for instance in self._expand_event_instances(
                            event,
                            base_event_id=event.get("event_id", ""),
                            event_idx=event_idx,
                        ):
                            instance["domain_name"] = domain_name
                            instance["window_id"] = window_id
                            instance["time_range"] = time_range
                            instance["chain_id"] = chain_id
                            instance["chain_idx"] = chain_idx
                            # Preserve state_refs for golden evidence traceability
                            instance["state_refs"] = state_refs
                            # Preserve evidence_for_states from Stage 2 event
                            instance["evidence_for_states"] = event.get("evidence_for_states", [])
                            all_events.append(instance)

        return all_events

    def _expand_event_instances(
        self,
        event: Dict[str, Any],
        base_event_id: str = "",
        event_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """Expand an event with multiple schedule_dates into individual instances."""
        time_spec = event.get("time_specification") or {}
        schedule_dates = time_spec.get("schedule_dates") or []
        app_api_variations = event.get("app_api_variations") or []

        if schedule_dates:
            time_value = time_spec.get("time") or time_spec.get("start_time")
            instances = []

            for instance_idx, date_value in enumerate(schedule_dates):
                payload = dict(event)
                payload["timestamp"] = self._build_timestamp(date_value, time_value)
                payload["event_date"] = date_value
                payload["atomic_event_id"] = self._build_atomic_event_id(
                    base_event_id=base_event_id or payload.get("event_id", ""),
                    date_value=date_value,
                    instance_idx=instance_idx,
                    event_idx=event_idx,
                )

                # Sample from app_api_variations if available
                if app_api_variations:
                    sampled = random.choice(app_api_variations)
                    payload["app_name"] = sampled.get("app_name")
                    payload["api_name"] = sampled.get("api_name")
                    payload["original_app_api_variations"] = app_api_variations

                instances.append(payload)

            return instances

        # Single occurrence
        timestamp = event.get("timestamp")
        if timestamp:
            payload = dict(event)
            payload["atomic_event_id"] = self._build_atomic_event_id(
                base_event_id=base_event_id or payload.get("event_id", ""),
                date_value=None,
                instance_idx=0,
                event_idx=event_idx,
            )
            return [payload]

        return []

    def _extract_chain_id_from_event_id(self, event_id: str) -> str:
        """Extract chain_id from Stage 2 event_id format."""
        if not event_id:
            return ""
        parts = event_id.rsplit("_", 2)
        if len(parts) != 3:
            return ""
        return f"{parts[0]}_{parts[1]}"

    def _normalize_time(self, time_value: Optional[str]) -> str:
        """Normalize time to HH:MM:SS format."""
        if not time_value:
            return "00:00:00"
        if len(time_value) == 5:
            return f"{time_value}:00"
        return time_value

    def _build_timestamp(self, date_value: str, time_value: Optional[str]) -> str:
        """Build timestamp from date and time."""
        return f"{date_value} {self._normalize_time(time_value)}"

    def _build_atomic_event_id(
        self,
        base_event_id: str,
        date_value: Optional[str],
        instance_idx: int,
        event_idx: int,
    ) -> str:
        """Build a per-instance event id to avoid de-dup across schedule_dates."""
        base = base_event_id or f"event_{event_idx + 1:03d}"
        if date_value:
            date_tag = date_value.replace("-", "")
            return f"{base}_d{date_tag}_i{instance_idx + 1:03d}"
        return f"{base}_i{instance_idx + 1:03d}"
    # =========================================================================
    # Step 3.2: Sort Events
    # =========================================================================

    def _sort_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort events chronologically."""
        def sort_key(event: Dict[str, Any]) -> tuple:
            timestamp = event.get("timestamp") or "9999-12-31 23:59:59"
            return (timestamp, event.get("window_id", ""))

        return sorted(events, key=sort_key)

    # =========================================================================
    # Step 3.3: Filter by Cutoff
    # =========================================================================

    def _filter_by_cutoff(
        self, events: List[Dict[str, Any]], cutoff_date: str
    ) -> List[Dict[str, Any]]:
        """Filter events by cutoff date."""
        # Parse cutoff_date (handle both YYYY-MM-DD and MM.DD formats)
        if "." in cutoff_date and "-" not in cutoff_date:
            parts = cutoff_date.split(".")
            if len(parts) == 2:
                month, day = parts
                cutoff_date = f"2024-{int(month):02d}-{int(day):02d}"

        filtered = []
        for event in events:
            timestamp = event.get("timestamp", "")
            event_date = timestamp.split(" ")[0] if timestamp else ""
            if event_date and event_date <= cutoff_date:
                filtered.append(event)

        return filtered

    def _print_event_stats(self, events: List[Dict[str, Any]]) -> None:
        """Print event statistics."""
        domain_counts: Dict[str, int] = {}
        app_counts: Dict[str, int] = {}
        api_counts: Dict[str, int] = {}

        for event in events:
            domain = event.get("domain_name", "Unknown")
            app = event.get("app_name", "Unknown")
            api = event.get("api_name", "Unknown")

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            app_counts[app] = app_counts.get(app, 0) + 1
            api_counts[f"{app}.{api}"] = api_counts.get(f"{app}.{api}", 0) + 1

        self.logger.info(f"Total events: {len(events)}")
        self.logger.info("Events by domain:")
        for domain, count in sorted(domain_counts.items()):
            self.logger.info(f"  - {domain}: {count}")

        self.logger.info("Events by app:")
        for app, count in sorted(app_counts.items(), key=lambda x: -x[1])[:10]:
            self.logger.info(f"  - {app}: {count}")

    # =========================================================================
    # Step 3.4: Generate App Logs
    # =========================================================================

    def _save_checkpoint(
        self,
        checkpoint: Stage3Checkpoint,
        app_logs: List[Dict[str, Any]],
    ) -> None:
        """Save checkpoint to file for resuming later."""
        checkpoint_data = checkpoint.to_dict()
        _write_json(self.checkpoint_path, checkpoint_data)

        # Also save intermediate app_logs
        _write_json(
            self.output_dir / "app_logs_intermediate.json",
            _json_safe({"app_logs": app_logs}),
        )

        if self.debug_mode:
            self.logger.debug(f"[Checkpoint] Saved at event {checkpoint.last_processed_event_id} "
                  f"(index {checkpoint.last_processed_index}, {len(app_logs)} logs)")

    def _load_checkpoint(self) -> Optional[Stage3Checkpoint]:
        """Load checkpoint from file if exists."""
        if not self.checkpoint_path.exists():
            return None

        try:
            data = json.loads(self.checkpoint_path.read_text())
            checkpoint = Stage3Checkpoint.from_dict(data)
            self.logger.info(f"[Checkpoint] Loaded: last event {checkpoint.last_processed_event_id}, "
                  f"{len(checkpoint.processed_event_ids)} events processed")
            return checkpoint
        except Exception as e:
            self.logger.warning(f"[Checkpoint] Failed to load: {e}")
            return None

    def _load_intermediate_logs(self) -> List[Dict[str, Any]]:
        """Load intermediate app logs if exists."""
        intermediate_path = self.output_dir / "app_logs_intermediate.json"
        if not intermediate_path.exists():
            return []

        try:
            data = json.loads(intermediate_path.read_text())
            return data.get("app_logs", [])
        except Exception as e:
            self.logger.warning(f"Failed to load intermediate logs: {e}")
            return []

    def _restore_from_checkpoint(
        self,
        checkpoint: Stage3Checkpoint,
        app_registry: AppRegistry,
        user_id: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any], int, set]:
        """Restore full state from checkpoint.

        Returns:
            - app_logs: Previously generated logs
            - chain_logs_history: Restored chain history
            - aggregate_usage: Restored usage stats
            - app_log_counter: Counter for next log ID
            - processed_event_ids: Set of already processed event IDs
        """
        # Load intermediate logs
        app_logs = self._load_intermediate_logs()
        if not app_logs:
            # Fallback to existing logs
            app_logs = self._load_existing_app_logs()

        # Restore chain_logs_history from checkpoint (deepcopy to avoid reference issues)
        chain_logs_history = deepcopy(checkpoint.chain_logs_history)

        # Restore app states from checkpoint (deepcopy each state)
        for app_name, state in checkpoint.app_states.items():
            try:
                app = app_registry.get_app(app_name, user_id)
                app.state = deepcopy(state)
                app._initialized = True
            except ValueError:
                self.logger.warning(f"Could not restore state for unknown app: {app_name}")

        # Restore usage
        aggregate_usage = deepcopy(checkpoint.aggregate_usage)

        # Get counter and processed IDs
        app_log_counter = checkpoint.app_log_counter
        processed_event_ids = set(checkpoint.processed_event_ids)

        self.logger.info(f"[Resume] Restored {len(app_logs)} logs, {len(chain_logs_history)} chains, "
              f"{len(checkpoint.app_states)} app states")

        return app_logs, chain_logs_history, aggregate_usage, app_log_counter, processed_event_ids

    def _rebuild_chain_logs_history(
        self, app_logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Rebuild chain_logs_history from existing app logs."""
        chain_logs_history: Dict[str, List[Dict[str, Any]]] = {}

        # Sort logs by timestamp to ensure correct order
        sorted_logs = sorted(app_logs, key=lambda x: x.get("timestamp", ""))

        for log in sorted_logs:
            chain_id = log.get("metadata", {}).get("chain_id", "")
            if chain_id:
                if chain_id not in chain_logs_history:
                    chain_logs_history[chain_id] = []
                chain_logs_history[chain_id].append(log)

        return chain_logs_history

    def _generate_app_logs(
        self,
        events: List[Dict[str, Any]],
        app_registry: AppRegistry,
        user_profile: Dict[str, Any],
        user_id: str,
        resume_from_existing: bool = False,
        resume_from_event_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate app logs for all events with checkpoint support.

        Generates unique app_log_ids while preserving event_ids from Stage 2.
        Each app_log is traceable back to its source event and state items.

        Supports resuming from:
        1. Checkpoint file (recommended) - preserves chain history and app states
        2. Existing logs (fallback) - rebuilds state from logs
        3. Specific event_id - start from a particular event
        """
        from tqdm import tqdm

        app_logs: List[Dict[str, Any]] = []
        chain_logs_history: Dict[str, List[Dict[str, Any]]] = {}
        aggregate_usage: Dict[str, Any] = {}
        required_fields = ("timestamp", "app_name", "api_name")

        app_log_counter = 0
        processed_event_ids: set[str] = set()
        start_index = 0

        # Try to resume from checkpoint first
        checkpoint = self._load_checkpoint() if resume_from_existing else None

        if checkpoint:
            # Full restore from checkpoint
            (app_logs, chain_logs_history, aggregate_usage,
             app_log_counter, processed_event_ids) = self._restore_from_checkpoint(
                checkpoint, app_registry, user_id
            )
            start_index = checkpoint.last_processed_index + 1

        elif resume_from_existing:
            # Fallback: rebuild from existing logs
            existing_logs = self._load_existing_app_logs()
            if existing_logs:
                app_logs.extend(existing_logs)
                processed_event_ids = {
                    log.get("event_id") for log in existing_logs if log.get("event_id")
                }
                app_log_counter = self._max_app_log_counter(existing_logs)

                # Rebuild chain_logs_history
                chain_logs_history = self._rebuild_chain_logs_history(existing_logs)

                # Restore app states by replaying logs
                for log in sorted(existing_logs, key=lambda x: x.get("timestamp", "")):
                    app_name = log.get("app_name")
                    api_name = log.get("api_name")
                    if not app_name or not api_name:
                        continue
                    try:
                        app = app_registry.get_app(app_name, user_id)
                        event_stub = {
                            "timestamp": log.get("timestamp"),
                            "app_name": app_name,
                            "api_name": api_name,
                        }
                        app.record_api_call(event_stub, log.get("request", {}), log.get("response", {}))
                    except ValueError:
                        pass

                    if log.get("metadata", {}).get("llm_usage"):
                        log_id = log.get("app_log_id", "unknown")
                        aggregate_usage[log_id] = log["metadata"]["llm_usage"]

                self.logger.info(f"[Resume] Rebuilt state from {len(existing_logs)} existing logs")

        # Handle resume_from_event_id
        if resume_from_event_id:
            for idx, event in enumerate(events):
                if event.get("event_id") == resume_from_event_id:
                    start_index = idx
                    self.logger.info(f"[Resume] Starting from event {resume_from_event_id} at index {idx}")
                    break
            else:
                self.logger.warning(f"resume_from_event_id {resume_from_event_id} not found")

        # Build event_id to index mapping for efficient lookup
        event_id_to_idx = {e.get("event_id"): i for i, e in enumerate(events)}

        # Process events
        logs_since_checkpoint = 0
        total_events = len(events)

        for idx in tqdm(range(start_index, total_events), desc="Generating app logs",
                        initial=start_index, total=total_events):
            event = events[idx]

            # Validate required fields
            if any(field not in event for field in required_fields):
                self.logger.warning(f"Skipping event {event.get('event_id')} - missing required fields")
                continue

            event_id = event.get("event_id")
            atomic_event_id = event.get("atomic_event_id") or event_id or f"idx_{idx:05d}"

            # Skip already processed events
            if atomic_event_id in processed_event_ids:
                continue

            app_log_counter += 1
            app_log_id = f"log_{app_log_counter:05d}"

            try:
                log = self._generate_single_app_log(
                    event=event,
                    app_registry=app_registry,
                    user_profile=user_profile,
                    user_id=user_id,
                    chain_logs_history=chain_logs_history,
                    app_log_id=app_log_id,
                )

                if log:
                    app_logs.append(log)
                    processed_event_ids.add(atomic_event_id)
                    logs_since_checkpoint += 1

                    # Update usage
                    if log.get("metadata", {}).get("llm_usage"):
                        aggregate_usage[app_log_id] = log["metadata"]["llm_usage"]

                    # Save checkpoint periodically
                    if logs_since_checkpoint >= self.checkpoint_interval:
                        # Collect current app states
                        app_states = {}
                        for (app_name, uid), app in app_registry.apps.items():
                            if uid == user_id:
                                app_states[app_name] = deepcopy(app.state)

                        checkpoint = Stage3Checkpoint(
                            last_processed_event_id=atomic_event_id,
                            last_processed_index=idx,
                            app_log_counter=app_log_counter,
                            chain_logs_history=deepcopy(chain_logs_history),
                            app_states=app_states,
                            aggregate_usage=deepcopy(aggregate_usage),
                            processed_event_ids=list(processed_event_ids),
                        )
                        self._save_checkpoint(checkpoint, app_logs)
                        logs_since_checkpoint = 0

            except Exception as e:
                # Save checkpoint on error for recovery
                app_states = {}
                for (app_name, uid), app in app_registry.apps.items():
                    if uid == user_id:
                        app_states[app_name] = deepcopy(app.state)

                checkpoint = Stage3Checkpoint(
                    last_processed_event_id=atomic_event_id,
                    last_processed_index=idx - 1,  # Last successful index
                    app_log_counter=app_log_counter - 1,  # Revert counter
                    chain_logs_history=deepcopy(chain_logs_history),
                    app_states=app_states,
                    aggregate_usage=deepcopy(aggregate_usage),
                    processed_event_ids=list(processed_event_ids),
                )
                self._save_checkpoint(checkpoint, app_logs)

                self.logger.error(f"Failed at event {event_id}: {e}")
                self.logger.info(f"[Checkpoint] Saved for recovery. Resume with resume_from_existing=True")
                raise

        # Final checkpoint save
        if app_logs:
            app_states = {}
            for (app_name, uid), app in app_registry.apps.items():
                if uid == user_id:
                    app_states[app_name] = deepcopy(app.state)

            final_event = events[-1] if events else {}
            final_checkpoint = Stage3Checkpoint(
                last_processed_event_id=final_event.get("atomic_event_id")
                or final_event.get("event_id", ""),
                last_processed_index=len(events) - 1,
                app_log_counter=app_log_counter,
                chain_logs_history=deepcopy(chain_logs_history),
                app_states=app_states,
                aggregate_usage=deepcopy(aggregate_usage),
                processed_event_ids=list(processed_event_ids),
            )
            self._save_checkpoint(final_checkpoint, app_logs)

        return app_logs, aggregate_usage

    def _generate_single_app_log(
        self,
        event: Dict[str, Any],
        app_registry: AppRegistry,
        user_profile: Dict[str, Any],
        user_id: str,
        chain_logs_history: Dict[str, List[Dict[str, Any]]],
        app_log_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single app log for an event.

        Args:
            event: The event from Stage 2 with state evidence info
            app_registry: Registry for app state management
            user_profile: User's basic profile
            user_id: User identifier
            chain_logs_history: Previous logs in the same chain
            app_log_id: Unique ID for this app log (e.g., "log_00001")

        Returns:
            App log dict with full traceability to state items, or None on failure
        """
        app_name = event.get("app_name", "")
        api_name = event.get("api_name", "")
        event_id = event.get("event_id", "")  # Preserved from Stage 2
        atomic_event_id = event.get("atomic_event_id", "") or event_id
        chain_id = event.get("chain_id", "")
        event_description = event.get("user_intent") or event.get("description") or ""

        # Get app and state
        try:
            app = app_registry.get_app(app_name, user_id)
        except ValueError as e:
            self.logger.warning(f"Skipping event {event_id} - {e}")
            return None
        app_state_snapshot = deepcopy(app.state)
        app.ensure_initialized()

        # Get API schema
        api_schema = APP_API_SCHEMAS.get(app_name, {}).get(api_name)
        if api_schema is None:
            self.logger.warning(f"Missing API schema for {app_name}.{api_name}")
            return None

        # Build prompt context
        domain_context = {
            "domain_name": event.get("domain_name", ""),
            "current_window_id": event.get("window_id", ""),
        }

        user_profile_payload = {
            "basic_profile": user_profile,
            "domain_context": domain_context,
        }

        event_payload = {
            "event": {
                "user_intent": event_description,
                "time_specification": event.get("time_specification", {}),
                "event_date": event.get("event_date"),
                "app": {
                    "app_name": app_name,
                    "api_name": api_name,
                },
            },
            "chain": {
                "chain_id": chain_id,
                "related_state_items": event.get("related_state_items", []),
            },
        }

        if event.get("original_app_api_variations"):
            event_payload["event"]["original_app_api_variations"] = event.get(
                "original_app_api_variations"
            )

        # Get previous logs from same chain
        MAX_PREVIOUS_LOGS = 20
        previous_logs = chain_logs_history.get(chain_id, [])[-MAX_PREVIOUS_LOGS:]
        previous_logs_formatted = [
            {
                "app_name": log.get("app_name"),
                "api_name": log.get("api_name"),
                "request": log.get("request"),
                "response": log.get("response"),
            }
            for log in previous_logs
        ]

        # Generate prompt
        prompt = app_log_prompt.render(
            api_schema=json.dumps(api_schema, indent=2, ensure_ascii=False),
            user_profile=json.dumps(user_profile_payload, indent=2, ensure_ascii=False),
            app_state=json.dumps(_json_safe(app_state_snapshot), indent=2, ensure_ascii=False),
            event_payload=json.dumps(event_payload, indent=2, ensure_ascii=False),
            previous_chain_logs=json.dumps(
                _json_safe(previous_logs_formatted), indent=2, ensure_ascii=False
            ),
        )
        # print(previous_logs_formatted)

        if self.debug_mode:
            _write_text(
                self.debug_dir / f"prompt_app_log_{app_log_id}_{event_id}.txt",
                prompt,
            )

        # Call LLM
        try:
            response_schema = self._build_response_schema(api_schema)
            llm_result = self.llm_client.generate_json(prompt, response_schema=response_schema)

            # Normalize and validate response
            normalized = self._normalize_llm_response(llm_result.data or {})
            request_payload, response_payload = self._validate_and_extract(
                app_name, api_name, normalized
            )

        except Exception as exc:
            self.logger.warning(f"Failed to generate app log for {event_id}: {exc}")
            if self.debug_mode:
                _write_json(
                    self.debug_dir / f"error_app_log_{event_id}.json",
                    {
                        "event_id": event_id,
                        "error": str(exc),
                        "event": _json_safe(event),
                    },
                )
            return None

        # Record API call to app state
        app.record_api_call(event, request_payload, response_payload)

        # Build golden evidence info from state_refs and evidence_for_states
        golden_evidence = self._build_golden_evidence(event)

        # Build log entry with full traceability
        log = {
            "app_log_id": app_log_id,  # Unique ID for this app log
            "event_id": event_id,      # Preserved from Stage 2
            "atomic_event_id": atomic_event_id,  # Per-instance ID for expanded events
            "timestamp": event.get("timestamp"),
            "app_name": app_name,
            "api_name": api_name,
            "request": request_payload,
            "response": response_payload,
            "golden_evidence": golden_evidence,  # State items this log is evidence for
            "metadata": {
                "user_intent": event_description,
                "domain": event.get("domain_name", ""),
                "window_id": event.get("window_id", ""),
                "chain_id": chain_id,
                "llm_usage": llm_result.usage,
            },
        }

        if self.debug_mode:
            _write_json(
                self.debug_dir / f"app_log_{app_log_id}_{event_id}_{app_name}_{api_name}.json",
                _json_safe(log),
            )

        # Update chain history
        if chain_id:
            if chain_id not in chain_logs_history:
                chain_logs_history[chain_id] = []
            chain_logs_history[chain_id].append(log)

        return log

    def _build_golden_evidence(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build golden evidence mapping from event's state references.

        This creates a structured mapping that allows tracing which state items
        this app_log serves as evidence for.

        Args:
            event: Event containing state_refs and evidence_for_states

        Returns:
            List of evidence entries, each containing:
            - state_category: The category (user_attributes_state, habits_state, preferences_state)
            - state_name: The name of the state item
            - evidenced_fields: List of fields this log provides evidence for
            - state_value: The actual state value (for verification)
            - change_type: Whether this is unchanged, add, modify, etc.
        """
        golden_evidence: List[Dict[str, Any]] = []

        # Get evidence_for_states from the event (generated in Stage 2)
        evidence_for_states = event.get("evidence_for_states", [])
        state_refs = event.get("state_refs", [])

        # Build a lookup from state_name to state_ref data
        state_ref_lookup: Dict[str, Dict[str, Any]] = {}
        for ref in state_refs:
            state_name = ref.get("state_name")
            if state_name:
                state_ref_lookup[state_name] = ref

        # Create evidence entries
        for evidence in evidence_for_states:
            state_name = evidence.get("state_name")
            evidenced_fields = evidence.get("evidenced_fields", [])

            if not state_name:
                continue

            # Get the state_ref for additional context
            state_ref = state_ref_lookup.get(state_name, {})
            state_category = state_ref.get("state_category", "")
            resolved_items = state_ref.get("resolved_items", [])

            # Create an entry for each resolved item (there may be multiple for collections)
            for resolved_item in resolved_items:
                evidence_entry = {
                    "state_category": state_category,
                    "state_name": state_name,
                    "evidenced_fields": evidenced_fields,
                    "change_type": resolved_item.get("change_type", "unchanged"),
                }

                # Include the actual state value for verification
                if "current_value" in resolved_item:
                    evidence_entry["state_value"] = resolved_item["current_value"]
                elif "new_value" in resolved_item:
                    evidence_entry["state_value"] = resolved_item["new_value"]
                elif "delta" in resolved_item:
                    evidence_entry["state_value"] = resolved_item["delta"]

                golden_evidence.append(evidence_entry)

        return golden_evidence

    def _build_response_schema(self, api_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON schema for LLM response.

        Handles Pydantic-generated schemas that contain $defs and $ref by:
        1. Extracting $defs from input and output schemas
        2. Merging them into a single $defs at the root level
        3. Removing $defs from nested schemas to avoid invalid references
        """
        input_schema = deepcopy(api_schema.get("input", {}))
        output_schema = deepcopy(api_schema.get("output", {}))

        # Convert to proper JSON schema if needed
        if "$defs" not in input_schema and "properties" not in input_schema:
            input_schema = self._spec_to_json_schema(input_schema)
        if "$defs" not in output_schema and "properties" not in output_schema:
            output_schema = self._spec_to_json_schema(output_schema)

        # Collect and merge $defs from both schemas to root level
        merged_defs = {}
        if "$defs" in input_schema:
            merged_defs.update(input_schema.pop("$defs"))
        if "$defs" in output_schema:
            merged_defs.update(output_schema.pop("$defs"))

        result = {
            "type": "object",
            "properties": {
                "input": input_schema,
                "output": output_schema,
            },
            "required": ["input", "output"],
            "additionalProperties": False,
        }

        # Add merged $defs at root level if any exist
        if merged_defs:
            result["$defs"] = merged_defs

        return result

    def _spec_to_json_schema(self, spec: Any) -> Dict[str, Any]:
        """Convert a simple spec to JSON schema."""
        if isinstance(spec, dict):
            properties = {key: self._spec_to_json_schema(value) for key, value in spec.items()}
            return {
                "type": "object",
                "properties": properties,
                "required": list(spec.keys()),
                "additionalProperties": False,
            }
        if isinstance(spec, list):
            items_schema = self._spec_to_json_schema(spec[0]) if spec else {"type": "object"}
            return {"type": "array", "items": items_schema}
        if isinstance(spec, str):
            if "|" in spec:
                options = [p.strip() for p in spec.split("|") if p.strip()]
                if options:
                    return {"type": "string", "enum": options}
            type_map = {
                "string": "string",
                "integer": "integer",
                "number": "number",
                "boolean": "boolean",
                "object": "object",
            }
            if spec in type_map:
                return {"type": type_map[spec]}
            return {"type": "string", "description": spec}
        return {"type": "string"}

    def _normalize_llm_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM response to standard format."""
        if data is None:
            return {}

        normalized = dict(data)

        # Map request -> input
        if "input" not in normalized and "request" in normalized:
            req = normalized.pop("request")
            if isinstance(req, dict) and "input" in req:
                normalized["input"] = req["input"]
            else:
                normalized["input"] = req

        # Map response -> output
        if "output" not in normalized and "response" in normalized:
            normalized["output"] = normalized.pop("response")

        # Remove extra fields
        for key in ["event_id", "timestamp", "app_name", "api_name"]:
            normalized.pop(key, None)

        return normalized

    def _validate_and_extract(
        self,
        app_name: str,
        api_name: str,
        normalized: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate and extract input/output payloads."""
        raw_input = normalized.get("input", {})
        raw_output = normalized.get("output", {})

        # Try to validate with specific API models
        input_model, output_model = get_api_input_output_models(app_name, api_name)

        validated_input = raw_input
        validated_output = raw_output

        if self.pydantic_available:
            if input_model:
                try:
                    validated_input = self._pydantic_dump(
                        self._pydantic_validate(input_model, raw_input)
                    )
                except self.ValidationError:
                    pass

            if output_model:
                try:
                    validated_output = self._pydantic_dump(
                        self._pydantic_validate(output_model, raw_output)
                    )
                except self.ValidationError:
                    pass

        return validated_input, validated_output

    def _pydantic_validate(self, model: Any, payload: Dict[str, Any]) -> Any:
        """Validate payload against Pydantic model."""
        if hasattr(model, "model_validate"):
            return model.model_validate(payload)
        return model.parse_obj(payload)

    def _pydantic_dump(self, model_obj: Any) -> Dict[str, Any]:
        """Dump Pydantic model to dict."""
        if hasattr(model_obj, "model_dump"):
            return model_obj.model_dump()
        return model_obj.dict()

    # =========================================================================
    # Save Results
    # =========================================================================

    def _save_results(
        self,
        app_logs: List[Dict[str, Any]],
        user_id: str,
        events: List[Dict[str, Any]],
    ) -> None:
        """Save generated app logs with golden evidence index."""
        # Sort logs chronologically
        app_logs.sort(key=lambda log: log.get("timestamp") or "9999-12-31 23:59:59")

        # Build golden evidence index: state_item -> list of app_log_ids
        golden_evidence_index = self._build_golden_evidence_index(app_logs)

        output_data = {
            "user_id": user_id,
            "total_events": len(events),
            "total_app_logs": len(app_logs),
            "domains": sorted(
                {e.get("domain_name") for e in events if e.get("domain_name")}
            ),
            "golden_evidence_index": golden_evidence_index,
            "app_logs": app_logs,
        }

        _write_json(
            self.output_dir / "app_logs_final.json",
            _json_safe(output_data),
        )

        # Also save a separate golden evidence index file for easy lookup
        _write_json(
            self.output_dir / "golden_evidence_index.json",
            _json_safe({
                "user_id": user_id,
                "description": "Maps state items to their golden evidence app_logs",
                "index": golden_evidence_index,
            }),
        )

        if app_logs:
            self.logger.info(f"Generated {len(app_logs)} app logs")
            self.logger.info(f"First: {app_logs[0]['timestamp']} - {app_logs[0]['app_name']}.{app_logs[0]['api_name']}")
            self.logger.info(f"Last: {app_logs[-1]['timestamp']} - {app_logs[-1]['app_name']}.{app_logs[-1]['api_name']}")
            self.logger.info(f"Golden evidence index: {len(golden_evidence_index)} state items tracked")

    def _build_golden_evidence_index(
        self, app_logs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build an index mapping state items to their golden evidence app_logs.

        This creates a reverse index for easy lookup:
        - Given a state item, find all app_logs that serve as evidence for it

        Returns:
            Dict mapping "state_category:state_name" to evidence info:
            {
                "user_attributes_state:primary_work_environment": {
                    "state_category": "user_attributes_state",
                    "state_name": "primary_work_environment",
                    "evidence_logs": [
                        {
                            "app_log_id": "log_00001",
                            "event_id": "work_education_w0_1_1",
                            "timestamp": "2023-10-06 13:00:00",
                            "app_name": "Notion",
                            "api_name": "UpdatePage",
                            "evidenced_fields": ["current_value"],
                            "change_type": "unchanged"
                        },
                        ...
                    ]
                },
                ...
            }
        """
        index: Dict[str, Dict[str, Any]] = {}

        for log in app_logs:
            app_log_id = log.get("app_log_id", "")
            event_id = log.get("event_id", "")
            timestamp = log.get("timestamp", "")
            app_name = log.get("app_name", "")
            api_name = log.get("api_name", "")
            golden_evidence = log.get("golden_evidence", [])

            for evidence in golden_evidence:
                state_category = evidence.get("state_category", "")
                state_name = evidence.get("state_name", "")
                evidenced_fields = evidence.get("evidenced_fields", [])
                change_type = evidence.get("change_type", "unchanged")

                if not state_category or not state_name:
                    continue

                key = f"{state_category}:{state_name}"

                if key not in index:
                    index[key] = {
                        "state_category": state_category,
                        "state_name": state_name,
                        "evidence_logs": [],
                    }

                index[key]["evidence_logs"].append({
                    "app_log_id": app_log_id,
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "app_name": app_name,
                    "api_name": api_name,
                    "evidenced_fields": evidenced_fields,
                    "change_type": change_type,
                })

        # Sort evidence_logs by timestamp for each state item
        for entry in index.values():
            entry["evidence_logs"].sort(key=lambda x: x.get("timestamp", ""))
            entry["total_evidence_count"] = len(entry["evidence_logs"])

        return index

    def _load_existing_app_logs(self) -> List[Dict[str, Any]]:
        """Load existing app logs from final output and debug files."""
        logs_by_event_id: Dict[str, Dict[str, Any]] = {}

        final_path = self.output_dir / "app_logs_final.json"
        if final_path.exists():
            try:
                payload = json.loads(final_path.read_text())
                for log in payload.get("app_logs", []):
                    event_id = log.get("event_id")
                    if event_id and event_id not in logs_by_event_id:
                        logs_by_event_id[event_id] = log
            except Exception as exc:
                self.logger.warning(f"Failed to read {final_path}: {exc}")

        if self.debug_mode and hasattr(self, "debug_dir") and self.debug_dir.exists():
            for path in self.debug_dir.glob("app_log_*.json"):
                if path.name.startswith("error_app_log_"):
                    continue
                try:
                    log = json.loads(path.read_text())
                except Exception:
                    continue
                event_id = log.get("event_id")
                if event_id and event_id not in logs_by_event_id:
                    logs_by_event_id[event_id] = log

        return list(logs_by_event_id.values())

    def _max_app_log_counter(self, logs: List[Dict[str, Any]]) -> int:
        """Find the highest app_log_id counter in existing logs."""
        max_counter = 0
        for log in logs:
            app_log_id = log.get("app_log_id", "")
            match = re.match(r"log_(\d+)$", app_log_id)
            if match:
                max_counter = max(max_counter, int(match.group(1)))
        return max_counter
