"""
Stage 2: Events Chain Generation

Responsibilities:
- Generate events chains for each domain, window by window (w0 → w4)
- Maintain proper context flow between windows
- Convert dynamic profile state items into concrete behavioral events

This stage produces event sequences for each domain that will be converted
to app logs in Stage 3.
"""

from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajectory_synthesis.llm_client import GeminiJSONClient, LLMResult
from trajectory_synthesis.events_chain_generator import EventsChainRequest, generate_events_chain


def _slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "domain"


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    """Write JSON to file with formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_text(path: Path, text: str) -> None:
    """Write text to file."""
    path.write_text(text)


# Keys to skip entirely from required_observable_fields
_SKIP_OBSERVABLE_KEYS = {"schedule_dates", "priority", "signals"}
# Keys to treat as atomic (don't recurse into sub-fields)
_ATOMIC_OBSERVABLE_KEYS = {"timing", "schedule"}


def _extract_observable_field_paths(
    value: object,
    prefix: str = "",
    max_depth: int = 4,
) -> List[str]:
    """
    Extract all leaf field paths from a nested dict/list structure.
    Returns paths like "current_value.schedule", "current_value.location", etc.

    - Skips: schedule_dates (derived), priority, signals
    - Treats as atomic: timing, schedule (no sub-field splitting)
    """
    paths: List[str] = []
    if max_depth <= 0:
        if prefix:
            paths.append(prefix)
        return paths

    if isinstance(value, dict):
        for key, val in value.items():
            # Skip keys that should not be in required_observable_fields
            if key in _SKIP_OBSERVABLE_KEYS:
                continue

            new_prefix = f"{prefix}.{key}" if prefix else key

            # Treat certain keys as atomic (don't recurse into sub-fields)
            if key in _ATOMIC_OBSERVABLE_KEYS:
                paths.append(new_prefix)
                continue

            # For nested dicts, recurse
            if isinstance(val, dict):
                paths.extend(_extract_observable_field_paths(val, new_prefix, max_depth - 1))
            elif isinstance(val, list):
                # For lists, just add the path to the list itself (not individual items)
                paths.append(new_prefix)
            else:
                # Leaf value
                paths.append(new_prefix)
    elif isinstance(value, list):
        if prefix:
            paths.append(prefix)
    else:
        if prefix:
            paths.append(prefix)

    return paths


def _find_changed_fields(
    current: object,
    previous: object,
    prefix: str = "current_value",
) -> List[str]:
    """
    Find fields that differ between current and previous values.
    Returns paths to changed fields only.
    """
    changed: List[str] = []

    # If previous is None, all current fields are "new"
    if previous is None:
        return _extract_observable_field_paths(current, prefix)

    # If types differ, the whole thing changed
    if type(current) != type(previous):
        if prefix:
            return [prefix]
        return _extract_observable_field_paths(current, prefix)

    if isinstance(current, dict) and isinstance(previous, dict):
        all_keys = set(current.keys()) | set(previous.keys())
        for key in all_keys:
            # Skip keys that should not be in required_observable_fields
            if key in _SKIP_OBSERVABLE_KEYS:
                continue

            new_prefix = f"{prefix}.{key}" if prefix else key
            curr_val = current.get(key)
            prev_val = previous.get(key)

            # If key only in one, it changed
            if key not in current or key not in previous:
                changed.append(new_prefix)
                continue

            # Treat certain keys as atomic
            if key in _ATOMIC_OBSERVABLE_KEYS:
                if curr_val != prev_val:
                    changed.append(new_prefix)
                continue

            # Recurse for nested dicts
            if isinstance(curr_val, dict) and isinstance(prev_val, dict):
                changed.extend(_find_changed_fields(curr_val, prev_val, new_prefix))
            elif curr_val != prev_val:
                changed.append(new_prefix)
    elif current != previous:
        # Non-dict values that differ
        if prefix:
            changed.append(prefix)

    return changed


@dataclass
class Domain:
    """Domain specification."""
    domain_name: str
    domain_scope_definition: str


@dataclass
class Stage2Result:
    """Result from Stage 2: Events Chain Generation."""
    events_chains_by_domain: Dict[str, List[Dict]]
    usage: Dict[str, Any]


class EventsChainStage:
    """
    Stage 2: Events Chain Generation

    This stage converts dynamic profile state items into concrete behavioral
    events for each domain, processing window by window to maintain context.
    """

    def __init__(
        self,
        llm_client: GeminiJSONClient,
        output_dir: Path,
        debug_mode: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.debug_mode = debug_mode
        self.logger = logger or logging.getLogger(__name__)
        _ensure_dir(self.output_dir)

        if self.debug_mode:
            self.debug_dir = self.output_dir / "debug" / "stage2_events_chain"
            _ensure_dir(self.debug_dir)

    def run(
        self,
        domains: Sequence[Domain],
        user_basic_profile: Dict[str, Any],
        dynamic_profiles: Dict[str, Dict],
        world_background: Optional[Any] = None,
        user_life_contexts: Optional[Dict] = None,
    ) -> Stage2Result:
        """
        Run the complete Stage 2 pipeline.

        Args:
            domains: List of domains to generate events for
            user_basic_profile: User's basic profile
            dynamic_profiles: Dynamic profiles from Stage 1
            world_background: Optional world context
            user_life_contexts: Optional life context information

        Returns:
            Stage2Result with events chains for all domains
        """
        aggregate_usage: Dict[str, Any] = {}
        events_chains_by_domain: Dict[str, List[Dict]] = {}

        # Build cross-domain summaries for context
        user_full_state_summaries = self._build_user_full_state_summaries(dynamic_profiles)
        initial_all_domains_summary = self._build_initial_summary(dynamic_profiles)

        for domain in domains:
            slug = _slugify(domain.domain_name)
            cache_path = self.output_dir / f"{slug}_events_chain.json"

            # Check if domain events chain is already cached
            if cache_path.exists():
                self.logger.info(f"=== Loading cached events chain for {domain.domain_name} ===")
                try:
                    cached_data = json.loads(cache_path.read_text())
                    events_chains_by_domain[domain.domain_name] = cached_data
                    aggregate_usage[domain.domain_name] = {}
                    self.logger.info(f"  Loaded {len(cached_data)} windows from cache")
                    continue
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load cache: {e}, regenerating...")

            self.logger.info(f"=== Generating events chain for {domain.domain_name} ===")

            events_chain_windows, domain_usage = self._generate_events_chain_for_domain(
                domain=domain,
                user_basic_profile=user_basic_profile,
                dynamic_profiles=dynamic_profiles,
                world_background=world_background,
                user_life_contexts=user_life_contexts,
                user_full_state_summaries=user_full_state_summaries,
                initial_all_domains_summary=initial_all_domains_summary,
            )

            events_chains_by_domain[domain.domain_name] = events_chain_windows
            aggregate_usage[domain.domain_name] = domain_usage

            # Save per-domain events chain
            _write_json(
                cache_path,
                events_chain_windows,
            )

        # Save combined events chains
        _write_json(
            self.output_dir / "all_events_chains.json",
            events_chains_by_domain,
        )

        return Stage2Result(
            events_chains_by_domain=events_chains_by_domain,
            usage=aggregate_usage,
        )

    def _generate_events_chain_for_domain(
        self,
        domain: Domain,
        user_basic_profile: Dict,
        dynamic_profiles: Dict[str, Dict],
        world_background: str,
        user_life_contexts: Optional[Dict],
        user_full_state_summaries: Dict[str, Dict],
        initial_all_domains_summary: str,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Generate events chain for a single domain, window by window."""

        slug = _slugify(domain.domain_name)
        domain_profile = dynamic_profiles.get(domain.domain_name, {})

        if not domain_profile:
            self.logger.warning(f"No profile found for {domain.domain_name}")
            return [], {}

        # Resolve window states
        resolved_windows = self._resolve_window_states(domain_profile)

        if self.debug_mode:
            _write_json(
                self.debug_dir / f"{slug}_resolved_windows.json",
                resolved_windows,
            )

        # Initialize state tracker for this domain
        state_tracker = self._init_state_tracker(domain_profile)

        # Build context maps
        window_ids = [w.get("window_id") for w in resolved_windows if w.get("window_id")]

        # Try to load domain-specific world background from file
        domain_world_background = self._load_domain_world_background(slug, world_background)
        world_background_map = self._map_world_background_to_windows(domain_world_background, window_ids)

        # Build summary maps
        summary_all_by_window, summary_by_window_domain = self._build_summary_maps(
            user_full_state_summaries, domain.domain_name
        )

        # Process life contexts
        life_context_baseline = {}
        life_context_by_window = {}
        if user_life_contexts:
            life_context_baseline = user_life_contexts.get("baseline", {})
            life_context_by_window = user_life_contexts.get("by_window", {})

        # Initial summaries for w1 fallbacks
        domain_initial_summary = (domain_profile.get("initial_state") or {}).get("summary", "")

        events_chain_windows: List[Dict] = []
        usage: Dict[str, Any] = {}
        rng = random.Random()
        stable_state_convert_probability = 0.5

        # Process each window
        for idx, window_state in enumerate(resolved_windows):
            window_id = window_state.get("window_id")
            if not window_id:
                continue

            # Check for cached window result
            window_cache_path = self.debug_dir / f"{slug}_events_chain_{window_id}.json" if self.debug_mode else None
            if window_cache_path and window_cache_path.exists():
                try:
                    cached_window = json.loads(window_cache_path.read_text())
                    self.logger.info(f"  Loading cached window {window_id}...")
                    if isinstance(cached_window, list) and cached_window:
                        cached_window = cached_window[0]
                    # Still need to update state tracker for subsequent windows
                    # Process conversion targets to update tracker state
                    next_window_state = resolved_windows[idx + 1] if idx + 1 < len(resolved_windows) else None
                    conversion_targets = self._select_conversion_targets_for_window(
                        window_state=window_state,
                        next_window_state=next_window_state,
                        state_tracker=state_tracker,
                        rng=rng,
                        stable_probability=stable_state_convert_probability,
                        is_window0=(idx == 0),
                    )
                    # Enrich cached window with state values (in case cache was from before enrichment)
                    self._enrich_events_chain_with_state_values(cached_window, conversion_targets)
                    events_chain_windows.append(cached_window)
                    usage[window_id] = {}
                    self._update_tracker_after_conversion(state_tracker, conversion_targets)
                    continue
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load window cache: {e}, regenerating...")

            self.logger.info(f"  Processing window {window_id}...")

            # Build prompt context
            prompt_context = self._build_window_prompt_context(
                idx=idx,
                window_id=window_id,
                window_state=window_state,
                resolved_windows=resolved_windows,
                summary_all_by_window=summary_all_by_window,
                summary_by_window_domain=summary_by_window_domain,
                domain_initial_summary=domain_initial_summary,
                initial_all_domains_summary=initial_all_domains_summary,
                life_context_baseline=life_context_baseline,
                life_context_by_window=life_context_by_window,
                domain_name=domain.domain_name,
            )

            # Get next window for look-ahead
            next_window_state = None
            if idx + 1 < len(resolved_windows):
                next_window_state = resolved_windows[idx + 1]

            # Select conversion targets
            # For window0 (idx=0), treat all items as unchanged with sampling
            conversion_targets = self._select_conversion_targets_for_window(
                window_state=window_state,
                next_window_state=next_window_state,
                state_tracker=state_tracker,
                rng=rng,
                stable_probability=stable_state_convert_probability,
                is_window0=(idx == 0),
            )

            # Build payload for prompt
            domain_window_state_payload = {
                "window_id": window_id,
                "time_range": window_state.get("time_range"),
                "state_table": conversion_targets,
            }

            # Clean payload for prompt (remove metadata)
            payload_for_prompt = self._clean_payload_for_prompt(domain_window_state_payload)

            if self.debug_mode:
                _write_json(
                    self.debug_dir / f"{slug}_window_state_payload_{window_id}.json",
                    domain_window_state_payload,
                )
                _write_json(
                    self.debug_dir / f"{slug}_window_state_for_prompt_{window_id}.json",
                    payload_for_prompt,
                )

            # Generate events chain
            window_world_background = world_background_map.get(
                window_id, "No world background available for this window."
            )

            events_result = generate_events_chain(
                self.llm_client,
                EventsChainRequest(
                    domain_name=domain.domain_name,
                    user_basic_profile=json.dumps(user_basic_profile, indent=2, ensure_ascii=False),
                    user_previous_window_summary=prompt_context["user_previous_window_summary"],
                    user_domain_previous_window_summary=prompt_context["user_domain_previous_window_summary"],
                    user_this_window_description=prompt_context["user_this_window_description"],
                    domain_window_state=payload_for_prompt,
                    world_background=window_world_background,
                ),
            )

            events_payload = events_result.data
            # Safety check: if LLM returns a list, take the first dict
            if isinstance(events_payload, list) and len(events_payload) > 0:
                events_payload = events_payload[0]
            if isinstance(events_payload, dict):
                events_payload.setdefault("window_id", window_id)
                events_payload.setdefault("time_range", window_state.get("time_range"))
                self._assign_event_ids(events_payload, domain.domain_name)
                # Enrich state_refs with actual state values for downstream processing
                self._enrich_events_chain_with_state_values(events_payload, conversion_targets)

            events_chain_windows.append(events_payload)
            usage[window_id] = events_result.usage

            # Save window cache (always save for resumability)
            if self.debug_mode:
                _write_text(
                    self.debug_dir / f"{slug}_events_chain_{window_id}_prompt.txt",
                    events_result.prompt,
                )
                _write_json(
                    self.debug_dir / f"{slug}_events_chain_{window_id}.json",
                    events_payload,
                )

            # Update tracker for converted items
            self._update_tracker_after_conversion(
                state_tracker, conversion_targets
            )

        return events_chain_windows, usage

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_user_full_state_summaries(
        self, dynamic_profiles: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Build cross-domain state summaries per window."""
        try:
            from trajectory_synthesis.generation_pipeline import _build_user_full_state_summaries
            return _build_user_full_state_summaries(dynamic_profiles)
        except ImportError:
            # Fallback implementation
            summaries: Dict[str, Dict] = {}
            for domain_name, profile in dynamic_profiles.items():
                for window in profile.get("time_windows", []):
                    window_id = window.get("window_id")
                    if not window_id:
                        continue
                    if window_id not in summaries:
                        summaries[window_id] = {
                            "window_profile_summary": "",
                            "summary_by_domain": {},
                        }
                    summaries[window_id]["summary_by_domain"][domain_name] = (
                        window.get("window_description", "")
                    )
            return summaries

    def _build_initial_summary(self, dynamic_profiles: Dict[str, Dict]) -> str:
        """Build cross-domain initial summary."""
        parts = []
        for domain_name, profile in dynamic_profiles.items():
            init_summary = (profile.get("initial_state") or {}).get("summary", "")
            if init_summary:
                parts.append(f"{domain_name}: {init_summary}")
        return "\n\n".join(parts)

    def _resolve_window_states(self, domain_profile: Dict) -> List[Dict]:
        """Resolve window states from domain profile."""
        try:
            from trajectory_synthesis.generation_pipeline import _resolve_window_states
            return _resolve_window_states(domain_profile)
        except ImportError:
            # Fallback: return time_windows directly
            return domain_profile.get("time_windows", [])

    def _init_state_tracker(self, domain_profile: Dict) -> Dict:
        """Initialize state tracker from initial_state."""
        tracker = {
            "user_attributes_state": {},
            "habits_state": {},
            "preferences_state": {},
        }

        initial_state = domain_profile.get("initial_state", {})
        for state_type in tracker.keys():
            section = initial_state.get(state_type, {})
            if isinstance(section, dict):
                for name, value in section.items():
                    values = value if isinstance(value, list) else [value]
                    for val in values:
                        key = self._make_item_key(val)
                        tracker[state_type].setdefault(name, {})[key] = {
                            "value": val,
                            "metadata": {
                                "already_converted_to_events_chain": False,
                            },
                        }

        return tracker

    def _make_item_key(self, value: Any) -> str:
        """Create a hashable key for tracking state items."""
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _load_domain_world_background(
        self, slug: str, fallback_world_background: Any
    ) -> Any:
        """
        Load domain-specific world background from file if it exists.

        Looks for {slug}_world_background.json in the output directory.
        Falls back to the provided world_background if not found.

        Args:
            slug: Domain slug (e.g., "family_close_relationships")
            fallback_world_background: Fallback value if domain-specific file not found

        Returns:
            Domain-specific world background dict or fallback value
        """
        domain_wb_path = self.output_dir / f"{slug}_world_background.json"
        if domain_wb_path.exists():
            try:
                domain_wb = json.loads(domain_wb_path.read_text())
                self.logger.info(f"  Loaded domain-specific world background from {domain_wb_path.name}")
                return domain_wb
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Failed to load {domain_wb_path.name}: {e}")
                self.logger.info(f"  Using fallback world background")

        return fallback_world_background

    def _map_world_background_to_windows(
        self, world_background: Any, window_ids: List[str]
    ) -> Dict[str, str]:
        """Map world background to windows."""
        try:
            from trajectory_synthesis.generation_pipeline import _map_world_background_to_windows
            return _map_world_background_to_windows(world_background, window_ids)
        except ImportError:
            # Fallback: use same background for all windows
            if isinstance(world_background, str):
                return {wid: world_background for wid in window_ids}
            return {wid: "" for wid in window_ids}

    def _build_summary_maps(
        self,
        user_full_state_summaries: Dict[str, Dict],
        domain_name: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build summary maps from full state summaries."""
        summary_all_by_window: Dict[str, str] = {}
        summary_by_window_domain: Dict[str, str] = {}

        for window_id, entry in user_full_state_summaries.items():
            if not window_id or not isinstance(entry, dict):
                continue
            summary_all_by_window[window_id] = entry.get("window_profile_summary", "")
            summary_by_window_domain[window_id] = (
                entry.get("summary_by_domain", {}).get(domain_name, "")
                or entry.get("window_description_by_domain", {}).get(domain_name, "")
            )

        return summary_all_by_window, summary_by_window_domain

    def _build_window_prompt_context(
        self,
        idx: int,
        window_id: str,
        window_state: Dict,
        resolved_windows: List[Dict],
        summary_all_by_window: Dict[str, str],
        summary_by_window_domain: Dict[str, str],
        domain_initial_summary: str,
        initial_all_domains_summary: str,
        life_context_baseline: Dict,
        life_context_by_window: Dict,
        domain_name: str,
    ) -> Dict[str, str]:
        """Build context for window prompt."""

        if idx == 0:
            previous_all_summary = initial_all_domains_summary
            previous_domain_summary = domain_initial_summary
        else:
            prev_window_id = resolved_windows[idx - 1].get("window_id")
            previous_all_summary = summary_all_by_window.get(prev_window_id, "")
            previous_domain_summary = summary_by_window_domain.get(prev_window_id, "")

        user_previous_window_summary = (
            "<previous window summary (all domains)>\n"
            f"{previous_all_summary or 'No previous window summary available.'}\n"
            "</previous window summary (all domains)>"
        )

        user_domain_previous_window_summary = (
            f"<previous window summary in {domain_name}>\n"
            f"{previous_domain_summary or 'No previous window summary available.'}\n"
            f"</previous window summary in {domain_name}>"
        )

        domain_window_description = window_state.get("window_description", "")
        user_this_window_description = (
            f"<this window description in {domain_name}>\n"
            f"{domain_window_description or 'No description available.'}\n"
            f"</this window description in {domain_name}>"
        )

        return {
            "user_previous_window_summary": user_previous_window_summary,
            "user_domain_previous_window_summary": user_domain_previous_window_summary,
            "user_this_window_description": user_this_window_description,
        }

    def _select_conversion_targets_for_window(
        self,
        window_state: Dict,
        next_window_state: Optional[Dict],
        state_tracker: Dict,
        rng: random.Random,
        stable_probability: float,
        is_window0: bool = False,
    ) -> Dict[str, List[Dict]]:
        """Select state items to convert to events for this window.

        For window0, all state items are treated as 'unchanged' and sampled
        using the same probability-based approach as other windows.

        IMPORTANT: For user_attributes_state with list values (collections),
        each item in the list is treated as a separate entry and sampled independently.
        """

        conversion_targets = {
            "user_attributes_state": [],
            "habits_state": [],
            "preferences_state": [],
        }

        # Build set of items that will change in next window
        next_window_changing = set()
        if next_window_state:
            for state_type in conversion_targets.keys():
                for item in next_window_state.get(state_type, []):
                    name = item.get("name")
                    change_type = (item.get("change_type") or item.get("op") or "").lower()
                    if name and change_type and change_type not in {"", "stable", "unchanged"}:
                        next_window_changing.add((state_type, name))

        for state_type in conversion_targets.keys():
            window_items = window_state.get(state_type, [])
            seen_keys: Dict[str, set] = {}

            for item in window_items:
                name = item.get("name")
                if not name:
                    continue

                item_change_type = (item.get("change_type") or item.get("op") or "").lower()
                current_value = item.get("current_value")
                previous_value = item.get("previous_value")
                change_reason = item.get("change_reason")

                # For user_attributes_state with collection-level changes (remove/modify/replace/set/drop),
                # emit a single collection-level change entry
                if (
                    state_type == "user_attributes_state"
                    and isinstance(current_value, list)
                    and item_change_type in {"remove", "modify", "replace", "set", "drop"}
                ):
                    metadata = {
                        "updated_this_window": True,
                        "freshness": True,
                        "already_converted_to_events_chain": False,
                        "should_convert_to_events_chain": True,
                        "reason": change_reason,
                    }

                    # For remove: only record the removed item (atomic operation)
                    if item_change_type == "remove" and previous_value is not None:
                        # Find the removed item by comparing previous and current
                        removed_items = [v for v in previous_value if v not in current_value]
                        removed_value = removed_items[0] if len(removed_items) == 1 else removed_items
                        change_entry: Dict[str, object] = {
                            "name": name,
                            "change_type": item_change_type,
                            "removed_value": removed_value,
                            "metadata": metadata,
                        }
                    else:
                        change_entry = {
                            "name": name,
                            "current_value": current_value,
                            "change_type": item_change_type,
                            "metadata": metadata,
                        }
                        if previous_value is not None:
                            change_entry["previous_value"] = previous_value

                    if change_reason:
                        change_entry["change_reason"] = change_reason
                    # Add required_observable_fields
                    change_entry["required_observable_fields"] = self._build_required_observable_fields(change_entry)
                    conversion_targets[state_type].append(change_entry)

                # For user_attributes_state with list values, expand to individual items
                # Each item is tracked and sampled independently
                values = (
                    current_value
                    if state_type == "user_attributes_state" and isinstance(current_value, list)
                    else [current_value]
                )

                for val in values:
                    # Process value for prompt
                    val_for_prompt = val
                    if state_type == "habits_state":
                        val_for_prompt = self._expand_habit_schedule_dates(
                            val, window_state.get("time_range")
                        )
                    elif state_type == "preferences_state":
                        val_for_prompt = self._strip_signals(val)

                    # Check tracker for existing entry
                    key = self._make_item_key(val)
                    tracker_entries = state_tracker[state_type].setdefault(name, {})
                    existing = tracker_entries.get(key)
                    already_converted = bool(
                        existing and existing.get("metadata", {}).get("already_converted_to_events_chain")
                    )
                    updated_this_window = existing is None
                    freshness = updated_this_window
                    reason = change_reason if updated_this_window else None

                    # For window0, treat all items as unchanged and apply sampling
                    if is_window0:
                        # Window0: all items are treated as unchanged, apply sampling
                        will_change_next = (state_type, name) in next_window_changing

                        should_convert = False
                        if state_type == "habits_state":
                            should_convert = True  # Always convert habits
                        elif will_change_next:
                            should_convert = True  # Must convert if changing next window
                        elif rng.random() < stable_probability:
                            should_convert = True  # Probabilistic sampling

                        metadata = {
                            "is_window0": True,
                            "updated_this_window": updated_this_window,
                            "freshness": freshness,
                            "already_converted_to_events_chain": already_converted,
                            "should_convert_to_events_chain": should_convert,
                        }

                        # Update tracker entry
                        tracker_entries[key] = {"value": val, "metadata": deepcopy(metadata)}
                        seen_keys.setdefault(name, set()).add(key)

                        if should_convert:
                            # Window0: always unchanged, only current_value needed
                            entry = {
                                "name": name,
                                "change_type": "unchanged",
                                "current_value": val_for_prompt,
                                "metadata": metadata,
                            }

                            # Build required observable fields
                            entry["required_observable_fields"] = self._build_required_observable_fields(entry)

                            conversion_targets[state_type].append(entry)

                            # Mark as converted in tracker
                            tracker_entries[key]["metadata"]["already_converted_to_events_chain"] = True
                    else:
                        # Non-window0: original logic
                        will_change_next = (state_type, name) in next_window_changing

                        # Determine the effective change type
                        # For individual items from a list, use item_change_type if fresh, otherwise unchanged
                        effective_change_type = item_change_type if freshness and item_change_type and item_change_type not in {"", "stable"} else "unchanged"
                        is_actual_change = effective_change_type != "unchanged"

                        should_convert = False
                        if state_type == "habits_state":
                            should_convert = True  # Always convert habits
                        elif freshness:
                            should_convert = True  # Always convert fresh items
                        elif will_change_next:
                            should_convert = True
                        elif not already_converted and rng.random() < stable_probability:
                            should_convert = True

                        metadata = {
                            "updated_this_window": updated_this_window,
                            "freshness": freshness,
                            "already_converted_to_events_chain": already_converted,
                            "should_convert_to_events_chain": should_convert,
                            "reason": reason if freshness else None,
                        }

                        # Update tracker entry
                        tracker_entries[key] = {"value": val, "metadata": deepcopy(metadata)}
                        seen_keys.setdefault(name, set()).add(key)

                        if should_convert:
                            entry = {
                                "name": name,
                                "change_type": effective_change_type,
                                "metadata": metadata,
                            }

                            # For unchanged: only current_value needed
                            # For changes: provide semantically appropriate delta information
                            if effective_change_type == "unchanged":
                                entry["current_value"] = val_for_prompt
                            elif effective_change_type in {"add", "acquire"}:
                                # New item: use new_value (no "from" concept)
                                entry["new_value"] = val_for_prompt
                                entry["previous_value"] = None
                                if change_reason:
                                    entry["change_reason"] = change_reason
                            elif effective_change_type == "drop":
                                # Dropped item: show what was dropped
                                entry["dropped_value"] = val_for_prompt
                                if change_reason:
                                    entry["change_reason"] = change_reason
                            else:
                                # Modification: use from/to delta
                                entry["delta"] = {
                                    "to": val_for_prompt,
                                }
                                if previous_value is not None:
                                    prev_for_prompt = previous_value
                                    if state_type == "habits_state":
                                        prev_for_prompt = self._expand_habit_schedule_dates(
                                            previous_value, window_state.get("time_range")
                                        )
                                    elif state_type == "preferences_state":
                                        prev_for_prompt = self._strip_signals(previous_value)
                                    entry["delta"]["from"] = prev_for_prompt
                                if change_reason:
                                    entry["change_reason"] = change_reason

                            # Build required observable fields
                            entry["required_observable_fields"] = self._build_required_observable_fields(entry)

                            conversion_targets[state_type].append(entry)

                            # Mark as converted in tracker
                            tracker_entries[key]["metadata"]["already_converted_to_events_chain"] = True

            # Drop tracker entries that no longer exist in the current snapshot (e.g., removals)
            for name, entries in list(state_tracker[state_type].items()):
                keep_keys = seen_keys.get(name, set())
                for key in list(entries.keys()):
                    if key not in keep_keys:
                        entries.pop(key, None)
                if not entries:
                    state_tracker[state_type].pop(name, None)

        return conversion_targets

    def _expand_habit_schedule_dates(
        self, habit_value: Any, time_range: Optional[List[str]]
    ) -> Any:
        """Expand habit schedule dates within time range."""
        try:
            from trajectory_synthesis.generation_pipeline import _expand_habit_schedule_dates
            return _expand_habit_schedule_dates(habit_value, time_range)
        except ImportError:
            return habit_value

    def _strip_signals(self, value: Any) -> Any:
        """Strip signals from preference values."""
        if isinstance(value, dict):
            result = deepcopy(value)
            result.pop("signals", None)
            return result
        elif isinstance(value, list):
            return [
                {**deepcopy(item), "signals": None}
                if isinstance(item, dict) else item
                for item in value
            ]
        return value

    def _build_required_observable_fields(self, entry: Dict[str, object]) -> List[str]:
            """
            Build the required_observable_fields list for a state item entry.

            - For 'unchanged': include all fields from current_value
            - For 'add'/'acquire': include all fields from new_value + change_reason
            - For 'drop': include dropped_value + change_reason
            - For modifications (modify/adjust/shift/refine): include changed fields from delta + change_reason
            """
            fields: List[str] = []

            change_type = (entry.get("change_type") or entry.get("op") or "").lower()

            if change_type == "unchanged":
                # For unchanged: show all fields from current_value
                current_value = entry.get("current_value")
                if current_value is not None:
                    current_paths = _extract_observable_field_paths(current_value, "current_value")
                    fields.extend(current_paths)

            elif change_type in {"add", "acquire"}:
                # For new items: show all fields from new_value
                new_value = entry.get("new_value")
                if new_value is not None:
                    new_paths = _extract_observable_field_paths(new_value, "new_value")
                    fields.extend(new_paths)
                # Add change_reason if present
                if entry.get("change_reason"):
                    fields.append("change_reason")

            elif change_type == "drop":
                # For dropped items: show the dropped value
                dropped_value = entry.get("dropped_value")
                if dropped_value is not None:
                    dropped_paths = _extract_observable_field_paths(dropped_value, "dropped_value")
                    fields.extend(dropped_paths)
                # Add change_reason if present
                if entry.get("change_reason"):
                    fields.append("change_reason")

            elif change_type == "remove":
                # For removed items: show the removed value (atomic operation)
                removed_value = entry.get("removed_value")
                if removed_value is not None:
                    removed_paths = _extract_observable_field_paths(removed_value, "removed_value")
                    fields.extend(removed_paths)
                # Add change_reason if present
                if entry.get("change_reason"):
                    fields.append("change_reason")

            else:
                # For modifications (modify, adjust, shift, refine): use delta structure
                delta = entry.get("delta", {})
                to_value = delta.get("to")
                from_value = delta.get("from")

                if to_value is not None:
                    if from_value is not None:
                        # Only include changed fields between from and to
                        changed_paths = _find_changed_fields(to_value, from_value, "delta.to")
                        fields.extend(changed_paths)
                    else:
                        # No previous value, include all fields from to
                        to_paths = _extract_observable_field_paths(to_value, "delta.to")
                        fields.extend(to_paths)

                # Add change_reason if present
                if entry.get("change_reason"):
                    fields.append("change_reason")

            return fields

    def _clean_payload_for_prompt(self, payload: Dict) -> Dict:
        """Remove internal metadata from payload for prompt."""
        cleaned = deepcopy(payload)

        for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
            entries = cleaned.get("state_table", {}).get(state_type, [])
            for entry in entries:
                entry.pop("metadata", None)

        return cleaned

    def _assign_event_ids(self, window_payload: Dict, domain_name: str) -> None:
        """Assign unique IDs to events and chains."""
        event_chains = window_payload.get("event_chains") or window_payload.get("events_chain", [])
        window_id = window_payload.get("window_id", "window")
        domain_slug = _slugify(domain_name)

        for chain_idx, chain in enumerate(event_chains, 1):
            if not isinstance(chain, dict):
                continue
            # Assign chain_id if not present
            if "chain_id" not in chain:
                chain["chain_id"] = f"{domain_slug}_{window_id}_{chain_idx}"
            events = chain.get("events", [])
            for event_idx, event in enumerate(events, 1):
                if isinstance(event, dict) and "event_id" not in event:
                    event["event_id"] = f"{domain_slug}_{window_id}_{chain_idx}_{event_idx}"

    def _enrich_events_chain_with_state_values(
        self,
        events_chain: Dict,
        conversion_targets: Dict[str, List[Dict]],
    ) -> Dict:
        """
        Enrich events_chain by resolving state_refs to actual state values.

        This adds the original state data (current_value, delta, change_reason, etc.)
        back to each state_ref, enabling downstream stages to:
        1. Generate realistic app logs with concrete values
        2. Create questions that reference specific state information

        Args:
            events_chain: The LLM-generated events chain with state_refs
            conversion_targets: The state_table from _select_conversion_targets_for_window
                              containing full state data

        Returns:
            Enriched events_chain with resolved_items in each state_ref
        """
        # Build lookup: (state_category, state_name) -> list of state items
        # Note: same name can have multiple items (e.g., multiple certifications)
        state_lookup: Dict[Tuple[str, str], List[Dict]] = {}
        for state_type in ["user_attributes_state", "habits_state", "preferences_state"]:
            for item in conversion_targets.get(state_type, []):
                name = item.get("name")
                if not name:
                    continue
                key = (state_type, name)
                if key not in state_lookup:
                    state_lookup[key] = []
                # Create a copy without internal metadata for cleaner output
                item_copy = {k: v for k, v in item.items() if k != "metadata"}
                state_lookup[key].append(item_copy)

        # Enrich each event_chain's state_refs
        if isinstance(events_chain, list):
            events_chain = events_chain[0] if events_chain else None
        if isinstance(events_chain, dict):
            event_chains = events_chain.get("event_chains") or events_chain.get("events_chain", [])
        else:
            event_chains = []
        for chain in event_chains:
            if not isinstance(chain, dict):
                continue

            enriched_refs = []
            for ref in chain.get("state_refs", []):
                category = ref.get("state_category")
                name = ref.get("state_name")

                if not category or not name:
                    enriched_refs.append(ref)
                    continue

                items = state_lookup.get((category, name), [])

                enriched_ref = {
                    "state_category": category,
                    "state_name": name,
                    "resolved_items": items,  # Full state data
                }
                enriched_refs.append(enriched_ref)

            chain["state_refs"] = enriched_refs

        return events_chain

    def _update_tracker_after_conversion(
        self,
        state_tracker: Dict,
        conversion_targets: Dict[str, List[Dict]],
    ) -> None:
        """Update tracker after conversion to mark items as converted.

        Note: This is now mostly a no-op since _select_conversion_targets_for_window
        already updates the tracker internally. Kept for safety/compatibility.
        """
        # The tracker is now updated in _select_conversion_targets_for_window
        # This method is kept for backwards compatibility but does nothing
        pass
