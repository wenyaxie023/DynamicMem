"""
App Log Generator

Converts events chains into concrete app logs using the app system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from mem_bench.behavior_and_conversation.app_system import AppLogEntry, AppRegistry


@dataclass
class EventChain:
    """An event chain from the events chain generator."""
    chain_id: str
    related_state_items: List[Dict[str, Any]]
    events: List[Dict[str, Any]]


@dataclass
class WindowEventChains:
    """All event chains for one window."""
    window_id: str
    domain_name: str
    time_range: List[str]  # ["YYYY-MM-DD", "YYYY-MM-DD"]
    event_chains: List[EventChain]


class AppLogGenerator:
    """
    Generates concrete app logs from event chains.

    Uses AppRegistry to maintain stateful consistency across all apps.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.app_registry = AppRegistry()

    def generate_logs_for_window(
        self,
        window_event_chains: WindowEventChains,
        user_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate app logs for all events in a window.

        All events are now app logs (including LLM Chat which replaces dialogue).

        Args:
            window_event_chains: All event chains for this window
            user_profile: User profile for context

        Returns:
            Dict with app_logs (all digital footprints)
        """
        app_logs = []

        context = {
            "user_id": self.user_id,
            "domain": window_event_chains.domain_name,
            "window_id": window_event_chains.window_id,
            "user_profile": user_profile or {}
        }

        # Process each event chain
        for chain in window_event_chains.event_chains:
            chain_context = {
                **context,
                "chain_id": chain.chain_id,
                "related_state_items": chain.related_state_items
            }

            # Process each event in the chain
            # All events are app logs now (no more dialogue distinction)
            for event in chain.events:
                log_entry = self._generate_app_log(event, chain_context)
                app_logs.append(log_entry)

        return {
            "window_id": window_event_chains.window_id,
            "domain_name": window_event_chains.domain_name,
            "time_range": window_event_chains.time_range,
            "app_logs": app_logs
        }

    def _generate_app_log(
        self,
        event: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a concrete app log from an event."""
        app_name = event["app_name"]
        api_name = event["api_name"]
        timestamp = event["timestamp"]
        description = event["description"]

        # Call the app API through the registry
        log_entry = self.app_registry.call_api(
            app_name=app_name,
            api_name=api_name,
            user_id=self.user_id,
            timestamp=timestamp,
            description=description,
            context=context
        )

        # Convert to dict and add metadata
        return {
            "event_id": event["event_id"],
            "timestamp": log_entry.timestamp,
            "app_name": log_entry.app_name,
            "api_name": log_entry.api_name,
            "request": log_entry.request,
            "response": log_entry.response,
            "metadata": {
                **log_entry.metadata,
                "purpose": event.get("purpose", ""),
                "chain_id": context.get("chain_id", "")
            }
        }

    def generate_logs_for_all_windows(
        self,
        all_window_event_chains: List[WindowEventChains],
        user_profile: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate app logs for all windows.

        This processes windows in order to maintain state consistency
        across the entire timeline.
        """
        all_logs = []

        for window_chains in all_window_event_chains:
            window_logs = self.generate_logs_for_window(
                window_chains,
                user_profile=user_profile
            )
            all_logs.append(window_logs)

        return all_logs


def parse_event_chains_from_json(data: Dict[str, Any]) -> WindowEventChains:
    """
    Parse event chains from LLM JSON output.

    Expected format:
    {
      "window_id": "w1",
      "domain_name": "health",
      "time_range": ["2024-01-01", "2024-03-31"],
      "event_chains": [
        {
          "chain_id": "...",
          "related_state_items": [...],
          "events": [...]
        }
      ]
    }
    """
    chains = []
    for chain_data in data.get("event_chains", []):
        chain = EventChain(
            chain_id=chain_data["chain_id"],
            related_state_items=chain_data["related_state_items"],
            events=chain_data["events"]
        )
        chains.append(chain)

    return WindowEventChains(
        window_id=data["window_id"],
        domain_name=data["domain_name"],
        time_range=data["time_range"],
        event_chains=chains
    )


# Example usage function
def generate_app_logs_from_events_chain_json(
    events_chain_json: Dict[str, Any],
    user_id: str,
    user_profile: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate app logs from events chain JSON.

    Args:
        events_chain_json: Output from events_chain_generator
        user_id: User identifier
        user_profile: Optional user profile for context

    Returns:
        Dict with app_logs and dialogue_logs
    """
    # Parse event chains
    window_chains = parse_event_chains_from_json(events_chain_json)

    # Create log generator
    log_generator = AppLogGenerator(user_id)

    # Generate logs
    return log_generator.generate_logs_for_window(
        window_chains,
        user_profile=user_profile
    )
