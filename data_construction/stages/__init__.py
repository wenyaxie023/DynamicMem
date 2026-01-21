"""
Stages for the batch generation pipeline.

Stage 1: Dynamic Profile Generation
    - Generate raw dynamic profiles per domain
    - In-domain conflict resolution (rule 1-5 cascade)
    - Cross-domain key alignment
    - Cross-domain conflict resolution

Stage 2: Events Chain Generation
    - Generate events chains for each domain, window by window (w0 → w4)

Stage 3: App Logs Generation
    - Merge all domain events chains
    - Sort events chronologically
    - Convert each event to structured app log via LLM
"""

from .stage1_dynamic_profile import DynamicProfileStage
from .stage2_events_chain import EventsChainStage
from .stage3_app_logs import AppLogsStage

__all__ = [
    "DynamicProfileStage",
    "EventsChainStage",
    "AppLogsStage",
]
