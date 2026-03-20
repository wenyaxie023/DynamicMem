"""Shared retrieval-query builder for retrieval-based TCE baselines."""

from typing import Any, Dict, List

from .task_spec import build_prediction_task_from_checkpoint

def build_retrieval_query(checkpoint: Dict[str, Any], target_keys: List[str]) -> str:
    return build_prediction_task_from_checkpoint(checkpoint, target_keys)
