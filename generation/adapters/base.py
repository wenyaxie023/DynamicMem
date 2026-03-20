from __future__ import print_function

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class BaselineConcurrencyPolicy:
    checkpoint_parallelism: str
    within_checkpoint_parallelism: str


_DEFAULT_CONCURRENCY_POLICY = BaselineConcurrencyPolicy(
    checkpoint_parallelism="forbidden",
    within_checkpoint_parallelism="forbidden",
)


BASELINE_CONCURRENCY_POLICIES: Dict[str, BaselineConcurrencyPolicy] = {
    "rag": BaselineConcurrencyPolicy("allowed", "allowed"),
    "oracle": BaselineConcurrencyPolicy("allowed", "allowed"),
    "icl": BaselineConcurrencyPolicy("allowed", "allowed"),
    "hipporag": BaselineConcurrencyPolicy("allowed", "allowed"),
    "hipporag2": BaselineConcurrencyPolicy("allowed", "allowed"),
    "amem_baseline": BaselineConcurrencyPolicy("allowed", "allowed"),
    # Letta's agent-loop implementation mutates shared memory state while answering,
    # so both checkpoint-level and within-checkpoint parallelism are disabled.
    "letta": BaselineConcurrencyPolicy("forbidden", "forbidden"),
}


def get_baseline_concurrency_policy(baseline_name: str) -> BaselineConcurrencyPolicy:
    return BASELINE_CONCURRENCY_POLICIES.get(str(baseline_name or "").strip(), _DEFAULT_CONCURRENCY_POLICY)


@dataclass
class TceAdapterArgs:
    baseline: str
    benchmark: Path
    app_logs_path: Path
    output: Path
    max_visible_logs: Optional[int] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    llm_max_workers: int = 1
    resume: bool = False
    max_checkpoints: Optional[int] = None
    debug: bool = False
    debug_dir: Optional[Path] = None
    save_prompt_and_raw: bool = False
    enable_rq3_apply_service_qa: bool = False
    rq3_apply_fail_on_missing_pack: bool = False
    rq3_apply_save_prompt_and_raw: bool = True
    extras: Dict[str, str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}
