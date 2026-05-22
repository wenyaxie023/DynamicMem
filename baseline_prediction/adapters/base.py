from __future__ import print_function

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


SHARED_PIPELINE_SNAPSHOT_ROUTE = "shared_pipeline_snapshot"
AGENT_LOOP_ROUTE = "agent_loop"


@dataclass(frozen=True)
class BaselineConcurrencyPolicy:
    checkpoint_parallelism: str
    within_checkpoint_parallelism: str


_DEFAULT_CONCURRENCY_POLICY = BaselineConcurrencyPolicy(
    checkpoint_parallelism="forbidden",
    within_checkpoint_parallelism="forbidden",
)

SHARED_PIPELINE_SNAPSHOT_BASELINES = {
    "rag",
    "oracle",
    "hipporag2",
    "amem",
    "memoryos",
    "simplemem",
}


BASELINE_CONCURRENCY_POLICIES: Dict[str, BaselineConcurrencyPolicy] = {
    "rag": BaselineConcurrencyPolicy("allowed", "allowed"),
    "oracle": BaselineConcurrencyPolicy("allowed", "allowed"),
    "hipporag2": BaselineConcurrencyPolicy("allowed", "allowed"),
    "amem": BaselineConcurrencyPolicy("forbidden", "allowed"),
    "memoryos": BaselineConcurrencyPolicy("allowed", "allowed"),
    "simplemem": BaselineConcurrencyPolicy("forbidden", "forbidden"),
}


def get_baseline_concurrency_policy(baseline_name: str) -> BaselineConcurrencyPolicy:
    return BASELINE_CONCURRENCY_POLICIES.get(str(baseline_name or "").strip(), _DEFAULT_CONCURRENCY_POLICY)


def get_baseline_route_class(baseline_name: str) -> str:
    key = str(baseline_name or "").strip().lower()
    if key in SHARED_PIPELINE_SNAPSHOT_BASELINES:
        return SHARED_PIPELINE_SNAPSHOT_ROUTE
    return SHARED_PIPELINE_SNAPSHOT_ROUTE


@dataclass
class TceAdapterArgs:
    baseline: str
    user_id: Optional[str]
    benchmark: Path
    app_logs_path: Path
    output: Path
    max_visible_logs: Optional[int] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    llm_max_workers: int = 1
    llm_temperature: Optional[float] = 0.0
    llm_top_p: Optional[float] = 1.0
    llm_top_k: Optional[int] = None
    resume: bool = False
    allow_destructive_rebuild: bool = False
    max_checkpoints: Optional[int] = None
    debug: bool = False
    debug_dir: Optional[Path] = None
    save_prompt_and_raw: bool = False
    enable_change_reasoning: bool = False
    enable_rq3_apply_service_qa: bool = False
    rq3_apply_save_prompt_and_raw: bool = True
    checkpoint_workers: int = 1
    within_checkpoint_workers: int = 1
    save_every_generation_keys: int = 1
    retriever_provider: str = "openai"
    retriever_model: str = "text-embedding-3-large"
    retriever_batch_size: int = 64
    retrieval_top_k: int = 5
    rq3_apply_retrieval_top_k: Optional[int] = None
    enable_final_qa: bool = False
    final_qa_path: Optional[str] = None
    final_qa_output_path: Optional[str] = None
    final_qa_retrieval_top_k: Optional[int] = None
    final_qa_save_prompt_and_raw: bool = False
    extras: Dict[str, str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}
