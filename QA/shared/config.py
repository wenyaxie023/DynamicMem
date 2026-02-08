from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    root: Path = Path(__file__).resolve().parents[2]
    user_id: int = 10
    provider: str = "gemini"
    model_name: str = "gemini-3-flash-preview"
    max_workers: int = 1
    retry_times: int = 1
    flush_every: int = 1
    sample_per_group: int = 5
    sample_seed: int = 42

    tasks_filename: str = "tasks.json"

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / f"user{self.user_id}"

    @property
    def tasks_path(self) -> Path:
        return self.data_dir / self.tasks_filename

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw.json"

    @property
    def sampled_tasks_path(self) -> Path:
        return self.data_dir / "tasks_sampled.json"

    @property
    def qa_output_path(self) -> Path:
        return self.data_dir / "qa.json"

    @property
    def qa_context_path(self) -> Path:
        return self.data_dir / "qa_context.json"

    @property
    def run_state_path(self) -> Path:
        return self.data_dir / "run_state.json"

    @property
    def judge_refine_output_path(self) -> Path:
        return self.data_dir / "qa_refine.json"

    @property
    def judge_state_path(self) -> Path:
        return self.data_dir / "run_state_judge.json"

    @property
    def doublecheck_output_path(self) -> Path:
        return self.data_dir / "qa_doublecheck.json"

    @property
    def doublecheck_state_path(self) -> Path:
        return self.data_dir / "run_state_doublecheck.json"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "qa_generation.log"

    def with_updates(self, **kwargs: object) -> "GenerationConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ContextConfig:
    root: Path = Path(__file__).resolve().parents[2]
    user_id: int = 10

    schema_filename: str = "schema.json"
    app_logs_final_filename: str = "app_logs_final.json"
    states_filename: str = "states.json"
    events_filename: str = "events.json"
    logs_filename: str = "logs.json"
    raw_filename: str = "raw.json"
    tasks_filename: str = "tasks.json"

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / f"user{self.user_id}"

    @property
    def schema_path(self) -> Path:
        return self.data_dir / self.schema_filename

    @property
    def app_logs_final_path(self) -> Path:
        return self.data_dir / self.app_logs_final_filename

    @property
    def states_path(self) -> Path:
        return self.data_dir / self.states_filename

    @property
    def events_path(self) -> Path:
        return self.data_dir / self.events_filename

    @property
    def logs_path(self) -> Path:
        return self.data_dir / self.logs_filename

    @property
    def raw_path(self) -> Path:
        return self.data_dir / self.raw_filename

    @property
    def tasks_path(self) -> Path:
        return self.data_dir / self.tasks_filename

    def task_path_for(self, tag: str) -> Path:
        return self.data_dir / f"task_{tag}.json"

    def task_final_path_for(self, tag: str) -> Path:
        return self.data_dir / f"task_final_{tag}.json"
