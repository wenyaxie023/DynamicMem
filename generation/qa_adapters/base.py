from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class QaAdapterArgs:
    baseline: str
    user_id: str
    input_root_dir: Optional[Path]
    output_root_dir: Optional[Path]
    qa_dir: Optional[Path]
    output_path: Optional[Path]

    llm_provider: Optional[str]
    llm_model: Optional[str]
    llm_max_workers: Optional[int]

    command: str
    subcommand: Optional[str]
    retry_times: Optional[int]
    flush_every: Optional[int]
    sample_per_group: Optional[int]
    sample_seed: Optional[int]
    qtypes: Optional[str]
    categories: Optional[str]
    fix_links_inplace: bool

    run_record_path: Path
    experiment_name: Optional[str]
    extras: Dict[str, str]
