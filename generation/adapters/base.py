from __future__ import print_function

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class DspAdapterArgs:
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
    extras: Dict[str, str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}
