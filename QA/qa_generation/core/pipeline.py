from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from shared.config import GenerationConfig

from .context_cache_builder import run_build_qa_context
from .doublecheck_generator import run_doublecheck
from .generator import run_generation
from .judge_generator import run_judge_refine
from .sampler import run_sampling
from qa_generation.tools.export_checker_csv import export_checker_csv


def run_pipeline(
    *,
    config: GenerationConfig,
    qtypes: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
) -> int:
    run_sampling(config=config, qtypes=qtypes, categories=categories)
    run_generation(config=config)
    run_build_qa_context(config=config)
    run_judge_refine(config=config)
    run_doublecheck(config=config)
    export_checker_csv(config=config)
    return 0
