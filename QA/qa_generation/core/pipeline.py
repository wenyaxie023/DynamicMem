from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from shared.config import GenerationConfig

from .generator import run_generation
from .sampler import run_sampling


def run_pipeline(
    *,
    config: GenerationConfig,
    qtypes: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
) -> int:
    run_sampling(config=config, qtypes=qtypes, categories=categories)
    run_generation(config=config)
    return 0
