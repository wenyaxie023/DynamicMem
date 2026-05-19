#!/usr/bin/env python3

from baseline_prediction.HippoRAG2.generation_tce.online_tce import (
    HippoRAG2Runner,
    main,
    run_generation,
    run_online_generation,
)

__all__ = [
    "HippoRAG2Runner",
    "main",
    "run_generation",
    "run_online_generation",
]


if __name__ == "__main__":
    main()
