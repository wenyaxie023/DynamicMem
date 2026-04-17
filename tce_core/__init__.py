"""Shared utilities for TCE evaluation.

Keep package import lightweight so callers that only need `pipeline` helpers do
not eagerly import the full evaluation stack.
"""

from .pipeline import (
    build_target_templates,
    drop_excluded_fields,
    normalize_app_logs,
    observed_logs_for_checkpoint,
    parse_ts,
    run_pipeline,
    to_log_text,
)


def evaluate_checkpoints(*args, **kwargs):
    from .evaluation import evaluate_checkpoints as _impl

    return _impl(*args, **kwargs)


def flatten_snapshot(*args, **kwargs):
    from .pipeline import flatten_snapshot as _impl

    return _impl(*args, **kwargs)


def mean_numeric_fields(*args, **kwargs):
    from .evaluation import mean_numeric_fields as _impl

    return _impl(*args, **kwargs)


def normalize_predictions(*args, **kwargs):
    from .evaluation import normalize_predictions as _impl

    return _impl(*args, **kwargs)


def score_snapshot(*args, **kwargs):
    from .evaluation import score_snapshot as _impl

    return _impl(*args, **kwargs)


def value_f1(*args, **kwargs):
    from .evaluation import value_f1 as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "evaluate_checkpoints",
    "flatten_snapshot",
    "mean_numeric_fields",
    "normalize_predictions",
    "score_snapshot",
    "value_f1",
    "run_pipeline",
    "parse_ts",
    "to_log_text",
    "drop_excluded_fields",
    "normalize_app_logs",
    "observed_logs_for_checkpoint",
    "build_target_templates",
]
