import json
from pathlib import Path
from typing import Any, Dict, List

from tce_core.evaluation import normalize_predictions

from .contracts import AppLogsBundle, BenchmarkBundle, PredictionBundle


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark(path: Path) -> BenchmarkBundle:
    raw = _load_json(Path(path))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid benchmark format: expected object, got {type(raw).__name__}")

    checkpoints = raw.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("Invalid benchmark format: top-level 'checkpoints' must be a list")

    total_checkpoints = raw.get("total_checkpoints")
    if not isinstance(total_checkpoints, int):
        total_checkpoints = len(checkpoints)

    user_id = raw.get("user_id")
    if user_id is not None:
        user_id = str(user_id)

    return BenchmarkBundle(
        benchmark_path=Path(path),
        user_id=user_id,
        total_checkpoints=total_checkpoints,
        checkpoints=[x for x in checkpoints if isinstance(x, dict)],
    )


def load_app_logs(path: Path) -> AppLogsBundle:
    raw = _load_json(Path(path))
    app_logs: List[Dict[str, Any]]

    if isinstance(raw, list):
        app_logs = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        logs = raw.get("app_logs", [])
        if not isinstance(logs, list):
            raise ValueError("Invalid app logs format: 'app_logs' must be a list")
        app_logs = [x for x in logs if isinstance(x, dict)]
    else:
        raise ValueError(f"Invalid app logs format: expected list or object, got {type(raw).__name__}")

    return AppLogsBundle(app_logs_path=Path(path), app_logs=app_logs)


def load_prediction(path: Path) -> PredictionBundle:
    raw = _load_json(Path(path))
    predictions_by_id = normalize_predictions(raw)

    if isinstance(raw, dict) and "predictions" in raw:
        pred_raw = raw.get("predictions", [])
    elif isinstance(raw, list):
        pred_raw = raw
    else:
        pred_raw = []

    pred_raw = [x for x in pred_raw if isinstance(x, dict)]

    return PredictionBundle(
        prediction_path=Path(path),
        predictions_raw=pred_raw,
        predictions_by_id=predictions_by_id,
    )
