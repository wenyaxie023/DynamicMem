
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunConfig:
    benchmark_path: Path
    app_logs_path: Optional[Path] = None
    prediction_path: Optional[Path] = None
    output_path: Optional[Path] = None


@dataclass(frozen=True)
class BenchmarkBundle:
    benchmark_path: Path
    user_id: Optional[str]
    total_checkpoints: int
    checkpoints: List[Dict[str, Any]]


@dataclass(frozen=True)
class PredictionBundle:
    prediction_path: Path
    predictions_raw: List[Dict[str, Any]]
    predictions_by_id: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class AppLogsBundle:
    app_logs_path: Path
    app_logs: List[Dict[str, Any]]
