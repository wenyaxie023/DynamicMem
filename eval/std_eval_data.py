from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import json
import logging

from .logger import setup_logger

from .config import (
    LOG_DIR,
)

logger = setup_logger("evaldata", LOG_DIR)


@dataclass
class EvalSample:
    """single evaluation sample"""
    query: str
    reference: str
    prediction: str
    id: Optional[str] = None
    metadata: Optional[Dict] = None
    scores: Optional[Dict[str, float]] = None
    # Evidence fields for retrieval evaluation
    reference_app_logs: Optional[List[Dict]] = None  # golden evidence
    evidence_prediction: Optional[List[Dict]] = None  # predicted evidence


@dataclass
class EvalResult:
    """evaluation result for an experiment"""
    experiment_name: str
    samples: List[EvalSample]
    summary: Dict[str, float]

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=ensure_ascii, indent=indent)


def load_data_preview(json_path):
    logger.info(f"Loading data preview from: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"Total samples loaded: {len(raw_data)}")

        samples = []
        for i, item in enumerate(raw_data):
            # Extract evidence_prediction from metadata if present
            metadata = item.get("metadata", {})
            evidence_prediction = metadata.pop("evidence_prediction", None) if metadata else None

            sample = EvalSample(
                id=item.get("id"),
                query=item["query"],
                reference=item["reference"],
                prediction=item.get("prediction", ""),
                metadata=metadata,
                scores={},
                reference_app_logs=item.get("reference_app_logs"),
                evidence_prediction=evidence_prediction,
            )
            samples.append(sample)

        return samples

    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        return []

    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON from file: {json_path} | error={e}"
        )
        return []

    except Exception as e:
        logger.exception(
            f"Unexpected error while loading data from: {json_path}"
        )
        raise
