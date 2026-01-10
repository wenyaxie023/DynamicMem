from dataclasses import dataclass
from typing import Optional, List, Dict
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvalSample:
    """single evaluation sample"""
    query: str
    reference: str
    prediction: str
    id: Optional[str] = None
    metadata: Optional[Dict] = None
    scores: Optional[Dict[str, float]] = None


@dataclass
class EvalResult:
    """evaluation result for an experiment"""
    experiment_name: str
    samples: List[EvalSample]
    summary: Dict[str, float]


def load_data_preview(json_path):
    logger.info(f"Loading data preview from: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"Total samples loaded: {len(raw_data)}")

        samples = []
        for i, item in enumerate(raw_data):
            sample = EvalSample(
                id=item.get("id"),
                query=item["query"],
                reference=item["reference"],
                prediction=item.get("prediction", ""),
                metadata=item.get("metadata", {}),
                scores={}
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
