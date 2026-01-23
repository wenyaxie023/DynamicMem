import json
import pprint
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .metrics import EvaluationManager
from .std_eval_data import EvalSample, EvalResult
from .logger import setup_logger
from dotenv import load_dotenv

load_dotenv()

from .config import (
    LOG_DIR,
    EXPERIMENT_NAME,
    EVAL_TABLE_PATH,
    EVAL_INPUT_PATH,
    EVAL_OUTPUT_PATH,
)

logger = setup_logger("eval", LOG_DIR)

# Base path for generation results
GENERATION_BASE_PATH = Path(__file__).resolve().parent.parent / "generation"


def load_generation(path: Path) -> list[EvalSample]:
    """Load evaluation samples from a single file."""
    if not path.exists():
        raise FileNotFoundError(f"Generation artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Extract evidence_prediction from metadata if present
        metadata = item.get("metadata", {}) or {}
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


def discover_prediction_files(
    baselines: Optional[List[str]] = None,
    users: Optional[List[str]] = None,
    base_path: Path = GENERATION_BASE_PATH,
) -> List[dict]:
    """
    Discover prediction files from baselines.

    Path structure: <base_path>/<baseline_name>/results/<user>/prediction/*.json

    Returns a list of dicts with keys: baseline, user, file_path, file_name
    """
    discovered = []

    # If no baselines specified, discover all
    if baselines is None:
        baselines = [
            d.name for d in base_path.iterdir()
            if d.is_dir() and (d / "results").exists()
        ]

    for baseline in baselines:
        baseline_path = base_path / baseline / "results"
        if not baseline_path.exists():
            logger.warning(f"Baseline path not found: {baseline_path}")
            continue

        # Discover users
        user_dirs = [d for d in baseline_path.iterdir() if d.is_dir()]
        if users:
            user_dirs = [d for d in user_dirs if d.name in users]

        for user_dir in user_dirs:
            prediction_dir = user_dir / "prediction"
            if not prediction_dir.exists():
                continue

            for json_file in prediction_dir.glob("*.json"):
                discovered.append({
                    "baseline": baseline,
                    "user": user_dir.name,
                    "file_path": json_file,
                    "file_name": json_file.stem,
                })

    return discovered


def summarize(samples: list[EvalSample]) -> dict:
    """Compute mean of all scores."""
    acc = {}
    for s in samples:
        if s.scores:
            for k, v in s.scores.items():
                acc.setdefault(k, []).append(v)

    return {f"{k}_mean": sum(v) / len(v) for k, v in acc.items()}


def run_single(
    eval_path: Path,
    experiment_name: str = EXPERIMENT_NAME,
    selected_metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
) -> EvalResult:
    """Run evaluation on a single prediction file."""
    logger.info(f"Starting evaluation for: {eval_path}")

    # Default metrics
    if selected_metrics is None:
        selected_metrics = [
            "exact_match",
            "rouge",
            "bert_score",
            "llm_judge",
            "evidence_recall",
        ]

    # 1. Load generated samples
    samples = load_generation(eval_path)
    logger.info(f"Loaded {len(samples)} generated samples")

    # 2. Run evaluation
    manager = EvaluationManager(selected_metrics=selected_metrics)
    evaluated_samples = manager.evaluate(samples)
    logger.info("Metrics evaluation completed")

    # 3. Build result object
    result = EvalResult(
        experiment_name=experiment_name,
        samples=evaluated_samples,
        summary=summarize(evaluated_samples)
    )

    # 4. Print summary
    print(f"=== SUMMARY ({experiment_name}) ===")
    pprint.pprint(result.summary)

    # 5. Table view
    df = pd.DataFrame([
        {
            "id": s.id,
            "category": s.metadata.get("category") if s.metadata else None,
            **s.scores
        }
        for s in evaluated_samples
    ])

    print(f"\n=== PER-SAMPLE SCORES ({experiment_name}) ===")
    print(df)

    # Save outputs if paths provided
    if table_path:
        df.to_csv(table_path, index=False, encoding="utf-8")
        logger.info(f"Evaluation table saved to {table_path}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        logger.info(f"Evaluation result saved to {output_path}")

    return result


def run_batch(
    baselines: Optional[List[str]] = None,
    users: Optional[List[str]] = None,
    selected_metrics: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    base_path: Path = GENERATION_BASE_PATH,
) -> List[dict]:
    """
    Run evaluation on multiple baselines and users.

    Args:
        baselines: List of baseline names to evaluate. If None, discover all.
        users: List of user names to evaluate. If None, evaluate all users.
        selected_metrics: List of metric names to compute.
        output_dir: Directory to save results. If None, save next to prediction files.
        base_path: Base path for generation results.

    Returns:
        List of summary dicts with baseline, user, file_name, and metrics.
    """
    logger.info("Starting BATCH EVALUATION")

    # Default metrics
    if selected_metrics is None:
        selected_metrics = [
            "exact_match",
            "rouge",
            "bert_score",
            "llm_judge",
            "evidence_recall",
        ]

    # Discover prediction files
    prediction_files = discover_prediction_files(
        baselines=baselines,
        users=users,
        base_path=base_path,
    )

    if not prediction_files:
        logger.warning("No prediction files found!")
        return []

    logger.info(f"Found {len(prediction_files)} prediction files to evaluate")

    all_summaries = []

    for item in prediction_files:
        baseline = item["baseline"]
        user = item["user"]
        file_path = item["file_path"]
        file_name = item["file_name"]

        experiment_name = f"{baseline}/{user}/{file_name}"

        # Determine output paths
        if output_dir:
            eval_output_dir = output_dir / baseline / user
        else:
            eval_output_dir = file_path.parent.parent / "eval"

        eval_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = eval_output_dir / f"{file_name}_eval.json"
        table_path = eval_output_dir / f"{file_name}_eval.csv"

        try:
            result = run_single(
                eval_path=file_path,
                experiment_name=experiment_name,
                selected_metrics=selected_metrics,
                output_path=output_path,
                table_path=table_path,
            )

            summary = {
                "baseline": baseline,
                "user": user,
                "file_name": file_name,
                **result.summary,
            }
            all_summaries.append(summary)

        except Exception as e:
            logger.error(f"Failed to evaluate {experiment_name}: {e}")
            continue

    # Create overall summary DataFrame
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        print("\n=== OVERALL BATCH SUMMARY ===")
        print(summary_df.to_string())

        # Save overall summary
        if output_dir:
            summary_path = output_dir / "batch_summary.csv"
            summary_df.to_csv(summary_path, index=False, encoding="utf-8")
            logger.info(f"Batch summary saved to {summary_path}")

    logger.info("Batch evaluation finished")
    return all_summaries


def run(eval_path: Path = EVAL_INPUT_PATH):
    """Legacy run function for backward compatibility."""
    logger.info("Starting EVALUATION stage")

    result = run_single(
        eval_path=eval_path,
        experiment_name=EXPERIMENT_NAME,
        output_path=EVAL_OUTPUT_PATH,
        table_path=EVAL_TABLE_PATH,
    )

    logger.info("Evaluation finished successfully")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Evaluation mode: single file or batch across baselines",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path (for single mode)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help="List of baselines to evaluate (for batch mode)",
    )
    parser.add_argument(
        "--users",
        type=str,
        nargs="+",
        default=None,
        help="List of users to evaluate (for batch mode)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to compute",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.mode == "single":
        eval_path = Path(args.input) if args.input else EVAL_INPUT_PATH
        run_single(
            eval_path=eval_path,
            selected_metrics=args.metrics,
            output_path=EVAL_OUTPUT_PATH,
            table_path=EVAL_TABLE_PATH,
        )
    else:
        run_batch(
            baselines=args.baselines,
            users=args.users,
            selected_metrics=args.metrics,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
