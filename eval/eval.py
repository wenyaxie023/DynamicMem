import json
import pprint
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from .metrics import EvaluationManager
from .std_eval_data import EvalSample, EvalResult
from .logger import setup_logger

from . import config
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

    def _parse_jsonish(value):
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return value
            if "\n<->\n" in text:
                parts = [p.strip() for p in text.split("\n<->\n") if p.strip()]
                parsed_parts = []
                for part in parts:
                    if part.startswith("{") or part.startswith("["):
                        try:
                            parsed_parts.append(json.loads(part))
                            continue
                        except Exception:
                            pass
                    parsed_parts.append(part)
                return parsed_parts
            if text.startswith("{") or text.startswith("["):
                try:
                    return json.loads(text)
                except Exception:
                    return value
        return value

    def _normalize_evidence_prediction(evidence):
        if not isinstance(evidence, list):
            return evidence
        normalized = []
        for item in evidence:
            if isinstance(item, dict) and "supporting_content" in item:
                item = {**item, "supporting_content": _parse_jsonish(item["supporting_content"])}
            normalized.append(item)
        return normalized

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Extract evidence_prediction from metadata if present
        metadata = item.get("metadata", {}) or {}
        evidence_prediction = metadata.pop("evidence_prediction", None) if metadata else None
        if metadata and "context" in metadata:
            metadata["context"] = _parse_jsonish(metadata["context"])
        evidence_prediction = _normalize_evidence_prediction(evidence_prediction)

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
    rag_topk: Optional[List[int]] = None,
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

    baseline_allowed_files = {
        "oracle": {"oracle_results.json"},
        "rag": {"rag_results_top5.json", "rag_results_top10.json", "rag_results_top20.json"},
        "MemoryOS": {"memoryos_results.json"},
        "HippoRAG2": {"hipporag2_results_top5.json", "hipporag2_results_top10.json", "hipporag2_results_top20.json"},
    }

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

            if baseline == "rag":
                if rag_topk:
                    allowed_files = {f"rag_results_top{k}.json" for k in rag_topk}
                else:
                    allowed_files = baseline_allowed_files["rag"]
            else:
                allowed_files = baseline_allowed_files.get(
                    baseline,
                    {f"{baseline}_results.json"},
                )
            for json_file in prediction_dir.glob("*.json"):
                if json_file.name not in allowed_files:
                    continue
                discovered.append({
                    "baseline": baseline,
                    "user": user_dir.name,
                    "file_path": json_file,
                    "file_name": json_file.stem,
                })

    return discovered


def load_eval_result(path: Path) -> EvalResult:
    """Load an evaluation result JSON (with scores) from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Eval result not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        sample = EvalSample(
            id=item.get("id"),
            query=item["query"],
            reference=item["reference"],
            prediction=item.get("prediction", ""),
            metadata=item.get("metadata"),
            scores=item.get("scores") or {},
            reference_app_logs=item.get("reference_app_logs"),
            evidence_prediction=item.get("evidence_prediction"),
        )
        samples.append(sample)

    return EvalResult(
        experiment_name=data.get("experiment_name", ""),
        samples=samples,
        summary=data.get("summary", {}),
    )


def _llm_score_failed(scores: Dict[str, Any], prefix: str) -> bool:
    score_key = f"llm_{prefix}_score"
    reason_key = f"llm_{prefix}_reason"
    if score_key not in scores:
        return False
    reason = (scores.get(reason_key) or "").strip()
    try:
        score = float(scores.get(score_key, 0.0))
    except Exception:
        return True
    return score == 0.0 and reason == ""


def find_failed_llm_indices(samples: List[EvalSample]) -> List[int]:
    failed_indices = []
    for i, sample in enumerate(samples):
        scores = sample.scores or {}
        if _llm_score_failed(scores, "gpt") or _llm_score_failed(scores, "gemini"):
            failed_indices.append(i)
    return failed_indices


def summarize(samples: list[EvalSample]) -> dict:
    """Compute mean of all scores."""
    # Skip non-numeric score keys (e.g., reason, criteria)
    skip_keywords = ("reason", "criteria")
    acc = {}
    for s in samples:
        if s.scores:
            for k, v in s.scores.items():
                if any(kw in k.lower() for kw in skip_keywords):
                    continue
                if isinstance(v, str):
                    v = float(v)
                acc.setdefault(k, []).append(v)

    return {f"{k}_mean": sum(v) / len(v) for k, v in acc.items()}


def run_single(
    eval_path: Path,
    experiment_name: str = EXPERIMENT_NAME,
    selected_metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
    rerun_failed_llm: bool = False,
    existing_eval_path: Optional[Path] = None,
    enable_partial: bool = False,
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
    
    ## partial re-evaluation
    if enable_partial:
        eval_source_path = existing_eval_path or output_path
        if eval_source_path and eval_source_path.exists():
            logger.info(f"Partial re-eval enabled; loading existing eval: {eval_source_path}")
            existing = load_eval_result(eval_source_path)
            samples = existing.samples
            refreshed = load_generation(eval_path)

            # Use 1-based indices to match "idx 151-180" in the request
            start_idx = 151
            end_idx = 180
            indices = [
                i for i in range(min(len(samples), len(refreshed)))
                if start_idx <= (i + 1) <= end_idx
            ]

            if not indices:
                logger.info("No samples in idx 151-180 range; returning existing results.")
                result = EvalResult(
                    experiment_name=existing.experiment_name or experiment_name,
                    samples=samples,
                    summary=summarize(samples),
                )
            else:
                logger.info(f"Re-evaluating idx {start_idx}-{end_idx} ({len(indices)} items).")
                # Replace samples with refreshed predictions for the targeted indices
                for i in indices:
                    updated = refreshed[i]
                    samples[i].query = updated.query
                    samples[i].reference = updated.reference
                    samples[i].prediction = updated.prediction
                    samples[i].metadata = updated.metadata
                    samples[i].reference_app_logs = updated.reference_app_logs
                    samples[i].evidence_prediction = updated.evidence_prediction
                    samples[i].scores = {}

                subset = [samples[i] for i in indices]
                manager = EvaluationManager(selected_metrics=selected_metrics)
                evaluated_subset = manager.evaluate(subset)
                for idx, evaluated in zip(indices, evaluated_subset):
                    samples[idx].scores = evaluated.scores

                result = EvalResult(
                    experiment_name=existing.experiment_name or experiment_name,
                    samples=samples,
                    summary=summarize(samples),
                )

            df = pd.DataFrame([
                {
                    "id": s.id,
                    "category": s.metadata.get("category") if s.metadata else None,
                    **(s.scores or {})
                }
                for s in samples
            ])

            if table_path:
                df.to_csv(table_path, index=False, encoding="utf-8")
                logger.info(f"Evaluation table saved to {table_path}")

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.to_json())
                logger.info(f"Evaluation result saved to {output_path}")

            return result
        logger.warning("enable_partial requested but existing eval file not found; running full eval.")

    if rerun_failed_llm:
        eval_source_path = existing_eval_path or output_path
        if eval_source_path and eval_source_path.exists():
            logger.info(f"Re-running failed LLM judge items from: {eval_source_path}")
            existing = load_eval_result(eval_source_path)
            samples = existing.samples
            failed_indices = find_failed_llm_indices(samples)
            if not failed_indices:
                logger.info("No failed LLM-judge items found; skipping re-run.")
                result = EvalResult(
                    experiment_name=existing.experiment_name or experiment_name,
                    samples=samples,
                    summary=summarize(samples),
                )
            else:
                logger.info(f"Re-evaluating {len(failed_indices)} failed LLM-judge items.")
                subset = [samples[i] for i in failed_indices]
                manager = EvaluationManager(selected_metrics=["llm_judge"])
                manager.evaluate(subset)
                result = EvalResult(
                    experiment_name=existing.experiment_name or experiment_name,
                    samples=samples,
                    summary=summarize(samples),
                )

            df = pd.DataFrame([
                {
                    "id": s.id,
                    "category": s.metadata.get("category") if s.metadata else None,
                    **(s.scores or {})
                }
                for s in samples
            ])

            if table_path:
                df.to_csv(table_path, index=False, encoding="utf-8")
                logger.info(f"Evaluation table saved to {table_path}")

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.to_json())
                logger.info(f"Evaluation result saved to {output_path}")

            return result
        logger.warning("rerun_failed_llm requested but existing eval file not found; running full eval.")

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
    rerun_failed_llm: bool = False,
    rag_topk: Optional[List[int]] = None,
    enable_partial: bool = False,
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
        rag_topk=rag_topk,
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

        # import pdb; pdb.set_trace()

        try:
            result = run_single(
                eval_path=file_path,
                experiment_name=experiment_name,
                selected_metrics=selected_metrics,
                output_path=output_path,
                table_path=table_path,
                rerun_failed_llm=rerun_failed_llm,
                existing_eval_path=output_path,
                enable_partial=enable_partial,
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
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider for GPT judge: openai or azure",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name for GPT judge",
    )
    parser.add_argument(
        "--rerun-llm-failed",
        action="store_true",
        help="Re-run LLM judge only for items with empty reason and score 0.0",
    )
    parser.add_argument(
        "--existing-eval",
        type=str,
        default=None,
        help="Existing eval JSON to patch when re-running failed LLM judge items",
    )
    parser.add_argument(
        "--rag-topk",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate rag_results_topK.json for the given K values",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Re-evaluate only idx 151-180 and load existing eval for others",
    )

    args = parser.parse_args()

    if args.llm_provider:
        config.LLM_JUDGE_GPT_PROVIDER = args.llm_provider
        if "gpt" not in [p.lower() for p in config.LLM_JUDGE_PROVIDERS]:
            config.LLM_JUDGE_PROVIDERS = ["gpt"] + list(config.LLM_JUDGE_PROVIDERS)
    if args.llm_model:
        config.LLM_JUDGE_GPT_MODEL = args.llm_model
        if "gpt" not in [p.lower() for p in config.LLM_JUDGE_PROVIDERS]:
            config.LLM_JUDGE_PROVIDERS = ["gpt"] + list(config.LLM_JUDGE_PROVIDERS)

    if args.mode == "single":
        eval_path = Path(args.input) if args.input else EVAL_INPUT_PATH
        run_single(
            eval_path=eval_path,
            selected_metrics=args.metrics,
            output_path=EVAL_OUTPUT_PATH,
            table_path=EVAL_TABLE_PATH,
            rerun_failed_llm=args.rerun_llm_failed,
            existing_eval_path=Path(args.existing_eval) if args.existing_eval else None,
            enable_partial=args.enable,
        )
    else:
        run_batch(
            baselines=args.baselines,
            users=args.users,
            selected_metrics=args.metrics,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            rerun_failed_llm=args.rerun_llm_failed,
            rag_topk=args.rag_topk,
            enable_partial=args.enable,
        )
