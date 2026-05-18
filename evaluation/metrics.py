import torch
import evaluate
import logging
from typing import List, Dict, Callable, Any

import sys
from pathlib import Path
# current_dir = Path(__file__).resolve().parent
# parent_dir = current_dir.parent
# sys.path.insert(0, str(parent_dir))


logger = logging.getLogger(__name__)


METRIC_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: str):
    def decorator(func: Callable):
        METRIC_REGISTRY[name] = func
        return func
    return decorator


@register_metric("exact_match")
def exact_match_metric(samples: List[Any]) -> List[float]:
    preds = [sample.prediction for sample in samples]
    refs = [sample.reference for sample in samples]
    return [1.0 if p == r else 0.0 for p, r in zip(preds, refs)]


@register_metric("evidence_recall")
def evidence_recall_metric(samples: List[Any]) -> Dict[str, List[float]]:
    """
    Calculate evidence recall: |predicted ∩ golden| / |golden|

    - reference_app_logs: golden evidence (list of dicts with 'app_log_id')
    - evidence_prediction: predicted evidence (list of dicts with 'app_log_id')
    """
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []

    for sample in samples:
        golden_logs = sample.reference_app_logs or []
        pred_logs = sample.evidence_prediction or []

        # Extract app_log_ids
        golden_ids = set(log.get("app_log_id") for log in golden_logs if log.get("app_log_id"))
        pred_ids = set(log.get("app_log_id") for log in pred_logs if log.get("app_log_id"))

        # Calculate metrics
        if len(golden_ids) == 0:
            recall = 1.0 if len(pred_ids) == 0 else 0.0
        else:
            recall = len(golden_ids & pred_ids) / len(golden_ids)

        if len(pred_ids) == 0:
            precision = 1.0 if len(golden_ids) == 0 else 0.0
        else:
            precision = len(golden_ids & pred_ids) / len(pred_ids)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "evidence_recall": recalls,
        "evidence_precision": precisions,
        "evidence_f1": f1s,
    }


@register_metric("rouge")
def rouge_metric(samples: List[Any]) -> Dict[str, List[float]]:
    rouge = evaluate.load("rouge")

    rouge1: List[float] = []
    rouge2: List[float] = []
    rougeL: List[float] = []

    for sample in samples:
        result = rouge.compute(
            predictions=[sample.prediction],
            references=[sample.reference],
            use_stemmer=False
        )
        rouge1.append(float(result["rouge1"]))
        rouge2.append(float(result["rouge2"]))
        rougeL.append(float(result["rougeL"]))

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


@register_metric("bert_score")
def bert_score_metric(samples: List[Any]) -> Dict[str, List[float]]:
    preds = [sample.prediction for sample in samples]
    refs = [sample.reference for sample in samples]

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type="bert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return {
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"]
    }


from .client import LLMClient
from . import config
from .prompts import BASIC_JUDGE_PROMPT


def _is_valid_judge_output(out: Any) -> bool:
    if isinstance(out, Exception) or not isinstance(out, dict):
        return False
    if "score" not in out:
        return False
    try:
        score = float(out["score"])
    except Exception:
        return False
    return 1.0 <= score <= 10.0


def _collect_with_retries(
    client: LLMClient,
    prompts: List[str],
    *,
    max_retries: int,
    provider_label: str,
) -> List[Any]:
    results: List[Any] = [None] * len(prompts)
    remaining = list(range(len(prompts)))
    attempt = 0
    while remaining:
        futures = [client.ask_async(prompts[i]) for i in remaining]
        batch = client.collect(futures)
        next_remaining = []
        for idx, out in zip(remaining, batch):
            if _is_valid_judge_output(out):
                results[idx] = out
            else:
                if attempt < max_retries:
                    next_remaining.append(idx)
                else:
                    results[idx] = out
        if next_remaining and attempt < max_retries:
            logger.warning(
                f"{provider_label} judge retrying {len(next_remaining)} items "
                f"(attempt {attempt + 1}/{max_retries})"
            )
        remaining = next_remaining
        attempt += 1
        if attempt > max_retries:
            break
    return results


@register_metric("llm_judge")
def llm_judge_metric(samples: List[Any]) -> Dict[str, List[float]]:
    gpt = None
    gemini = None
    providers = {p.lower() for p in config.LLM_JUDGE_PROVIDERS}
    if "gpt" in providers:
        gpt = LLMClient(
            provider=config.LLM_JUDGE_GPT_PROVIDER,
            model_name=config.LLM_JUDGE_GPT_MODEL,
            max_workers=config.LLM_MAX_WORKERS,
        )
    if "gemini" in providers:
        gemini = LLMClient(
            provider="gemini",
            model_name=config.LLM_JUDGE_GEMINI_MODEL,
            max_workers=config.LLM_MAX_WORKERS,
        )

    gpt_scores: List[float] = []
    gemini_scores: List[float] = []
    gpt_reasons: List[str] = []
    gemini_reasons: List[str] = []

    preds = [sample.prediction for sample in samples]
    refs = [sample.reference for sample in samples]
    querys = [sample.query for sample in samples]

    prompts: List[str] = []

    for query, pred, ref in zip(querys, preds, refs):
        prompt = BASIC_JUDGE_PROMPT.format(
            query=query,
            prediction=pred,
            reference=ref
        )
        prompts.append(prompt)

    max_retries = getattr(config, "LLM_JUDGE_MAX_RETRIES", 3)

    if gpt is not None:
        gpt_results = _collect_with_retries(
            gpt,
            prompts,
            max_retries=max_retries,
            provider_label="GPT",
        )
        for idx, out_gpt in enumerate(gpt_results):
            if isinstance(out_gpt, Exception):
                logger.warning(f"GPT judge failed at index {idx}: {out_gpt}")
                gpt_scores.append(0.0)
                gpt_reasons.append("")
                continue
            try:
                gpt_scores.append(float(out_gpt["score"]))
                gpt_reasons.append(str(out_gpt.get("reason", "")))
            except Exception as e:
                logger.warning(f"GPT judge parse failed at index {idx}: {e}")
                gpt_scores.append(0.0)
                gpt_reasons.append("")

    if gemini is not None:
        gemini_results = _collect_with_retries(
            gemini,
            prompts,
            max_retries=max_retries,
            provider_label="Gemini",
        )
        for idx, out_gem in enumerate(gemini_results):
            if isinstance(out_gem, Exception):
                logger.warning(f"Gemini judge failed at index {idx}: {out_gem}")
                gemini_scores.append(0.0)
                gemini_reasons.append("")
                continue
            try:
                gemini_scores.append(float(out_gem["score"]))
                gemini_reasons.append(str(out_gem.get("reason", "")))
            except Exception as e:
                logger.warning(f"Gemini judge parse failed at index {idx}: {e}")
                gemini_scores.append(0.0)
                gemini_reasons.append("")

    results: Dict[str, List[float]] = {}
    if gpt is not None:
        results["llm_gpt_score"] = gpt_scores
        results["llm_gpt_reason"] = gpt_reasons
    if gemini is not None:
        results["llm_gemini_score"] = gemini_scores
        results["llm_gemini_reason"] = gemini_reasons
    return results


class EvaluationManager:
    def __init__(self, selected_metrics: List[str]):
        self.active_metrics = selected_metrics or list(METRIC_REGISTRY.keys())

    def _update_samples(self, samples, key, scores):
        for i, score in enumerate(scores):
            if samples[i].scores is None:
                samples[i].scores = {}
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                samples[i].scores[key] = round(float(score), 4)
            else:
                samples[i].scores[key] = score

    def evaluate(self, samples: List[Any]) -> List[Any]:
        for metric_name in self.active_metrics:
            if metric_name not in METRIC_REGISTRY:
                raise ValueError(f"Metric '{metric_name}' is not registered.")

            logger.info(f"Computing metric: {metric_name}")
            metric_func = METRIC_REGISTRY[metric_name]
            result = metric_func(samples)

            if isinstance(result, dict):
                for key, values in result.items():
                    self._update_samples(samples, key, values)
            else:
                self._update_samples(samples, metric_name, result)

        return samples
