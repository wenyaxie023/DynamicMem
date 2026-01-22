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
from .config import LLM_MAX_WORKERS
from .prompts import BASIC_JUDGE_PROMPT
@register_metric("llm_judge")
def llm_judge_metric(samples: List[Any]) -> Dict[str, List[float]]:
    gpt = LLMClient(
        provider="openai",
        model_name="gpt-5-mini",
        max_workers=LLM_MAX_WORKERS,
    )
    gemini = LLMClient(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        max_workers=LLM_MAX_WORKERS,
    )

    gpt_scores: List[float] = []
    gemini_scores: List[float] = []

    preds = [sample.prediction for sample in samples]
    refs = [sample.reference for sample in samples]
    querys = [sample.query for sample in samples]

    gpt_futures = []
    gemini_futures = []
    prompts: List[str] = []

    for query, pred, ref in zip(querys, preds, refs):
        prompt = BASIC_JUDGE_PROMPT.format(
            query=query,
            prediction=pred,
            reference=ref
        )
        prompts.append(prompt)
        gpt_futures.append(gpt.ask_async(prompt))
        gemini_futures.append(gemini.ask_async(prompt))

    gpt_results = gpt.collect(gpt_futures)
    for idx, out_gpt in enumerate(gpt_results):
        if isinstance(out_gpt, Exception):
            logger.warning(f"GPT judge failed at index {idx}: {out_gpt}")
            gpt_scores.append(0.0)
            continue
        try:
            gpt_scores.append(float(out_gpt["score"]))
        except Exception as e:
            logger.warning(f"GPT judge parse failed at index {idx}: {e}")
            gpt_scores.append(0.0)

    gemini_results = gemini.collect(gemini_futures)
    for idx, out_gem in enumerate(gemini_results):
        if isinstance(out_gem, Exception):
            logger.warning(f"Gemini judge failed at index {idx}: {out_gem}")
            gemini_scores.append(0.0)
            continue
        try:
            gemini_scores.append(float(out_gem["score"]))
        except Exception as e:
            logger.warning(f"Gemini judge parse failed at index {idx}: {e}")
            gemini_scores.append(0.0)

    return {
        "llm_gpt_score": gpt_scores,
        "llm_gemini_score": gemini_scores,
    }


class EvaluationManager:
    def __init__(self, selected_metrics: List[str]):
        self.active_metrics = selected_metrics or list(METRIC_REGISTRY.keys())

    def _update_samples(self, samples, key, scores):
        for i, score in enumerate(scores):
            if samples[i].scores is None:
                samples[i].scores = {}
            samples[i].scores[key] = round(float(score), 4)

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
