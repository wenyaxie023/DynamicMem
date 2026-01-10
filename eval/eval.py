import json
import pprint
from pathlib import Path
import pandas as pd

from metrics import EvaluationManager
from std_eval_data import EvalSample, EvalResult
from logger import setup_logger

from config import (
    LOG_DIR,
    EXPERIMENT_NAME,
    EVAL_TABLE_PATH,
    EVAL_INPUT_PATH
)



logger = setup_logger("eval", LOG_DIR)


def load_generation(path: Path) -> list[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Generation artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [EvalSample(**x) for x in data]



def summarize(samples: list[EvalSample]) -> dict:
    acc = {}
    for s in samples:
        for k, v in s.scores.items():
            acc.setdefault(k, []).append(v)

    return {f"{k}_mean": sum(v) / len(v) for k, v in acc.items()}

def run(eval_path: Path = EVAL_INPUT_PATH):
    logger.info("📊 Starting EVALUATION stage")

    # 1. Load generated samples
    samples = load_generation(eval_path)
    logger.info(f"✅ Loaded {len(samples)} generated samples")

    # 2. Run evaluation
    manager = EvaluationManager(
        selected_metrics=[
            "exact_match",
            "rouge",
            "bert_score",
            "llm_judge",
        ]
    )

    evaluated_samples = manager.evaluate(samples)
    logger.info("✅ Metrics evaluation completed")

    # 3. Build result object
    result = EvalResult(
        experiment_name=EXPERIMENT_NAME,
        samples=evaluated_samples,
        summary=summarize(evaluated_samples)
    )

    # 4. Print summary
    print("📊 === SUMMARY ===")
    pprint.pprint(result.summary)

    # 5. Table view
    df = pd.DataFrame([
        {
            "id": s.id,
            "category": s.metadata.get("category"),
            **s.scores
        }
        for s in evaluated_samples
    ])

    print("\n📋 === PER-SAMPLE SCORES ===")
    print(df)
    df.to_csv(
        EVAL_TABLE_PATH,
        index=False,
        encoding="utf-8"
    )
    logger.info(f"🧾 Evaluation table saved to {EVAL_TABLE_PATH}")

    logger.info("🎉 Evaluation finished successfully")


if __name__ == "__main__":
    run()
