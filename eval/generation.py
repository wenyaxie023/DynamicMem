import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Literal

from .client import LLMClient
from .std_eval_data import EvalSample, load_data_preview
from .prompts import QUERY_PROMPT
from .logger import setup_logger
from ..config import (
    BG_PATH,
    QA_PATH,
    GEN_OUTPUT_PATH,
    LOG_DIR,
    GEN_PROVIDER,
    GEN_MODEL_NAME,
    HISTORY_DIR,
    LLM_MAX_WORKERS,
)


logger = setup_logger("generate", LOG_DIR)


GenerationMode = Literal["auto", "overwrite"]
SaveMode = Literal["real_time", "final", "none"]
BackupMode = Literal["none", "per_run"]


def build_generation_prompt(bg: str, query: str) -> str:
    return QUERY_PROMPT.format(bg=bg, query=query)


def load_background(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Background file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def save_generation(samples: list[EvalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(s) for s in samples],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"[GEN][SAVE] current QA saved to {path}")


def maybe_save(
    *,
    samples: list[EvalSample],
    path: Path | None,
    mode: SaveMode,
    when: Literal["step", "final"],
) -> None:
    if mode == "none" or path is None:
        return

    if mode == "real_time" and when == "step":
        save_generation(samples, path)

    if mode == "final" and when == "final":
        save_generation(samples, path)


def backup_generation(
    *,
    samples: list[EvalSample],
    history_dir: Path,
    base_name: str,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_dir.mkdir(parents=True, exist_ok=True)
    backup_path = history_dir / f"{base_name}_{ts}.json"

    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(s) for s in samples],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"[GEN][BACKUP] history QA archived to {backup_path}")


def run_generation(
    samples: list[EvalSample],
    bg: str,
    llm: LLMClient,
    *,
    mode: GenerationMode = "auto",
    save_path: Path | None = None,
    save_mode: SaveMode = "final",
) -> None:
    total = len(samples)
    tasks: list[tuple[int, EvalSample, str]] = []

    for idx, sample in enumerate(samples, start=1):
        has_pred = bool(sample.prediction and sample.prediction.strip())

        if mode == "auto" and has_pred:
            logger.info(f"[GEN][SKIP][{idx}/{total}] id={sample.id}")
            continue

        logger.info(f"[GEN][RUN][{idx}/{total}] id={sample.id}")

        prompt = build_generation_prompt(bg=bg, query=sample.query)
        tasks.append((idx, sample, prompt))

    if not tasks:
        return

    prompts = [prompt for _, _, prompt in tasks]
    futures = llm.ask_many_async(prompts)
    results = llm.collect(futures)

    for task_idx, out in enumerate(results):
        idx, sample, _ = tasks[task_idx]
        try:
            if isinstance(out, Exception):
                raise out
            if not isinstance(out, dict) or "answer" not in out:
                raise ValueError(f"Bad output: {out}")

            sample.prediction = out["answer"]
            sample.metadata = sample.metadata or {}
            sample.metadata.update({
                "gen_model": llm.model_name,
                "gen_time": datetime.now().isoformat(timespec="seconds"),
                "gen_mode": mode,
            })

            if "reason" in out:
                sample.metadata["gen_reason"] = out["reason"]

            maybe_save(
                samples=samples,
                path=save_path,
                mode=save_mode,
                when="step",
            )
        except Exception:
            logger.exception(f"[GEN][FAILED] id={sample.id}")
            raise


def run(
    *,
    write_mode: GenerationMode = "auto",
    save_mode: SaveMode = "final",
    backup_mode: BackupMode = "none",
) -> None:
    logger.info("🚀 Starting GENERATION stage")

    # 1. Load background
    bg = load_background(BG_PATH)
    logger.info("✅ Background loaded")

    # 2. Load QA samples
    samples = load_data_preview(QA_PATH)
    logger.info(f"✅ Loaded {len(samples)} QA samples")

    # 3. Init LLM
    gen_llm = LLMClient(
        provider=GEN_PROVIDER,
        model_name=GEN_MODEL_NAME,
        max_workers=LLM_MAX_WORKERS,
    )

    # 4. Run generation
    run_generation(
        samples,
        bg,
        gen_llm,
        mode=write_mode,
        save_path=GEN_OUTPUT_PATH,
        save_mode=save_mode,
    )

    # 5. Save current QA (final)
    maybe_save(
        samples=samples,
        path=GEN_OUTPUT_PATH,
        mode=save_mode,
        when="final",
    )

    # 6. Backup history QA
    if backup_mode == "per_run":
        backup_generation(
            samples=samples,
            history_dir=HISTORY_DIR,
            base_name=GEN_OUTPUT_PATH.stem,
        )

    logger.info("🎉 Generation finished successfully")


if __name__ == "__main__":
    run(
        write_mode="overwrite",
        save_mode="final",
        backup_mode="none",
    )
