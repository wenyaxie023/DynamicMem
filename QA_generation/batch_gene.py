import json
import logging
import random
from typing import List, Dict, Any

from client import LLMClient
from context_fetch import fetch_context
from config import BG_PATH, REAL_ATOMS_PATH, LLM_MAX_WORKERS, LOG_DIR

from test import (
    QUESTION1_TEMPLATE,
    BIND_PROMPT_TEMPLATE,
)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("batch_gene")
    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_DIR / "batch_gene.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def pick_offset_atoms(atoms: List[Dict[str, Any]], offset: int = 5) -> tuple[Dict, Dict]:
    if len(atoms) <= offset:
        raise ValueError(f"Need at least {offset + 1} atoms to pick offset pair.")
    start_idx = random.randrange(0, len(atoms) - offset)
    return atoms[start_idx], atoms[start_idx + offset]


def extract_qa(llm_response: Any) -> Dict[str, str]:
    """
    LLM response is a JSON array.
    We extract question + answer from the FIRST element.
    """
    if isinstance(llm_response, Exception):
        raise llm_response

    if isinstance(llm_response, str):
        try:
            parsed = json.loads(llm_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON response: {llm_response}") from exc
    else:
        parsed = llm_response

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Expected non-empty JSON array, got: {type(parsed)}")

    item = parsed[0]
    if not isinstance(item, dict) or "question" not in item or "answer" not in item:
        raise ValueError(f"Invalid QA item: {item}")
    return {
        "question": item["question"],
        "answer": item["answer"],
    }


def generate_batch_questions(
    output_path: str,
    seed: int | None = None,
):
    logger = _get_logger()
    if seed is not None:
        random.seed(seed)

    # ---------- load data ----------
    with open(REAL_ATOMS_PATH, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    with open(BG_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    llm = LLMClient(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        max_workers=LLM_MAX_WORKERS,
    )

    results = []

    # ---------- 1–10: random single-atom questions ----------
    sample_count = min(10, len(atoms))
    sampled_atoms = random.sample(atoms, sample_count)
    prompts = []

    for atom in sampled_atoms:
        context = fetch_context(atom=atom, schema=schema)
        prompt = QUESTION1_TEMPLATE.format(
            ATOM_JSON=json.dumps(atom, ensure_ascii=True),
            CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
        )
        prompts.append(prompt)

    logger.info("Dispatching %s single-atom prompts.", len(prompts))
    responses = llm.ask_many(prompts)
    for atom, resp in zip(sampled_atoms, responses):
        try:
            qa = extract_qa(resp)
        except Exception:
            logger.exception("Single-atom QA failed. atom_id=%s", atom.get("atom_id"))
            raise
        results.append(qa)

    # ---------- 11: bound question ----------
    atom1, atom2 = pick_offset_atoms(atoms, offset=5)

    context1 = fetch_context(atom=atom1, schema=schema)
    context2 = fetch_context(atom=atom2, schema=schema)

    # generate base questions
    prompt1 = QUESTION1_TEMPLATE.format(
        ATOM_JSON=json.dumps(atom1, ensure_ascii=True),
        CONTEXT_TEXT=json.dumps(context1, ensure_ascii=True, indent=2),
    )
    prompt2 = QUESTION1_TEMPLATE.format(
        ATOM_JSON=json.dumps(atom2, ensure_ascii=True),
        CONTEXT_TEXT=json.dumps(context2, ensure_ascii=True, indent=2),
    )

    logger.info("Dispatching 2 base prompts for bind question.")
    base_results = llm.ask_many([prompt1, prompt2])
    if len(base_results) != 2:
        raise RuntimeError("Expected 2 base prompt responses.")
    r1, r2 = base_results
    if isinstance(r1, Exception):
        logger.exception("Base prompt 1 failed. atom_id=%s", atom1.get("atom_id"))
        raise r1
    if isinstance(r2, Exception):
        logger.exception("Base prompt 2 failed. atom_id=%s", atom2.get("atom_id"))
        raise r2

    # bind
    bind_prompt = BIND_PROMPT_TEMPLATE.format(
        QUESTION1_JSON=r1,
        QUESTION2_JSON=r2,
        CONTEXT_TEXT=json.dumps(
            {
                "context_1": context1,
                "context_2": context2,
            },
            ensure_ascii=True,
            indent=2,
        ),
    )

    bind_resp = llm.ask(bind_prompt)
    try:
        bind_qa = extract_qa(bind_resp)
    except Exception:
        logger.exception(
            "Bind QA failed. atom1_id=%s atom2_id=%s",
            atom1.get("atom_id"),
            atom2.get("atom_id"),
        )
        raise
    results.append(bind_qa)

    # ---------- save ----------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
if __name__ == "__main__":
    generate_batch_questions(
        output_path="eval/data/sample_atomQA.json",
        seed=42,
    )
