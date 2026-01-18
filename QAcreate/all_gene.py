import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval.client import LLMClient
from .config import (
    BG_PATH,
    LLM_MAX_WORKERS,
    GEN_PROVIDER,
    GEN_MODEL_NAME,
    REAL_ATOMS_PATH,
    LOG_DIR,
)
from .context_fetch import fetch_context
from ..eval.logger import setup_logger
from .test import QUESTION1_TEMPLATE


def _time_window_rank(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value == "init":
        return 0
    if value.startswith("w") and value[1:].isdigit():
        return int(value[1:])
    return None


def _filter_atoms(atoms: List[Dict[str, Any]], max_window: str) -> List[Dict[str, Any]]:
    max_rank = _time_window_rank(max_window)
    if max_rank is None:
        raise ValueError(f"Invalid max_window: {max_window}")
    kept: List[Dict[str, Any]] = []
    for atom in atoms:
        rank = _time_window_rank(atom.get("time_window"))
        if rank is None:
            continue
        if rank <= max_rank:
            kept.append(atom)
    return kept


def _extract_qa(llm_response: Any) -> Dict[str, str]:
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


def generate_all_questions(
    *,
    atoms_path: Path = REAL_ATOMS_PATH,
    output_path: Path = Path("eval/data/QA_all.json"),
    max_window: str = "w2",
    batch_size: int | None = None,
) -> List[Dict[str, Any]]:
    logger = setup_logger("all_gene", LOG_DIR)

    with open(atoms_path, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    with open(BG_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    atoms = _filter_atoms(atoms, max_window=max_window)
    logger.info("Filtered atoms: %s", len(atoms))

    llm = LLMClient(
        provider=GEN_PROVIDER,
        model_name=GEN_MODEL_NAME,
        max_workers=LLM_MAX_WORKERS,
    )

    if batch_size is None:
        batch_size = max(1, LLM_MAX_WORKERS * 4)

    results: List[Dict[str, Any]] = []

    for start in range(0, len(atoms), batch_size):
        batch = atoms[start : start + batch_size]
        prompts: List[str] = []
        for atom in batch:
            context = fetch_context(atom=atom, schema=schema)
            prompt = QUESTION1_TEMPLATE.format(
                ATOM_JSON=json.dumps(atom, ensure_ascii=True),
                CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
            )
            prompts.append(prompt)

        logger.info("Dispatching prompts %s-%s", start + 1, start + len(batch))
        responses = llm.ask_many(prompts)

        for resp, atom in zip(responses, batch):
            qa = _extract_qa(resp)
            results.append(
                {
                    "id": f"Q{len(results) + 1}",
                    "query": qa["question"],
                    "reference": qa["answer"],
                    "prediction": "",
                    "metadata": {"category": atom.get("domain")},
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def runcase():
    with open(REAL_ATOMS_PATH, "r", encoding="utf-8") as f:
        atoms = json.load(f)
    atoms = _filter_atoms(atoms, max_window='w2')
    with open('eval/data/QA_all.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QA pairs.")
    print(f"Loaded {len(atoms)} atoms.")
    for i, item in enumerate(qa_data):
        item['metadata']['path'] = atoms[i]['path']
    with open('eval/data/QA_all_revised.json', 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    runcase()
