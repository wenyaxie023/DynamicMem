import json
import re
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import Future, FIRST_COMPLETED, wait

from tqdm import tqdm

from client import LLMClient
from config import QAConfig
from context_fetch_by_anchor import fetch_context_by_anchor
from logger import setup_logger
import prompts
from shared import RegistryEntry, atom_uid, atomic_write_json


def _clean_json_str(text: str) -> str:
    text = text.strip()
    if "```" in text:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return text


def _extract_qa(resp: Any) -> Dict[str, str]:
    if isinstance(resp, Exception):
        raise resp

    if isinstance(resp, str):
        cleaned_resp = _clean_json_str(resp)
        try:
            parsed = json.loads(cleaned_resp)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode failed: {str(e)}. Content: {cleaned_resp[:100]}...")
    else:
        parsed = resp

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Expected non-empty list from LLM")

    item = parsed[0]
    required_keys = ("question", "answer", "evidence")
    for k in required_keys:
        if k not in item:
            raise ValueError(f"Missing key '{k}' in LLM response")

    return {
        "question": item["question"],
        "answer": item["answer"],
        "evidence": item["evidence"],
        "draft1": item.get("draft1", ""),
        "draft2": item.get("draft2", ""),
    }


def load_existing_qas(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["atom_uid"]: item for item in data}
    except json.JSONDecodeError:
        print(f"Warning: {path} is corrupted or empty. Starting fresh.")
        return {}


def generate_qas_with_resume(
    *,
    config: QAConfig,
    target_atoms_path: Path,
    all_atoms_source: List[Dict],
    schema: Dict[str, Any],
    output_path: Path,
    max_window: str,
    flush_size: int,
    prompt_template: str,
):
    logger = setup_logger(
        f"qa_gen_{max_window}",
        config.log_dir,
        enabled=config.enable_logging,
    )

    target_atoms = json.loads(target_atoms_path.read_text(encoding="utf-8"))

    existing = load_existing_qas(output_path)
    done_uids = set(existing.keys())

    pending_atoms = [a for a in target_atoms if atom_uid(a) not in done_uids]
    logger.info(
        "Total Target Atoms=%d | Done=%d | Pending=%d",
        len(target_atoms), len(done_uids), len(pending_atoms)
    )
    if not pending_atoms:
        logger.info("All tasks completed.")
        return

    llm = LLMClient(
        provider=config.gen_provider,
        model_name=config.gen_model_name,
        max_workers=config.llm_max_workers,
    )

    results = list(existing.values())
    inflight: Dict[Future, Dict[str, Any]] = {}
    idx = 0

    pbar = tqdm(total=len(pending_atoms), desc=f"Gen {max_window}", unit="atom")

    try:
        while idx < len(pending_atoms) or inflight:
            while idx < len(pending_atoms) and len(inflight) < config.llm_max_workers:
                atom = pending_atoms[idx]

                context = fetch_context_by_anchor(
                    anchor=atom.get("anchor", ""),
                    atoms=all_atoms_source,
                    schema=schema,
                    max_window=atom.get("time_window"),
                )

                prompt = prompt_template.format(
                    ATOM_JSON=json.dumps(atom, ensure_ascii=True),
                    CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
                )

                fut = llm.ask_async(prompt)
                inflight[fut] = atom
                idx += 1

            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                atom = inflight.pop(fut)
                uid = atom_uid(atom)
                pbar.update(1)

                try:
                    qa = _extract_qa(fut.result())

                    record = {
                        "atom_uid": uid,
                        "query": qa["question"],
                        "reference": qa["answer"],
                        "prediction": "",
                        "metadata": {
                            "category": atom.get("domain"),
                            "path": atom.get("path"),
                            "reference_evidence": qa["evidence"],
                            "draft_question": qa["draft1"],
                            "draft_answer": qa["draft2"],
                            "max_window": max_window,
                        },
                    }

                    results.append(record)

                    if len(results) % flush_size == 0:
                        atomic_write_json(results, output_path)

                except Exception as e:
                    logger.error("Atom %s failed: %s", uid, e)
                    continue

    finally:
        pbar.close()
        llm.close()

    atomic_write_json(results, output_path)
    logger.info("Done. Total QAs: %d", len(results))


def generate_qas(registry: List[RegistryEntry], config: QAConfig | None = None) -> None:
    config = config or QAConfig()
    print("Loading full atoms dataset...")
    all_atoms = json.loads(config.real_atoms_path.read_text(encoding="utf-8"))
    schema = json.loads(config.schema_path.read_text(encoding="utf-8"))

    config.data_dir.mkdir(parents=True, exist_ok=True)

    for entry in registry:
        tag = entry.tag
        max_window = entry.max_window
        target_atoms_path = Path(entry.atoms_path)
        qa_output_path = Path(entry.qa_output_path)
        if entry.suffix:
            qa_output_path = qa_output_path.with_name(
                f"{qa_output_path.stem}{entry.suffix}{qa_output_path.suffix}"
            )
        flush_size = entry.flush_size
        if not flush_size or flush_size <= 0:
            raise ValueError(f"Invalid flush_size for tag {tag}: {flush_size}")
        prompt_name = entry.prompt_name or "NOW_TEMPLATE"
        if not hasattr(prompts, prompt_name):
            raise ValueError(f"Unknown prompt name for tag {tag}: {prompt_name}")
        prompt_template = getattr(prompts, prompt_name)

        print(f"\nGenerating QAs for window group: {tag} (max_window={max_window})")

        if not target_atoms_path.exists():
            raise FileNotFoundError(
                f"Sampled atoms not found for {tag}. Expected: {target_atoms_path}"
            )

        generate_qas_with_resume(
            config=config,
            target_atoms_path=target_atoms_path,
            all_atoms_source=all_atoms,
            schema=schema,
            output_path=qa_output_path,
            max_window=max_window,
            flush_size=flush_size,
            prompt_template=prompt_template,
        )
