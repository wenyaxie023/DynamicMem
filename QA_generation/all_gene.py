import json
import random
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import Future, FIRST_COMPLETED, wait
from tqdm import tqdm

from client import LLMClient
from config import QAConfig
from context_fetch_by_anchor import fetch_context_by_anchor
from logger import setup_logger
from new_test import NOW_TEMPLATE


# =========================================================
# Helpers: Time window & UID
# =========================================================

def _time_window_rank(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value == "init":
        return 0
    if isinstance(value, str) and value.startswith("w") and value[1:].isdigit():
        return int(value[1:])
    return None


def _filter_atoms(atoms: List[Dict[str, Any]], max_window: str) -> List[Dict[str, Any]]:
    max_rank = _time_window_rank(max_window)
    if max_rank is None:
        raise ValueError(f"Invalid max_window: {max_window}")

    kept = []
    for atom in atoms:
        rank = _time_window_rank(atom.get("time_window"))
        if rank is not None and rank <= max_rank:
            kept.append(atom)
    return kept


def atom_uid(atom: Dict[str, Any]) -> str:
    return f"{atom.get('domain')}::{atom.get('path')}::{atom.get('time_window')}"


# =========================================================
# Helpers: Safe File Operations & JSON Cleaning
# =========================================================

def atomic_write_json(data: Any, path: Path):
    """
    原子写入：先写临时文件，再重命名，防止写入中断导致文件损坏。
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Windows 下 os.replace 是原子的 (Python 3.3+)
    os.replace(tmp_path, path)


def _clean_json_str(text: str) -> str:
    """
    清洗 LLM 返回的字符串，移除可能存在的 Markdown 代码块标记。
    """
    text = text.strip()
    # 匹配 ```json ... ``` 或 ``` ... ```
    if "```" in text:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return text


# =========================================================
# LLM QA parsing
# =========================================================

def _extract_qa(resp: Any) -> Dict[str, str]:
    if isinstance(resp, Exception):
        raise resp

    # 1. 尝试解析 JSON
    if isinstance(resp, str):
        cleaned_resp = _clean_json_str(resp)
        try:
            parsed = json.loads(cleaned_resp)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode failed: {str(e)}. Content: {cleaned_resp[:100]}...")
    else:
        parsed = resp

    # 2. 结构校验
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


# =========================================================
# Stage 1: Atom sampling & dump (罗盘针)
# =========================================================

def dump_sampled_atoms(
    *,
    atoms: List[Dict[str, Any]],
    output_path: Path,
    max_window: str,
    sample_n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    random.seed(seed)
    filtered = _filter_atoms(atoms, max_window=max_window)
    
    # 如果数量不足 sample_n，则全量取
    sampled = random.sample(filtered, min(sample_n, len(filtered)))

    atomic_write_json(sampled, output_path)
    return sampled


# =========================================================
# Stage 2: Sliding-window QA generation (可断点续传)
# =========================================================

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
    flush_every: int,
):
    logger = setup_logger(
        f"qa_gen_{max_window}",
        config.log_dir,
        enabled=config.enable_logging,
    )

    # 读取采样后的任务列表
    target_atoms = json.loads(target_atoms_path.read_text(encoding="utf-8"))
    
    # 读取已有进度
    existing = load_existing_qas(output_path)
    done_uids = set(existing.keys())
    
    # 过滤出未完成的任务
    pending_atoms = [a for a in target_atoms if atom_uid(a) not in done_uids]
    print(pending_atoms)
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

    # [新增] 初始化进度条，总数为待处理的任务数
    pbar = tqdm(total=len(pending_atoms), desc=f"Gen {max_window}", unit="atom")

    try:
        # 只要还有任务没分配，或者还有任务在运行中
        while idx < len(pending_atoms) or inflight:
            
            # ---- 1. 补满并发池 ----
            while idx < len(pending_atoms) and len(inflight) < config.llm_max_workers:
                atom = pending_atoms[idx]
                
                context = fetch_context_by_anchor(
                    anchor=atom.get("anchor", ""),
                    atoms=all_atoms_source,
                    schema=schema,
                    max_window=atom.get("time_window"),
                )

                prompt = NOW_TEMPLATE.format(
                    ATOM_JSON=json.dumps(atom, ensure_ascii=True),
                    CONTEXT_TEXT=json.dumps(context, ensure_ascii=True, indent=2),
                )

                fut = llm.ask_async(prompt)
                inflight[fut] = atom
                idx += 1

            # ---- 2. 等待至少一个完成 ----
            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                atom = inflight.pop(fut)
                uid = atom_uid(atom)
                
                # [新增] 无论成功失败，这里都算完成一个任务，进度条+1
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

                    # 定期刷盘
                    if len(results) % flush_every == 0:
                        atomic_write_json(results, output_path)
                        # 使用 tqdm.write 避免打断进度条
                        # pbar.write(f"Flushed progress: {len(results)} QAs saved")

                except Exception as e:
                    logger.error("Atom %s failed: %s", uid, e)
                    continue

    finally:
        pbar.close()  # [新增] 关闭进度条
        llm.close()

    # 最终保存
    atomic_write_json(results, output_path)
    logger.info("Done. Total QAs: %d", len(results))


# =========================================================
# Entry point (multi-window)
# =========================================================

def main(config: QAConfig | None = None):
    config = config or QAConfig()
    print("Loading full atoms dataset...")
    all_atoms = json.loads(config.real_atoms_path.read_text(encoding="utf-8"))
    schema = json.loads(config.bg_path.read_text(encoding="utf-8"))

    config.data_dir.mkdir(parents=True, exist_ok=True)

    for tag, max_window in config.window_presets.items():
        print(f"\nProcessing window group: {tag} (max_window={max_window})")
        
        target_atoms_path = config.sampled_atoms_path(tag)
        qa_output_path = config.qa_output_path_for(tag)

        if not target_atoms_path.exists():
            print(f"Sampling atoms for {tag}...")
            dump_sampled_atoms(
                atoms=all_atoms,
                output_path=target_atoms_path,
                max_window=max_window,
                sample_n=config.sample_n,
                seed=config.sample_seed,
            )

        generate_qas_with_resume(
            config=config,
            target_atoms_path=target_atoms_path,
            all_atoms_source=all_atoms, 
            schema=schema,
            output_path=qa_output_path,
            max_window=max_window,
            flush_every=config.flush_every,
        )


if __name__ == "__main__":
    main()
