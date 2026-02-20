import subprocess
import sys
from pathlib import Path
from typing import List

from .base import QaAdapterArgs


def _append_if(cmd: List[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.extend([flag, str(value)])


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(args: QaAdapterArgs):
    script = Path(__file__).resolve().parents[1] / "rag" / "rag.py"
    cmd: List[str] = [sys.executable, str(script), "--user-idx", str(args.user_id)]

    _append_if(cmd, "--input-root-dir", args.input_root_dir)
    _append_if(cmd, "--output-root-dir", args.output_root_dir)
    _append_if(cmd, "--qa-dir", args.qa_dir)

    _append_if(cmd, "--llm-provider", args.llm_provider)
    _append_if(cmd, "--llm-model", args.llm_model)
    _append_if(cmd, "--llm-max-workers", args.llm_max_workers)

    _append_if(cmd, "--gen-topk", args.extras.get("gen_topk", "5"))
    _append_if(cmd, "--chunk-size", args.extras.get("chunk_size", "-1"))
    _append_if(cmd, "--retriever-type", args.extras.get("retriever_type", "openai"))
    _append_if(cmd, "--retriever-model", args.extras.get("retriever_model", "text-embedding-3-large"))

    if _is_true(args.extras.get("skip_retrieve", "false")):
        cmd.append("--skip-retrieve")
    if _is_true(args.extras.get("retrieve_only", "false")):
        cmd.append("--retrieve-only")
    if _is_true(args.extras.get("write_each", "false")):
        cmd.append("--write-each")
    if _is_true(args.extras.get("resume", "false")):
        cmd.append("--resume")

    repo_root = Path(__file__).resolve().parents[2]
    rc = subprocess.call(cmd, cwd=repo_root)
    if rc != 0:
        raise RuntimeError("QA RAG baseline run failed with code {}".format(rc))
    return {"command": cmd, "returncode": rc}
