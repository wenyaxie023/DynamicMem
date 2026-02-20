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


def _flag(value: str) -> str:
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "y", "on"} else "false"


def run(args: QaAdapterArgs):
    script = Path(__file__).resolve().parents[1] / "oracle" / "generation.py"
    cmd: List[str] = [sys.executable, str(script), "--user-idx", str(args.user_id)]

    _append_if(cmd, "--root-dir", args.input_root_dir)
    _append_if(cmd, "--output-root-dir", args.output_root_dir)
    _append_if(cmd, "--qa-dir", args.qa_dir)
    _append_if(cmd, "--output-path", args.output_path)

    _append_if(cmd, "--llm-provider", args.llm_provider)
    _append_if(cmd, "--llm-model", args.llm_model)
    _append_if(cmd, "--llm-max-workers", args.llm_max_workers)

    log_size = args.extras.get("log_size", "large")
    _append_if(cmd, "--log-size", log_size)

    if _flag(args.extras.get("write_each", "false")) == "true":
        cmd.append("--write-each")
    if _flag(args.extras.get("resume", "false")) == "true":
        cmd.append("--resume")

    _append_if(cmd, "--only-idx-151-180", _flag(args.extras.get("only_idx_151_180", "false")))

    repo_root = Path(__file__).resolve().parents[2]
    rc = subprocess.call(cmd, cwd=repo_root)
    if rc != 0:
        raise RuntimeError("QA oracle baseline run failed with code {}".format(rc))
    return {"command": cmd, "returncode": rc}
