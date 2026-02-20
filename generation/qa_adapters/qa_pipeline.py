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


def _build_command(args: QaAdapterArgs) -> List[str]:
    cmd: List[str] = [sys.executable, "-m", "QA"]

    if args.command == "pipeline":
        cmd.extend(["pipeline", "--user-id", str(args.user_id)])
        _append_if(cmd, "--provider", args.llm_provider)
        _append_if(cmd, "--model", args.llm_model)
        _append_if(cmd, "--max-workers", args.llm_max_workers)
        _append_if(cmd, "--retry-times", args.retry_times)
        _append_if(cmd, "--flush-every", args.flush_every)
        _append_if(cmd, "--sample-per-group", args.sample_per_group)
        _append_if(cmd, "--sample-seed", args.sample_seed)
        _append_if(cmd, "--qtypes", args.qtypes)
        _append_if(cmd, "--categories", args.categories)
        return cmd

    if args.command == "context":
        cmd.extend(["context", args.subcommand or "pipeline", "--user-id", str(args.user_id)])
        return cmd

    if args.command == "generation":
        cmd.extend(["generation", args.subcommand or "pipeline", "--user-id", str(args.user_id)])
        _append_if(cmd, "--provider", args.llm_provider)
        _append_if(cmd, "--model", args.llm_model)
        _append_if(cmd, "--max-workers", args.llm_max_workers)
        _append_if(cmd, "--retry-times", args.retry_times)
        _append_if(cmd, "--flush-every", args.flush_every)
        _append_if(cmd, "--sample-per-group", args.sample_per_group)
        _append_if(cmd, "--sample-seed", args.sample_seed)
        _append_if(cmd, "--qtypes", args.qtypes)
        _append_if(cmd, "--categories", args.categories)
        return cmd

    if args.command == "fix-links":
        cmd.extend(["fix-links", "--user-id", str(args.user_id)])
        if args.fix_links_inplace:
            cmd.append("--inplace")
        return cmd

    if args.command == "export-csv":
        cmd.extend(["generation", "export-csv", "--user-id", str(args.user_id)])
        return cmd

    raise ValueError("Unknown QA command: {}".format(args.command))


def run(args: QaAdapterArgs):
    cmd = _build_command(args)
    repo_root = Path(__file__).resolve().parents[2]
    rc = subprocess.call(cmd, cwd=repo_root)
    if rc != 0:
        raise RuntimeError("QA run failed with code {}".format(rc))
    return {"command": cmd, "returncode": rc}
