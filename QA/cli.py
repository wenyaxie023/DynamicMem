from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_module(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parent)


def main() -> int:
    parser = argparse.ArgumentParser(description="QA unified CLI")
    sub = parser.add_subparsers(dest="command")

    p_context = sub.add_parser("context", help="Run qa_context CLI")
    p_context.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to qa_context")

    p_generation = sub.add_parser("generation", help="Run qa_generation CLI")
    p_generation.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to qa_generation")

    p_fix_links = sub.add_parser("fix-links", help="Run qa_generation.fix_qa_links")
    p_fix_links.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to fix_qa_links")

    p_pipeline = sub.add_parser("pipeline", help="Run context pipeline then generation pipeline")
    p_pipeline.add_argument("--user-id", type=int, default=10)
    p_pipeline.add_argument("--provider", type=str, default=None)
    p_pipeline.add_argument("--model", type=str, default=None)
    p_pipeline.add_argument("--max-workers", type=int, default=None)
    p_pipeline.add_argument("--retry-times", type=int, default=None)
    p_pipeline.add_argument("--flush-every", type=int, default=None)
    p_pipeline.add_argument("--sample-per-group", type=int, default=None)
    p_pipeline.add_argument("--sample-seed", type=int, default=None)
    p_pipeline.add_argument("--qtypes", type=str, default=None)
    p_pipeline.add_argument("--categories", type=str, default=None)

    args = parser.parse_args()

    if args.command == "context":
        return _run_module("qa_context", args.args)
    if args.command == "generation":
        return _run_module("qa_generation", args.args)
    if args.command == "fix-links":
        return _run_module("qa_generation.fix_qa_links", args.args)
    if args.command == "pipeline":
        ctx_args = ["pipeline", "--user-id", str(args.user_id)]
        rc = _run_module("qa_context", ctx_args)
        if rc != 0:
            return rc

        gen_args: List[str] = ["pipeline", "--user-id", str(args.user_id)]
        if args.provider:
            gen_args += ["--provider", args.provider]
        if args.model:
            gen_args += ["--model", args.model]
        if args.max_workers is not None:
            gen_args += ["--max-workers", str(args.max_workers)]
        if args.retry_times is not None:
            gen_args += ["--retry-times", str(args.retry_times)]
        if args.flush_every is not None:
            gen_args += ["--flush-every", str(args.flush_every)]
        if args.sample_per_group is not None:
            gen_args += ["--sample-per-group", str(args.sample_per_group)]
        if args.sample_seed is not None:
            gen_args += ["--sample-seed", str(args.sample_seed)]
        if args.qtypes:
            gen_args += ["--qtypes", args.qtypes]
        if args.categories:
            gen_args += ["--categories", args.categories]
        return _run_module("qa_generation", gen_args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
