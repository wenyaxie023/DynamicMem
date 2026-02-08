from __future__ import annotations

import argparse
from typing import List, Optional

from qa_generation.core.generator import run_generation
from qa_generation.core.pipeline import run_pipeline
from qa_generation.core.sampler import run_sampling
from shared.config import GenerationConfig


def _parse_int_list(value: str | None) -> Optional[List[int]]:
    if not value:
        return None
    items: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return items or None


def _parse_str_list(value: str | None) -> Optional[List[str]]:
    if not value:
        return None
    items = [token.strip() for token in value.split(",") if token.strip()]
    return items or None


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--retry-times", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--sample-per-group", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--qtypes", type=str, default=None, help="comma-separated, e.g. 1,2,6")
    parser.add_argument("--categories", type=str, default=None, help="comma-separated category names")
    return parser


def _build_config(args: argparse.Namespace) -> GenerationConfig:
    config = GenerationConfig(user_id=args.user_id) if args.user_id is not None else GenerationConfig()
    if args.provider:
        config = config.with_updates(provider=args.provider)
    if args.model:
        config = config.with_updates(model_name=args.model)
    if args.max_workers is not None:
        config = config.with_updates(max_workers=args.max_workers)
    if args.retry_times is not None:
        config = config.with_updates(retry_times=args.retry_times)
    if args.flush_every is not None:
        config = config.with_updates(flush_every=args.flush_every)
    if args.sample_per_group is not None:
        config = config.with_updates(sample_per_group=args.sample_per_group)
    if args.sample_seed is not None:
        config = config.with_updates(sample_seed=args.sample_seed)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="qa_generation: tasks sampling + QA generation")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("sample", parents=[_base_parser()])
    sub.add_parser("generate", parents=[_base_parser()])
    sub.add_parser("pipeline", parents=[_base_parser()])

    args = parser.parse_args()
    config = _build_config(args)

    qtypes = _parse_int_list(args.qtypes)
    categories = _parse_str_list(args.categories)

    if args.command == "sample":
        return run_sampling(config=config, qtypes=qtypes, categories=categories)
    if args.command == "generate":
        return run_generation(config=config)
    return run_pipeline(
        config=config,
        qtypes=qtypes,
        categories=categories,
    )


if __name__ == "__main__":
    raise SystemExit(main())
