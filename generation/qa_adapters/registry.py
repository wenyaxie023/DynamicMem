from typing import Callable, Dict

from . import oracle, qa_pipeline, rag, stubs
from .base import QaAdapterArgs


_ADAPTERS: Dict[str, Callable[[QaAdapterArgs], object]] = {
    "qa_pipeline": qa_pipeline.run,
    "default": qa_pipeline.run,
    "oracle": oracle.run,
    "rag": rag.run,
    "hipporag2": stubs.not_implemented("hipporag2"),
    "memoryos": stubs.not_implemented("memoryos"),
    "amem": stubs.not_implemented("amem"),
}


def list_adapters():
    return sorted(_ADAPTERS.keys())


def run_adapter(args: QaAdapterArgs):
    key = (args.baseline or "").strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(
            "Unknown QA baseline '{}'. Available: {}".format(
                args.baseline,
                ", ".join(list_adapters()),
            )
        )
    return _ADAPTERS[key](args)
