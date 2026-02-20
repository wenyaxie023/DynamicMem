from typing import Callable, Dict

from . import icl, oracle, qa_pipeline, rag, stubs
from .base import QaAdapterArgs


_ADAPTERS: Dict[str, Callable[[QaAdapterArgs], object]] = {
    "qa_pipeline": qa_pipeline.run,
    "default": qa_pipeline.run,
    "oracle": oracle.run,
    "icl": icl.run,
    "rag": rag.run,
    "hipporag2": stubs.not_implemented("hipporag2"),
    "memoryos": stubs.not_implemented("memoryos"),
    "mem0": stubs.not_implemented("mem0"),
    "amem": stubs.not_implemented("amem"),
    "letta": stubs.not_implemented("letta"),
    "memgpt": stubs.not_implemented("memgpt"),
    "nemori": stubs.not_implemented("nemori"),
    "zep": stubs.not_implemented("zep"),
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
