from typing import Callable, Dict

from .base import TceAdapterArgs
from . import amem, hipporag2, icl, letta, oracle, rag, stubs


_ADAPTERS: Dict[str, Callable[[TceAdapterArgs], object]] = {
    "oracle": oracle.run,
    "icl": icl.run,
    "rag": rag.run,
    "hipporag2": hipporag2.run,
    "memoryos": stubs.not_implemented("memoryos"),
    "amem": amem.run,
    "letta": letta.run,
    "memgpt": letta.run,
    # Research baseline names requested by paper list
    "nemori": stubs.not_implemented("nemori"),
    "zep": stubs.not_implemented("zep"),
    "mem0": stubs.not_implemented("mem0"),
}


def list_adapters():
    return sorted(_ADAPTERS.keys())


def run_adapter(args: TceAdapterArgs):
    key = (args.baseline or "").strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(
            "Unknown baseline '{}'. Available: {}".format(
                args.baseline,
                ", ".join(list_adapters()),
            )
        )
    return _ADAPTERS[key](args)
