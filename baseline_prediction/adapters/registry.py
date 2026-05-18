from typing import Callable, Dict

from .base import TceAdapterArgs
from . import amem, hipporag2, icl, letta, mem0, memoryos, oracle, rag, simplemem, stubs, zep


_ADAPTERS: Dict[str, Callable[[TceAdapterArgs], object]] = {
    "oracle": oracle.run,
    "icl": icl.run,
    "rag": rag.run,
    "hipporag2": hipporag2.run,
    "memoryos": memoryos.run,
    "amem": amem.run,
    "letta": letta.run,
    "memgpt": letta.run,
    "nemori": stubs.not_implemented("nemori"),
    "zep": zep.run,
    "mem0": mem0.run,
    "simplemem": simplemem.run,
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
