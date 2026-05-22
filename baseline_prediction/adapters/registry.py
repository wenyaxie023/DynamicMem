from typing import Callable, Dict

from .base import TceAdapterArgs
from . import amem, hipporag2, memoryos, oracle, rag, simplemem


_ADAPTERS: Dict[str, Callable[[TceAdapterArgs], object]] = {
    "oracle": oracle.run,
    "rag": rag.run,
    "hipporag2": hipporag2.run,
    "memoryos": memoryos.run,
    "amem": amem.run,
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
