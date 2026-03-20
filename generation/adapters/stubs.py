from .base import TceAdapterArgs


def not_implemented(name: str):
    def _run(_args: TceAdapterArgs):
        raise NotImplementedError(
            "Baseline '{}' is not wired for TCE in this repo yet. "
            "Please add an adapter module and baseline implementation first.".format(name)
        )

    return _run
