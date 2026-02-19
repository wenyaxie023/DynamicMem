from .base import DspAdapterArgs


def not_implemented(name: str):
    def _run(_args: DspAdapterArgs):
        raise NotImplementedError(
            "Baseline '{}' is not wired for DSP in this repo yet. "
            "Please add an adapter module and baseline implementation first.".format(name)
        )

    return _run
