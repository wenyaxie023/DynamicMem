from .base import QaAdapterArgs


def not_implemented(name: str):
    def _run(_args: QaAdapterArgs):
        raise NotImplementedError(
            "Baseline '{}' is not wired for QA in this repo yet. "
            "Please add a QA adapter module and baseline implementation first.".format(name)
        )

    return _run
