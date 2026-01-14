from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Atom:
    atom_id: str
    domain: str
    type: str          # singular | collection | habit | preferences | operation_*
    anchor: str        # semantic anchor (key / habit_name / preference_name)
    time_window: str   # init | w1 | w2 | ...
    key: str           # atomic key (field name / delta.xxx / op / ...)
    value: Any         # scalar | list | dict
    path: Optional[str] = None  #
