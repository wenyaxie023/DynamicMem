from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Task:
    uid: str
    qtype: int
    category: str
    payload: Dict[str, Any]
