from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class Atom:
    atom_id: str
    domain: str
    anchor: str
    time_window: str
    key: str
    value: Any
    path: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)
