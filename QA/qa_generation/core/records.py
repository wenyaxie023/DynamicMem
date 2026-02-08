from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class QAMetadata:
    qtype: int
    category: str
    event_ids: List[str]
    reference_evidence: List[str]
    draft_question: str = ""
    draft_answer: str = ""


@dataclass
class BaseRecord:
    uid: str
    query: str
    reference: str
    prediction: str
    metadata: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QARecord(BaseRecord):
    metadata: QAMetadata


@dataclass
class JudgeRecord(BaseRecord):
    metadata: Dict[str, Any]


@dataclass
class DoublecheckRecord(BaseRecord):
    metadata: Dict[str, Any]
