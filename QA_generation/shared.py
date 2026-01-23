import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from prompts import NOW_TEMPLATE


def atomic_write_json(data: Any, path: Path) -> None:
    """
    Atomic write: write to temp file, then replace target to avoid partial writes.
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atom_uid(atom: Dict[str, Any]) -> str:
    return f"{atom.get('domain')}::{atom.get('path')}::{atom.get('time_window')}"


@dataclass
class RegistryEntry:
    tag: str
    max_window: str
    atoms_path: str
    qa_output_path: str
    sample_count: int
    flush_size: int | None = None
    prompt_name: str = "NOW_TEMPLATE"
    suffix: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "max_window": self.max_window,
            "atoms_path": self.atoms_path,
            "qa_output_path": self.qa_output_path,
            "sample_count": self.sample_count,
            "flush_size": self.flush_size,
            "prompt_name": self.prompt_name,
            "suffix": self.suffix,
        }
