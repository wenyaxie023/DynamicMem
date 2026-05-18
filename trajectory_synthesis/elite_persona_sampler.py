from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 42
# A modest buffer keeps the shuffle reasonably random without downloading the full set.
DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parent
    / "persona_data"
    / f"elite_personas_sample_seed{DEFAULT_SEED}.jsonl"
)


class PersonaSamplerError(RuntimeError):
    """Raised when sampling elite personas fails."""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sampled_personas(
    path: str | Path = DEFAULT_SAMPLE_PATH, limit: Optional[int] = None
) -> List[Dict]:
    """
    Load sampled personas from a JSONL file.

    Args:
        path: Path to the JSONL file.
        limit: Optional cap on how many personas to return.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona sample file not found: {path}")

    personas: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(personas) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            personas.append(json.loads(line))
    return personas


def sample_elite_personas(
    *,
    output_path: str | Path = DEFAULT_SAMPLE_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    overwrite: bool = False,
) -> Path:
    """
    Stream the PersonaHub elite_persona split and persist a deterministic sample.

    Sampling uses dataset streaming with a shuffle buffer so we only download
    a small slice of the 370M-persona release.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is optional
        raise PersonaSamplerError(
            "The 'datasets' package is required to sample elite personas. "
            "Install it with `pip install datasets`."
        ) from exc

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    _ensure_dir(output_path.parent)

    dataset = load_dataset(
        "proj-persona/PersonaHub",
        "elite_persona",
        split="train",
        streaming=True,
    )
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shuffled):
            if idx >= sample_size:
                break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = output_path.with_suffix(".meta.json")
    meta_payload = {
        "source": "proj-persona/PersonaHub:elite_persona",
        "sample_size": sample_size,
        "seed": seed,
        "buffer_size": buffer_size,
        "output_path": str(output_path),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False))
    return output_path
