import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import QAConfig
from shared import RegistryEntry
from shared import atomic_write_json


def _time_window_rank(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value == "init":
        return 0
    if isinstance(value, str) and value.startswith("w") and value[1:].isdigit():
        return int(value[1:])
    return None


def _filter_atoms(atoms: List[Dict[str, Any]], max_window: str) -> List[Dict[str, Any]]:
    max_rank = _time_window_rank(max_window)
    if max_rank is None:
        raise ValueError(f"Invalid max_window: {max_window}")

    kept = []
    for atom in atoms:
        rank = _time_window_rank(atom.get("time_window"))
        if rank is not None and rank <= max_rank:
            kept.append(atom)
    return kept


def dump_sampled_atoms(
    *,
    atoms: List[Dict[str, Any]],
    output_path: Path,
    max_window: str,
    sample_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    filtered = _filter_atoms(atoms, max_window=max_window)
    sampled = random.sample(filtered, min(sample_n, len(filtered)))
    atomic_write_json(sampled, output_path)
    return sampled


def _load_sampled_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid sampled atoms file: {path}")
    return len(data)


def sample_atoms(config: QAConfig | None = None) -> List[RegistryEntry]:
    config = config or QAConfig()
    print("Loading full atoms dataset...")
    all_atoms = json.loads(config.real_atoms_path.read_text(encoding="utf-8"))

    config.data_dir.mkdir(parents=True, exist_ok=True)

    registry: List[RegistryEntry] = []
    for tag, max_window in config.window_presets.items():
        print(f"\nSampling window group: {tag} (max_window={max_window})")
        target_atoms_path = config.sampled_atoms_path(tag)
        if not target_atoms_path.exists():
            print(f"Sampling atoms for {tag}...")
            sampled = dump_sampled_atoms(
                atoms=all_atoms,
                output_path=target_atoms_path,
                max_window=max_window,
                sample_n=config.sample_n,
                seed=config.sample_seed,
            )
            sample_count = len(sampled)
        else:
            print(f"Sampling atoms for {tag} already done. Using {target_atoms_path}.")
            sample_count = _load_sampled_count(target_atoms_path)

        registry.append(
            RegistryEntry(
                tag=tag,
                max_window=max_window,
                atoms_path=str(target_atoms_path),
                qa_output_path=str(config.qa_output_path_for(tag)),
                sample_count=sample_count,
            )
        )

    return registry


def build_registry_from_existing(config: QAConfig | None = None) -> List[RegistryEntry]:
    config = config or QAConfig()
    registry: List[RegistryEntry] = []
    for tag, max_window in config.window_presets.items():
        target_atoms_path = config.sampled_atoms_path(tag)
        if not target_atoms_path.exists():
            raise FileNotFoundError(
                f"Sampled atoms not found for {tag}. Expected: {target_atoms_path}"
            )
        sample_count = _load_sampled_count(target_atoms_path)
        registry.append(
            RegistryEntry(
                tag=tag,
                max_window=max_window,
                atoms_path=str(target_atoms_path),
                qa_output_path=str(config.qa_output_path_for(tag)),
                sample_count=sample_count,
            )
        )
    return registry
