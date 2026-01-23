from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class QAConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    qa_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    gen_provider: str = "gemini"
    gen_model_name: str = "gemini-3-flash-preview"
    llm_max_workers: int = 5
    experiment_name: str = "MemBench_Evaluation_01"
    enable_logging: bool = False

    sample_n: int = 10
    sample_seed: int = 42
    flush_rate: float = 0.05

    window_presets: Dict[str, str] = field(
        default_factory=lambda: {
            "w0": "w0",
            "w0_w1": "w1",
            "w0_w4": "w4",
        }
    )

    # batch_sample_count: int = 10
    # batch_offset: int = 5
    # batch_seed: int = 42
    # batch_output_relpath: str = "/data/sample_atomQA.json"
    # ez_target_index: int = 0
    # bind_target_indices: tuple[int, int] = (300, 500)

    macros_filename: str = "atom_macros.yaml"
    ir_filename: str = "atom_ir.yaml"
    schema_filename: str = "schema.json"
    atoms_filename: str = "atoms.json"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def schema_path(self) -> Path:
        return self.data_dir / self.schema_filename

    @property
    def qa_path(self) -> Path:
        return self.data_dir / "QA.json"

    @property
    def real_atoms_path(self) -> Path:
        return self.data_dir / self.atoms_filename

    @property
    def log_dir(self) -> Path:
        return self.data_dir /  "logs"

    @property
    def macros_path(self) -> Path:
        return self.qa_dir / self.macros_filename

    @property
    def ir_path(self) -> Path:
        return self.qa_dir / self.ir_filename

    def sampled_atoms_path(self, tag: str) -> Path:
        return self.data_dir / f"atoms_{tag}.json"

    def qa_output_path_for(self, tag: str) -> Path:
        return self.data_dir / f"qa_{tag}.json"

    # @property
    # def batch_output_path(self) -> Path:
    #     return self.root / self.batch_output_relpath
