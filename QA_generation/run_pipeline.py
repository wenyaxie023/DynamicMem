import argparse
import json
import math
import time
from pprint import pprint

try:
    import yaml
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required to run the QA pipeline") from exc

from config import QAConfig
from engine import atom_to_dict, atomize, load_ir, load_schema
from expand_macros import expand_macros
from generation import generate_qas
from sampling import build_registry_from_existing, sample_atoms


def expand_macros_file(config: QAConfig) -> None:
    with open(config.macros_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    expanded = expand_macros(data)
    with open(config.ir_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(expanded, f, sort_keys=False)


def build_atoms_file(config: QAConfig) -> None:
    schema = load_schema(str(config.schema_path))
    specs = load_ir(str(config.ir_path))
    atoms = atomize(schema, specs)
    config.real_atoms_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.real_atoms_path, "w", encoding="utf-8") as f:
        json.dump([atom_to_dict(a) for a in atoms], f, indent=2, ensure_ascii=False)


def run_pipeline(
    config: QAConfig | None = None,
    *,
    expand: bool = True,
    build_atoms: bool = True,
    sample_atoms_step: bool = True,
    generate_qa: bool = True,
    flush_size: int | None = None,
    output_suffix: str | None = None,
) -> None:
    config = config or QAConfig()
    if expand:
        print("Step 1/4: expand macros")
        start = time.perf_counter()
        expand_macros_file(config)
        elapsed = time.perf_counter() - start
        print(f"Step 1/4: expand macros done in {elapsed:.2f}s")
    else:
        print("Step 1/4: expand macros (skipped)")
    if build_atoms:
        print("Step 2/4: build atoms")
        start = time.perf_counter()
        build_atoms_file(config)
        elapsed = time.perf_counter() - start
        print(f"Step 2/4: build atoms done in {elapsed:.2f}s")
    else:
        print("Step 2/4: build atoms (skipped)")
    if sample_atoms_step:
        print("Step 3/4: sample atoms")
        start = time.perf_counter()
        registry = sample_atoms(config)
        elapsed = time.perf_counter() - start
        print(f"Step 3/4: sample atoms done in {elapsed:.2f}s")
    else:
        print("Step 3/4: sample atoms (skipped)")
        registry = build_registry_from_existing(config) if generate_qa else []
    if generate_qa:
        print("Step 4/4: generate QA")
        start = time.perf_counter()
        for entry in registry:
            sample_count = entry.sample_count
            if flush_size is None:
                computed = math.ceil(sample_count * config.flush_rate)
                entry.flush_size = max(1, computed)
            else:
                entry.flush_size = flush_size
            if output_suffix:
                entry.suffix = output_suffix
        # for entry in registry:
        #     pprint(entry.to_dict())
        generate_qas(registry, config)
        elapsed = time.perf_counter() - start
        print(f"Step 4/4: generate QA done in {elapsed:.2f}s")
    else:
        print("Step 4/4: generate QA (skipped)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QA generation pipeline")
    parser.add_argument("--skip-expand", action="store_true", help="Skip macro expansion")
    parser.add_argument("--skip-atoms", action="store_true", help="Skip atoms.json generation")
    parser.add_argument("--skip-sample", action="store_true", help="Skip sampled atoms generation")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA generation")
    parser.add_argument("--flush-size", type=int, default=None, help="Flush size for QA generation")
    parser.add_argument("--output-suffix", type=str, default=None, help="Suffix for QA output files")
    args = parser.parse_args()

    run_pipeline(
        expand=not args.skip_expand,
        build_atoms=not args.skip_atoms,
        sample_atoms_step=not args.skip_sample,
        generate_qa=not args.skip_qa,
        flush_size=args.flush_size,
        output_suffix=args.output_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
