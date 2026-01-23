import argparse
import json

try:
    import yaml
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required to run the QA pipeline") from exc

from config import QAConfig
from engine import atom_to_dict, atomize, load_ir, load_schema
from expand_macros import expand_macros
from all_gene import main as run_all_gene


def expand_macros_file(config: QAConfig) -> None:
    with open(config.macros_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    expanded = expand_macros(data)
    with open(config.ir_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(expanded, f, sort_keys=False)


def build_atoms_file(config: QAConfig) -> None:
    schema = load_schema(str(config.bg_path))
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
    generate_qa: bool = True,
) -> None:
    config = config or QAConfig()
    if expand:
        expand_macros_file(config)
    if build_atoms:
        build_atoms_file(config)
    if generate_qa:
        run_all_gene(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QA generation pipeline")
    parser.add_argument("--skip-expand", action="store_true", help="Skip macro expansion")
    parser.add_argument("--skip-atoms", action="store_true", help="Skip atoms.json generation")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA generation")
    args = parser.parse_args()

    run_pipeline(
        expand=not args.skip_expand,
        build_atoms=not args.skip_atoms,
        generate_qa=not args.skip_qa,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
