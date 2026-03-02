from pathlib import Path
from typing import List

from .base import QaAdapterArgs


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _candidate_user_dirs(user: str) -> List[str]:
    raw = str(user).strip()
    if not raw:
        return []
    out = [raw]
    if raw.isdigit():
        n = int(raw)
        out.append(f"user{n}")
        out.append(f"{n:03d}_user_{n:03d}")
    return list(dict.fromkeys(out))


def _resolve_user_dir(root: Path, user: str) -> str:
    for c in _candidate_user_dirs(user):
        if (root / c).exists():
            return c
    raise FileNotFoundError(f"Cannot resolve user directory for '{user}' under {root}")


def run(args: QaAdapterArgs):
    from generation.letta.letta import run_qa

    if args.input_root_dir is None:
        raise ValueError("letta QA adapter requires data.input_root_dir")

    input_root = Path(args.input_root_dir)
    user_dir = _resolve_user_dir(input_root, args.user_id)
    input_user_dir = input_root / user_dir

    qa_root = Path(args.qa_dir) if args.qa_dir is not None else input_root
    qa_user_dir = qa_root / user_dir

    if args.output_path is not None:
        output_path = Path(args.output_path)
        prediction_dir = output_path.parent
    else:
        if args.output_root_dir is None:
            raise ValueError("letta QA adapter needs output.output_path or data.output_root_dir")
        prediction_dir = Path(args.output_root_dir) / user_dir / "prediction"
        output_path = None

    app_logs_filename = args.extras.get("app_logs_filename", "app_log_large.json")
    qa_filename = args.extras.get("qa_filename", "qa.json")
    checkpoint_state_path = args.extras.get("checkpoint_state_path")
    agentfile_path = args.extras.get("agentfile_path")

    run_qa(
        input_user_dir=input_user_dir,
        qa_user_dir=qa_user_dir,
        prediction_dir=prediction_dir,
        output_path=output_path,
        app_logs_filename=app_logs_filename,
        qa_filename=qa_filename,
        resume=_is_true(args.extras.get("resume", "false")),
        checkpoint_state_path=Path(checkpoint_state_path) if checkpoint_state_path else None,
        agentfile_path=Path(agentfile_path) if agentfile_path else None,
    )

    cmd = [
        "python3",
        "-m",
        "generation.letta.letta",
        "--input-user-dir",
        str(input_user_dir),
        "--qa-user-dir",
        str(qa_user_dir),
        "--prediction-dir",
        str(prediction_dir),
        "--app-logs-filename",
        app_logs_filename,
        "--qa-filename",
        qa_filename,
    ]
    return {"command": cmd, "returncode": 0}
