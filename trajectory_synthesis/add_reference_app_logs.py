import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def add_reference_app_logs(
    *,
    qa_path: Path,
    app_logs_path: Path,
    output_path: Optional[Path] = None,
    include_full_logs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Add `reference_app_logs` to each QA item's metadata by mapping
    `reference_evidence` (event_id list) to app logs in app_logs_final.json.
    `reference_app_logs` is always a list of app log dicts.
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    with open(app_logs_path, "r", encoding="utf-8") as f:
        app_logs_data = json.load(f)

    app_logs = app_logs_data.get("app_logs", [])

    event_to_logs: Dict[str, List[Any]] = {}
    for log in app_logs:
        event_id = log.get("event_id")
        if not event_id:
            continue
        # Always store full app log dicts in reference_app_logs.
        event_to_logs.setdefault(event_id, []).append(log)

    def _dedupe_keep_order(items: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for item in items:
            if isinstance(item, dict):
                key = item.get("app_log_id") or id(item)
            else:
                key = item
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    for item in qa_list:
        metadata = item.get("metadata") or {}
        evidence = metadata.get("reference_evidence") or []
        reference_app_logs: List[Any] = []
        for ev in evidence:
            reference_app_logs.extend(event_to_logs.get(ev, []))
        metadata["reference_app_logs"] = _dedupe_keep_order(reference_app_logs)
        item["metadata"] = metadata

    if output_path is None:
        output_path = qa_path.with_name(
            qa_path.stem + "_with_app_logs" + qa_path.suffix
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    return qa_list


def process_all_users(
    *,
    root_dir: Path,
    qa_filenames: Optional[List[str]] = None,
    app_logs_filename: str = "app_logs_final.json",
    include_full_logs: bool = False,
    output_suffix: str = "_with_app_logs",
) -> List[Path]:
    """
    Traverse all user directories under root_dir. If qa/app_logs exist,
    write a new QA file with reference_app_logs appended.
    """
    if qa_filenames is None:
        qa_filenames = ["qa_w0.json", "qa_w0_w1.json", "qa_w0_w4.json"]

    written: List[Path] = []
    for user_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        app_logs_path = user_dir / app_logs_filename
        if not app_logs_path.exists():
            continue
        for qa_name in qa_filenames:
            qa_path = user_dir / qa_name
            if not qa_path.exists():
                continue
            output_path = qa_path.with_name(qa_path.stem + output_suffix + qa_path.suffix)
            add_reference_app_logs(
                qa_path=qa_path,
                app_logs_path=app_logs_path,
                output_path=output_path,
                include_full_logs=include_full_logs,
            )
            written.append(output_path)
    return written


if __name__ == "__main__":
    # Default root is the gemini_3_flash_preview outputs (repo-root outputs/).
    default_root = (
        Path(__file__).resolve().parent.parent / "outputs" / "gemini_3_flash_preview"
    )
    process_all_users(root_dir=default_root)
