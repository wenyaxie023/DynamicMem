from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from qa_generation.io import read_json_list
from shared.config import GenerationConfig


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _qtype_category(item: Dict[str, Any]) -> tuple[int, str]:
    meta = item.get("metadata")
    if not isinstance(meta, dict):
        return 0, ""
    return _as_int(meta.get("qtype")), _as_str(meta.get("category"))


def _checker(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata")
    if not isinstance(meta, dict):
        return {}
    checker = meta.get("checker_info")
    return checker if isinstance(checker, dict) else {}


def _doublecheck(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata")
    if not isinstance(meta, dict):
        return {}
    dc = meta.get("doublecheck_info")
    return dc if isinstance(dc, dict) else {}


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def export_checker_csv(*, config: GenerationConfig) -> None:
    refine_rows = read_json_list(config.judge_refine_output_path, allow_missing=False)
    double_rows = read_json_list(config.doublecheck_output_path, allow_missing=True)
    double_by_uid = {
        row["uid"]: row for row in double_rows if isinstance(row.get("uid"), str) and row.get("uid")
    }

    refine_full: List[Dict[str, Any]] = []
    double_full: List[Dict[str, Any]] = []
    simplify_refine: List[Dict[str, Any]] = []
    simplify_with_double: List[Dict[str, Any]] = []
    checker_summary: List[Dict[str, Any]] = []

    for item in refine_rows:
        uid = _as_str(item.get("uid"))
        if not uid:
            continue
        qtype, category = _qtype_category(item)
        checker = _checker(item)
        judge = _as_int(checker.get("judge"))
        rationale = _as_str(checker.get("rationale"))
        refine = checker.get("refine")
        refine = refine if isinstance(refine, dict) else {}
        refine_check = _as_int(refine.get("refine_check"))
        draft = _as_str(refine.get("draft"))
        refine_query = _as_str(refine.get("refine_query"))
        refine_answer = _as_str(refine.get("refine_answer"))
        history = refine.get("history")
        history = history if isinstance(history, dict) else {}
        history_query = _as_str(history.get("query"), "null")
        history_answer = _as_str(history.get("answer"), "null")

        query_final = _as_str(item.get("query"))
        answer_final = _as_str(item.get("reference"))

        refine_full.append(
            {
                "uid": uid,
                "qtype": qtype,
                "category": category,
                "judge": judge,
                "rationale": rationale,
                "refine_check": refine_check,
                "refine_draft": draft,
                "refine_query": refine_query,
                "refine_answer": refine_answer,
                "history_query": history_query,
                "history_answer": history_answer,
                "query_final": query_final,
                "answer_final": answer_final,
            }
        )
        simplify_refine.append(
            {
                "uid": uid,
                "qtype": qtype,
                "category": category,
                "judge": judge,
                "refine_check": refine_check,
            }
        )

        dc_item = double_by_uid.get(uid, {})
        dc = _doublecheck(dc_item)
        dc_judge = _as_int(dc.get("judge"), -1) if dc_item else -1
        dc_rationale = _as_str(dc.get("rationale")) if dc_item else ""

        simplify_with_double.append(
            {
                "uid": uid,
                "qtype": qtype,
                "category": category,
                "judge": judge,
                "refine_check": refine_check,
                "doublecheck_judge": dc_judge,
            }
        )
        checker_summary.append(
            {
                "uid": uid,
                "qtype": qtype,
                "category": category,
                "judge": judge,
                "refine_check": refine_check,
                "doublecheck_judge": dc_judge,
                "history_query": history_query,
                "history_answer": history_answer,
                "query_final": query_final,
                "answer_final": answer_final,
            }
        )

        if dc_item:
            double_full.append(
                {
                    "uid": uid,
                    "qtype": qtype,
                    "category": category,
                    "doublecheck_judge": dc_judge,
                    "doublecheck_rationale": dc_rationale,
                    "query_final": _as_str(dc_item.get("query")),
                    "answer_final": _as_str(dc_item.get("reference")),
                }
            )

    out_dir = config.data_dir
    _write_csv(
        out_dir / "qa_refine.csv",
        refine_full,
        [
            "uid",
            "qtype",
            "category",
            "judge",
            "rationale",
            "refine_check",
            "refine_draft",
            "refine_query",
            "refine_answer",
            "history_query",
            "history_answer",
            "query_final",
            "answer_final",
        ],
    )
    _write_csv(
        out_dir / "qa_doublecheck.csv",
        double_full,
        [
            "uid",
            "qtype",
            "category",
            "doublecheck_judge",
            "doublecheck_rationale",
            "query_final",
            "answer_final",
        ],
    )
    _write_csv(
        out_dir / "qa_refine_simple.csv",
        simplify_refine,
        ["uid", "qtype", "category", "judge", "refine_check"],
    )
    _write_csv(
        out_dir / "qa_checker_simple.csv",
        simplify_with_double,
        ["uid", "qtype", "category", "judge", "refine_check", "doublecheck_judge"],
    )
    _write_csv(
        out_dir / "qa_checker_summary.csv",
        checker_summary,
        [
            "uid",
            "qtype",
            "category",
            "judge",
            "refine_check",
            "doublecheck_judge",
            "history_query",
            "history_answer",
            "query_final",
            "answer_final",
        ],
    )

    print(f"Saved: {out_dir / 'qa_refine.csv'}")
    print(f"Saved: {out_dir / 'qa_doublecheck.csv'}")
    print(f"Saved: {out_dir / 'qa_refine_simple.csv'}")
    print(f"Saved: {out_dir / 'qa_checker_simple.csv'}")
    print(f"Saved: {out_dir / 'qa_checker_summary.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export qa_refine/qa_doublecheck to CSV")
    parser.add_argument("--user-id", type=int, default=10)
    args = parser.parse_args()
    export_checker_csv(config=GenerationConfig(user_id=args.user_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
