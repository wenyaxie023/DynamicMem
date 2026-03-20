#!/usr/bin/env python3
"""Build reproducible TCE manual-review sample artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tce_core.manual_review import (
    FIXED_SEED,
    build_apply_pool,
    build_change_tracking_pool,
    build_review_sheet_rows,
    build_stage1_pool,
    build_state_completion_pool,
    review_plan_markdown,
    select_manual_review_samples,
    write_csv,
    write_json,
)


PURPOSE_GROUPS = [
    ("stage1_l1_fail", lambda row: row.get("stage") == "stage1" and row.get("bucket") == "l1_fail"),
    ("stage1_l2_fail", lambda row: row.get("stage") == "stage1" and row.get("bucket") == "l2_fail"),
    ("stage1_pass", lambda row: row.get("stage") == "stage1" and row.get("bucket") == "pass"),
    (
        "stage2_state_completion",
        lambda row: row.get("stage") == "stage2_state_completion",
    ),
    (
        "stage2_change_tracking",
        lambda row: row.get("stage") == "stage2_change_tracking",
    ),
    (
        "stage2_apply_accepted",
        lambda row: row.get("stage") == "stage2_apply" and row.get("bucket") == "accepted",
    ),
    (
        "stage2_apply_discarded",
        lambda row: row.get("stage") == "stage2_apply" and row.get("bucket") == "discarded",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCE Stage1+Stage2 manual-review samples.")
    parser.add_argument("--validated", type=Path, required=True, help="Validated benchmark JSON path.")
    parser.add_argument("--task-packs", type=Path, required=True, help="Task-pack benchmark JSON path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    args = parser.parse_args()

    validated_payload = json.loads(args.validated.read_text(encoding="utf-8"))
    task_pack_payload = json.loads(args.task_packs.read_text(encoding="utf-8"))

    stage1_pool = build_stage1_pool(validated_payload)
    state_completion_pool = build_state_completion_pool(task_pack_payload)
    change_tracking_pool = build_change_tracking_pool(task_pack_payload)
    apply_pool = build_apply_pool(task_pack_payload)
    samples = select_manual_review_samples(
        stage1_pool=stage1_pool,
        state_completion_pool=state_completion_pool,
        change_tracking_pool=change_tracking_pool,
        apply_pool=apply_pool,
        seed=args.seed,
    )
    review_rows = build_review_sheet_rows(samples)
    review_row_by_unit = {
        sample["sample_unit_id"]: row
        for sample, row in zip(samples, review_rows)
    }

    summary = {
        "seed": args.seed,
        "counts": {
            "stage1_pool": len(stage1_pool),
            "state_completion_pool": len(state_completion_pool),
            "change_tracking_pool": len(change_tracking_pool),
            "apply_pool": len(apply_pool),
            "selected_samples": len(samples),
        },
        "selected_by_stage_bucket": {
            f"{stage}::{bucket}": count
            for (stage, bucket), count in Counter(
                (str(sample.get("stage")), str(sample.get("bucket"))) for sample in samples
            ).items()
        },
        "selected_by_category": dict(Counter(str(sample.get("state_category")) for sample in samples)),
        "selected_by_checkpoint": dict(Counter(str(sample.get("checkpoint_id")) for sample in samples)),
    }

    output_dir = args.output_dir
    write_json(output_dir / "tce_manual_review_sample_set_20260308.json", samples)
    write_json(output_dir / "tce_manual_review_sample_summary_20260308.json", summary)
    write_csv(
        output_dir / "tce_manual_review_sample_set_20260308.csv",
        samples,
        [
            "sample_unit_id",
            "stage",
            "bucket",
            "checkpoint_id",
            "state_key",
            "state_category",
            "changed_status",
            "source_reason",
            "review_checklist_ref",
            "risk_tag",
        ],
    )
    write_csv(
        output_dir / "tce_manual_review_sheet_20260308.csv",
        review_rows,
        [
            "sample_id",
            "stage",
            "bucket",
            "checkpoint_id",
            "state_key",
            "state_category",
            "changed_status",
            "source_reason",
            "human_verdict",
            "issue_type",
            "severity",
            "notes",
            "action_recommendation",
        ],
    )
    (output_dir / "tce_manual_review_plan_20260308.md").write_text(
        review_plan_markdown(),
        encoding="utf-8",
    )

    purpose_manifest = []
    for purpose_name, predicate in PURPOSE_GROUPS:
        purpose_samples = [sample for sample in samples if predicate(sample)]
        purpose_sheet = [
            review_row_by_unit[sample["sample_unit_id"]]
            for sample in purpose_samples
            if sample["sample_unit_id"] in review_row_by_unit
        ]
        write_json(
            output_dir / f"{purpose_name}_sample_set_20260308.json",
            purpose_samples,
        )
        write_csv(
            output_dir / f"{purpose_name}_sample_set_20260308.csv",
            purpose_samples,
            [
                "sample_unit_id",
                "stage",
                "bucket",
                "checkpoint_id",
                "state_key",
                "state_category",
                "changed_status",
                "source_reason",
                "review_checklist_ref",
                "risk_tag",
            ],
        )
        write_csv(
            output_dir / f"{purpose_name}_review_sheet_20260308.csv",
            purpose_sheet,
            [
                "sample_id",
                "stage",
                "bucket",
                "checkpoint_id",
                "state_key",
                "state_category",
                "changed_status",
                "source_reason",
                "human_verdict",
                "issue_type",
                "severity",
                "notes",
                "action_recommendation",
            ],
        )
        purpose_manifest.append(
            {
                "purpose_name": purpose_name,
                "sample_count": len(purpose_samples),
                "json_path": str(output_dir / f"{purpose_name}_sample_set_20260308.json"),
                "csv_path": str(output_dir / f"{purpose_name}_sample_set_20260308.csv"),
                "review_sheet_path": str(output_dir / f"{purpose_name}_review_sheet_20260308.csv"),
            }
        )
    write_json(
        output_dir / "tce_manual_review_purpose_manifest_20260308.json",
        purpose_manifest,
    )

    print("saved", output_dir)
    print("selected_samples", len(samples))


if __name__ == "__main__":
    main()
