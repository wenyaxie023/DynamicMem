#!/usr/bin/env python3
import json
import unittest
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
    build_stage1_pool,
    build_state_completion_pool,
    select_manual_review_samples,
)


class TceManualReviewSamplingAcceptance(unittest.TestCase):
    def test_selection_matches_planned_counts(self):
        validated = Path(
            "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/"
            "tce_benchmark_state_validated_20260308_applycrit_all.json"
        )
        task_packs = Path(
            "data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/"
            "tce_benchmark_task_packs_20260308_applycrit_all.json"
        )
        validated_payload = json.loads(validated.read_text(encoding="utf-8"))
        task_pack_payload = json.loads(task_packs.read_text(encoding="utf-8"))

        samples = select_manual_review_samples(
            stage1_pool=build_stage1_pool(validated_payload),
            state_completion_pool=build_state_completion_pool(task_pack_payload),
            change_tracking_pool=build_change_tracking_pool(task_pack_payload),
            apply_pool=build_apply_pool(task_pack_payload),
            seed=FIXED_SEED,
        )
        self.assertEqual(len(samples), 48)
        counts = Counter((sample["stage"], sample["bucket"]) for sample in samples)
        self.assertEqual(counts[("stage1", "l1_fail")], 5)
        self.assertEqual(counts[("stage1", "l2_fail")], 5)
        self.assertEqual(counts[("stage1", "pass")], 8)
        self.assertEqual(counts[("stage2_state_completion", "computed")], 4)
        self.assertEqual(counts[("stage2_state_completion", "reused")], 4)
        self.assertEqual(counts[("stage2_change_tracking", "changed")], 8)
        self.assertEqual(counts[("stage2_apply", "accepted")], 8)
        self.assertEqual(counts[("stage2_apply", "discarded")], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
