#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from data_construction.download_world_backgrounds import resolve_target_user_dirs


class DownloadWorldBackgroundsTest(unittest.TestCase):
    def test_resolve_target_user_dirs_from_sample_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample.jsonl"
            records = [
                {"user_id": "user_001", "persona": "alpha"},
                {"user_id": "user_002", "persona": "beta"},
                {"user_id": "user_003", "persona": "gamma"},
            ]
            with sample_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            user_dirs = resolve_target_user_dirs(
                explicit_users=None,
                sample_path=sample_path,
                user_count=2,
                sample_size=10,
                seed=42,
                user_index=None,
            )

            self.assertEqual(user_dirs, ["001_user_001", "002_user_002"])

    def test_resolve_target_user_dirs_with_user_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample.jsonl"
            records = [
                {"user_id": "user_001", "persona": "alpha"},
                {"user_id": "user_002", "persona": "beta"},
            ]
            with sample_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            user_dirs = resolve_target_user_dirs(
                explicit_users=None,
                sample_path=sample_path,
                user_count=2,
                sample_size=10,
                seed=42,
                user_index=2,
            )

            self.assertEqual(user_dirs, ["002_user_002"])

    def test_resolve_target_user_dirs_prefers_explicit_users(self):
        user_dirs = resolve_target_user_dirs(
            explicit_users=["001_user_001", "010_user_010"],
            sample_path="unused.jsonl",
            user_count=10,
            sample_size=10,
            seed=42,
            user_index=None,
        )

        self.assertEqual(user_dirs, ["001_user_001", "010_user_010"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
