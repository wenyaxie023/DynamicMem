#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


google_module = ModuleType("google")
google_generativeai = ModuleType("google.generativeai")
google_api_core = ModuleType("google.api_core")
google_api_core_exceptions = ModuleType("google.api_core.exceptions")

google_module.generativeai = google_generativeai
google_module.api_core = google_api_core
google_api_core.exceptions = google_api_core_exceptions

sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.generativeai", google_generativeai)
sys.modules.setdefault("google.api_core", google_api_core)
sys.modules.setdefault("google.api_core.exceptions", google_api_core_exceptions)

jinja2_module = ModuleType("jinja2")


class _FakeTemplate:
    def __init__(self, text: str):
        self.text = text

    def render(self, **_: object) -> str:
        return self.text


jinja2_module.Template = _FakeTemplate
sys.modules.setdefault("jinja2", jinja2_module)

from data_construction.stages.stage1_dynamic_profile import Domain, DynamicProfileStage


class Stage1WorldBackgroundModeTest(unittest.TestCase):
    def test_require_existing_loads_aggregated_world_backgrounds_and_syncs_domain_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            payload = {
                "Family Close Relationships": {
                    "world_backgrounds": [{"window_id": "w0", "background": "cached"}],
                    "combined_background": "cached aggregate background",
                }
            }
            (output_dir / "world_backgrounds.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            stage = DynamicProfileStage(
                llm_client=object(),
                output_dir=output_dir,
                debug_mode=False,
                world_bg_mode="require_existing",
            )

            world_backgrounds, usage = stage._generate_world_backgrounds(
                user_basic_profile={},
                domains=[
                    Domain(
                        domain_name="Family Close Relationships",
                        domain_scope_definition="test scope",
                    )
                ],
            )

            self.assertEqual(usage, {})
            self.assertEqual(
                world_backgrounds["Family Close Relationships"]["combined_background"],
                "cached aggregate background",
            )
            domain_cache = output_dir / "family_close_relationships_world_background.json"
            self.assertTrue(domain_cache.exists())
            cache_payload = json.loads(domain_cache.read_text(encoding="utf-8"))
            self.assertEqual(
                cache_payload["combined_background"],
                "cached aggregate background",
            )

    def test_require_existing_fails_fast_when_world_backgrounds_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = DynamicProfileStage(
                llm_client=object(),
                output_dir=Path(tmpdir),
                debug_mode=False,
                world_bg_mode="require_existing",
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                stage._generate_world_backgrounds(
                    user_basic_profile={},
                    domains=[
                        Domain(
                            domain_name="Family Close Relationships",
                            domain_scope_definition="test scope",
                        )
                    ],
                )

            message = str(ctx.exception)
            self.assertIn("world_backgrounds.json", message)
            self.assertIn("family_close_relationships_world_background.json", message)
            self.assertIn("--world-bg-mode generate_if_missing", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
