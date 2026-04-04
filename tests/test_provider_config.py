#!/usr/bin/env python3
import os
import unittest
from unittest import mock

from generation.common.provider_config import (
    apply_openai_compat_env,
    resolve_openai_compatible_credentials,
)


class ProviderConfigTest(unittest.TestCase):
    def test_resolve_openai_credentials_does_not_fallback_to_azure(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_BASE_URL": "https://azure.example",
            },
            clear=False,
        ):
            api_key, base_url = resolve_openai_compatible_credentials("openai", require_api_key=True)
            self.assertEqual(api_key, "openai-key")
            self.assertIsNone(base_url)

    def test_resolve_azure_credentials_does_not_fallback_to_openai(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://openai.example",
                "AZURE_OPENAI_API_KEY": "azure-key",
            },
            clear=False,
        ):
            api_key, base_url = resolve_openai_compatible_credentials("azure", require_api_key=True)
            self.assertEqual(api_key, "azure-key")
            self.assertIsNone(base_url)

    def test_apply_openai_compat_env_clears_stale_values(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "stale-key",
                "OPENAI_BASE_URL": "https://stale.example",
            },
            clear=False,
        ):
            apply_openai_compat_env(None, None)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_BASE_URL", os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
