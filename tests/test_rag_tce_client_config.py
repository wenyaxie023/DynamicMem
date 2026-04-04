#!/usr/bin/env python3
import os
import unittest
from unittest import mock

from generation.rag import rag_tce


class RagTceClientConfigTest(unittest.TestCase):
    def test_build_openai_client_uses_openai_env_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_BASE_URL": "https://azure.example/openai/v1",
            },
            clear=False,
        ), mock.patch("generation.rag.rag_tce.OpenAI") as openai_cls:
            rag_tce._build_openai_client("openai")

        openai_cls.assert_called_once_with(
            api_key="openai-key",
            base_url="https://openai.example/v1",
        )

    def test_build_openai_client_uses_azure_env_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_BASE_URL": "https://azure.example/openai/v1",
            },
            clear=False,
        ), mock.patch("generation.rag.rag_tce.OpenAI") as openai_cls:
            rag_tce._build_openai_client("azure")

        openai_cls.assert_called_once_with(
            api_key="azure-key",
            base_url="https://azure.example/openai/v1",
            default_headers={"api-key": "azure-key"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
