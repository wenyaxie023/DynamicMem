import unittest

from generation.common.llm_client import LLMClient


class LLMClientSamplingTest(unittest.TestCase):
    def _make_client(self, *, provider: str, model_name: str, temperature=0.0, top_p=1.0):
        client = object.__new__(LLMClient)
        client.provider = provider
        client.model_name = model_name
        client.temperature = temperature
        client.top_p = top_p
        client.top_k = None
        return client

    def test_gpt5_openai_family_omits_sampling_kwargs(self):
        client = self._make_client(provider="azure", model_name="gpt-5-mini")
        self.assertEqual(client._response_sampling_kwargs(), {})

    def test_non_gpt5_keeps_sampling_kwargs(self):
        client = self._make_client(provider="azure", model_name="gpt-4o-mini")
        self.assertEqual(
            client._response_sampling_kwargs(),
            {"temperature": 0.0, "top_p": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
