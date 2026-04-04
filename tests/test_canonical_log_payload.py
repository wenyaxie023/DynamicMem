import json
import unittest

from generation.rag.rag_tce import _log_search_text
from tce_core.pipeline import to_log_text


class CanonicalLogPayloadTest(unittest.TestCase):
    def test_to_log_text_preserves_raw_log_object(self) -> None:
        log = {
            "app_log_id": "log_00001",
            "timestamp": "2024-03-31 14:30:00",
            "app_name": "CoffeeApp",
            "api_name": "GetPreference",
            "request": {"user": "u1"},
            "response": {"favorite": "espresso"},
            "extra_field": {"nested": True},
        }

        self.assertEqual(json.loads(to_log_text(log)), log)

    def test_rag_search_text_uses_canonical_raw_log_payload(self) -> None:
        log = {
            "app_log_id": "log_00002",
            "timestamp": "2024-03-31 15:00:00",
            "app_name": "Calendar",
            "api_name": "CreateEvent",
            "request": {"title": "Walk"},
            "response": {"start_time": "06:30"},
            "extra_field": "keep-me",
        }

        self.assertEqual(json.loads(_log_search_text(log)), log)


if __name__ == "__main__":
    unittest.main()
