import json
import unittest

from generation.load_dataset import MemBenchEvent, build_membench_memory_from_event


class LoadDatasetPayloadTest(unittest.TestCase):
    def test_build_membench_memory_from_event_preserves_raw_log_payload(self) -> None:
        event = MemBenchEvent(
            event_id="log_00001",
            timestamp="2024-03-31 14:30:00",
            app_name="CoffeeApp",
            api_name="GetPreference",
            request={"user": "u1"},
            response={"favorite": "espresso"},
        )

        content, time_str = build_membench_memory_from_event(event)

        self.assertEqual(time_str, "2024-03-31 14:30:00")
        self.assertEqual(
            json.loads(content),
            {
                "app_log_id": "log_00001",
                "timestamp": "2024-03-31 14:30:00",
                "app_name": "CoffeeApp",
                "api_name": "GetPreference",
                "request": {"user": "u1"},
                "response": {"favorite": "espresso"},
            },
        )
        self.assertNotIn("memory_timestamp", json.loads(content))


if __name__ == "__main__":
    unittest.main()
