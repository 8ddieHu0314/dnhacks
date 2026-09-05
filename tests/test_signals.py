import unittest

from fastapi.testclient import TestClient

from vision_api.main import app


class AdvisorySignalApiTests(unittest.TestCase):
    def test_records_a_field_observation_without_an_isolation_decision(self) -> None:
        with TestClient(app) as client:
            session = client.post("/v1/sessions").json()["session_id"]
            response = client.post(
                f"/v1/sessions/{session}/advisory-field-signals",
                json={"level": 8.5, "state": "field_detected"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "field_detected")
        self.assertNotIn("safe", response.json())
