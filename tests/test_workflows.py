import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from vision_api.main import app
from vision_api.workflows import WorkflowRegistry


class WorkflowRegistryTests(unittest.TestCase):
    def test_loads_the_bundled_generic_workflow(self) -> None:
        directory = Path(__file__).parents[1] / "services/vision_api/workflow_definitions"
        workflow = WorkflowRegistry.from_directory(directory).get("generic-field-support")

        self.assertEqual(workflow.version, "0.1.0")
        self.assertEqual(len(workflow.checkpoints), 2)


class WorkflowApiTests(unittest.TestCase):
    def test_lists_workflows_and_binds_one_to_a_session(self) -> None:
        with TestClient(app) as client:
            workflows = client.get("/v1/workflows")
            self.assertEqual(workflows.status_code, 200)
            self.assertIn("generic-field-support", [item["id"] for item in workflows.json()])

            created = client.post("/v1/sessions", json={"workflow_id": "generic-field-support"})
            self.assertEqual(created.status_code, 201)
            self.assertEqual(created.json()["workflow_version"], "0.1.0")

            unknown = client.post("/v1/sessions", json={"workflow_id": "not-a-workflow"})
            self.assertEqual(unknown.status_code, 422)
