"""Loading and selecting versioned workflow definitions."""

from __future__ import annotations

from pathlib import Path

from .models import WorkflowDefinition


class WorkflowNotFoundError(LookupError):
    """The requested workflow is not in the configured registry."""


class WorkflowRegistry:
    def __init__(self, definitions: list[WorkflowDefinition]) -> None:
        self._definitions = {definition.id: definition for definition in definitions}
        if not self._definitions:
            raise ValueError("At least one workflow definition is required")
        if len(self._definitions) != len(definitions):
            raise ValueError("Workflow definition ids must be unique")

    @classmethod
    def from_directory(cls, directory: Path) -> WorkflowRegistry:
        definitions = [
            WorkflowDefinition.model_validate_json(path.read_text())
            for path in sorted(directory.glob("*.json"))
        ]
        return cls(definitions)

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._definitions[workflow_id]
        except KeyError as exc:
            raise WorkflowNotFoundError(workflow_id) from exc

    def list(self) -> list[WorkflowDefinition]:
        return list(self._definitions.values())
