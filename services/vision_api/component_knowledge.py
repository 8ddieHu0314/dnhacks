from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RetrievedComponent

_TOKEN = re.compile(r"[a-z0-9]+")

def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(value.lower())

@dataclass(frozen=True, slots=True)
class ComponentMatch:
    record: dict[str, Any]
    score: float

class ComponentKnowledgeBase:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("Component knowledge base must contain at least one record")
        self._records = records
        self._by_id = {str(record["id"]): record for record in records}
        if len(self._by_id) != len(records):
            raise ValueError("Component ids must be unique")
        self._search_text = [json.dumps(record, ensure_ascii=True).lower() for record in records]

    @classmethod
    def from_path(cls, path: Path) -> ComponentKnowledgeBase:
        try:
            records = json.loads(path.read_text())["components"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load component knowledge base at {path}") from exc
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("components.json must contain a components array of objects")
        return cls(records)

    def record_for(self, component_id: str) -> dict[str, Any] | None:
        return self._by_id.get(component_id)

    def search(self, query: str, *, limit: int = 3) -> list[ComponentMatch]:
        terms = _tokens(query)
        if limit < 1 or not terms:
            return []
        scores = [sum(text.count(term) for term in terms) for text in self._search_text]
        exact = [index for index, record in enumerate(self._records) if record["id"] in query.lower()]
        order = exact + sorted((i for i, score in enumerate(scores) if score and i not in exact), key=scores.__getitem__, reverse=True)
        return [ComponentMatch(self._records[index], max(float(scores[index]), 100.0 if index in exact else 0.0)) for index in order[:limit]]

    @staticmethod
    def summaries(matches: list[ComponentMatch]) -> list[RetrievedComponent]:
        summaries = []
        for match in matches:
            record = match.record
            confidence = str(record.get("confidence", "low")).lower()
            summaries.append(RetrievedComponent(
                component_id=str(record["id"]), canonical_name=str(record["canonical_name"]),
                data_confidence=confidence if confidence in {"high", "medium", "low"} else "low",
                retrieval_score=max(0.0, match.score),
            ))
        return summaries

    @staticmethod
    def prompt_records(matches: list[ComponentMatch]) -> list[dict[str, Any]]:
        fields = ("identity", "function", "visual_identification", "pins", "electrical",
                  "wiring_to_uno", "safety", "troubleshooting")
        records = []
        for match in matches:
            record, details = match.record, match.record.get("details", {})
            records.append({
                "id": record["id"], "canonical_name": record["canonical_name"],
                "data_confidence": record.get("confidence", "low"),
                "source_count": record.get("source_count", 0),
                **{field: details.get(field, {}) for field in fields},
            })
        return records
