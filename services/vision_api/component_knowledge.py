from __future__ import annotations

import base64
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from .models import (ComponentGuidance, ComponentIdentification, DebugStep,
                     EvidenceCheckResult, Frame, RetrievedComponent,
                     VisionAnalysis, VisionOutput)
from .vlm import VisionModelError, encoded_image_url, instruction_for

_TOKEN = re.compile(r"[a-z0-9]+")

def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(value.lower())


def _searchable_record(record: dict[str, Any]) -> str:
    details = record.get("details", {})
    selected = {key: record.get(key, "") for key in ("id", "canonical_name", "name_on_kit", "category", "pins", "interface", "function")}
    selected["details"] = {key: details.get(key, {}) for key in ("identity", "function", "visual_identification")}
    return json.dumps(selected, ensure_ascii=True).lower()

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
        self._search_text = [_searchable_record(record) for record in records]
        self._term_counts = [Counter(_tokens(text)) for text in self._search_text]
        document_frequency = Counter(term for counts in self._term_counts for term in counts)
        count = len(self._term_counts)
        self._idf = {term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5)) for term, frequency in document_frequency.items()}

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
        terms = set(_tokens(query))
        if limit < 1 or not terms:
            return []
        scores = [sum(self._idf.get(term, 0.0) * counts[term] / (counts[term] + 1) for term in terms) for counts in self._term_counts]
        exact = [index for index, record in enumerate(self._records) if record["id"] in query.lower()]
        named = [index for index, record in enumerate(self._records) if index not in exact
                 and terms.intersection(_tokens(str(record["id"])))]
        canonical = sorted((index for index, record in enumerate(self._records) if index not in exact + named
                            and terms.intersection(_tokens(str(record["canonical_name"])))),
                           key=lambda index: len(terms.intersection(_tokens(str(self._records[index]["canonical_name"])))), reverse=True)
        selected = set(exact + named + canonical)
        order = exact + named + canonical + sorted((i for i, score in enumerate(scores) if score and i not in selected), key=scores.__getitem__, reverse=True)
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
        fields = ("identity", "function", "visual_identification", "pins", "electrical", "key_specs",
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


class SceneDescription(BaseModel):
    scene_description: str = Field(min_length=1, max_length=1_500)
    visible_text: list[str] = Field(default_factory=list, max_length=20)
    likely_component_terms: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("visible_text", "likely_component_terms", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return list(value.values())
        return value

    def query(self, user_request: str | None) -> str:
        return " ".join([self.scene_description, *self.visible_text, *self.likely_component_terms, user_request or ""])


class ComponentKnowledgeVLM:
    """Use a VLM only after a frame description narrows the component context."""

    name = "component-knowledge-vlm"

    def __init__(self, *, base_url: str | None, api_key: str | None, model: str,
                 timeout_seconds: float, knowledge_base: ComponentKnowledgeBase, top_k: int = 3,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not base_url or not model:
            raise ValueError("VLM_BASE_URL and VLM_MODEL are required for component_knowledge")
        if not 1 <= top_k <= 5:
            raise ValueError("COMPONENT_KNOWLEDGE_TOP_K must be between 1 and 5")
        self._base_url, self._api_key, self._model = base_url.rstrip("/"), api_key, model
        self._timeout, self._knowledge_base, self._top_k = timeout_seconds, knowledge_base, top_k
        self._transport = transport

    def _body(self, *, system: str, text: str, frame: Frame) -> dict[str, Any]:
        return {"model": self._model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": encoded_image_url(frame)}},
                ]}]}

    async def _complete(self, *, system: str, text: str, frame: Frame) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(f"{self._base_url}/chat/completions", headers=headers,
                                             json=self._body(system=system, text=text, frame=frame))
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if item.get("type") == "text")
            if not isinstance(content, str):
                raise TypeError("completion content was not text")
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise TypeError("completion JSON was not an object")
            return payload
        except httpx.HTTPError as exc:
            raise VisionModelError(f"VLM request failed: {exc}") from exc
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisionModelError("VLM response did not contain a valid JSON object") from exc

    @staticmethod
    def _scene_prompt(user_request: str | None) -> str:
        return """Describe only what is visible in this electronics-kit image. Return JSON with
scene_description, visible_text, and likely_component_terms. Transcribe markings verbatim;
do not identify a part with certainty or give wiring advice. User request: """ + (user_request or "none")

    @staticmethod
    def _guidance_prompt(frame: Frame, records: list[dict[str, Any]]) -> str:
        ids = [record["id"] for record in records]
        return f"""You are an educational electronics-kit copilot. Return only JSON matching
VisionOutput. Use only the candidate records below; allowed component ids are {ids}. If the
image cannot distinguish them, identify none and ask a short clarifying question. Never invent
a specification, component id, or connection. Image evidence cannot prove a wire is electrically
connected, correct, or safe: state uncertainty and propose a human-confirmed check. Do not direct
mains/high-voltage work. Every proposed action requires_confirmation=true. Put component evidence,
wiring feedback, questions, and caveats in analysis.component_guidance. Keep non-debug lists to three
brief items or fewer. If the user request starts DEBUG MODE, set mode=debug and add debug_guidance:
state the apparent problem, give ordered reversible steps with the reason and expected evidence,
and mark every step requires_confirmation=true. List only remaining steps and put the current next
action first. Preserve a prior plan unless new visual or user-reported evidence changes it. If a
connection, marking, polarity, or rail break is not visible, do not guess: request one precise
alternative view in visual_clarification and explain how to frame it. Treat continuity and voltage
measurements as user-reported evidence, never visual facts.\n\nWorkflow:\n{instruction_for(frame)}
\n\nUser request: {frame.metadata.user_request or 'identify the part and give safe context.'}
\n\nCandidate records:\n{json.dumps(records, ensure_ascii=True)}"""

    @staticmethod
    def _ground(output: VisionOutput, matches: list[ComponentMatch]) -> VisionOutput:
        summaries = ComponentKnowledgeBase.summaries(matches)
        allowed_ids = {summary.component_id for summary in summaries}
        analysis = output.analysis or VisionAnalysis(summary="Component candidates are available for review.")
        guidance = analysis.component_guidance or ComponentGuidance()
        caveats = list(guidance.data_caveats)
        for summary in summaries:
            if summary.data_confidence != "high":
                caveat = f"{summary.canonical_name} has {summary.data_confidence}-confidence kit data; confirm critical values."
                if caveat not in caveats:
                    caveats.append(caveat)
        guidance = guidance.model_copy(update={
            "retrieved_components": summaries,
            "identified_components": [item for item in guidance.identified_components if item.component_id in allowed_ids],
            "data_caveats": caveats[:5],
        })
        return output.model_copy(update={"analysis": analysis.model_copy(update={"component_guidance": guidance})})

    async def _matches_for(self, frame: Frame) -> list[ComponentMatch]:
        matches = self._knowledge_base.search(frame.metadata.user_request or "", limit=self._top_k)
        if matches:
            return matches
        scene = SceneDescription.model_validate(await self._complete(
            system=self._scene_prompt(frame.metadata.user_request), text="Describe this frame for retrieval.", frame=frame,
        ))
        return self._knowledge_base.search(scene.query(frame.metadata.user_request), limit=self._top_k)

    async def analyze(self, frame: Frame) -> VisionOutput:
        matches = await self._matches_for(frame)
        records = self._knowledge_base.prompt_records(matches)
        output = VisionOutput.model_validate(await self._complete(
            system=self._guidance_prompt(frame, records), text="Give grounded component guidance.", frame=frame,
        ))
        return self._ground(output, matches)


class AnthropicComponentKnowledgeVLM(ComponentKnowledgeVLM):
    """Native Messages API variant of the grounded component workflow."""

    name = "anthropic-component-knowledge-vlm"

    def __init__(self, *, base_url: str | None, **kwargs: Any) -> None:
        kwargs["timeout_seconds"] = max(float(kwargs["timeout_seconds"]), 45.0)
        super().__init__(base_url=base_url or "https://api.anthropic.com", **kwargs)

    def _body(self, *, system: str, text: str, frame: Frame,
              schema: dict[str, Any] | None = None) -> dict[str, Any]:
        image = base64.b64encode(frame.image_bytes).decode("ascii")
        body = {"model": self._model, "max_tokens": 1400, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "source": {"type": "base64",
                     "media_type": f"image/{frame.metadata.encoding}", "data": image}},
                ]}]}
        if schema:
            body["tools"] = [{"name": "submit_result", "description": "Return the requested structured result.", "input_schema": schema}]
            body["tool_choice"] = {"type": "tool", "name": "submit_result"}
        return body

    @staticmethod
    def _json_object(content: str) -> dict[str, Any]:
        start, end = content.find("{"), content.rfind("}")
        payload = json.loads(content[start:end + 1])
        if not isinstance(payload, dict):
            raise TypeError("completion JSON was not an object")
        return payload

    @staticmethod
    def _guidance_schema(ids: list[str]) -> dict[str, Any]:
        string_list = {"type": "array", "items": {"type": "string"}, "maxItems": 2}
        identification = {"type": "object", "properties": {
            "component_id": {"type": "string", "enum": ids}, "confidence": {"type": "number"},
            "observed_evidence": string_list, "uncertainty": {"type": ["string", "null"]},
        }, "required": ["component_id", "confidence"]}
        guidance = {"type": "object", "properties": {
            "identified_components": {"type": "array", "items": identification},
            "wiring_feedback": string_list, "clarifying_questions": string_list, "data_caveats": string_list,
        }, "required": []}
        step = {"type": "object", "properties": {
            "instruction": {"type": "string"}, "reason": {"type": "string"},
            "expected_evidence": {"type": "string"}, "safety_note": {"type": ["string", "null"]},
            "requires_confirmation": {"type": "boolean"},
        }, "required": ["instruction", "reason", "expected_evidence", "requires_confirmation"]}
        clarification = {"type": "object", "properties": {
            "target": {"type": "string"}, "requested_view": {"type": "string"}, "reason": {"type": "string"},
        }, "required": ["target", "requested_view", "reason"]}
        connection = {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"},
            "status": {"type": "string", "enum": ["visible", "inferred", "unclear"]},
            "evidence": {"type": "string"},
        }, "required": ["source", "target", "status", "evidence"]}
        circuit = {"type": "object", "properties": {
            "summary": {"type": "string"},
            "components": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "connections": {"type": "array", "items": connection, "maxItems": 24},
            "unknowns": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": ["summary", "components", "connections", "unknowns", "confidence"]}
        comparison = {"type": "object", "properties": {
            "fits_purpose": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "explanation": {"type": "string"},
            "matches": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "mismatches": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "missing_or_unclear": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "safety_issues": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        }, "required": ["fits_purpose", "explanation", "matches", "mismatches",
                        "missing_or_unclear", "safety_issues"]}
        check = {"type": "object", "properties": {
            "check_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "pass", "fail"]},
            "evidence_source": {"type": "string", "enum": ["visual", "user_report", "measurement", "unknown"]},
            "evidence": {"type": "string"},
        }, "required": ["check_id", "status", "evidence_source", "evidence"]}
        debug = {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["needs_context", "in_progress", "ready_to_test", "resolved"]},
            "phase": {"type": "string", "enum": ["inspect_unpowered", "verify_unpowered", "ready_to_power", "test_powered", "resolved"]},
            "safe_to_energize": {"type": "boolean"},
            "problem": {"type": "string"}, "steps": {"type": "array", "items": step, "maxItems": 8},
            "checks": {"type": "array", "items": check, "maxItems": 12},
            "visual_clarification": {"anyOf": [clarification, {"type": "null"}]},
            "observed_circuit": circuit, "comparison": comparison,
        }, "required": ["status", "phase", "safe_to_energize", "problem", "checks", "steps", "visual_clarification",
                        "observed_circuit", "comparison"]}
        analysis = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["identification", "debug"]},
            "summary": {"type": "string"}, "observations": string_list, "safety_alerts": string_list,
            "component_guidance": guidance, "debug_guidance": debug},
            "required": ["summary", "component_guidance"]}
        return {"type": "object", "properties": {"analysis": analysis}, "required": ["analysis"]}

    async def _complete(self, *, system: str, text: str, frame: Frame,
                        schema: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(f"{self._base_url}/v1/messages", headers=headers,
                                             json=self._body(system=system, text=text, frame=frame, schema=schema))
                if response.is_error:
                    raise VisionModelError(
                        f"Anthropic request failed ({response.status_code}): {response.text[:500]}"
                    )
            response_body = response.json()
            blocks = response_body["content"]
            for block in blocks:
                if block.get("type") == "tool_use" and block.get("name") == "submit_result":
                    return block["input"]
            content = "".join(block.get("text", "") for block in blocks
                              if block.get("type") == "text")
            if not content:
                raise TypeError(f"completion contained no text blocks: {[block.get('type') for block in blocks]}")
            return self._json_object(content)
        except httpx.HTTPError as exc:
            raise VisionModelError(f"Anthropic request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisionModelError("Anthropic response did not contain a valid JSON object") from exc

    async def analyze(self, frame: Frame) -> VisionOutput:
        matches = self._knowledge_base.search(frame.metadata.user_request or "", limit=self._top_k)
        if not matches:
            scene = SceneDescription.model_validate(await self._complete(
                system=self._scene_prompt(frame.metadata.user_request), text="Describe this frame for retrieval.",
                frame=frame, schema=SceneDescription.model_json_schema(),
            ))
            matches = self._knowledge_base.search(scene.query(frame.metadata.user_request), limit=self._top_k)
        records = self._knowledge_base.prompt_records(matches)
        output = VisionOutput.model_validate(await self._complete(
            system=self._guidance_prompt(frame, records),
            text="Give grounded component guidance.", frame=frame,
            schema=self._guidance_schema([record["id"] for record in records]),
        ))
        return self._ground(output, matches)


class AnthropicCatalogIdentificationVLM(AnthropicComponentKnowledgeVLM):
    """Fast catalog identification patterned after the glasses relay receiver."""

    name = "anthropic-catalog-identification-vlm"

    def __init__(self, *, base_url: str | None, **kwargs: Any) -> None:
        super().__init__(base_url=base_url, **kwargs)
        self._debugger = AnthropicComponentKnowledgeVLM(base_url=base_url, **kwargs)
        self._debug_sessions: set[str] = set()
        self._debug_components: dict[str, list[str]] = {}
        self._debug_history: dict[str, dict[str, Any]] = {}
        self._debug_evidence: dict[str, dict[str, Any]] = {}
        target_path = Path(__file__).with_name("circuit_definitions") / "motor-fan-button-relay.json"
        self._target_circuit = json.loads(target_path.read_text())
        lines = []
        for record in self._knowledge_base._records:
            visual = record.get("details", {}).get("visual_identification", {})
            marking = visual.get("printed_text_to_look_for") or record.get("mpn") or "none"
            look = " ".join(str(value) for value in (visual.get("shape_and_size"), visual.get("color_and_markings")) if value)
            line = f"- {record['id']}: {record['canonical_name'][:50]} | text: {str(marking)[:70]} | looks: {look[:130] or 'n/a'}"
            confused = visual.get("easily_confused_with") or []
            if confused:
                line += f" | not: {str(confused[0])[:50]}"
            lines.append(line)
        self._catalog_index = "\n".join(lines)

    def _body(self, *, system: str, text: str, frame: Frame,
              schema: dict[str, Any] | None = None) -> dict[str, Any]:
        body = super()._body(system=system, text=text, frame=frame, schema=schema)
        body["max_tokens"] = 160
        body.pop("temperature", None)
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return body

    def _system_prompt(self) -> str:
        return ("Identify the electronic component in view against this catalog. Read printed markings first; "
                "never invent specifications. Include up to five clearly visible catalog ids. Reply with JSON only: "
                "{\"id\": \"primary catalog id or null\", \"visible_ids\": [\"catalog id\"], \"confidence\": number, "
                "\"name\": \"short name\", \"evidence\": \"12 words max\"}.\n\nCatalog:\n" + self._catalog_index)

    def _enforce_power_gate(self, analysis: VisionAnalysis, session_id: str = "") -> VisionAnalysis:
        debug = analysis.debug_guidance
        if debug is None:
            return analysis
        definitions = self._target_circuit["verification_checks"]
        allowed = {item["id"] for item in definitions}
        reported = {item.check_id: item for item in debug.checks if item.check_id in allowed}
        submitted = self._debug_evidence.get(session_id, {})
        def accepted(item: dict[str, Any], result: EvidenceCheckResult | None) -> bool:
            source = item["required_evidence"]
            return bool(result and result.status == "pass" and result.evidence_source == source
                        and (source == "visual" or submitted.get(item["id"], {}).get("evidence_source") == source))
        checks = []
        for item in definitions:
            result = reported.get(item["id"])
            if result is None or (result.status == "pass" and not accepted(item, result)):
                result = EvidenceCheckResult(check_id=item["id"], status="pending", evidence_source="unknown",
                    evidence=f"Waiting for accepted {item['required_evidence']} evidence.")
            checks.append(result)
        blocking = [item for item in definitions if item["blocks_power"]]
        safe = all(accepted(item, reported.get(item["id"])) for item in blocking)
        steps = debug.steps
        if not safe:
            power_action = r"\b(connect (?:usb )?power|turn on|power on|apply power|energize)\b"
            steps = [step for step in steps if not re.search(power_action, step.instruction.lower())]
            if not steps:
                pending = next(item for item in blocking if not reported.get(item["id"])
                               or reported[item["id"]].status != "pass")
                steps = [DebugStep(instruction=f"Keep power disconnected and complete check {pending['id']}.",
                    reason="The server blocks energized testing until every unpowered gate passes.",
                    expected_evidence=pending["pass_condition"])]
        phase = debug.phase if safe or debug.phase in {"inspect_unpowered", "verify_unpowered"} else "verify_unpowered"
        return analysis.model_copy(update={"debug_guidance": debug.model_copy(update={
            "checks": checks, "safe_to_energize": safe, "phase": phase, "steps": steps})})

    async def _debug_breadboard(self, frame: Frame) -> VisionOutput:
        request = frame.metadata.user_request or "Find visible wiring problems and give the next safe check."
        metadata = frame.metadata.model_copy(update={
            "user_request": f"DEBUG MODE: Breadboard jumper-wire circuit. {request}",
        })
        ids = self._debug_components.get(frame.session_id) or ["breadboard-830", "jumper-wire"]
        matches = [ComponentMatch(record, 100.0) for item in ids
                   if (record := self._knowledge_base.record_for(item)) is not None]
        records = self._knowledge_base.prompt_records(matches)
        debug_frame = replace(frame, metadata=metadata)
        previous = self._debug_history.get(frame.session_id)
        task = "Build or update the debug plan."
        evidence = self._debug_evidence.setdefault(frame.session_id, {})
        evidence.update({item.check_id: item.model_dump(mode="json") for item in frame.metadata.reported_evidence})
        if evidence:
            task += " Wearer-submitted evidence: " + json.dumps(list(evidence.values()), ensure_ascii=True)
        if previous:
            task += " Previous plan: " + json.dumps(previous, ensure_ascii=True)
        system = self._debugger._guidance_prompt(debug_frame, records) + f"""

Configured target circuit:
{json.dumps(self._target_circuit, ensure_ascii=True)}

Before suggesting a fix, reconstruct the observed circuit as components and pairwise connections.
Mark every edge visible, inferred, or unclear and cite image evidence. Then compare that graph with
every configured target connection and safety invariant. Do not treat a component's presence as proof
of a connection. Use visual_clarification when an essential endpoint, relay pin, or breadboard row is
not readable. Return one checks entry for every configured verification check. A check passes only
when this frame or the user's current report supplies its required evidence type; never convert a
visual guess into user_report or measurement evidence. Keep the phase unpowered and
safe_to_energize=false until every blocks_power check passes. Do not propose connecting power before
that gate opens. The observed_circuit and comparison objects are mandatory even when uncertain."""
        output = VisionOutput.model_validate(await self._debugger._complete(
            system=system, text=task,
            frame=debug_frame, schema=self._debugger._guidance_schema(ids),
        ))
        output = self._debugger._ground(output, matches)
        analysis = output.analysis or VisionAnalysis(summary="Show the breadboard wiring clearly.")
        analysis = self._enforce_power_gate(analysis, frame.session_id)
        if analysis.debug_guidance is not None:
            self._debug_history[frame.session_id] = analysis.debug_guidance.model_dump(mode="json")
        return output.model_copy(update={"analysis": analysis.model_copy(update={"mode": "debug"})})

    async def analyze(self, frame: Frame) -> VisionOutput:
        if frame.session_id in self._debug_sessions:
            request = (frame.metadata.user_request or "").lower()
            if any(command in request for command in ("exit debug mode", "stop debugging", "identification mode")):
                self._debug_sessions.discard(frame.session_id)
                self._debug_components.pop(frame.session_id, None)
                self._debug_history.pop(frame.session_id, None)
                self._debug_evidence.pop(frame.session_id, None)
                return VisionOutput(analysis=VisionAnalysis(
                    mode="identification", summary="Debug mode ended. Show me a component to identify."
                ))
            return await self._debug_breadboard(frame)
        try:
            payload = await self._complete(
                system=self._system_prompt(), text=frame.metadata.user_request or "Identify the component in view.",
                frame=frame,
            )
        except VisionModelError as exc:
            if "did not contain a valid JSON" not in str(exc):
                raise
            payload = {"id": None, "name": "No component is clear in this frame.", "evidence": ""}
        record = self._knowledge_base.record_for(str(payload.get("id"))) if payload.get("id") else None
        evidence = str(payload.get("evidence", "")).strip()
        if record is None:
            return VisionOutput(analysis=VisionAnalysis(
                summary=str(payload.get("name") or "No catalog component is clear in this frame."),
                observations=[evidence] if evidence else [], component_guidance=ComponentGuidance(
                    clarifying_questions=["Hold one component closer and make its printed label visible."])))
        match = ComponentMatch(record, 100.0)
        guidance = ComponentGuidance(identified_components=[ComponentIdentification(
            component_id=str(record["id"]), confidence=payload.get("confidence", 0),
            observed_evidence=[evidence] if evidence else [],
        )])
        mode = "debug" if record["id"] == "breadboard-830" else "identification"
        if mode == "debug":
            self._debug_sessions.add(frame.session_id)
            ids = [str(item) for item in payload.get("visible_ids", []) if self._knowledge_base.record_for(str(item))]
            self._debug_components[frame.session_id] = list(dict.fromkeys([record["id"], *ids]))[:5]
        output = VisionOutput(analysis=VisionAnalysis(mode=mode,
            summary=str(payload.get("name") or record["canonical_name"]),
            observations=[evidence] if evidence else [], component_guidance=guidance,
        ))
        return self._ground(output, [match])
