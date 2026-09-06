from __future__ import annotations

import asyncio
import subprocess
from collections import defaultdict
from collections.abc import Callable

from fastapi import WebSocket

from .models import VisionAnalysis


def spoken_guidance(analysis: VisionAnalysis) -> str:
    """Condense a full result to the one thing the wearer should hear next."""

    debug = analysis.debug_guidance
    if analysis.mode != "debug" or debug is None:
        return analysis.summary
    prefix = f"Safety: {analysis.safety_alerts[0]} " if analysis.safety_alerts else ""
    if not debug.safe_to_energize:
        prefix += "Keep all power disconnected. "
    if debug.visual_clarification is not None:
        view = debug.visual_clarification
        return f"{prefix}Show me a {view.requested_view} of {view.target}. {view.reason}".strip()
    if debug.steps:
        step = debug.steps[0]
        return f"{prefix}Next: {step.instruction} Expected: {step.expected_evidence}".strip()
    return f"{prefix}{analysis.summary}".strip()


class SpeechRouter:
    """Mirror speech locally or forward it to a connected glasses bridge."""

    def __init__(self, mode: str, launcher: Callable[..., object] = subprocess.Popen) -> None:
        self._mode = mode
        self._launcher = launcher
        self._sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self._last: dict[str, str] = {}

    def attach(self, session_id: str, websocket: WebSocket) -> None:
        self._sockets[session_id].add(websocket)

    def detach(self, session_id: str, websocket: WebSocket) -> None:
        self._sockets[session_id].discard(websocket)

    async def publish(self, session_id: str, frame_id: str, text: str) -> None:
        text = text.strip()
        if not text or self._last.get(session_id) == text:
            return
        self._last[session_id] = text
        if self._mode in {"computer", "both"}:
            try:
                self._launcher(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
        if self._mode in {"glasses", "both"}:
            await asyncio.gather(*(self._send(session_id, socket, frame_id, text) for socket in self._sockets[session_id]))

    async def publish_event(self, session_id: str, payload: dict[str, object]) -> None:
        """Forward non-speech analysis state to every attached client."""

        if self._mode in {"glasses", "both"}:
            await asyncio.gather(
                *(self._send_event(session_id, socket, payload) for socket in self._sockets[session_id])
            )

    async def _send_event(
        self, session_id: str, websocket: WebSocket, payload: dict[str, object]
    ) -> None:
        try:
            await websocket.send_json(payload)
        except Exception:
            self.detach(session_id, websocket)

    async def _send(self, session_id: str, websocket: WebSocket, frame_id: str, text: str) -> None:
        try:
            await websocket.send_json({"type": "speak", "frame_id": frame_id, "text": text})
            await websocket.send_json({"type": "speak_end", "frame_id": frame_id})
        except Exception:
            self.detach(session_id, websocket)
