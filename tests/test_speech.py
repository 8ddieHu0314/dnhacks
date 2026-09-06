import unittest

from vision_api.models import VisionAnalysis
from vision_api.speech import SpeechRouter, spoken_guidance


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send_json(self, message) -> None:
        self.messages.append(message)


class SpeechRouterTests(unittest.IsolatedAsyncioTestCase):
    def test_requests_visual_clarification_before_giving_a_step(self) -> None:
        analysis = VisionAnalysis.model_validate({"mode": "debug", "summary": "Wiring is obscured.",
            "safety_alerts": ["Disconnect USB power."], "debug_guidance": {
                "status": "needs_context", "problem": "Rail connection is hidden.",
                "steps": [{"instruction": "Move the wire.", "reason": "Inspect it.",
                    "expected_evidence": "The socket is visible."}],
                "visual_clarification": {"target": "upper power rail", "requested_view": "top-down close-up",
                    "reason": "Your hand hides the jumper socket."}}})

        spoken = spoken_guidance(analysis)
        self.assertIn("Safety: Disconnect USB power.", spoken)
        self.assertIn("top-down close-up of upper power rail", spoken)
        self.assertNotIn("Move the wire", spoken)

    def test_speaks_the_first_debug_step_when_the_view_is_sufficient(self) -> None:
        analysis = VisionAnalysis.model_validate({"mode": "debug", "summary": "LED is dark.",
            "debug_guidance": {"status": "in_progress", "problem": "LED polarity may be reversed.",
                "steps": [{"instruction": "Disconnect power.", "reason": "Change wiring safely.",
                    "expected_evidence": "The power LED turns off."}], "visual_clarification": None}})

        self.assertEqual(spoken_guidance(analysis),
            "Keep all power disconnected. Next: Disconnect power. Expected: The power LED turns off.")

    async def test_mirrors_the_same_message_to_mac_and_glasses(self) -> None:
        launched = []

        def launch(command, **_kwargs) -> None:
            launched.append(command)

        router = SpeechRouter("both", launcher=launch)
        socket = FakeWebSocket()
        router.attach("session", socket)
        await router.publish("session", "frame-1", "Breadboard visible.")
        await router.publish("session", "frame-2", "Breadboard visible.")

        self.assertEqual(launched, [["say", "Breadboard visible."]])
        self.assertEqual(socket.messages, [{
            "type": "speak", "frame_id": "frame-1", "text": "Breadboard visible.",
        }, {"type": "speak_end", "frame_id": "frame-1"}])

    async def test_hush_suppresses_only_frames_already_in_flight(self) -> None:
        router = SpeechRouter("glasses")
        socket = FakeWebSocket()
        router.attach("session", socket)
        router.track_frame("session", "old")
        await router.stop("session")
        await router.publish("session", "old", "Stale answer.")
        router.track_frame("session", "new")
        await router.publish("session", "new", "Fresh answer.")

        self.assertEqual(socket.messages, [
            {"type": "speak_stop"},
            {"type": "speak", "frame_id": "new", "text": "Fresh answer."},
            {"type": "speak_end", "frame_id": "new"},
        ])
