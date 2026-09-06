import unittest

from vision_api.speech import SpeechRouter


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send_json(self, message) -> None:
        self.messages.append(message)


class SpeechRouterTests(unittest.IsolatedAsyncioTestCase):
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
        }])
