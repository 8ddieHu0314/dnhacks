import unittest
from pathlib import Path


class MotorFanFirmwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).parents[1] / "firmware/motor_fan_relay/motor_fan_relay.ino"
        cls.source = path.read_text()

    def test_matches_the_target_pin_assignment(self) -> None:
        self.assertIn("BUTTON_PIN = 2", self.source)
        self.assertIn("RELAY_PIN = 8", self.source)
        self.assertIn("pinMode(BUTTON_PIN, INPUT_PULLUP)", self.source)

    def test_relay_starts_off_and_tracks_the_debounced_button(self) -> None:
        self.assertLess(self.source.index("digitalWrite(RELAY_PIN, LOW)"), self.source.index("void loop()"))
        self.assertIn("digitalWrite(RELAY_PIN, stablePressed ? HIGH : LOW)", self.source)
        self.assertIn("DEBOUNCE_MS = 25", self.source)


if __name__ == "__main__":
    unittest.main()
