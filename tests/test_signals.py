import unittest

from vision_api.models import AdvisoryFieldSignal
from vision_api.signals import AdvisorySignalStore


class AdvisorySignalStoreTests(unittest.TestCase):
    def test_retains_an_observation_without_an_isolation_decision(self) -> None:
        signal = AdvisorySignalStore().record_field_signal(
            "session", AdvisoryFieldSignal(level=8.5, state="field_detected")
        )

        self.assertEqual(signal.state, "field_detected")
        self.assertFalse(hasattr(signal, "safe"))
