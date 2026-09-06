import json
import unittest
from pathlib import Path


class TargetCircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parents[1] / "services/vision_api/circuit_definitions/motor-fan-button-relay.json"
        self.target = json.loads(path.read_text())
        self.edges = {(edge["source"], edge["target"]) for edge in self.target["connections"]}

    def test_uses_a_transistor_instead_of_driving_the_relay_from_gpio(self) -> None:
        self.assertIn(("UNO D8", "1 kOhm resistor"), self.edges)
        self.assertIn(("1 kOhm resistor", "PN2222 base"), self.edges)
        self.assertIn(("relay coil B", "PN2222 collector"), self.edges)
        self.assertNotIn(("UNO D8", "relay coil A"), self.edges)

    def test_motor_is_normally_off_and_externally_powered(self) -> None:
        self.assertIn(("external 5V +", "relay COM"), self.edges)
        self.assertIn(("relay NO", "motor +"), self.edges)
        self.assertIn(("motor -", "external supply -"), self.edges)

    def test_protects_both_inductive_loads(self) -> None:
        diode_edges = [edge for edge in self.target["connections"] if "diode" in edge["source"]]
        self.assertEqual(len(diode_edges), 4)


if __name__ == "__main__":
    unittest.main()
