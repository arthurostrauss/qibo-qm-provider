import unittest

from qibo_qm_provider import (
    QiboDeviceSpec,
    QibocalQualibrateMapper,
    QibolabPulseLowerer,
    QmBackendRepresentation,
    qasm2_to_qasm3,
)


class TestQibocalQualibrateMapper(unittest.TestCase):
    def test_register_and_resolve(self):
        mapper = QibocalQualibrateMapper()
        mapper.register("resonator_scan", "qualibrate.resonator_scan")

        self.assertEqual(
            mapper.resolve("resonator_scan"),
            "qualibrate.resonator_scan",
        )


class TestQasmConversion(unittest.TestCase):
    def test_uses_injected_qiskit_like_functions(self):
        loads_called = {}

        def fake_loads(source):
            loads_called["source"] = source
            return {"fake": "circuit"}

        def fake_dumps(circuit):
            self.assertEqual(circuit, {"fake": "circuit"})
            return "OPENQASM 3;"

        out = qasm2_to_qasm3(
            "OPENQASM 2.0;",
            qasm2_loader=fake_loads,
            qasm3_dumper=fake_dumps,
        )

        self.assertEqual(out, "OPENQASM 3;")
        self.assertEqual(loads_called["source"], "OPENQASM 2.0;")

    def test_empty_qasm2_rejected(self):
        with self.assertRaises(ValueError):
            qasm2_to_qasm3("  ", qasm2_loader=lambda s: s, qasm3_dumper=lambda c: c)


class TestBackendRepresentation(unittest.TestCase):
    def test_payload_contains_device_and_qua_sections(self):
        device = QiboDeviceSpec(name="qm_chip", qubits=5, native_gates=("rx", "cz"))
        backend = QmBackendRepresentation(device)

        payload = backend.to_qm_payload()

        self.assertEqual(payload["device"]["name"], "qm_chip")
        self.assertEqual(payload["device"]["qubits"], 5)
        self.assertEqual(payload["device"]["native_gates"], ["rx", "cz"])
        self.assertIn("qua", payload)


class TestPulseLowerer(unittest.TestCase):
    def test_lowers_qibolab_like_sequence(self):
        lowerer = QibolabPulseLowerer()

        lowered = lowerer.lower(
            [
                {"operation": "x90", "channel": "q0_xy", "duration": 20},
                {"operation": "measure", "channel": "q0_ro", "duration": 200},
            ]
        )

        self.assertEqual(
            lowered,
            [
                "play(x90, q0_xy, duration=20)",
                "play(measure, q0_ro, duration=200)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
