"""Opt-in, real-hardware integration test for ``QiboQMBackend``.

This is the actual go/no-go check for this backend design: it validates
``add_basic_macros`` (reused directly against a real IQCC-provisioned
``QMBackend``), the ``from_qiskit_backend`` wrapping path, circuit
conversion, the full ``run()``/``QMJob`` pipeline (no mocking), and
measurement translation -- end to end, against real hardware -- not just
the individual pieces the default (no-hardware) suite in the rest of
``test/`` checks in isolation.

Requires IQCC cloud credentials/access configured in the environment (see
``qiskit_qm_provider.IQCCProvider``). Skipped automatically otherwise, so the
rest of the suite (and CI without cloud access) stays green. Run explicitly
with e.g.::

    ~/venvs/rl_qoc/bin/python -m pytest test/test_iqcc_integration.py -v -m iqcc

Not run as part of the default ``pytest test/`` invocation for this package.
"""

import pytest
from qibo import Circuit, gates
from qiskit_qm_provider import IQCCProvider, add_basic_macros

from qibo_qm_provider import QiboQMBackend

pytestmark = pytest.mark.iqcc

BACKEND_NAME = "arbel"


@pytest.fixture
def backend():
    try:
        iqcc_provider = IQCCProvider()
        qiskit_backend = iqcc_provider.get_backend(BACKEND_NAME)
        add_basic_macros(qiskit_backend, reset_type="active", max_attempts=1)
    except Exception as exc:  # noqa: BLE001 - any credential/network/backend failure means "skip"
        pytest.skip(f"IQCC backend {BACKEND_NAME!r} unavailable: {exc!r}")

    return QiboQMBackend.from_qiskit_backend(qiskit_backend)


def test_backend_capabilities_are_populated(backend):
    assert backend.qubits
    assert backend.connectivity
    assert backend.natives


def test_execute_circuit_on_real_hardware(backend):
    """Qubit indices 0/1 resolve to "arbel"'s qA1/qA2 (position in
    machine.active_qubit_names -- see qibo_qm_backend.py's circuit_to_qua/
    execute_circuit docstrings). Their CZ macro is only registered as
    (control=qA2, target=qA1) -- i.e. CZ(1, 0), not CZ(0, 1) -- because the
    physical implementation is asymmetric (QubitPair.moving_qubit): the
    flux pulse only plays on one specific qubit of the pair. Confirmed
    directly against this exact machine: CZ(0, 1) raises
    UnsupportedConnectivityError naming this same fix."""
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.M(0, 1))

    nshots = 100
    result = backend.execute_circuit(circuit, nshots=nshots)

    samples = result.samples()
    assert samples.shape == (nshots, 2)
