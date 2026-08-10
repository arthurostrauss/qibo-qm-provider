"""Tests for qibo_qm_provider.backend.flux_tunable_transmon_backend."""

import pytest

from qibo_qm_provider import FluxTunableTransmonBackend


@pytest.fixture
def backend(dummy_machine):
    return FluxTunableTransmonBackend(dummy_machine)


def test_wraps_qiskit_flux_tunable_transmon_backend(backend):
    from qiskit_qm_provider.backend.flux_tunable_transmon_backend import (
        FluxTunableTransmonBackend as QiskitFluxTunableTransmonBackend,
    )

    assert isinstance(backend.qiskit_backend, QiskitFluxTunableTransmonBackend)


def test_get_qubit_channels_returns_quam_channels(backend, dummy_machine):
    qubit_name = next(iter(dummy_machine.qubits))
    channels = backend.get_qubit_channels(qubit_name)

    assert set(channels) == {"xy", "z", "resonator"}
    quam_qubit = dummy_machine.qubits[qubit_name]
    assert channels["xy"] is quam_qubit.xy
    assert channels["z"] is quam_qubit.z
    assert channels["resonator"] is quam_qubit.resonator


def test_execute_circuit_and_capabilities_inherited_unchanged(backend):
    # qubits/connectivity/natives, execute_circuit, and circuit conversion are
    # inherited from QiboQMBackend unchanged -- no topology-specific override.
    assert backend.qubits == list(backend.qiskit_backend.qubit_dict.values())
    assert backend.execute_circuit.__func__.__qualname__.startswith("QiboQMBackend.")
