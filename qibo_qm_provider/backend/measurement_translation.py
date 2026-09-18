"""Translate a Qiskit ``Result`` back onto a Qibo circuit's measurement gates.

``qibo.gates.M.register_samples`` has no equivalent in Qiskit's classical
register model, so this glue code is the one genuinely new piece of logic in
this backend (everything upstream -- macro installation, Target/operation
mapping, QUA program construction, job submission -- is inherited unchanged
from the wrapped ``qiskit_qm_provider.backend.qm_backend.QMBackend``).

Bit-ordering convention (confirmed empirically against qiskit 1.x, not
assumed -- see the design notes in the approved plan):

* Each Qibo ``M`` gate becomes one Qiskit classical register, named after
  ``gate.register_name``, with ``creg.size == len(gate.qubits)``.
* When a circuit has multiple classical registers, ``Result.get_memory()``
  concatenates them with the *last-declared* register as the leftmost
  (most-significant) characters of the per-shot bitstring, and the
  *first-declared* register as the rightmost chunk.
* Within one register's chunk, the bit for ``gate.qubits[j]`` sits at
  position ``len(gate.qubits) - 1 - j`` from the left of that chunk (i.e.
  reversing the chunk gives the bits in ``gate.qubits`` order).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from qibo.models import Circuit as QiboCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

__all__ = ["translate_measurements"]


def _register_chunks(qc: QuantumCircuit) -> Dict[str, Tuple[int, int]]:
    """Map each classical register name to its ``(start, end)`` slice within
    the per-shot bitstring returned by ``Result.get_memory()``.

    Registers are concatenated with the last-declared register occupying the
    leftmost (lowest Python string index) characters.
    """
    chunks: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for creg in reversed(qc.cregs):
        chunks[creg.name] = (pos, pos + creg.size)
        pos += creg.size
    return chunks


def translate_measurements(
    circuit: QiboCircuit,
    qc: QuantumCircuit,
    result: Result,
    experiment_index: int = 0,
) -> None:
    """Assign per-shot samples from a Qiskit ``Result`` onto each Qibo ``M``
    gate in ``circuit.measurements``, mirroring ``qibolab._core.backends.
    QibolabBackend.assign_measurements``'s role for this backend.

    Args:
        circuit: The original Qibo circuit (its ``.measurements`` gates are
            mutated in place via ``register_samples``).
        qc: The Qiskit ``QuantumCircuit`` produced by
            ``circuit_conversion.qibo_circuit_to_qiskit`` -- needed for its
            ``cregs`` declaration order.
        result: The ``qiskit.result.Result`` returned by
            ``QMBackend.run(qc, shots=..., memory=True).result()``. Requires
            the ``memory`` run option to have been set, since per-gate
            samples need per-shot bitstrings, not aggregate counts.
        experiment_index: Index of this circuit within ``result`` (0 for a
            single-circuit run, matching how this backend calls ``run``).
    """
    memory = result.get_memory(experiment_index)
    chunks = _register_chunks(qc)

    for gate in circuit.measurements:
        start, end = chunks[gate.register_name]
        n_qubits = len(gate.qubits)
        samples = np.empty((len(memory), n_qubits), dtype=np.uint8)
        for shot, bitstring in enumerate(memory):
            chunk = bitstring[start:end][::-1]
            for j in range(n_qubits):
                samples[shot, j] = int(chunk[j])
        gate.result.register_samples(samples)
