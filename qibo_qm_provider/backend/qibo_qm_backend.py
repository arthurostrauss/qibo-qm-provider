"""``QiboQMBackend``: a Qibo ``Backend`` that wraps a ``qiskit_qm_provider.
backend.qm_backend.QMBackend`` instance.

This is a translation shim, not an independent reimplementation. Macro
installation, Target/operation-mapping population, calibration-override
precedence, QUA program construction, QM connection handling, job submission,
and result fetching are all inherited unchanged by delegating to the wrapped
``QMBackend`` (see ``self.qiskit_backend``). The only new logic here is:

* converting a Qibo ``Circuit`` into the Qiskit ``QuantumCircuit`` that
  ``QMBackend`` expects (``circuit_conversion.qibo_circuit_to_qiskit``), and
* translating the resulting Qiskit ``Result`` back onto Qibo's per-gate
  measurement convention (``measurement_translation.translate_measurements``).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.models import Circuit as QiboCircuit
from qibo.result import MeasurementOutcomes
from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.parameter_table import InputType
from quam.core import QuamRoot

from .circuit_conversion import qibo_circuit_to_qiskit
from .measurement_translation import translate_measurements

try:
    from qm import QuantumMachinesManager
except ImportError:  # pragma: no cover - qm-qua always a declared dependency
    QuantumMachinesManager = None  # type: ignore[assignment,misc]

__all__ = ["QiboQMBackend"]


class QiboQMBackend(NumpyBackend):
    """Generic Qibo backend for Quantum Machines hardware, via QuAM.

    Wraps a ``qiskit_qm_provider.backend.qm_backend.QMBackend`` instance and
    exposes the Qibo ``Backend`` contract (``qubits``, ``connectivity``,
    ``natives``, ``execute_circuit``) as thin views/adapters over it.
    """

    def __init__(
        self,
        machine: QuamRoot,
        qmm: Optional["QuantumMachinesManager"] = None,
        init_macro: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.name = "qibo-qm-provider"
        self._qiskit_backend = QMBackend(machine, qmm=qmm, init_macro=init_macro, name=name)

    @classmethod
    def from_qiskit_backend(cls, qiskit_backend: QMBackend) -> "QiboQMBackend":
        """Wrap an already-instantiated ``QMBackend`` (or subclass), e.g. one
        obtained from ``qiskit_qm_provider.IQCCProvider.get_backend(...)``,
        without reconstructing a fresh ``QMBackend`` from a bare ``machine``.

        ``QiboQMBackend``'s own logic is duck-typed against the wrapped
        backend's public surface (``run``, ``qubit_dict``, ``qubit_pair_dict``,
        ``target``, ``update_target``, ``quantum_circuit_to_qua``) and does
        not care whether the wrapped instance is a plain ``QMBackend`` or a
        topology-specific subclass such as ``FluxTunableTransmonBackend``.
        """
        obj = cls.__new__(cls)
        NumpyBackend.__init__(obj)
        obj.name = "qibo-qm-provider"
        obj._qiskit_backend = qiskit_backend
        return obj

    @property
    def qiskit_backend(self) -> QMBackend:
        """The wrapped ``qiskit_qm_provider.backend.qm_backend.QMBackend``.

        Exposed read-only so callers can reach ``update_target()``,
        ``update_calibrations()``, ``qm``/``qmm``, and ``target`` directly
        without ``QiboQMBackend`` needing to re-wrap every method.
        """
        return self._qiskit_backend

    # ------------------------------------------------------------------
    # Qibo Backend contract
    # ------------------------------------------------------------------

    @property
    def qubits(self) -> List[int]:
        return list(self._qiskit_backend.qubit_dict.values())

    @property
    def connectivity(self) -> List[Tuple[int, int]]:
        return list(self._qiskit_backend.qubit_pair_dict.values())

    @property
    def natives(self) -> List[str]:
        return list(self._qiskit_backend.target.operation_names)

    def update_target(self, input_type: Optional[InputType] = None) -> None:
        """Resynchronize the qm-qasm operation registry with whatever macros
        are currently stored on the wrapped QuAM machine's qubits/pairs.

        Thin delegate to ``self.qiskit_backend.update_target()`` -- see its
        docstring for the precedence rule (machine macro < backend/target
        override) and the sync steps. Exposed directly on ``QiboQMBackend``
        so callers don't need to reach into ``.qiskit_backend`` just to
        re-sync after mutating the machine (e.g. after ``register_gate``, or
        after installing macros by any other means, including
        ``add_basic_macros`` or ``qibolab_bridge.import_qibolab_natives_as_macros``).
        """
        self._qiskit_backend.update_target(input_type=input_type)

    def register_gate(
        self,
        name: str,
        qubits: Union[int, str, Tuple[Union[int, str], Union[int, str]]],
        macro: Any,
    ) -> None:
        """Dynamically add (or replace) a gate implementation.

        Installs ``macro`` under ``name`` on the QuAM qubit (single ``qubits``
        argument) or qubit-pair (a 2-tuple) identified by ``qubits`` --
        resolved the same way ``get_qubit``/``get_qubit_pair`` already
        resolve int/str/object identifiers -- then calls ``update_target()``
        so the new operation is immediately visible in ``natives`` and
        reachable by ``execute_circuit``. ``macro`` is whatever the wrapped
        backend already accepts as a qubit/pair macro: a ``quam.core.macro.
        QuamMacro`` instance, or a bare callable taking the gate's own
        parameters (matching how ``add_basic_macros``-installed macros work).

        This mutates the QuAM machine object itself (the same effect as
        setting ``qubit.macros[name] = macro`` by hand) -- it is not an
        ephemeral, backend-only override. For a non-persistent override
        instead (e.g. swapping in a different implementation for one
        circuit/job without touching the QuAM machine), use the wrapped
        backend's own ``Target``-level calibration mechanism directly via
        ``self.qiskit_backend`` (see ``qiskit_qm_provider``'s
        ``examples/custom_gate.py``) rather than this method.
        """
        if isinstance(qubits, tuple):
            component = self._qiskit_backend.get_qubit_pair(qubits)
        else:
            component = self._qiskit_backend.get_qubit(qubits)
        component.macros[name] = macro
        self.update_target()

    def apply_gate(self, *args, **kwargs):
        raise_error(
            NotImplementedError,
            "QiboQMBackend cannot apply gates directly; it executes circuits on QM hardware.",
        )

    def apply_gate_density_matrix(self, *args, **kwargs):
        raise_error(
            NotImplementedError,
            "QiboQMBackend cannot apply gates directly; it executes circuits on QM hardware.",
        )

    def execute_circuit(
        self,
        circuit: QiboCircuit,
        initial_state=None,
        nshots: int = 1000,
    ) -> MeasurementOutcomes:
        if isinstance(initial_state, QiboCircuit):
            return self.execute_circuit(initial_state + circuit, nshots=nshots)
        if initial_state is not None:
            raise_error(ValueError, "QiboQMBackend only supports circuits as initial states.")

        qc = qibo_circuit_to_qiskit(circuit)
        job = self._qiskit_backend.run(qc, shots=nshots, memory=True)
        result = job.result()

        outcome = MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
        translate_measurements(circuit, qc, result)
        return outcome
