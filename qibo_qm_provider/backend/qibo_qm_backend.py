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
from qiskit_qm_provider.backend.qua_circuit_compilation import QuaCircuitCompilation
from qiskit_qm_provider.parameter_table import InputType, Parameter, ParameterTable
from quam.core import QuamRoot

from .circuit_conversion import qibo_circuit_to_qiskit, validate_two_qubit_connectivity
from .measurement_translation import translate_measurements
from .parameter_table import QiboParameterTable
from .symbolic_parameters import circuit_has_symbols, validate_symbol_name

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
        so the new operation is visible in ``natives``.

        ``macro`` must be a ``quam.core.macro.QuamMacro`` instance (e.g. a
        ``QubitMacro`` subclass) whose ``apply`` takes the gate's own
        parameters. A **bare callable does not work**, contrary to what this
        docstring previously claimed: ``QMBackend._populate_target`` reads
        ``macro.apply`` unconditionally, so a plain function raises
        ``AttributeError: 'function' object has no attribute 'apply'``
        (verified). ``apply``'s arity must also match the number of parameters
        the emitted operation carries, or compilation fails with
        ``NumberOfParametersMismatch``.

        .. warning::
           **Registering a name that already exists may not reach the
           compiler, depending on the installed ``qiskit-qm-provider``
           version.** Root cause: ``qm_qasm.OperationIdentifier`` defines
           neither ``__eq__`` nor ``__hash__``, so it falls back to Python's
           default identity comparison -- two separately-constructed
           ``OperationIdentifier("rz", 1, (0,))`` instances are never equal
           (verified directly against qm-qasm 1.7.7). ``qiskit-qm-provider``
           used to key its internal QUA-operation cache by these objects
           directly, so re-populating an operation that already had an entry
           silently added a second, functionally duplicate one instead of
           overwriting it -- ``register_gate("rz", ...)`` on a machine that
           already had an ``rz`` macro made the change visible in ``natives``
           while ``quantum_circuit_to_qua`` kept calling the original
           implementation. Fixed upstream (``qiskit_qm_provider.backend.
           backend_utils.operation_key``, used throughout ``QMBackend``), but
           only in source as of this writing -- this package's declared
           dependency floor (``qiskit-qm-provider>=0.3.4``) predates the fix.
           Until a release carrying it is available, the workaround remains:
           install the macro on the QuAM machine *before* constructing the
           backend, rather than via this method against an existing name.
           Adding a genuinely new operation name is unaffected either way.

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

    def circuit_to_qua(
        self,
        circuit: QiboCircuit,
        *,
        input_type: Optional[InputType] = None,
        param_table: Optional[Union[ParameterTable, List[Union[ParameterTable, Parameter]], dict]] = None,
    ) -> QuaCircuitCompilation:
        """Compile a Qibo circuit to QUA, once, with its symbolic parameters
        (if any) bound to a real-time parameter table.

        The Qibo-facing counterpart of
        ``qiskit_qm_provider.QMBackend.quantum_circuit_to_qua`` -- unlike
        ``execute_circuit``, this accepts a circuit with symbolic (``sympy``)
        gate parameters, since it returns the compiled QUA program directly
        rather than a fixed-shot-count ``MeasurementOutcomes``.

        Converts ``circuit`` to a Qiskit ``QuantumCircuit`` exactly once and
        builds ``param_table`` from that same object, rather than the two
        Qibo->Qiskit conversions a caller composing ``QiboParameterTable.
        from_qibo_circuit`` and ``qibo_circuit_to_qiskit`` separately would
        otherwise perform.

        Args:
            circuit: A Qibo circuit, symbolic or concrete.
            input_type: Forwarded to ``QiboParameterTable.from_qiskit`` when
                ``param_table`` is not supplied. Ignored if ``param_table`` is
                given.
            param_table: An already-built parameter table (or bare
                ``Parameter``/sequence/dict), for the case where several
                circuits share one table so that one set of QUA variables
                drives all of them. When omitted, a table is built from
                ``circuit``'s own symbolic parameters (``None`` if it has
                none).

        Returns:
            The upstream ``QuaCircuitCompilation`` unchanged -- it already
            carries ``.qua_program`` and the wired measurement outputs.

        Raises:
            UnsupportedParameterError: If a parameter name collides with an
                operation installed on the wrapped machine (checked against
                ``self.qiskit_backend.qm_qasm_basis_gates``, the full machine
                ``Target`` -- a precise version of the lower-bound check
                already applied when the circuit itself was converted, which
                only knows the operations *this circuit* emits).
            UnsupportedConnectivityError: If a two-qubit gate addresses a
                physical qubit pair with no registered connectivity in that
                direction (many QM two-qubit natives are physically
                asymmetric). Set ``circuit.wire_names`` to route logical
                qubits onto the physical qubits/direction that is actually
                calibrated -- see :func:`~qibo_qm_provider.backend.
                circuit_conversion.qibo_circuit_to_qiskit`.
        """
        qubit_dict = self._qiskit_backend.qubit_dict
        qc = qibo_circuit_to_qiskit(circuit, qubit_dict=qubit_dict)
        validate_two_qubit_connectivity(qc, self._qiskit_backend.qubit_pair_dict, qubit_dict)
        table = (
            param_table
            if param_table is not None
            else QiboParameterTable.from_qiskit(qc, input_type=input_type)
        )
        basis_gates = frozenset(self._qiskit_backend.qm_qasm_basis_gates)
        for parameter in qc.parameters:
            validate_symbol_name(parameter.name, forbidden_names=basis_gates)
        return self._qiskit_backend.quantum_circuit_to_qua(qc, table)

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
        if circuit_has_symbols(circuit):
            raise_error(
                ValueError,
                "QiboQMBackend.execute_circuit does not support a circuit with "
                "symbolic parameters -- it returns MeasurementOutcomes, which "
                "presupposes one concrete circuit and shot count. Bind the "
                "parameters first (circuit.set_parameters(...)), or use "
                "circuit_to_qua(circuit) directly for real-time parameterization.",
            )

        qubit_dict = self._qiskit_backend.qubit_dict
        qc = qibo_circuit_to_qiskit(circuit, qubit_dict=qubit_dict)
        validate_two_qubit_connectivity(qc, self._qiskit_backend.qubit_pair_dict, qubit_dict)
        job = self._qiskit_backend.run(qc, shots=nshots, memory=True)
        result = job.result()

        outcome = MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
        translate_measurements(circuit, qc, result)
        return outcome
