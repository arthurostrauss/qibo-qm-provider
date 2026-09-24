"""``QiboQMBackend``: a Qibo ``Backend`` that wraps a ``qiskit_qm_provider.
backend.qm_backend.QMBackend`` instance.

This is a translation shim, not an independent reimplementation. Macro
installation, Target/operation-mapping population, calibration-override
precedence, QUA program construction, QM connection handling, job submission,
and result fetching are all inherited unchanged by delegating to the wrapped
``QMBackend`` (see ``self.qiskit_backend``). The only new logic here is:

* decomposing any gate ``execute_circuit`` receives that isn't already native
  to this machine (``default_transpile.default_transpile``, opt out via
  ``transpile=False``),
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

from .circuit_conversion import (
    normalize_symmetric_two_qubit_directions,
    qibo_circuit_to_qiskit,
    validate_two_qubit_connectivity,
)
from .default_transpile import default_transpile
from .gate_map import enum_compatible_quam_natives
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
        self.name = name or "qibo-qm-provider"
        self._qiskit_backend = QMBackend(machine, qmm=qmm, init_macro=init_macro, name=name)
    
    @property
    def machine(self) -> QuamRoot:
        """
        The QuAM machine object.
        """
        return self._qiskit_backend.machine

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

    @classmethod
    def from_iqcc(
        cls,
        backend_name: str,
        *,
        quam_state_folder_path: Optional[str] = None,
        quam_cls: Optional[type[QuamRoot]] = None,
        api_token: Optional[str] = None,
    ) -> "QiboQMBackend":
        """Fetch ``backend_name``'s latest state from IQCC and build a backend.

        No ``$QIBOLAB_PLATFORMS`` folder registration required -- this is
        the native-Python entrypoint, going through the same
        ``_create_iqcc_with_machine`` a registered
        ``qibo-qm-iqcc-<backend_name>`` folder's ``platform.py`` would call.
        """
        from qiskit_qm_provider.providers import IQCCProvider
        from qibo_qm_provider.quam_macros.superconducting import add_basic_macros
        provider = IQCCProvider(api_token=api_token)
        machine = provider.get_machine(backend_name, quam_state_folder_path=quam_state_folder_path, quam_cls=quam_cls)
        add_basic_macros(machine)
        return cls(machine=machine)
    
    @classmethod
    def from_local(
        cls,
        state_path: str,
        *,
        quam_class: Optional[str] = None,
    ) -> "QiboQMBackend":
        """Load an already-existing local QuAM state and build a backend.

        No ``$QIBOLAB_PLATFORMS`` folder registration required -- the
        native-Python entrypoint counterpart to ``from_iqcc``.
        """
        from qibo_qm_provider.quam_macros.superconducting import add_basic_macros
        from qiskit_qm_provider.providers import QMProvider
        provider = QMProvider(state_folder_path=state_path, quam_cls=quam_class)
        machine = provider.get_machine()
        add_basic_macros(machine)
        def init_macro(*args, **kwargs):
            machine.initialize_qpu(*args, **kwargs)
            if hasattr(machine, "apply_qdac_offsets"):
                machine.apply_qdac_offsets()
        return cls(machine=machine, init_macro=init_macro)
        

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
        """The Enum-compatible, QuAM-backed native gate names for this machine.

        Shared definition with ``QiboQMPlatformBackend.natives`` (issue #6):
        the intersection of QuAM macros installed on the wrapped machine
        (via the wrapped ``QMBackend``'s ``target.operation_names``) and
        :data:`~qibo_qm_provider.backend.gate_map.ENUM_COMPATIBLE_NATIVE_GATES`
        -- the most restrictive set both Qibo's ``NativeGates`` and
        qibolab's default ``Compiler`` accept. Custom macros outside that
        Enum intersection are intentionally omitted; inspect
        ``self.qiskit_backend.target.operation_names`` for the unfiltered
        machine-level view.
        """
        return enum_compatible_quam_natives(self._qiskit_backend.target.operation_names)

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

        Replacing an existing name (e.g. ``register_gate("rz", ...)`` on a
        machine that already has an ``rz`` macro) is supported: the compiler
        picks up the new implementation (this relies on the ``operation_key``
        cache fix shipped in ``qiskit-qm-provider`` 0.3.5).

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

    def _default_transpile(self, circuit: QiboCircuit) -> QiboCircuit:
        """The gate-decomposition step :meth:`execute_circuit` runs by
        default (``transpile=True``) -- see :func:`~qibo_qm_provider.
        backend.default_transpile.default_transpile`.

        Both ``already_native`` and ``decomposition_targets`` use
        :attr:`natives` (the shared Enum∩QuAM-macro set from issue #6).
        Gates outside that set -- including QuAM macros that are not
        Enum-compatible -- are not treated as already-native here; pass
        ``transpile=False`` if you need to compile such a circuit as-is.
        """
        return default_transpile(
            circuit, already_native=self.natives, decomposition_targets=self.natives
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

        Unlike ``execute_circuit``, this path does **not** run the default
        gate-decomposition step -- pass an already-native circuit (or
        decompose yourself) before calling.

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
                direction. Qubit-exchange-symmetric gates (``CZ``, ``iSWAP``
                -- :data:`~qibo_qm_provider.backend.gate_map.
                SYMMETRIC_TWO_QUBIT_NATIVE_GATES`) are reordered onto the
                registered direction first and never raise this for
                direction alone. For an asymmetric gate, set
                ``circuit.wire_names`` to route logical qubits onto the
                physical qubits/direction that is actually calibrated -- see
                :func:`~qibo_qm_provider.backend.circuit_conversion.
                qibo_circuit_to_qiskit`.
        """
        qubit_dict = self._qiskit_backend.qubit_dict
        qc = qibo_circuit_to_qiskit(circuit, qubit_dict=qubit_dict)
        qc = normalize_symmetric_two_qubit_directions(qc, self._qiskit_backend.qubit_pair_dict)
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
        transpile: bool = True,
    ) -> MeasurementOutcomes:
        """Execute ``circuit`` on the wrapped QM machine.

        Physical qubit placement is driven entirely by ``circuit.wire_names``
        (see :func:`~qibo_qm_provider.backend.circuit_conversion.
        qibo_circuit_to_qiskit`):

        * **Explicit** (``Circuit(n, wire_names=[...])`` set to physical
          qubit names): honoured verbatim -- resolved through
          ``self.qiskit_backend.qubit_dict`` into a
          :class:`~qiskit.transpiler.Layout`, applied via
          :func:`qiskit.compiler.transpile` with no target/basis involved,
          so nothing about the requested assignment (including a two-qubit
          gate's qubit order) is reinterpreted. A reversed asymmetric gate
          (e.g. ``CNOT``) still raises :class:`~qibo_qm_provider.exceptions.
          UnsupportedConnectivityError`; a reversed ``CZ``/``iSWAP`` is
          reordered (exact, and placement-preserving -- see
          :func:`~qibo_qm_provider.backend.circuit_conversion.
          normalize_symmetric_two_qubit_directions`), on every path.
        * **Left unset**, with ``transpile=True`` (default): an ordinary
          ``transpile(qc, backend=self.qiskit_backend, optimization_level=0)``
          runs against the wrapped machine's real ``Target`` -- Qiskit's own
          preset pass manager picks layout/routing/gate-direction
          automatically.

        Args:
            circuit: A concrete Qibo circuit (no symbolic parameters -- use
                :meth:`circuit_to_qua` for those).
            initial_state: A Qibo circuit to prepend, or ``None``.
            nshots: Number of shots to sample.
            transpile: When ``True`` (default): (1) decompose any gate not
                already native to this machine into its native gate set,
                via :func:`~qibo_qm_provider.backend.default_transpile.
                default_transpile` -- e.g. a plain ``H`` on a machine with
                no ``h`` macro (emits a ``UserWarning`` naming the gates
                decomposed); and (2), only when ``circuit.wire_names`` is
                left unset, run the automatic backend-targeted layout step
                described above. When ``False``, skip *both* -- the fully
                manual contract this parameter has always had: no pre-check
                that the circuit is already native or already placed: a
                non-native gate or an uncalibrated asymmetric two-qubit
                direction (a reversed ``CZ``/``iSWAP`` is still reordered) still
                fails, but via the prior error path (e.g. a qm_qasm
                ``CompilationException``, not a guaranteed
                ``UnsupportedGateError``/``UnsupportedConnectivityError``
                from this method) -- unless ``circuit.wire_names`` is
                explicit, which is still honoured regardless of
                ``transpile`` (it is a placement directive from the caller,
                not something this flag governs).
        """
        if isinstance(initial_state, QiboCircuit):
            return self.execute_circuit(initial_state + circuit, nshots=nshots, transpile=transpile)
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
        if transpile:
            circuit = self._default_transpile(circuit)

        qubit_dict = self._qiskit_backend.qubit_dict
        qc = qibo_circuit_to_qiskit(
            circuit,
            qubit_dict=qubit_dict,
            backend=self._qiskit_backend if transpile else None,
        )
        qc = normalize_symmetric_two_qubit_directions(qc, self._qiskit_backend.qubit_pair_dict)
        validate_two_qubit_connectivity(qc, self._qiskit_backend.qubit_pair_dict, qubit_dict)
        job = self._qiskit_backend.run(qc, shots=nshots, memory=True)
        result = job.result()

        outcome = MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
        translate_measurements(circuit, qc, result)
        return outcome

    def execute_circuits(
        self,
        circuits: List[QiboCircuit],
        initial_state=None,
        nshots: int = 1000,
        transpile: bool = True,
    ) -> List[MeasurementOutcomes]:
        """Execute several circuits on the wrapped QM machine as one job.

        Batched analog of :meth:`execute_circuit`: every circuit goes
        through the exact same steps (initial-state prepending, symbolic-
        parameter rejection, optional default transpilation, Qibo->Qiskit
        conversion, connectivity validation) and is then submitted together
        as a single ``QMBackend.run(list_of_qc, ...)`` job -- one job for
        the whole batch rather than one per circuit -- with each circuit's
        measurements translated back from its own slot in the shared
        ``Result`` (``experiment_index``).

        Args:
            circuits: See :meth:`execute_circuit`; applied identically to every circuit in the batch.
                every circuit in ``circuits``.
            initial_state: A Qibo circuit prepended to every circuit in
                ``circuits``, or ``None``.
            nshots: Number of shots to sample, shared by every circuit.
            transpile: See :meth:`execute_circuit`.
        """
        if isinstance(initial_state, QiboCircuit):
            return self.execute_circuits(
                [initial_state + circuit for circuit in circuits],
                nshots=nshots,
                transpile=transpile,
            )
        if initial_state is not None:
            raise_error(ValueError, "QiboQMBackend only supports circuits as initial states.")
        for circuit in circuits:
            if circuit_has_symbols(circuit):
                raise_error(
                    ValueError,
                    "QiboQMBackend.execute_circuits does not support a circuit with "
                    "symbolic parameters -- it returns MeasurementOutcomes, which "
                    "presupposes one concrete circuit and shot count. Bind the "
                    "parameters first (circuit.set_parameters(...)), or use "
                    "circuit_to_qua(circuit) directly for real-time parameterization.",
                )

        if transpile:
            circuits = [self._default_transpile(circuit) for circuit in circuits]

        qubit_dict = self._qiskit_backend.qubit_dict
        qubit_pair_dict = self._qiskit_backend.qubit_pair_dict
        qcs = [
            normalize_symmetric_two_qubit_directions(
                qibo_circuit_to_qiskit(circuit, qubit_dict=qubit_dict), qubit_pair_dict
            )
            for circuit in circuits
        ]
        for qc in qcs:
            validate_two_qubit_connectivity(qc, qubit_pair_dict, qubit_dict)
        job = self._qiskit_backend.run(qcs, shots=nshots, memory=True)
        result = job.result()

        outcomes = []
        for index, (circuit, qc) in enumerate(zip(circuits, qcs)):
            outcome = MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
            translate_measurements(circuit, qc, result, experiment_index=index)
            outcomes.append(outcome)
        return outcomes
