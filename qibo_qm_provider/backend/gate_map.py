"""The Qibo-gate -> Qiskit-gate correspondence used by the direct (symbolic)
lowering path.

Every entry here was **verified numerically**, not read off documentation.
Two-qubit gates were compared as 2-qubit *circuit* unitaries with
``QuantumCircuit.reverse_bits()`` applied, so what is checked is the
argument-role mapping (which Qibo argument is the control) rather than raw
matrix basis ordering. Comparing bare ``gate.matrix()`` arrays instead reports
false differences for every asymmetric 2-qubit gate, because Qibo is big-endian
and Qiskit is little-endian -- a real trap when extending this table.

Findings worth keeping visible:

- ``qibo.CU3(t,p,l) == CUGate(t, p, l, gamma=-(p+l)/2)`` and
  ``qibo.CU2(p,l)   == CUGate(pi/2, p, l, gamma=-(p+l)/2)``.
  Both ``CU3Gate`` and ``CUGate(..., gamma=0)`` are **wrong**. The ``gamma``
  term is a *derived expression*, which is why the sympy translator has to
  handle arithmetic and not just bare symbols: with a symbolic ``p``/``l``
  these two gates are unrepresentable otherwise.
- ``qibo.GIVENS`` and ``qibo.RBS`` are *real* rotations and are **not**
  ``XXPlusYYGate`` at any angle scaling, despite looking like they should be.
  Deliberately absent below rather than mapped approximately.
- ``fSim`` reuses ``qiskit_qm_provider.additional_gates.FSimGate`` (verified
  equal), so there is one definition of it in the stack rather than two.

Operation naming
----------------
The Qiskit gate's ``.name`` is what ends up in the exported OpenQASM3 and what
``qm_qasm`` looks up. Building a *bare* Qiskit standard gate for ``U1``, ``U2``,
``U3``, ``PRX``, ``U1q``, ``CU1``, ``CU2``, ``CU3``, or ``RXXYY`` would rename
the operation away from Qibo's own name -- e.g. both ``PRX`` and ``U1q`` would
become ``r``, and a QuAM macro installed under the Qibo name a user actually
wrote would get a missing-operation error for a Qiskit name they never used.
:mod:`~qibo_qm_provider.backend.qibo_qiskit_gates` avoids this at the source
for exactly these nine gates: each is a thin subclass of the real standard
gate (so the unitary is inherited, not re-derived) with ``.name`` fixed to
Qibo's own name in ``__init__``. So every entry in :data:`QIBO_TO_QISKIT`
below emits the name in :data:`QIBO_TO_OPERATION_NAME`, and every entry in
that table equals the mapped Qibo gate's own ``gate.name`` -- there is no
renamed gate left to alias.
"""

from __future__ import annotations

import qiskit.circuit.library as qlib
from qibo.transpiler.unroller import NativeGates

from .qibo_qiskit_gates import (
    FSimGate,
    GPI2Gate,
    GPIGate,
    QiboCU1Gate,
    QiboCU2Gate,
    QiboCU3Gate,
    QiboMSGate,
    QiboPRXGate,
    QiboRXXYYGate,
    QiboU1Gate,
    QiboU1qGate,
    QiboU2Gate,
    QiboU3Gate,
)

__all__ = [
    "QIBO_TO_QISKIT",
    "QIBO_TO_OPERATION_NAME",
    "EMITTED_OPERATION_NAMES",
    "DEFERRED_GATES",
    "OPERATION_NAME_TO_NATIVE_GATE",
]

#: ``{qibo gate class name: builder(*translated_params) -> Qiskit gate}``.
#: Qubit arguments are *not* passed here -- the caller appends the returned
#: gate onto ``gate.qubits`` in Qibo's own order, which is what makes the
#: control-first convention line up (verified for CRX/CRY/CRZ/CU1/RZX).
QIBO_TO_QISKIT = {
    # --- single-qubit, non-parametric ---------------------------------------
    "I": lambda: qlib.IGate(),
    "H": lambda: qlib.HGate(),
    "X": lambda: qlib.XGate(),
    "Y": lambda: qlib.YGate(),
    "Z": lambda: qlib.ZGate(),
    "S": lambda: qlib.SGate(),
    "SDG": lambda: qlib.SdgGate(),
    "T": lambda: qlib.TGate(),
    "TDG": lambda: qlib.TdgGate(),
    "SX": lambda: qlib.SXGate(),
    "SXDG": lambda: qlib.SXdgGate(),
    # --- single-qubit, parametric -------------------------------------------
    "RX": lambda theta: qlib.RXGate(theta),
    "RY": lambda theta: qlib.RYGate(theta),
    "RZ": lambda theta: qlib.RZGate(theta),
    "U1": lambda theta: QiboU1Gate(theta),
    "U2": lambda phi, lam: QiboU2Gate(phi, lam),
    "U3": lambda theta, phi, lam: QiboU3Gate(theta, phi, lam),
    "PRX": lambda theta, phi: QiboPRXGate(theta, phi),
    "U1q": lambda theta, phi: QiboU1qGate(theta, phi),
    "GPI": lambda phi: GPIGate(phi),
    "GPI2": lambda phi: GPI2Gate(phi),
    # --- two-qubit, non-parametric ------------------------------------------
    "CNOT": lambda: qlib.CXGate(),
    "CZ": lambda: qlib.CZGate(),
    "CY": lambda: qlib.CYGate(),
    "SWAP": lambda: qlib.SwapGate(),
    "iSWAP": lambda: qlib.iSwapGate(),
    "CSX": lambda: qlib.CSXGate(),
    # --- two-qubit, parametric ----------------------------------------------
    "CRX": lambda theta: qlib.CRXGate(theta),
    "CRY": lambda theta: qlib.CRYGate(theta),
    "CRZ": lambda theta: qlib.CRZGate(theta),
    "CU1": lambda theta: QiboCU1Gate(theta),
    # gamma = -(phi + lam) / 2 -- verified exactly; see module docstring.
    "CU2": lambda phi, lam: QiboCU2Gate(phi, lam),
    "CU3": lambda theta, phi, lam: QiboCU3Gate(theta, phi, lam),
    "RXX": lambda theta: qlib.RXXGate(theta),
    "RYY": lambda theta: qlib.RYYGate(theta),
    "RZZ": lambda theta: qlib.RZZGate(theta),
    "RZX": lambda theta: qlib.RZXGate(theta),
    "RXXYY": lambda theta: QiboRXXYYGate(theta),
    "fSim": lambda theta, phi: FSimGate(theta, phi),
    "MS": lambda phi0, phi1, theta: QiboMSGate(phi0, phi1, theta),
    # --- three-qubit --------------------------------------------------------
    "TOFFOLI": lambda: qlib.CCXGate(),
    "CCZ": lambda: qlib.CCZGate(),
}

#: Qibo gate class name -> the operation name actually emitted into OpenQASM3
#: (i.e. the name a QuAM macro must be registered under). Every entry here
#: equals the mapped gate's own Qibo ``gate.name`` -- there is no renaming, by
#: construction, for the nine gates that would otherwise land on a Qiskit
#: standard name (see the module docstring and
#: :mod:`~qibo_qm_provider.backend.qibo_qiskit_gates`).
QIBO_TO_OPERATION_NAME = {
    "I": "id",
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "SDG": "sdg",
    "T": "t",
    "TDG": "tdg",
    "SX": "sx",
    "SXDG": "sxdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "U1": "u1",
    "U2": "u2",
    "U3": "u3",
    "PRX": "prx",
    "U1q": "u1q",
    "GPI": "gpi",
    "GPI2": "gpi2",
    "CNOT": "cx",
    "CZ": "cz",
    "CY": "cy",
    "SWAP": "swap",
    "iSWAP": "iswap",
    "CSX": "csx",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "CU1": "cu1",
    "CU2": "cu2",
    "CU3": "cu3",
    "RXX": "rxx",
    "RYY": "ryy",
    "RZZ": "rzz",
    "RZX": "rzx",
    "RXXYY": "rxxyy",
    "fSim": "fsim",
    "MS": "ms",
    "TOFFOLI": "ccx",
    "CCZ": "ccz",
}

#: Every operation name this lowering can emit. A sympy symbol sharing one of
#: these names would be silently renamed by Qiskit's exporter (see
#: :func:`~qibo_qm_provider.backend.symbolic_parameters.validate_symbol_name`),
#: so the converter passes this set in as ``forbidden_names``.
EMITTED_OPERATION_NAMES = frozenset(QIBO_TO_OPERATION_NAME.values()) | {"measure"}

#: Operation name -> Qibo native-gate name, restricted to
#: ``qibo.transpiler.unroller.NativeGates``'s own nine members (``I``, ``Z``,
#: ``RZ``, ``M``, ``GPI2``, ``U3``, ``CZ``, ``iSWAP``, ``CNOT``). Built by
#: inverting :data:`QIBO_TO_OPERATION_NAME` and keeping only entries whose
#: Qibo name is one of those nine -- so e.g. ``"cx"`` (the operation Qiskit
#: emits for Qibo's ``CNOT``) maps back to ``"CNOT"``, while ``"x"``, ``"gpi"``,
#: or a QuAM macro name with no Qibo-gate meaning at all are simply absent.
#:
#: This exists because ``NativeGates`` is what Qibo's own default transpiler
#: construction looks operation names up against
#: (``NativeGates[backend.natives]``, see ``qibo.backends.__init__``'s
#: ``_default_transpiler``) -- and lookup is by exact, case-sensitive member
#: name. A lowercase Qiskit operation string like ``"cz"`` never matches the
#: member ``NativeGates.CZ``, so anything not translated through this table
#: would silently resolve to ``NativeGates.NONE`` instead of raising.
#:
#: ``"measure"`` is added explicitly since ``M`` (Qibo's measurement gate) has
#: no entry in :data:`QIBO_TO_OPERATION_NAME` -- measurement is handled
#: separately throughout this package, not through the gate table.
OPERATION_NAME_TO_NATIVE_GATE = {
    op_name: qibo_name
    for qibo_name, op_name in QIBO_TO_OPERATION_NAME.items()
    if qibo_name in NativeGates.__members__
}
OPERATION_NAME_TO_NATIVE_GATE["measure"] = "M"

#: Qibo gates deliberately *not* mapped, with the reason. Used to produce an
#: actionable error rather than a bare KeyError.
DEFERRED_GATES = {
    "GIVENS": "a real Givens rotation; not XXPlusYYGate at any angle scaling (verified)",
    "RBS": "a real reconfigurable-beam-splitter rotation; not XXPlusYYGate (verified)",
    "GeneralizedRBS": "multi-qubit, no Qiskit standard equivalent",
    "GeneralizedfSim": "carries an arbitrary 2x2 unitary block, no Qiskit standard equivalent",
    "DEUTSCH": "three-qubit, no Qiskit standard equivalent",
    "Align": "no OpenQASM representation at all (qibo's own to_qasm() also refuses it)",
}
