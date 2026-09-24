"""Qiskit ``Gate`` subclasses for Qibo gates, each carrying Qibo's own
OpenQASM name.

Every gate's ``.name`` is deliberately **exactly** Qibo's own ``gate.name``:
``"gpi"``, ``"gpi2"``, ``"ms"``, ``"u1"``, ``"u2"``, ``"u3"``, ``"prx"``,
``"u1q"``, ``"cu1"``, ``"cu2"``, ``"cu3"``, ``"rxxyy"``. That string is what
``qiskit.qasm3.Exporter(basis_gates=...)`` emits and what ``qm_qasm`` looks up
in its operations database, so it must match the QuAM macro name installed via
``QiboQMBackend.register_gate``.

Two families of gate live here, for two different reasons:

1. **No Qiskit standard-library equivalent at all** (``GPI``, ``GPI2``,
   ``QiboMSGate``): plain ``Gate`` subclasses with a hand-provided
   ``_define()``/``__array__``. ``GPIGate``/``GPI2Gate`` (like ``FSimGate``)
   are re-exported from ``qiskit_qm_provider.additional_gates`` rather than
   defined here.
2. **A Qiskit standard gate exists, but under a different name**
   (``QiboU1Gate``, ``QiboU2Gate``, ``QiboU3Gate``, ``QiboPRXGate``,
   ``QiboU1qGate``, ``QiboCU1Gate``, ``QiboCU2Gate``, ``QiboCU3Gate``,
   ``QiboRXXYYGate``): these **subclass the real standard gate**
   (``PhaseGate``, ``UGate``, ``RGate``, ``CPhaseGate``, ``CUGate``,
   ``XXPlusYYGate``) and only override ``.name`` in ``__init__``. This is
   deliberate, not a shortcut: renaming ``gate_map.py``'s builders onto Qiskit
   standard gates means the *emitted operation name* stops matching the Qibo
   gate a user actually wrote (``PRX``/``U1q`` -> ``r``, ``U2``/``U3`` -> ``u``,
   ``CU2``/``CU3`` -> ``cu``, ``RXXYY`` -> ``xx_plus_yy``) -- a QuAM macro
   installed under the Qibo name then gets a missing-operation error naming a
   gate the user never wrote. Subclassing instead of building a bare
   ``PhaseGate``/etc. and mutating ``.name`` afterwards keeps ``isinstance``
   checks against the real gate class working (verified: ``.name`` is a plain
   mutable attribute, not a property, and renaming this way survives the whole
   export + ``qm_qasm`` dispatch pipeline unchanged) and gets the correct
   ``__array__``/``_define()`` for free from the parent class -- there is
   nothing to get wrong about the unitary since it *is* the standard gate's.

**``_define()`` is best-effort everywhere, and correctness on the QUA path does
not depend on it.** Because these names are passed in ``Exporter``'s
``basis_gates``, they are emitted *opaquely* -- verified directly: a ``gpi``
listed in ``basis_gates`` exports as a bare ``gpi(phi) q[0];`` call with no
``gate gpi(...) {...}`` definition block, and ``qm_qasm`` then dispatches it to
the registered ``gpi`` operation. The definition bodies exist so that Qiskit's
transpiler and ``Operator()``/simulation still work if a caller wants them,
and so the unitary-equivalence tests have something to check against.

The decompositions in family 1 are not hand-derived. ``GPI``/``GPI2`` use
Qibo's own ``qasm_label`` bodies verbatim (``qibo.gates.gates.GPI.qasm_label``
-> ``"gate gpi(phi) q {u3(pi, phi - pi/2, pi/2 - phi) q;}"``), and ``MS`` uses
the identity documented on :class:`QiboMSGate`, all confirmed numerically
against ``qibo.gates.*``'s own ``matrix()`` (see
``test/test_symbolic_lowering.py``).
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.library import CPhaseGate, CUGate, PhaseGate, RGate, UGate, XXPlusYYGate

# fSim, GPI and GPI2 are already defined upstream (GPI/GPI2 since
# qiskit-qm-provider 0.3.5, with the same "gpi"/"gpi2" names and Qibo's own
# qasm_label decompositions) and verified equal to their qibo counterparts --
# re-exported rather than redefined, so there is exactly one definition of each
# in the stack, and the ones QMBackend puts in its Target are the same classes.
from qiskit_qm_provider.additional_gates import FSimGate, GPI2Gate, GPIGate

__all__ = [
    "GPIGate",
    "GPI2Gate",
    "QiboMSGate",
    "FSimGate",
    "QiboU1Gate",
    "QiboU2Gate",
    "QiboU3Gate",
    "QiboPRXGate",
    "QiboU1qGate",
    "QiboCU1Gate",
    "QiboCU2Gate",
    "QiboCU3Gate",
    "QiboRXXYYGate",
]


class QiboMSGate(Gate):
    r"""Qibo's ``MS`` gate: the phased Mølmer-Sørensen gate,
    :math:`\exp\!\left(-i\frac{\theta}{2}\, X_{\phi_0}\otimes X_{\phi_1}\right)`
    where :math:`X_\phi = \cos\phi\,X + \sin\phi\,Y`.

    **Qiskit's own ``MSGate``/``GMS`` cannot be reused here.** Those take a
    symmetric ``theta`` *matrix* over ``num_qubits`` and have nowhere to put the
    two individual phases. Verified: ``qibo.MS(q0, q1, 0, 0, theta)`` *does*
    equal ``MSGate(2, theta)`` (itself equal to ``RXXGate(theta)``), but with
    non-zero ``phi0``/``phi1`` it does not. So this is a genuine 3-parameter
    custom gate, and Qiskit's ``MSGate`` is only its ``phi0 = phi1 = 0``
    special case.

    Named ``QiboMSGate`` rather than ``MSGate`` purely to avoid an import
    collision -- Qiskit's standard gate is *also* named ``"ms"`` at the
    ``.name`` level, which is the name we want emitted.

    Note ``qibo.gates.MS`` validates ``0 <= theta <= pi/2`` numerically in its
    own ``__init__``, so a *symbolic* ``theta`` cannot be constructed on the
    Qibo side at all (it raises ``TypeError: cannot determine truth value of
    Relational``). ``phi0``/``phi1`` may be symbolic. Nothing here enforces
    that -- it is Qibo's constraint, upstream of this class.
    """

    def __init__(
        self,
        phi0: ParameterValueType,
        phi1: ParameterValueType,
        theta: ParameterValueType,
        label: str | None = None,
    ):
        super().__init__("ms", 2, [phi0, phi1, theta], label=label)

    def _define(self) -> None:
        # X_phi = RZ(phi) X RZ(phi)^dag, so
        #   MS = (RZ(phi0) @ RZ(phi1)) . RXX(theta) . (RZ(-phi0) @ RZ(-phi1))
        # Verified numerically against qibo.gates.MS.matrix().
        phi0, phi1, theta = self.params
        qc = QuantumCircuit(2, name=self.name)
        qc.rz(-phi0, 0)
        qc.rz(-phi1, 1)
        qc.rxx(theta, 0, 1)
        qc.rz(phi0, 0)
        qc.rz(phi1, 1)
        self.definition = qc

    def __array__(self, dtype=complex, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi0, phi1, theta = (complex(p) for p in self.params)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        p, m = phi0 + phi1, phi0 - phi1
        # Qiskit is little-endian, so this is the qubit-reversed form of the
        # big-endian matrix in qibo.gates.MS's docstring.
        return np.array(
            [
                [c, 0, 0, -1j * np.exp(-1j * p) * s],
                [0, c, -1j * np.exp(1j * m) * s, 0],
                [0, -1j * np.exp(-1j * m) * s, c, 0],
                [-1j * np.exp(1j * p) * s, 0, 0, c],
            ],
            dtype=dtype,
        )


class QiboU1Gate(PhaseGate):
    r"""Qibo's ``U1`` gate. Identical to Qiskit's ``PhaseGate``
    (:math:`\mathrm{diag}(1, e^{i\theta})`), kept as its own class so its
    ``.name`` stays ``"u1"`` -- Qiskit's own ``PhaseGate`` emits ``"p"``.
    """

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        super().__init__(theta, label=label)
        self.name = "u1"


class QiboU2Gate(UGate):
    r"""Qibo's ``U2`` gate: ``UGate(pi/2, phi, lam)``, named ``"u2"`` --
    Qiskit's own ``UGate`` emits ``"u"``, which is also what :class:`QiboU3Gate`
    would emit, so the two would otherwise collide onto one operation name.
    """

    def __init__(self, phi: ParameterValueType, lam: ParameterValueType, label: str | None = None):
        super().__init__(np.pi / 2, phi, lam, label=label)
        self.name = "u2"


class QiboU3Gate(UGate):
    r"""Qibo's ``U3`` gate: identical to Qiskit's ``UGate(theta, phi, lam)``,
    named ``"u3"`` instead of ``"u"`` -- see :class:`QiboU2Gate`.
    """

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: str | None = None,
    ):
        super().__init__(theta, phi, lam, label=label)
        self.name = "u3"


class QiboPRXGate(RGate):
    r"""Qibo's ``PRX`` (phased-x) gate: identical to Qiskit's
    ``RGate(theta, phi)``, named ``"prx"`` -- Qiskit's own ``RGate`` emits
    ``"r"``, which is also what :class:`QiboU1qGate` would emit despite being a
    distinct Qibo gate, so the two would otherwise collide onto one name.
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, label: str | None = None):
        super().__init__(theta, phi, label=label)
        self.name = "prx"


class QiboU1qGate(RGate):
    r"""Qibo's ``U1q`` gate: the same unitary as ``RGate(theta, phi)`` (and
    therefore as :class:`QiboPRXGate`), kept as a distinct class/name
    (``"u1q"``) because Qibo exposes ``PRX`` and ``U1q`` as two separate
    gates -- see :class:`QiboPRXGate`.
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, label: str | None = None):
        super().__init__(theta, phi, label=label)
        self.name = "u1q"


class QiboCU1Gate(CPhaseGate):
    r"""Qibo's ``CU1`` gate: identical to Qiskit's ``CPhaseGate(theta)``,
    named ``"cu1"`` -- Qiskit's own ``CPhaseGate`` emits ``"cp"``.
    """

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        super().__init__(theta, label=label)
        self.name = "cu1"


class QiboCU2Gate(CUGate):
    r"""Qibo's ``CU2`` gate: ``CUGate(pi/2, phi, lam, gamma=-(phi+lam)/2)``,
    named ``"cu2"`` -- Qiskit's own ``CUGate`` emits ``"cu"``, which is also
    what :class:`QiboCU3Gate` would emit, so the two would otherwise collide.

    The ``gamma`` correction is *derived*, not approximate -- verified
    numerically as a full 4x4 *circuit* unitary (``reverse_bits()`` applied,
    since Qibo is big-endian and Qiskit little-endian; comparing raw gate
    matrices reports a false mismatch here): ``CUGate(..., gamma=0)`` is
    **wrong**. See ``gate_map``'s module docstring. With symbolic ``phi``/
    ``lam``, ``gamma`` becomes a Qiskit ``ParameterExpression`` built from
    ordinary ``+``/``-``/``/`` -- exactly what
    :func:`~qibo_qm_provider.backend.symbolic_parameters.
    sympy_to_qiskit_parameter` produces for those two operands, so this gate is
    what exercises that translator's ``Add``/``Mul`` folding rather than only
    bare symbols.
    """

    def __init__(self, phi: ParameterValueType, lam: ParameterValueType, label: str | None = None):
        super().__init__(np.pi / 2, phi, lam, -(phi + lam) / 2, label=label)
        self.name = "cu2"


class QiboCU3Gate(CUGate):
    r"""Qibo's ``CU3`` gate: ``CUGate(theta, phi, lam, gamma=-(phi+lam)/2)``,
    named ``"cu3"`` instead of ``"cu"`` -- see :class:`QiboCU2Gate` for the
    ``gamma`` derivation and why the name would otherwise collide.
    """

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: str | None = None,
    ):
        super().__init__(theta, phi, lam, -(phi + lam) / 2, label=label)
        self.name = "cu3"


class QiboRXXYYGate(XXPlusYYGate):
    r"""Qibo's ``RXXYY`` gate: identical to Qiskit's ``XXPlusYYGate(theta)``,
    named ``"rxxyy"`` -- Qiskit's own ``XXPlusYYGate`` emits ``"xx_plus_yy"``.
    """

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        super().__init__(theta, label=label)
        self.name = "rxxyy"
