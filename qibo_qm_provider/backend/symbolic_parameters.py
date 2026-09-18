"""Translate Qibo's ``sympy`` gate parameters into Qiskit ``Parameter``s.

Qibo stores a gate parameter as whatever object was passed in, with no
numeric coercion, so a ``sympy`` expression survives construction intact::

    Circuit(1).add(gates.RX(0, theta=2 * sympy.Symbol("theta") + np.pi / 2))
    # -> gate.parameters == (2*theta + 1.5707963267949,)

There is no Qibo API for lowering that: ``Circuit.to_qasm()`` raises
``TypeError: Cannot convert expression to float``, because OpenQASM **2** has
no symbolic ``input`` declaration. Everything downstream of a Qiskit
``QuantumCircuit`` handles symbols correctly, though, so this module supplies
the missing hop: sympy expression -> Qiskit ``ParameterExpression``.

Supported grammar
-----------------
``Symbol``, ``Add``, ``Mul``, and anything sympy considers a number
(``Float``, ``Integer``, ``Rational``, ``pi``, ...). Everything else raises
:class:`~qibo_qm_provider.exceptions.UnsupportedParameterError`.

**That boundary is `qm_qasm`'s, not sympy's and not Qiskit's.** Qiskit's
``ParameterExpression`` is perfectly happy with ``sin``/``cos``/``**``, and
``qiskit.qasm3`` exports them without complaint -- but ``qm_qasm`` then refuses
them (verified directly against qm-qasm 1.7.7):

- ``rz(sin(theta))`` -> ``DisallowedNodeTypeException: Nodes of type
  <class 'openqasm3.ast.FunctionCall'> are not allowed``
- ``rz(theta**2)`` -> ``NotImplementedError: Binary expression undefined: **``

whereas ``+ - * /`` all compile, arriving in the QUA macro as a live
``QuaBinaryOperation``. Rejecting early, naming the gate, beats surfacing one
of those two errors from deep inside the compiler.

One sympy quirk worth knowing: sympy folds ``t*t`` into ``Pow(t, 2)``, so
squares are unreachable rather than merely unimplemented. There is no way to
express them within what ``qm_qasm`` accepts.
"""

from __future__ import annotations

from typing import Union

import sympy as sp
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression

from qibo_qm_provider.exceptions import UnsupportedParameterError

__all__ = [
    "circuit_has_symbols",
    "gate_has_symbols",
    "sympy_to_qiskit_parameter",
    "validate_symbol_name",
    "QASM3_RESERVED_KEYWORDS",
    "QASM3_BUILTIN_CONSTANTS",
]


def _qiskit_reserved_keywords() -> frozenset:
    """Qiskit's *own* QASM3 reserved-word set, rather than a hand-maintained
    copy of it.

    It is private API (``qiskit.qasm3.exporter._RESERVED_KEYWORDS``), but it is
    the exact set the exporter tests against in ``SymbolTable.symbol_defined``,
    so reusing it is strictly more accurate than re-deriving the list from the
    OpenQASM3 spec and hoping the two agree. Falls back to a small literal set
    if the private name ever moves.
    """
    try:
        from qiskit.qasm3.exporter import _RESERVED_KEYWORDS

        return frozenset(_RESERVED_KEYWORDS)
    except ImportError:  # pragma: no cover - defensive against upstream churn
        return frozenset(
            {
                "OPENQASM", "angle", "array", "barrier", "bit", "bool", "box",
                "break", "cal", "complex", "const", "continue", "creg", "ctrl",
                "def", "defcal", "defcalgrammar", "delay", "duration",
                "durationof", "else", "end", "extern", "float", "for", "gate",
                "gphase", "if", "in", "include", "input", "int", "inv", "let",
                "measure", "mutable", "negctrl", "output", "pow", "qreg",
                "qubit", "reset", "return", "sizeof", "stretch", "uint",
                "while",
            }
        )


#: Words the OpenQASM3 exporter will not let a variable shadow.
QASM3_RESERVED_KEYWORDS = _qiskit_reserved_keywords()

#: OpenQASM3 built-in constants. These are *not* in Qiskit's reserved set, so
#: the exporter happily emits ``input float[64] pi;`` -- but ``qm_qasm`` then
#: raises ``RedeclarationException: The name "pi" already exists and cannot be
#: redeclared`` (verified). Rejected here so the error names the gate.
QASM3_BUILTIN_CONSTANTS = frozenset({"pi", "tau", "euler", "im", "π", "τ", "ℇ"})


def validate_symbol_name(name: str, forbidden_names: frozenset = frozenset()) -> None:
    """Reject a symbol name that cannot survive the lowering as-written.

    The failure this prevents is *not* a loud exporter error -- it is a silent
    **rename**. Qiskit's exporter calls ``register_variable(..., allow_rename=
    True)`` for circuit parameters, so a name that collides with a reserved
    word or with a gate name in scope is quietly re-emitted as e.g. ``x_0``.
    The ``inputs`` dict handed to ``qm_qasm`` is still keyed by the original
    ``parameter.name``, so the compile then fails with
    ``MissingInputException: The input x_0 was not provided to the compiler``
    -- an error that names neither the gate nor the real cause. All verified
    directly against qiskit 2.5.2 / qm-qasm 1.7.7.

    Args:
        name: The sympy symbol's name.
        forbidden_names: Additional names that would collide with a gate in
            scope. Callers pass the operation names the lowering can emit; see
            :data:`qibo_qm_provider.backend.gate_map.EMITTED_OPERATION_NAMES`.

    Raises:
        UnsupportedParameterError: If the name is unusable.
    """
    if not name.isidentifier():
        raise UnsupportedParameterError(
            f"Symbol name {name!r} is not a valid identifier, so it cannot be "
            f"exported as an OpenQASM3 'input' declaration. Rename the sympy "
            f"symbol to a plain identifier (letters, digits, underscore, not "
            f"starting with a digit)."
        )
    if name in QASM3_RESERVED_KEYWORDS:
        raise UnsupportedParameterError(
            f"Symbol name {name!r} is an OpenQASM3 reserved keyword. Qiskit's "
            f"exporter would silently rename it (to {name}_0 or similar), and "
            f"the compile would then fail with a MissingInputException that "
            f"does not mention this gate. Rename the sympy symbol, e.g. "
            f"{name}_ or my_{name}."
        )
    if name in QASM3_BUILTIN_CONSTANTS:
        raise UnsupportedParameterError(
            f"Symbol name {name!r} is an OpenQASM3 built-in constant, so "
            f"qm_qasm rejects redeclaring it (RedeclarationException). Rename "
            f"the sympy symbol, e.g. {name}_ or my_{name}."
        )
    if name in forbidden_names:
        raise UnsupportedParameterError(
            f"Symbol name {name!r} collides with a gate/operation name in scope, "
            f"which makes Qiskit's exporter silently rename the parameter and "
            f"the compile fail with a MissingInputException. Rename the sympy "
            f"symbol, e.g. {name}_ or my_{name}."
        )


def gate_has_symbols(gate) -> bool:
    """Whether any of ``gate``'s parameters is a symbolic sympy expression."""
    for param in getattr(gate, "parameters", ()) or ():
        if isinstance(param, sp.Basic) and param.free_symbols:
            return True
    return False


def circuit_has_symbols(circuit) -> bool:
    """Whether ``circuit`` carries at least one symbolic gate parameter.

    Preferred over a ``try``/``except`` around ``Circuit.to_qasm()``: it is
    cheap, it does not depend on which exception type Qibo happens to raise
    (today a bare ``TypeError``), and it does not conflate "has symbols" with
    the several other reasons ``to_qasm()`` can fail (e.g. ``Align``).
    """
    return any(gate_has_symbols(gate) for gate in circuit.queue)


def sympy_to_qiskit_parameter(
    expr,
    registry: dict[str, Parameter],
    forbidden_names: frozenset = frozenset(),
) -> Union[ParameterExpression, float]:
    """Rebuild a sympy expression as a Qiskit ``ParameterExpression``.

    Args:
        expr: The value held in ``gate.parameters``. A plain ``int``/``float``
            passes straight through; a sympy expression is rebuilt node by
            node.
        registry: Mutable ``{symbol_name: Parameter}`` cache, shared across a
            whole circuit so that one sympy symbol maps to exactly one Qiskit
            ``Parameter`` object no matter how many gates mention it. Callers
            pass the same dict for every gate in a circuit.
        forbidden_names: Forwarded to :func:`validate_symbol_name`.

    Returns:
        A ``ParameterExpression`` if ``expr`` involves free symbols, otherwise
        a plain ``float``.

    Raises:
        UnsupportedParameterError: For any node outside the supported grammar
            (see the module docstring), or a symbol whose name is not a usable
            OpenQASM3 identifier.
    """
    if not isinstance(expr, sp.Basic):
        # Already a concrete Python/NumPy number.
        return expr

    if isinstance(expr, sp.Symbol):
        validate_symbol_name(expr.name, forbidden_names)
        return registry.setdefault(expr.name, Parameter(expr.name))

    # NOTE: `is_number` (lowercase), not the class-check `is_Number`. The
    # former is True for `pi`, `Rational`, and any symbol-free expression;
    # the latter is False for `pi`, which would send it down the unsupported
    # branch below.
    if expr.is_number:
        return float(expr)

    args = [sympy_to_qiskit_parameter(arg, registry, forbidden_names) for arg in expr.args]

    if expr.func is sp.Add:
        result = args[0]
        for arg in args[1:]:
            result = result + arg
        return result

    if expr.func is sp.Mul:
        result = args[0]
        for arg in args[1:]:
            result = result * arg
        return result

    raise UnsupportedParameterError(
        f"Cannot lower sympy expression {expr!r} (node type "
        f"{type(expr).__name__}) to a real-time QUA parameter. qm_qasm accepts "
        f"only '+ - * /' on a gate argument -- function calls (sin, cos, exp, "
        f"abs) and '**' are rejected by its own compiler. Note sympy folds "
        f"'t*t' into 'Pow(t, 2)', so squares cannot be expressed either. "
        f"Either bind this parameter to a concrete value before lowering, or "
        f"restructure it as a sum/product of symbols."
    )
