"""Exceptions raised by :mod:`qibo_qm_provider`."""

from __future__ import annotations


class GateNameMappingError(ValueError):
    """Raised when a Qibo gate has no known OpenQASM2/Qiskit equivalent."""


class UnsupportedGateError(NotImplementedError):
    """Raised when a Qibo gate cannot be lowered through the OpenQASM2/3 detour.

    ``qibo.gates.Align`` is the primary example: it has no OpenQASM
    representation, and ``Circuit.to_qasm()`` raises ``NotImplementedError``
    for it directly. This exception wraps that (and similar) failures with a
    message that names the offending gate and qubits.
    """


class UnsupportedParameterError(NotImplementedError):
    """Raised when a symbolic (``sympy``) gate parameter cannot be lowered to a
    Qiskit ``ParameterExpression`` that ``qm_qasm`` is able to compile.

    The binding constraint is **qm_qasm's**, not sympy's or Qiskit's: verified
    directly, ``qm_qasm`` accepts only ``+ - * /`` on a gate argument. A
    function call (``sin``, ``cos``, ``exp``, ``abs``) raises
    ``DisallowedNodeTypeException: Nodes of type <class
    'openqasm3.ast.FunctionCall'> are not allowed``, and ``**`` raises
    ``NotImplementedError: Binary expression undefined: **``. Both would
    otherwise surface deep inside the compiler with no reference to the gate
    that caused them, so they are rejected up front here instead.

    Also raised for a symbol whose name is not a usable OpenQASM3 identifier
    (a reserved word, or not a valid identifier at all) -- otherwise the
    failure appears as an opaque ``QASM3ExporterError`` about a variable name.

    See :mod:`qibo_qm_provider.backend.symbolic_parameters`.
    """


class UnsupportedConnectivityError(NotImplementedError):
    """Raised when a two-qubit gate addresses a physical qubit pair with no
    registered connectivity on the wrapped machine, in that direction.

    Many QM two-qubit natives (e.g. a flux-tunable ``CZ``) are physically
    **asymmetric** -- the flux pulse plays on one specific qubit of the pair
    (``QubitPair.moving_qubit``) -- so ``CZ(a, b)`` and ``CZ(b, a)`` are *not*
    interchangeable, even though Qibo itself treats the gate as
    order-independent and never checks this. ``qiskit_qm_provider``'s
    ``_populate_target`` registers a qubit-pair macro under exactly one
    ordered ``(control, target)`` tuple, taken from the QuAM ``QubitPair``'s
    own roles, with no reverse entry and no symmetric fallback anywhere in
    ``qm_qasm``'s own qubit-pattern matching (verified directly). Raised
    before compilation reaches that point, naming the physical qubits
    involved and, when only the direction is wrong, the order that would work.
    """


class InvalidPlatformName(ValueError):
    """Raised when a ``qibo-qm-provider`` platform name doesn't match the
    ``qibo-qm-{local,iqcc-<backend_name>}`` grammar.

    See :mod:`qibo_qm_provider.qibolab_bridge.platform_naming`.
    """


class UnsupportedEnvelopeError(NotImplementedError):
    """Raised when a QuAM ``Pulse``'s envelope has no qibolab ``Pulse``
    equivalent supported by :func:`qibo_qm_provider.qibolab_bridge.
    platform_from_quam.quam_to_qibolab_platform`.

    Mirrors :class:`UnsupportedGateError`'s role, for the opposite
    (QuAM -> qibolab) conversion direction.
    """


class MissingQuamAttributeError(AttributeError):
    """Raised when converting a QuAM object to a qibolab ``Platform`` needs
    an attribute that isn't present on the given machine (e.g. a qubit
    missing its ``.xy``/``.resonator``/``.z`` channel, or a macro with no
    corresponding pulse). Names the missing attribute and the offending
    qubit/pair/root object in the message.
    """


class UnsupportedWiringError(NotImplementedError):
    """Raised when a QuAM channel's or port's wiring has no supported
    qibolab equivalent in
    :mod:`qibo_qm_provider.qibolab_bridge._quam_wiring`.

    Only OPX1000 MW-FEM (drive/readout) and LF-FEM (flux) wiring is
    supported. Octave/IQ-mixer channels (``IQChannel``/``InOutIQChannel``),
    OPX+ ports, and 2 GSa/s ports raise this instead of being silently
    mis-wired -- a ``Platform`` that misconfigures real hardware is worse
    than one that refuses to build.

    Mirrors :class:`UnsupportedEnvelopeError`'s role, for wiring rather than
    pulse envelopes.
    """


class AmplitudeOutOfRangeError(ValueError):
    """Raised when a QuAM pulse amplitude cannot be expressed as a qibolab
    amplitude.

    qibolab's ``Pulse.amplitude`` is dimensionless and normalised to
    ``[-1, 1]``, and is multiplied by the channel's maximum output voltage
    at playback. A QuAM amplitude exceeding that maximum therefore has no
    valid qibolab representation; raising is preferred over silently
    clipping, which would play a different pulse than the one calibrated.
    Names the offending pulse and channel in the message.
    """
