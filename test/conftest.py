"""Synthetic, in-memory pytest fixtures for ``qibo_qm_provider`` tests.

``qiskit_qm_provider``'s own test suite requires a ``QUAM_STATE_PATH`` env var
pointing at a real, on-disk QuAM state (wiring, calibrated LO/mixer settings,
octave config, etc.) because faking that convincingly -- to the point where
``generate_config()``/``connect()`` would produce something QOP-shaped -- is
nontrivial. This package needs tests that run in CI with zero external state
and no dependency on a real (or even plausibly fake) hardware setup, so
instead these fixtures build a ``FluxTunableQuam`` entirely in Python:

- Ports are fake ``("con1", <n>)`` tuples -- never resolved against a real
  controller.
- LO/mixer/frequency-converter objects hold made-up numbers.
- Nothing here ever calls ``.generate_config()``, ``.connect()``, or opens a
  ``QuantumMachine``. Only quam dataclass construction happens, so no OPX
  hardware or QOP connection is required to use these fixtures.

The result is "hardware-shaped enough" for ``quam``/``quam_builder`` object
construction and for ``qiskit_qm_provider.quam_macros.superconducting.
add_basic_macros`` to run its macro-seeding logic (which only touches Python
objects), without being hardware-accurate.
"""

from __future__ import annotations

import warnings

import pytest
from quam.components.hardware import FrequencyConverter, LocalOscillator, Mixer
from quam.components.pulses import SquarePulse
from quam_builder.architecture.superconducting.components.flux_line import FluxLine
from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorIQ,
)
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import FluxTunableQuam
from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import (
    FluxTunableTransmon,
)
from quam_builder.architecture.superconducting.qubit_pair.flux_tunable_transmon_pair import (
    FluxTunableTransmonPair,
)


def _make_qubit(name: str, port_offset: int) -> FluxTunableTransmon:
    """Build a fully in-memory ``FluxTunableTransmon`` with fake wiring.

    ``port_offset`` just keeps the (fake) controller ports distinct between
    qubits -- nothing here is ever resolved against a real controller.
    """
    xy_i, xy_q = 2 * port_offset + 1, 2 * port_offset + 2
    z_port = 3 * port_offset + 3

    xy = XYDriveIQ(
        opx_output_I=("con1", xy_i),
        opx_output_Q=("con1", xy_q),
        frequency_converter_up=FrequencyConverter(
            local_oscillator=LocalOscillator(frequency=5.0e9 + port_offset * 1e8, power=10),
            mixer=Mixer(),
        ),
        intermediate_frequency=100e6,
        operations={
            "x180": SquarePulse(length=40, amplitude=0.1),
            "x90": SquarePulse(length=40, amplitude=0.05),
            "y90": SquarePulse(length=40, amplitude=0.05),
            "-y90": SquarePulse(length=40, amplitude=-0.05),
        },
    )
    resonator = ReadoutResonatorIQ(
        opx_output_I=("con1", xy_i),
        opx_output_Q=("con1", xy_q),
        opx_input_I=("con1", xy_i),
        opx_input_Q=("con1", xy_q),
        frequency_converter_up=FrequencyConverter(
            local_oscillator=LocalOscillator(frequency=7.0e9 + port_offset * 1e8, power=10),
            mixer=Mixer(),
        ),
        intermediate_frequency=50e6,
        operations={"readout": SquarePulse(length=100, amplitude=0.1)},
    )
    z = FluxLine(
        opx_output=("con1", z_port),
        # "const" is what quam_builder's CZGate uses as the default flux pulse
        # on the moving qubit -- harmless to define even though, in this
        # installed qiskit_qm_provider version, add_basic_macros never
        # successfully constructs a CZGate (see add_basic_macros_installed
        # below).
        operations={"const": SquarePulse(length=40, amplitude=0.1)},
    )
    return FluxTunableTransmon(id=name, xy=xy, resonator=resonator, z=z)


@pytest.fixture
def dummy_machine() -> FluxTunableQuam:
    """A fully constructed, in-memory ``FluxTunableQuam`` with 2 qubits and 1 pair.

    ``q0``/``q1`` are real ``FluxTunableTransmon`` instances with non-None
    ``xy`` (``XYDriveIQ``), ``resonator`` (``ReadoutResonatorIQ``), and ``z``
    (``FluxLine``) channels -- all real ``quam.components.channels``
    subclasses (``IQChannel``, ``InOutIQChannel``, ``SingleChannel``
    respectively), just wired to fake ports. ``q0-q1`` is a
    ``FluxTunableTransmonPair`` with ``coupler=None`` (no tunable coupler
    wired up -- this package doesn't need one for its tests).

    Scope is ``function`` (the default) so each test gets its own machine and
    mutations (e.g. installing macros) never leak between tests.
    """
    machine = FluxTunableQuam()

    machine.qubits["q0"] = _make_qubit("q0", port_offset=0)
    machine.qubits["q1"] = _make_qubit("q1", port_offset=1)

    # Qubits must already be attached to `machine.qubits` before the pair
    # references them -- a quam object can only have one parent, so the pair
    # points at them via string reference rather than embedding the same
    # objects twice (this is the same pattern quam_builder's own test suite
    # uses, e.g. `qubit_control="#/qubits/q0"` in quam-builder's
    # test_modify_quam.py).
    machine.qubit_pairs["q0-q1"] = FluxTunableTransmonPair(
        id="q0-q1",
        qubit_control="#/qubits/q0",
        qubit_target="#/qubits/q1",
        coupler=None,
    )

    machine.active_qubit_names = ["q0", "q1"]
    machine.active_qubit_pair_names = ["q0-q1"]

    return machine


@pytest.fixture
def add_basic_macros_installed(dummy_machine: FluxTunableQuam) -> FluxTunableQuam:
    """``dummy_machine`` after running ``add_basic_macros`` on it.

    Exercises the macro-installation path end to end. Each qubit's
    ``.macros`` dict ends up populated with ``x, sx, sy, sydg, rz, measure,
    reset, delay, id``, since that part of ``add_basic_macros`` only touches
    plain Python objects and has no coupler/hardware dependency.

    The pair's ``cz`` macro previously (qiskit-qm-provider 0.3.2) hit a real
    upstream version-mismatch bug: ``add_basic_macros`` called
    ``CZGate(flux_pulse_control=...)``, but the installed quam_builder's
    ``CZGate`` dataclass field is named ``flux_pulse_qubit`` -- and since the
    call was wrapped in ``except ValueError`` while it actually raised
    ``TypeError``, the exception was never caught there. **Fixed upstream in
    qiskit-qm-provider 0.3.3** (confirmed against the currently installed
    version: ``CZGate(flux_pulse_qubit=...)`` now installs cleanly, no
    exception, no dangling ``None`` entry). The ``try/except`` below is kept
    as a defensive fallback (matching the "warn rather than raise if
    two-qubit macro installation doesn't succeed" behavior
    ``add_basic_macros`` itself is clearly meant to have for e.g. a genuinely
    coupler-less pair) rather than removed, since it is still reachable for
    other failure modes (e.g. no coupler wired up at all) -- but it is no
    longer masking the ``flux_pulse_control``/``flux_pulse_qubit`` bug
    specifically.

    If this ``try/except`` ever fires again, note upstream leaves
    ``qubit_pair.macros["cz"] = None`` dangling on failure (it pre-assigns
    ``None`` immediately before attempting the real ``CZGate(...)``
    construction, per its own source) -- confirmed empirically that this
    broken ``None`` entry, left in place, crashes
    ``QMBackend.__init__``/``_populate_target()`` with an unrelated
    ``AttributeError: 'NoneType' object has no attribute 'apply'`` for *any*
    test that goes on to construct a backend from this machine. So the
    cleanup below explicitly pops that dangling entry, restoring the
    fixture to "no cz macro installed" rather than "a broken one."
    """
    from qibo_qm_provider import add_basic_macros

    try:
        add_basic_macros(dummy_machine)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring.
        warnings.warn(
            "add_basic_macros raised while installing the two-qubit 'cz' macro on "
            f"the synthetic 'q0-q1' pair (no coupler wired up): {exc!r}. Single-qubit "
            "macros were already installed on both qubits before this point.",
            stacklevel=2,
        )
        pair = dummy_machine.qubit_pairs["q0-q1"]
        if pair.macros.get("cz") is None:
            pair.macros.pop("cz", None)

    return dummy_machine
