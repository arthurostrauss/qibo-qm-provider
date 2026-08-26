# Integration status vs. `architecture_preliminary_insights.md`

**This document reads newest-first.** Each `##` section is a dated slice of
work; the most recent is immediately below, the original 2026-08-07 slice and
its framing are near the bottom, and the `Align` appendix is last. Earlier
sections were written when they were current, so where a later section
contradicts an earlier one, **the later one wins** — the corrections are
called out explicitly rather than by silently editing history.

| Date | Slice | Section |
|---|---|---|
| 2026-08-26 | Gate renaming eliminated at the source; `qiskit-qm-provider` `OperationIdentifier` bug root-caused and fixed upstream | [Gate renaming and an upstream `qiskit-qm-provider` fix](#gate-renaming-and-an-upstream-qiskit-qm-provider-fix-2026-08-26) |
| 2026-08-26 | Symbolic (sympy-parameter) Qibo-circuit → QUA lowering; bug #3 closed | [Symbolic Qibo-circuit lowering](#symbolic-qibo-circuit-lowering-2026-08-26) |
| 2026-08-24 | `sequence_to_qua_macro`/`circuit_to_qua_macro`; revised target architecture | `QiboQMPlatformBackend` |
| 2026-08-23 | Platform-name resolution gap traced | Platform name resolution |
| 2026-08-19/23 | `QmController` instrument wiring, hardware-validated | `QiboQMPlatformBackend` |
| 2026-08-18 | `QiboQMPlatformBackend`, the qibolab-native second path | `QiboQMPlatformBackend` |
| 2026-08-07 | First vertical slice, validated on `"arbel"` | Executive summary and below |

This note tracks what actually exists in `qibo_qm_provider` today against the
objectives laid out in `architecture_preliminary_insights.md`, including
where the implementation deliberately deviated from that note's
recommendations, and what was learned by running the current code against a
real IQCC-hosted QM backend (`"arbel"`).

Companion documents: `symbolic_circuit_lowering.md` (the 2026-08-26 path, in
depth, plus the findings and compromises behind it), `slice2_plan.md` (what is
next for it), `qibo_backend_vs_qibolab_platform.md`,
`qibolab_platform_from_quam_plan.md`, `qibocal_multi_qubit_handling.md`.

## Gate renaming and an upstream `qiskit-qm-provider` fix (2026-08-26)

Direct follow-up on the section immediately below, prompted by review of it.
Two independent pieces of work, both landing the same day as the lowering
itself.

### 1. The nine-gate renaming problem is now structurally impossible

The section below originally closed the renaming gap (`U1`/`U2`/`U3`/`PRX`/
`U1q`/`CU1`/`CU2`/`CU3`/`RXXYY` landing on a Qiskit standard gate's own name
instead of Qibo's) with `QiboQMBackend.install_qibo_gate_aliases()` — walk the
machine, install a QuAM reference alias under the emitted name wherever a
macro existed under the Qibo name. That method, `gate_map.QIBO_NAME_TO_EMITTED`,
and its six tests are now **removed**, replaced by a fix at the source:
`qibo_qiskit_gates.py` gained nine thin subclasses (`QiboU1Gate`, `QiboU2Gate`,
`QiboU3Gate`, `QiboPRXGate`, `QiboU1qGate`, `QiboCU1Gate`, `QiboCU2Gate`,
`QiboCU3Gate`, `QiboRXXYYGate`) of the real standard gate (`PhaseGate`,
`UGate`, `RGate`, `CPhaseGate`, `CUGate`, `XXPlusYYGate`), each overriding only
`.name` to Qibo's own name in `__init__`.

This is not a workaround-of-a-workaround: `.name` is a plain mutable instance
attribute on a Qiskit `Gate` (not a property), and subclassing rather than
instantiate-then-mutate keeps `isinstance` against the real gate class true —
verified directly that renaming this way survives export + `qm_qasm` dispatch
unchanged, and that it was already established elsewhere in this same path
that a custom gate's `_define()`/unitary is unused on the QUA route (opaque
`basis_gates` dispatch), so there is nothing to get wrong by subclassing. Net
effect: `QIBO_TO_OPERATION_NAME[name] == mapped_gate.name` for every entry in
`QIBO_TO_QISKIT`, with no exceptions — the invariant is pinned directly by
`test_no_gate_is_ever_renamed`.

### 2. `qiskit-qm-provider`: `OperationIdentifier` has no value equality — found, root-caused, fixed upstream

While tracing exactly why `register_gate` cannot rebind an existing operation
(recorded as an open, undiagnosed limitation in the section below), the actual
mechanism was found: `qm_qasm.OperationIdentifier` defines neither `__eq__`
nor `__hash__`, so it falls back to Python's default — object identity.
Verified directly against qm-qasm 1.7.7:

```python
a = OperationIdentifier("rz", 1, (0,))
b = OperationIdentifier("rz", 1, (0,))
a == b                  # False
hash(a) == hash(b)      # False
```

`qiskit_qm_provider.backend.qm_backend.QMBackend` keyed its internal
`_operation_mapping_QUA`/`_calibration_operation_mapping_QUA` caches — the
ones actually consumed by `Compiler.compile()` — directly by these objects.
Every call to `_populate_target()` (at construction, and again inside
`update_target()`) builds a **fresh** `OperationIdentifier` for the same
logical operation and assigns it into the dict; since that key is never equal
to the one already present, the assignment silently **adds a duplicate**
rather than overwriting. Traced with a live dict dump: after one
`register_gate("rz", ...)` call, three "rz" entries coexist with identical
`repr()`. Whatever internal matching `qm_qasm.Compiler` does at compile time
resolves to the *first* one ever inserted — the original macro from
construction — which is exactly the previously-unexplained symptom.

**Same root cause, second bug, found in passing:**
`backend_utils.has_conflicting_calibrations` builds a `set()` of
`OperationIdentifier` objects to detect a repeated calibration; with no value
equality, `op_id not in custom_gates` is unconditionally `True`, so the
function could never detect a real conflict — it always silently returned
`False`. Confirmed directly: the exact old logic, run against a manufactured
conflicting-calibration scenario, returns `False`; the fixed logic returns
`True` for the identical scenario.

**Fixed upstream, in the real source repository**
(`~/Library/CloudStorage/OneDrive-QMMachinesLTD/GitHub/qiskit-qm-provider`,
not the installed copy): added `backend_utils.operation_key(name,
number_of_params, qubits) -> tuple`, a plain value-hashable stand-in, and
rewired every write site in `_populate_target`/`update_target`/
`update_calibrations`/`has_conflicting_calibrations` to key on it instead.
Real `OperationIdentifier` objects — which `HardwareConfig` does require — are
now materialized fresh, once, only in the `compiler` property, *after* the
canonical mapping is known to be correctly deduplicated. Added regression
tests in that repo's own `test/test_backend_utils.py`
(`TestOperationKey`, `TestHasConflictingCalibrations`).

**Not yet released.** This package's declared dependency floor
(`qiskit-qm-provider>=0.3.4`) predates the fix — a plain `pip install` today
still has the bug. Verified end to end by mirroring the two patched files into
this repo's own `.venv` site-packages copy: the previously-`xfail`
(`strict=False`) `test_register_gate_cannot_override_an_existing_operation`
now `XPASS`es, with no other regressions (full suite: 213 passed, 5
deselected, 1 xpassed). `register_gate`'s docstring updated to describe the
fix and its release status rather than an unexplained limitation; the
workaround (install the macro on the machine *before* constructing the
backend) is unchanged and still necessary until a fixed release ships.

**Not filed as an upstream issue yet** — this section records the diagnosis
precisely enough to file one against `qm_qasm` (missing `__eq__`/`__hash__`
on `OperationIdentifier`) if that turns out to be preferred over carrying the
`operation_key` workaround indefinitely in `qiskit-qm-provider`.

## Symbolic Qibo-circuit lowering (2026-08-26)

**Built.** A second lowering route inside `QiboQMBackend` for circuits whose
gate parameters are `sympy` expressions, so a Qibo circuit can be compiled once
and re-parameterized in real time from QUA — the Qibo-side counterpart of what
`ParameterTable.from_qiskit` + `QMBackend.quantum_circuit_to_qua` already do for
Qiskit. Full write-up, including every verified claim and the compromises taken,
is in `symbolic_circuit_lowering.md`; this entry is the status summary.

The motivating wall: Qibo happily stores a sympy expression as a gate parameter,
but `Circuit.to_qasm()` then raises `TypeError: Cannot convert expression to
float`, because OpenQASM**2** has no symbolic `input`. Everything downstream of a
Qiskit `QuantumCircuit` already handles symbols correctly, so the fix is narrow:
for symbolic circuits, skip `to_qasm()` and build the Qiskit circuit directly.

New modules under `qibo_qm_provider/backend/`:

- `symbolic_parameters.py` — `circuit_has_symbols` (the branch predicate) and
  `sympy_to_qiskit_parameter` (sympy → Qiskit `ParameterExpression`), plus
  parameter-name validation.
- `gate_map.py` — the verified Qibo→Qiskit gate correspondence and the
  deliberately-unmapped gate list. Every emitted name equals the mapped gate's
  own Qibo name (see the section above); no renaming table survives.
- `qibo_qiskit_gates.py` — `GPIGate`, `GPI2Gate`, `QiboMSGate`; re-exports
  upstream `FSimGate`; plus the nine renamed-standard-gate subclasses from the
  section above.
- `parameter_table.py` — `QiboParameterTable.from_qibo_circuit`, delegating to
  `from_qiskit`.
- `circuit_conversion.py` — gains `build_qiskit_circuit_directly` and the
  automatic branch; the existing OpenQASM2 route is otherwise untouched.

**Deliberately `QiboQMBackend`, not `QiboQMPlatformBackend`.** Symbolic
parameters are a property of Qibo *circuits* and cannot be expressed in Qibolab
at all (`qibolab._core.native.rotation()` raises `TypeError` on a symbolic
angle; every rule in `_core/compilers/default.py` does eager arithmetic). This
is the same split already argued in `qibolab_platform_from_quam_plan.md` §8, and
it is the concrete capability that makes keeping `QiboQMBackend` permanently
worthwhile rather than transitional.

**Validated** (`test/test_symbolic_lowering.py`, 112 tests + 1 documented
xfail/xpass; full suite 213 passed, no regressions — see the section above for
why one test currently xpasses in this repo's own `.venv`). No hardware or
IQCC credentials needed — `qm_qasm.Compiler.compile` and `qm.generate_qua_script`
both run offline. Headline end-to-end check: a symbolic `RZ` compiles to
`frame_rotation_2pi(((0.0-v2)*0.15915494309189535), 'q0.xy')`, which asserts in
one line that the symbol survived as a live QUA *variable*, that the **minus
sign** of the virtual-Z was applied, and that it landed on the right qubit's
`xy` element. Unit-verified against Qibo's own `matrix()`: all 42 mapped gates,
and both the `__array__` and `_define()` bodies of every custom/renamed gate
class (thirteen, after the section above).

**Not yet done:** no live-hardware run of this path, and no
`QiboQMBackend.circuit_to_qua` convenience entry point yet — callers currently
compose `qibo_circuit_to_qiskit` + `QiboParameterTable` + the wrapped backend's
`quantum_circuit_to_qua` themselves. Both are slice 2 (`slice2_plan.md`).

### Corrections to earlier sections of this document

1. **Bug #3 (`I` → `u(0,0,0)`) is fixed.** Earlier sections list it as open with
   the root cause "inside Qiskit's own OQ2 parser normalization, unaffected by
   anything Qibo emits". The cause was right; the conclusion that it was not
   ours to fix was wrong. `Circuit.to_qasm()` always emits
   `include "qelib1.inc"`, and `qiskit.qasm2.loads` does not build that whole
   library in — passing `custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS`
   supplies it, and `id` then stays `id`. One line in `circuit_conversion.py`.
2. **The same fix closes a previously unnoticed, larger gap.** Without those
   definitions, `swap`, `crx`, `cry`, `crz`, `rzz`, `rxx`, `ryy` — ordinary
   `qelib1.inc` gates — all failed the OpenQASM2 route with
   `QASM2ParseError: '<gate>' is not defined in this scope`. This was never
   recorded as a limitation because nothing exercised those gates; the
   equivalence test between the two lowering routes found it. `iSWAP` still
   fails there (genuinely absent from `qelib1.inc`) and is supported on the
   symbolic route instead.
3. **`register_gate`'s documented contract was wrong in two ways**, both now
   corrected in its docstring: a bare callable is *not* accepted
   (`_populate_target` reads `macro.apply` unconditionally), and registering a
   name that **already exists** does not reach the compiler — root-caused
   precisely in the section above (`qm_qasm.OperationIdentifier` has no value
   equality) and fixed upstream in source, pending release. Installing the macro on
   the machine *before* constructing the backend does work, and is the
   documented workaround. Both behaviours are pinned by tests.

## `QiboQMPlatformBackend`: a second, qibolab-native execution path (2026-08-18)

The 2026-08-07 sections further down this note describe `QiboQMBackend`'s OpenQASM/QuAM-macro
path, which is unchanged and remains the only path that actually executes
circuits on hardware today. This section documents an **additional**,
separate backend added this session: `QiboQMPlatformBackend`, which wraps a
genuine `qibolab.Platform` built directly from a QuAM object, so that
Qibocal's existing calibration protocols (written purely against
`platform.execute(...)`/`platform.natives`, nothing driver-specific) work
against QM hardware by reusing qibolab's own `Compiler`/`QmController` stack
instead of an OpenQASM round-trip. Full design rationale lives in
`qibo_backend_vs_qibolab_platform.md`, `qibocal_multi_qubit_handling.md`, and
`qibolab_platform_from_quam_plan.md` (repo root); this section only tracks
what actually got built, mirroring this note's own style for the rest of the
package.

### What's built

- **Folder-naming contract** (`qibolab_bridge/platform_naming.py`):
  `qibo-qm-local` / `qibo-qm-iqcc-<backend_name>`, parsed by fixed-prefix
  stripping (not positional hyphen-splitting, so a hyphenated IQCC backend
  name is safe). `InvalidPlatformName` raised on anything else.
- **The converter** (`qibolab_bridge/_quam_platform_conversion.py`,
  `platform_from_quam.py`): `quam_to_qibolab_platform(machine, name)` builds
  a qibolab `Platform`'s `qubits`/`couplers` (QuAM qubit/pair names used
  verbatim as qibolab `QubitId`s, a deliberate deviation from
  `native_import.py`'s `str(qubit_id)` convention — see that module's
  docstring) and `parameters.native_gates`, reading QuAM's high-level
  `.macros` dict (not raw `.operations`) via the shared macro-name tables
  now in `qibolab_bridge/naming.py` — a deliberate choice, made after
  discussion, for robustness to custom gate implementations over reading
  fixed pulse names directly.
  - Verified empirically against the real `add_basic_macros_installed`
    fixture, not assumed: `CZGate.flux_pulse_qubit` resolves directly to a
    real QuAM `Pulse` object (not a string needing a secondary lookup), and
    `pair.moving_qubit` correctly selects which qubit's flux channel a `CZ`
    native plays on — more accurate than qibolab's own generic
    `initialize_parameters` scaffolding default, which doesn't have that
    information.
  - Pulse-envelope conversion (`_quam_envelope_to_qibolab_pulse`):
    `SquarePulse`/`GaussianPulse` convert losslessly to qibolab's symbolic
    `Rectangular`/`Gaussian` envelopes. Everything else (`DragCosinePulse`
    included — verified it has no `rel_sigma`-equivalent field at all, so a
    qibolab `Drag` envelope can't be reconstructed from it) falls back to
    sampling QuAM's own `Pulse.calculate_waveform()` (universal on every
    QuAM pulse type, verified directly) and wrapping the result as a
    qibolab `Custom` envelope (`amplitude=1.0`, since the samples come out
    already amplitude-scaled — confirmed empirically). `UnsupportedEnvelopeError`
    now only fires on a genuine conversion failure, not merely an
    unrecognized envelope class.
  - **No longer true as of 2026-08-19/2026-08-23** (was: "`Platform.instruments`
    is deliberately `{}`, no `QmController`, blocked on missing MW-FEM wiring
    details"). That blocker turned out to be wrong on inspection: qibolab
    already ships full, dedicated MW-FEM support
    (`MwFemOscillatorConfig`/`configure_mw_fem_line`/
    `configure_mw_fem_acquire_line`, `ModuleTypes = Literal["opx1","LF","MW"]`)
    — nothing was missing on qibolab's side, only the converter code. See
    "Instrument wiring: from deliberately-deferred to hardware-validated"
    below for what got built and how it was verified against real,
    21-qubit, multi-bank `"arbel"` hardware. Mixer calibration
    (`IqMixerConfig`) remains genuinely out of scope — QuAM's `Mixer`
    (`correction_gain`/`correction_phase`) has no field correspondence to
    qibolab's `IqMixerConfig` (`offset_i`/`offset_q`/`scale_q`/`phase_q`);
    real-hardware mixer correction is left to `QmController`'s own separate
    `calibration_path` mechanism, unchanged from the original assessment.
- **Fetch/load entrypoints** (`create_iqcc`/`create_local` in
  `platform_from_quam.py`): both folder-driven (a `quam_source.json`
  sidecar supplies `state_path`/`quam_class` defaults) and directly
  callable with explicit keyword arguments — one implementation
  (`_create_iqcc_with_machine`/`_create_local_with_machine`) serves both
  call styles. `create_iqcc` lazily imports and wraps
  `qiskit_qm_provider.providers.iqcc_cloud_provider.get_machine_from_iqcc`,
  re-raising its `ValueError`/`ConnectionError` (confirmed these are the
  actual types `iqcc_cloud_client`'s HTTP error handling raises) with the
  offending backend name attached.
- **`QiboQMPlatformBackend`** (`backend/qibo_qm_platform_backend.py`):
  subclasses `qibolab._core.backends.QibolabBackend` directly —
  `execute_circuit`/`execute_circuits` are inherited unchanged (qibolab's
  own `Compiler.compile` + `platform.execute`, no OpenQASM detour). Adds
  `from_iqcc`/`from_local`/`from_machine` classmethods (native-Python
  entrypoints, no `$QIBOLAB_PLATFORMS` folder registration required) and
  `refresh()` (full re-derivation of `self.platform` from `self.machine`
  after mutating the latter directly). No `register_gate`/`update_target`
  equivalent — see "Revised target architecture" below for why, and what's
  planned instead.
- **`qibo_qm_provider.MetaBackend`**: registers this package as a
  standard Qibo backend name, so `qibo.set_backend("qibo_qm_provider",
  platform=...)` resolves to `QiboQMPlatformBackend` — deliberately *not*
  routing through `qibolab`'s own `MetaBackend` (hardcoded to always
  construct a plain `QibolabBackend`). Verified end-to-end, not just unit
  tested: constructing a `Platform` from a QuAM object, registering it via
  `qibo.set_backend("qibo_qm_provider", platform=platform)`, and confirming
  `qibo.backends._Global.backend()` is a `QiboQMPlatformBackend` instance.
  Also verified that plain `qibo.set_backend("qibolab", platform=...)`
  continues to work unmodified on the exact same `Platform` object,
  returning a bare `QibolabBackend` — this package's converter output has
  no dependency on which backend class wraps it.

### Test status

As of 2026-08-19: 21 new tests across `test_platform_naming.py`,
`test_platform_from_quam.py`, `test_qibo_qm_platform_backend.py`. Full suite:
57 passed, 2 deselected (the `iqcc`-marked live-hardware test, unaffected by
this work) — no regressions to the existing `QiboQMBackend`/
`FluxTunableTransmonBackend`/`circuit_conversion`/`measurement_translation`/
`qibolab_bridge` tests. Everything above is unit-tested against the real,
in-memory `dummy_machine`/`add_basic_macros_installed` fixtures already in
`test/conftest.py` (no new fixture needed) — at that point, **none of it had
been exercised against real hardware or a real IQCC connection**, unlike
`QiboQMBackend`'s `X`-gate result in the 2026-08-07 sections below.

As of 2026-08-23, after the instrument-wiring work below: full offline suite
is **77 passed, 5 deselected** (`test/`, no `-m iqcc`), and the 3
`iqcc`-marked tests added for the wiring work (`test_iqcc_platform_wiring.py`)
now **pass live against real `"arbel"` hardware** — see "Instrument wiring"
below for what they check and what they found.

As of 2026-08-24, after the QUA-macro work (see "Built (2026-08-24)" above)
and the `qibolab_bridge` re-scope (`naming.py`/`_quam_pulses.py` extracted;
`pulse_sequence_import.py` renamed to `native_import.py`, no public-API
change): full offline suite is **101 passed, 5 deselected** — 24 new tests
across `test_quam_pulses.py` (all 7 qibolab envelope kinds round-trip
through the shared QuAM <-> qibolab converter, previously 3 of 7) and
`test_qua_macros.py` (operation-index shape-matching, missing-operation
registration, `Delay`/`VirtualZ`/`Align`/`Readout` emission, real-time
parameter overrides), plus new cases in `test_quam_wiring.py` and
`test_qibo_qm_platform_backend.py`. No regressions to any pre-existing test.

### Instrument wiring: from deliberately-deferred to hardware-validated (2026-08-19/2026-08-23)

The 2026-08-18 note above said building `Platform.instruments` was blocked
on real IQCC MW-FEM wiring details being unavailable. That premise was wrong:
inspecting qibolab's own QM driver (`qibolab._core.instruments.qm`) showed
full, dedicated MW-FEM support already shipped
(`MwFemOscillatorConfig`, `configure_mw_fem_line`,
`configure_mw_fem_acquire_line`, `ModuleTypes = Literal["opx1","LF","MW"]`)
— confirmed directly by fetching the real `"arbel"` machine and reading its
wiring (`machine.ports`, `machine.network`), which turned out to be **100%
OPX1000 MW-FEM/LF-FEM** (drive+readout on MW-FEM, flux on LF-FEM; no Octave
anywhere on this rig). Nothing was missing on qibolab's side; only the
converter was.

**What got built** (new module `qibolab_bridge/_quam_wiring.py`, plus
`platform_from_quam._build_qm_controller` assembling it into a `QmController`):

- Channel/config conversion dispatches on QuAM's own channel class
  (`MWChannel` -> MW-FEM, `SingleChannel` -> LF-FEM flux, `IQChannel` ->
  `UnsupportedWiringError`, deliberately out of scope — no Octave rig has
  been available to validate against). One shared LO config per *physical*
  MW-FEM port (not per qubit) — required because `"arbel"`'s 21 qubits share
  only 4 physical readout ports (one per bank A/B/C/D); qibolab's
  `MwFemOutput.update()` asserts matching band/power/sampling-rate across
  qubits sharing a port, which the real machine's data does satisfy.
  `IqConfig.frequency` is qibolab's **absolute RF**, derived as `LO + IF`
  from QuAM's signed `intermediate_frequency` — confirmed algebraically
  against every qubit on `"arbel"` (`f_01 == upconverter_frequency +
  intermediate_frequency`, exactly, for all 21).
- Flux (LF-FEM): `OpxOutputConfig.offset` is read from `FluxLine.flux_point`
  (`joint_offset` on `"arbel"`) rather than the port's own `offset` field
  (which QuAM always writes as `0.0`, since QuAM biases flux at *runtime* in
  QUA, not statically in config) — a deliberate inversion, since a qibolab
  `Platform` has no QUA prologue to apply a runtime bias in.
- **Correction (2026-08-24): the "feedforward_filter is not transferred"
  claim below was wrong — qibolab does support OPX1000 filter taps.**
  `qibolab._core.components.filters.FiniteImpulseResponseFilter` carries raw
  FIR taps verbatim, and `OpxOutputConfig.filter("LF"|"MW"|"opx1000")` emits
  both a `"feedforward"` key (every registered filter's `.feedforward`,
  convolved) and an `"exponential"` key (from `ExponentialFilter` terms
  alone) — present in the qibolab version this package already depends on.
  This was a converter gap, not an upstream limitation; QuAM's raw
  `feedforward_filter` taps (48 on every `"arbel"` flux port) are now
  transferred as a `FiniteImpulseResponseFilter`. **Still genuinely lossy,
  and still warned loudly:** a port carrying *both* `feedforward_filter`
  taps and `exponential_filter` terms may double-apply the exponential
  correction, since qibolab's convolved `"feedforward"` key includes
  `ExponentialFilter`'s own FIR approximation on top of the native
  `"exponential"` key already carrying the same terms — flagged with a
  dedicated warning naming the affected port, pending verification against
  real hardware's `machine.generate_config()`. Port `delay` (~60 samples on
  `"arbel"`) and `crosstalk` still have no qibolab field at all and are
  dropped. All three affect flux pulse shape / CZ timing fidelity, not
  correctness of what does get built.
- `quam_to_qibolab_platform` never regresses to raising where it previously
  succeeded: any `UnsupportedWiringError`/missing-`network`-data condition
  falls back to an instruments-less `Platform` with a warning (topology/
  native-gate inspection still works), preserving the pre-2026-08-19 fixture
  and test behavior.
- `qibolab._core.parameters.ConfigKinds.extend([QmConfigs])` is called once
  at import time — qibolab's `Parameters.configs` validates against a
  registered-kind discriminated union, and QM's own config kinds
  (`"opx-output"`, `"mw-fem-oscillator"`, `"qm-acquisition"`) aren't
  registered by default.

**A real, latent bug in the already-shipped native-gate converter, found and
fixed alongside this work:** qibolab's QM driver computes
`voltage_amp = pulse.amplitude * max_voltage` (0.5 V direct-mode / 2.5 V
amplified-mode) when registering a waveform — i.e. qibolab's
`Pulse.amplitude` is unitless, normalized to `[-1, 1]`. QuAM's amplitudes are
volts (LF ports) or already fully-normalized (MW ports via
`full_scale_power_dbm`). The existing `_quam_platform_conversion.py` passed
QuAM's raw values straight through unchanged — invisible while
`instruments={}` (nothing ever multiplied by `max_voltage`), but every real
pulse would have played at exactly half its calibrated amplitude the moment
instruments existed. Fixed: amplitudes (symbolic and `Custom`-sampled alike)
are now divided by the channel's `max_voltage` at conversion time, and a new
`AmplitudeOutOfRangeError` is raised (not silently clipped) when a
calibrated pulse's peak voltage exceeds what the target channel can produce.

**A genuine upstream qibolab limitation, discovered by that fix firing on
real data, not a bug in this package:** applying the corrected amplitude
scaling and re-deriving native gates for the whole `"arbel"` machine raises
`AmplitudeOutOfRangeError` for 6 of 21 qubits (`qA3`, `qB5`, `qC3`, `qC5`,
`qD3`, `qD5` — their calibrated `x180` peak voltages run 0.51-0.88 V).
Traced to `qibolab._core.instruments.qm.controller.channel_max_voltage`: it
only special-cases an *LF-FEM* config's `output_mode == "amplified"`; every
MW-FEM-configured (`IqConfig`) channel always gets the flat 0.5 V "direct"
ceiling regardless of that port's actual `full_scale_power_dbm` (which
genuinely varies per `"arbel"` port, -2 to +18 dBm) — qibolab has no
dBm-to-voltage conversion for MW-FEM at all. Raising here (rather than
scaling against a locally-invented, dBm-derived ceiling) is the correct
behavior pending an upstream qibolab fix: a higher local ceiling would only
desynchronize this converter's amplitude scaling from what qibolab's own
driver does at pulse-registration time, trading a loud, honest failure for
a silent, different one.

**Live-hardware verification (`test/test_iqcc_platform_wiring.py`, 3 tests,
all passing against real `"arbel"`, 2026-08-23):**

1. `test_instruments_and_channels_populated_for_every_qubit` — every one of
   `"arbel"`'s 21 qubits has all 4 channels (`drive`/`probe`/`acquisition`/
   `flux`) present in both `platform.channels` and `platform.parameters.configs`.
2. `test_generated_config_matches_quam_ground_truth` — the strongest
   available offline-equivalent check: for one representative qubit per
   readout bank (A/B/C/D, exercising the multiplexed-readout sharing across
   all 4 physical ports), diffs qibolab's `QmController`-generated QUA config
   directly against QuAM's own `machine.generate_config()` output for the
   same physical ports. Port tuples, signed `intermediate_frequency`,
   `time_of_flight`, `smearing`, FEM `type`, `band`, `full_scale_power_dbm`,
   and `upconverter` all match exactly; only the already-flagged,
   already-warned differences (flux `offset`, `filter.feedforward`, absent
   `delay`/`shareable`) diverge.
3. `test_compiles_real_native_gate_sequence_offline` — builds a real
   `PulseSequence` from `qA1`'s calibrated `RX` native (confirmed within
   qibolab's voltage ceiling) and calls `QmController.play(...)` with
   `manager=None` (no hardware connection): qibolab compiles a real QUA
   program and returns it instead of executing. Confirms the amplitude fix
   end-to-end: the peak voltage qibolab's compiled pulse would produce
   (`envelope * amplitude * max_voltage`) matches QuAM's own calibrated peak
   voltage (`calculate_waveform()`'s peak) to `rel=1e-6`.

Writing test 3 surfaced two bugs **in the test itself**, not in the
converter — worth recording since both are easy to hit again:
`qibolab.ExecutionParameters(nshots=1)` leaves `relaxation_time=None`, but
`QmController.play` reads `options.relaxation_time` directly rather than
through qibolab's own `default(...)`-guarded accessor, so `None > 0` raises
`TypeError` unless `relaxation_time=0` (or another int) is passed
explicitly; and `qibolab._core.pulses.pulse.Pulse.i(sampling_rate)` takes a
**sampling rate**, not a sample count (it computes
`samples = int(self.duration * sampling_rate)` internally) — calling it with
the pulse's own duration as if it were a sample count silently computed
`duration**2` samples and failed deep inside `Custom.i()`'s length check.
Both fixed in the test file; `sampling_rate=1` is the correct call at QM's
1 GSa/s.

### Revised target architecture (discussed and agreed this session)

The original framing — keep `QiboQMBackend` and `QiboQMPlatformBackend` as
two permanently-separate backends, one for "QUA-native real-time
capability," one for "Qibolab/Qibocal-ecosystem compatibility" — was
revisited and does not hold up:

1. **Real-time-randomized RB** (the concrete case for keeping a permanent
   QUA-native backend) turns out not to need `QiboQMBackend.execute_circuit`
   at all — it's expected to land as a native Qibocal/Qualibrate-node
   routine with direct QM/QUA access (the `CalibrationBinding` mechanism
   `architecture_preliminary_insights.md` already describes), not as a
   Qibo-`Backend`-level capability. A Qibocal routine was never constrained
   to call only `platform.execute(...)`.
2. **`QiboQMBackend.execute_circuit` and `QibolabBackend.execute_circuit`
   are the same *kind* of stub.** Confirmed directly against
   `qiskit_qm_provider.backend.qm_backend.QMBackend.quantum_circuit_to_qua`:
   it's the same OpenQASM-export + `qm_qasm.Compiler` step `QMBackend.run()`
   already calls internally, just exposed standalone. Confirmed qibolab's
   `Compiler` already lowers straight to concrete, pulse-level `Pulse`
   objects (not a symbolic native-gate IR needing a separate step). Now that
   `QmController` is built (see "Instrument wiring" above),
   `QibolabBackend.execute_circuit` **is** the general-purpose "compile and
   run" stub — maintaining `QiboQMBackend`'s parallel OpenQASM
   implementation of the same job duplicates qibolab's own compiler/driver
   stack.

**Correction (2026-08-24): `QiboQMBackend`'s OpenQASM path is not merely a
transitional executor — it stays for a permanent, independent reason.**
Traced directly (not assumed): `qiskit.qasm3.dumps` exports a Qiskit
`Parameter` as a first-class OpenQASM3 `input` symbol, and
`qm_qasm.Compiler.compile(code, inputs={...})` binds that symbol to an
already-declared QUA variable at compile time — a genuinely reusable,
real-time-parameterized compiled artifact. qibolab's native compiler cannot
do this: every rule in `qibolab._core.compilers.default` (confirmed by
testing directly, not reading) does eager Python/NumPy arithmetic on gate
parameters — `qibolab._core.native.rotation()` raises `TypeError` on a
`sympy.Symbol` angle (`_normalize_angles`'s `assert 0 <= theta < 2*np.pi`
cannot evaluate a symbolic value). So `QiboQMBackend`'s OpenQASM/`qm_qasm`
path remains the only part of this stack where a **gate-level symbolic
parameter** survives compilation at all — independent of, and in addition
to, the real-time-RB reason in point 1 above. See
`qibolab_platform_from_quam_plan.md`'s "Real-time parameterization" section
for the full trace.

**Target end-state:** `QiboQMPlatformBackend` becomes the canonical backend
for circuit-level execution with concrete (already-bound) gate parameters,
once it can actually connect to and execute on real hardware —
`QiboQMBackend`'s OpenQASM path retires from *that* role at that point, but
now never retires outright: it is the only path offering circuit-level
real-time symbolic parameters (previous paragraph). `Platform.instruments`/`QmController` (the piece this note
originally gated on) is now built and hardware-validated for **config
generation and compilation**, but stock `QmController.connect()`/`.play()`
still cannot talk to IQCC: they build a plain `QuantumMachinesManager` and
call `machine.compile()`/`queue.add_compiled()`, while IQCC's
`CloudQuantumMachinesManager`/`CloudQuantumMachine` only expose a single
`execute(program, options)` method with neither `compile` nor `queue` —
confirmed by reading `iqcc_cloud_client.qmm_cloud` directly. Closing this
(a `QmController` subclass overriding `connect()`/`play()`'s tail for the
cloud case, mirroring `qiskit_qm_provider`'s existing `CloudQMJob` pattern)
was scoped out of the instrument-wiring work on purpose and is the next
concrete step toward real execution through this path — see
`qibolab_platform_from_quam_plan.md`'s implementation plan (as amended
2026-08-23) for the deferred scope list.

**Built (2026-08-24):** `QiboQMPlatformBackend.sequence_to_qua_macro`/
`.circuit_to_qua_macro` (`backend/qibo_qm_platform_backend.py`), backed by
the new `qibolab_bridge/qua_macros.py` — the qibolab-native analogues of
`qiskit_qm_provider.QMBackend`'s `schedule_to_qua_macro`/
`quantum_circuit_to_qua`. `sequence_to_qua_macro` converts an
already-built qibolab `PulseSequence` into a reusable QUA macro by emitting
QuAM channel calls (`play`/`wait`/`align`/`frame_rotation_2pi`/`measure`) —
a direct, verified port of qibolab's own
`qibolab._core.instruments.qm.program.instructions.play` emitter, retargeted
from qibolab's generated QUA element names to QuAM's `Channel` methods, so
the config authority stays `machine.generate_config()` and the IQCC-cloud
`QmController.connect()`/`.play()` blocker (previous paragraph) is
sidestepped entirely for this path. `build_operation_index` matches an
incoming pulse against already-registered QuAM operations by shape
(duration + envelope, excluding amplitude — `native.rotation()` rescales
amplitude without touching envelope, so this correctly reuses e.g. an
`x180` pulse for an arbitrary-angle rotation via `amplitude_scale`), and
registers a new `WaveformPulse` when nothing matches. `circuit_to_qua_macro`
wraps this with the already-inherited `self.compiler.compile(circuit,
self.platform)` — same constraint as `Compiler.compile` itself: gate
parameters must already be concrete, not symbolic (see the correction
above). Real-time parameterization is deliberately pulse-field-level only
(`parameters={name: (pulse_id, "amplitude"|"duration"|"phase")}`), matching
qibolab's own `Sweeper.Parameter` granularity — not circuit-level symbolic
gate parameters, which only `QiboQMBackend`'s OpenQASM path offers (see
above). Unit-tested (`test/test_qua_macros.py`,
`test/test_qibo_qm_platform_backend.py`) against the `mw_fem_machine`
fixture; not yet exercised live against real hardware.

## Platform name resolution: what Qibolab's own convention needs, and what's missing (2026-08-23)

Raised directly: today, neither `qibo.set_backend("qibolab", platform="arbel")`
nor `qibo.set_backend("qibo_qm_provider", platform="arbel")` works — passing
a bare name string fails, even though `QiboQMPlatformBackend.from_iqcc("arbel")`
(a direct Python call, no name resolution involved) already builds a fully
wired, hardware-validated platform (see above). This section traces exactly
why, using qibolab's own real source (`qibolab._core.platform.load`), not
assumption, and pins down precisely what this package would need to add to
close the gap. Nothing below is new design — `qibolab_platform_from_quam_plan.md`
§1-§3 (repo root) already worked this out in detail; this section is the
"why it still doesn't work today" companion, written after actually running
it, since the plan's design was never implemented.

### The mechanism, traced and verified live

A qibolab platform **name is not a registry lookup** — `qibolab.create_platform(name)`
(`qibolab/_core/platform/load.py`) resolves it as a **literal folder name**
searched across the `:`-separated directory list in the `$QIBOLAB_PLATFORMS`
environment variable, then dynamically executes that folder's `platform.py`
file and calls its zero-argument `create() -> Platform | Hardware`:

```python
def _platforms_paths() -> list[Path]:
    paths = os.environ.get("QIBOLAB_PLATFORMS")
    if paths is None:
        raise RuntimeError("Platforms path $QIBOLAB_PLATFORMS unset.")
    return [Path(p) for p in paths.split(os.pathsep)]

def create_platform(name: str) -> Platform:
    if name == "dummy":
        return create_dummy()          # the one built-in exception
    path = _search(name, _platforms_paths())
    hardware = _load_platform(path)     # exec's <path>/platform.py, calls create()
    ...
```

Confirmed today, live, in this exact environment (`$QIBOLAB_PLATFORMS` unset,
which is the default state — nothing in this repo or its dependencies sets
it):

```pycon
>>> qibo.set_backend("qibo_qm_provider", platform="qibo-qm-iqcc-arbel")
RuntimeError: Platforms path $QIBOLAB_PLATFORMS unset.
```

This is **not** a bug anywhere in `qibo_qm_provider` — `QiboQMPlatformBackend.__init__`
forwards a bare string straight to `QibolabBackend.__init__`
(`self.platform = platform if isinstance(platform, Platform) else
create_platform(platform)`), and `create_platform` is qibolab's own function,
entirely unmodified. The failure is purely "no folder named
`qibo-qm-iqcc-arbel` exists on any path in `$QIBOLAB_PLATFORMS`" — in fact
`$QIBOLAB_PLATFORMS` doesn't even need to contain the real path; a folder
matching the searched name has to exist *somewhere* on it.

### Confirmed today: once that folder exists, both entrypoints work, fully wired, against real hardware

Hand-built (not generated by any tool in this repo — see "What's missing"
below) a two-file folder mirroring exactly the shape
`qibolab_platform_from_quam_plan.md` §2 already specified:

```text
<some dir>/qibo-qm-iqcc-arbel/platform.py         # 6 lines, byte-identical
                                                   #   across any iqcc-* folder
<some dir>/qibo-qm-iqcc-arbel/quam_source.json    # {"state_path": "quam_state"}
```

```python
# platform.py
from pathlib import Path
from qibo_qm_provider.qibolab_bridge.platform_from_quam import create_iqcc

FOLDER = Path(__file__).parent

def create():
    return create_iqcc(FOLDER.name, FOLDER)
```

With `$QIBOLAB_PLATFORMS` pointed at that folder's parent:

```pycon
>>> qibo.set_backend("qibo_qm_provider", platform="qibo-qm-iqcc-arbel")
>>> type(qibo.backends._Global.backend()).__name__
'QiboQMPlatformBackend'
>>> qibo.backends._Global.backend().platform.qubits.keys()  # 21 qubits
>>> qibo.backends._Global.backend().platform.instruments    # {'qm': QmController(...)}

>>> qibo.set_backend("qibolab", platform="qibo-qm-iqcc-arbel")   # separate call
>>> type(qibo.backends._Global.backend()).__name__
'QibolabBackend'   # plain qibolab backend, same converter output, zero qibo_qm_provider-specific code
```

Both ran live against real `"arbel"` (21 qubits, `instruments=['qm']`,
identical to `QiboQMPlatformBackend.from_iqcc("arbel")`'s direct-call result)
— confirming the *design* is sound end to end: `create_iqcc`, the
`platform_naming` grammar, `MetaBackend`, and plain `qibolab.MetaBackend` all
already compose correctly. There is no missing capability in the converter
or the backend classes.

### What's missing: nothing generates the folder

The only gap is that **no code in this repo ever writes a `platform.py` +
`quam_source.json` pair to disk** — `create_iqcc`/`create_local` (the
functions such a `platform.py` calls) exist and are tested; the scaffolding
tool that would write the two files themselves does not, despite being
specified in detail in `qibolab_platform_from_quam_plan.md` §2-§3 (a
`platform.py` template, a `quam_source.json` template, and a "validate before
writing" step that actually attempts an IQCC connection to catch a typo'd
backend name at scaffold time rather than at every future `create_platform`
call). Concretely, still needed:

1. **A scaffolding function/CLI** — e.g.
   `qibo_qm_provider.qibolab_bridge.scaffold_iqcc_platform(backend_name, dest_dir, ...)`
   /`scaffold_local_platform(state_path, dest_dir, ...)` — writing exactly the
   two files shown above into `<dest_dir>/qibo-qm-{iqcc-<name>,local}/`. Pure
   file-writing; every piece it needs (`platform_naming.parse_platform_name`
   for the folder-name grammar, `create_iqcc`/`create_local` for validation)
   already exists.
2. **A documented `$QIBOLAB_PLATFORMS` setup step** for anyone (a user, CI, a
   Qibocal runcard) wanting name-based resolution — this is inherent to how
   qibolab itself works (confirmed: even qibolab's own tutorials require
   setting this env var; it is not something `qibo_qm_provider` can remove or
   work around), so the ask here is a one-time documented step, not new code
   to write.
3. **Decide the default scaffold location** — the plan's design resolves a
   relative `state_path` against the platform's *own* folder
   (`<dest_dir>/qibo-qm-iqcc-arbel/quam_state/`) specifically so that
   scaffolding two different machines never collides on one shared
   `QUAM_STATE_PATH`-style global path (the real, confirmed gap in
   `qiskit_qm_provider.get_machine_from_iqcc`'s own default, documented
   earlier in this note) — this repo's own `test/` and scratch usage should
   pick one convention (e.g. a `platforms/` folder at the repo root, or a
   location under the user's home directory) and document it, once the
   scaffolding tool exists.

None of this is blocked on anything upstream — unlike the earlier
`Platform.instruments` gap (which turned out to be a false blocker) and the
still-real IQCC-cloud-`connect()` gap (which needs new code around
`CloudQuantumMachinesManager`), this is a small, self-contained, well-specified
piece of tooling with a design that has already been validated by hand. It is
the natural next step for making `qibo.set_backend(...)`/Qibocal-runcard-style
name resolution actually usable, as opposed to the always-available
`QiboQMPlatformBackend.from_iqcc(...)`/`.from_local(...)` direct-Python
entrypoints, which need no folder at all and already work today.

## What's validated vs. still open (as of 2026-08-07; later sections supersede)

**Validated (this session, against real hardware or in the default test
suite):**
- `QiboQMBackend`/`FluxTunableTransmonBackend` construction, both from a bare
  `QuamRoot`/`machine` and via `from_qiskit_backend` wrapping an
  already-provisioned backend (e.g. from `IQCCProvider`).
- `add_basic_macros` re-export/wrapper, single-qubit macros, and (as of
  `qiskit-qm-provider>=0.3.3`) `cz` two-qubit macro installation (bug #2,
  fixed and regression-tested; not yet re-verified live).
- Circuit conversion for `X`, `RZ`, `CZ`, `M` (unit-tested); live-executed for
  `X` and bare `M`.
- Measurement translation, single- and multi-register cases, both unit-level
  (synthetic `Result`) and live (real `Result` from `"arbel"`).
- `Align` fails explicitly and predictably rather than silently.
- `qibolab_bridge.import_qibolab_natives_as_macros` against qibolab's real
  `create_platform("dummy")` natives (unit-tested only, not live).
- `extended_compatibility=True` letting non-`qelib1.inc` gates (e.g. `gpi`)
  parse instead of raising `QASM2ParseError` (unit-tested only, not live).
- `QiboQMBackend.update_target()` and `.register_gate(name, qubits, macro)`
  (unit-tested only, not live).

**Open (explicitly out of scope for this slice, or newly found and
unresolved):**
- `I` gate round-trip (bug #3) — root cause now pinned precisely (Qiskit's
  own OQ2 parser normalizes `id` to `u(0,0,0)` before this package ever sees
  the circuit); **fixed 2026-08-26** via `qasm2.LEGACY_CUSTOM_INSTRUCTIONS`, see
  the top of this note. Two concrete fix options were identified here, neither
  implemented.
- Two-qubit (`CZ`) live execution against real hardware — bug #2 (the
  blocker) is now fixed and re-verified against the synthetic fixture, but a
  live `"arbel"` retry has not been done yet.
- The `compiler_options` cloud-execution bug (bug #1) is now fixed upstream
  (confirmed by reading `qiskit-qm-provider`'s current source, 2026-08-18)
  and the dependency floor bumped to match, but a fresh live IQCC cloud job
  with the diagnostic monkeypatch removed has not been re-run to confirm
  end-to-end.
- `register_gate`/`update_target` and `extended_compatibility` have not yet
  been re-verified against real `"arbel"` hardware, only the synthetic test
  fixture — e.g. installing a `gpi`-named macro via `register_gate` and
  executing a `GPI` circuit live would be the natural next live-hardware
  check, but was not run this session.
- Job-submission internals beyond direct delegation (result-streaming
  customization, multi-circuit batching behavior) are inherited from
  `qiskit_qm_provider.QMJob` wholesale and have not been independently
  stress-tested here.
- Everything under §7 steps 1-4 and 6-9 above (Qibolab snapshot/sync layer,
  SDK-neutral parameter table, Qibocal/Qualibrate calibration bindings).

## Follow-up additions (2026-08-07, after the live-hardware test)

Two gaps raised directly against the live-testing findings were closed:

1. **`extended_compatibility` wired into circuit conversion.**
   `qibo_circuit_to_qiskit(circuit, extended_compatibility=True)` now forwards
   this flag to `Circuit.to_qasm()`. Confirmed empirically (not assumed) what
   it actually fixes: gates absent from OpenQASM2's standard `qelib1.inc`
   library (e.g. `gpi`, `gpi2`) previously made `qiskit.qasm2.loads` raise
   `QASM2ParseError: '<gate>' is not defined in this scope` outright, because
   Qibo's `to_qasm(extended_compatibility=False)` emits the bare gate call
   with no definition. With `extended_compatibility=True`, `to_qasm()`
   additionally emits an inline `gate gpi(phi) q {u3(...) q;}` decomposition
   block, which `qiskit.qasm2.loads` accepts. Confirmed no regression for
   gates that already worked (`x`, `rz`, `cz`, `id`, `measure` all parse
   identically either way) — made the new default. This does **not** fix bug
   #3 (`I`/`u(0,0,0)`) — that failure is downstream of the OQ2 text entirely,
   inside Qiskit's own parser normalization, and is unaffected by anything
   Qibo emits. Whatever gate now successfully parses (e.g. `gpi`) still needs
   a matching macro installed (`add_basic_macros` does not install one) to
   actually execute — `extended_compatibility` moves the failure mode from a
   parse crash to `qm_qasm`'s own actionable "missing operation" error, which
   is the correct direction but not full support.
2. **Dynamic gate-rule registration**, addressing "something similar to
   `update_target` that keeps the qm-qasm registry in sync with the stored
   instructions in the backend" directly:
   - `QiboQMBackend.update_target(input_type=None)` — a thin delegate to
     `self.qiskit_backend.update_target(...)`, so callers don't need to reach
     into `.qiskit_backend` just to resync after mutating the QuAM machine by
     any means (`add_basic_macros`, `qibolab_bridge.
     import_qibolab_natives_as_macros`, or direct `qubit.macros[...] = ...`
     assignment).
   - `QiboQMBackend.register_gate(name, qubits, macro)` — installs `macro`
     under `name` on the QuAM qubit (`qubits` a single int/str/object) or
     qubit-pair (`qubits` a 2-tuple), then calls `update_target()`
     automatically, so a new gate is immediately visible in `.natives` and
     usable by `execute_circuit` in one call. This mutates the QuAM machine
     object itself (same effect as `qubit.macros[name] = macro` by hand) —
     for a non-persistent, backend-only override instead, the wrapped
     backend's own `Target`-level calibration mechanism
     (`self.qiskit_backend.target.add_instruction(...)`, as in
     `qiskit_qm_provider`'s `examples/custom_gate.py`) remains directly
     reachable and was deliberately not re-wrapped, to avoid duplicating an
     already-complete extension point.

Both are unit-tested (`test_qibo_qm_backend.py`,
`test_circuit_conversion.py`) against the synthetic fixture; not yet
re-verified live against `"arbel"` in this session.

## New findings from live-hardware testing (2026-08-07)

Executed via `QiboQMBackend.from_qiskit_backend(...)` wrapping a real
`qiskit_qm_provider.IQCCProvider().get_backend("arbel")`, macros installed
with `add_basic_macros(qiskit_backend, reset_type="active", max_attempts=1)`.

**Result — X gate, qubit `qA1`, 200 shots:**

| Circuit | measured `0` | measured `1` | fraction `1` |
|---|---|---|---|
| `X(0); M(0)` | 26 | 174 | 0.870 |
| `M(0)` only (control) | 175 | 25 | 0.125 |

The two results are near-mirror-images of each other, consistent with a real
superconducting qubit's readout/relaxation error rate (~12-13% in both
directions) rather than either a broken measurement path (which would show
no dependence on the `X` gate) or a broken `X` gate (which would show ~50/50
or no change from baseline). This is the strongest available confirmation
that circuit conversion, macro-based QUA emission, real QOP execution, and
measurement translation are all correct together, not just individually
unit-tested.

**Bug found #1 — blocked all real IQCC execution as shipped; root cause
confirmed, not a usage error on this package's part.**
`qiskit_qm_provider.job.qm_job.QMJob.submit()` (v0.3.2/0.3.3) calls
`self.qm.execute(prog, compiler_options=compiler_options)` on the cloud path.
The currently installed `iqcc_cloud_client` (0.19.3) defines
`CloudQuantumMachine.execute(self, program, terminal_output=False,
options={})` — no `compiler_options` parameter at all — so this raises
`TypeError: CloudQuantumMachine.execute() got an unexpected keyword argument
'compiler_options'` on every cloud job submission. The results above were
obtained only after applying a **diagnostic, in-process-only monkeypatch**
(never touching installed files) that drops the unsupported keyword before
calling the real `execute`.

The author asked whether this was actually a job-class *dispatch* issue on
this package's side (i.e. whether `QiboQMBackend` should have gone through
`IQCCJob` instead of the plain `QMJob`). Traced precisely, not assumed:
`QMJob.from_circuits` picks `job_cls = IQCCJob` only when `backend.qmm` is
**not** an instance of `QuantumMachinesManager`/`CloudQuantumMachinesManager`;
otherwise it picks the base `QMJob`. `IQCCProvider.get_backend(...)` never
sets `qmm` explicitly — the wrapped `QMBackend`'s `qmm` property lazily calls
`self.machine.connect()` the first time it's needed, and `quam_builder`'s
`Quam.connect()` (`quam_builder/architecture/superconducting/qpu/base_quam.py`)
is *config-driven*: it builds whatever `qmm_class`/`qmm_settings` the loaded
QuAM state's `network` config specifies, and for a machine fetched from IQCC
that resolves to `iqcc_cloud_client.qmm_cloud.CloudQuantumMachinesManager`
(deliberately QM-SDK-shaped, so `.open_qm()` behaves like the local SDK).
That means `isinstance(backend.qmm, (..., CloudQuantumMachinesManager))` is
`True` by design, and `job_cls = QMJob` is the *intended* path for any
`IQCCProvider`-obtained backend, not a path this package's `QiboQMBackend`
diverted onto — the exact same `TypeError` would occur calling
`qiskit_qm_provider.backend.QMBackend.run(...)` directly on a Qiskit circuit,
with no `qibo_qm_provider` code involved at all. So this looks like a real,
reproducible version-drift bug between `qiskit-qm-provider`'s `QMJob.submit()`
and the installed `iqcc_cloud_client`'s `CloudQuantumMachine.execute()`
signature, not a dispatch mistake — unless there is a different, intentional
way to obtain an IQCC-connected backend (e.g. constructing a raw `IQCC_Cloud`
client and assigning it to `backend.qmm` directly) that routes through
`IQCCJob` instead and was not tried here.

**Fixed upstream, confirmed directly against source (2026-08-18).** The
author fixed this in `qiskit-qm-provider` itself (commit `c4b3258`,
"Unify QM job submission: execute + OPX1000/OPX+ queue (#7)", version
bumped to `0.3.4`); verified by reading the new code, not taken on report
alone. `QMJob.submit()` no longer calls `self.qm.execute(prog,
compiler_options=compiler_options)` directly — it now goes through
`qm_execution_options.submit_qua_programs`, whose `_execute_kwargs` helper
branches explicitly on `is_cloud_quantum_machines_manager(qmm)`: the cloud
path builds `{"options": {"timeout": ...}}` (matching
`CloudQuantumMachine.execute(self, program, terminal_output=False,
options={})`'s real signature) and never includes `compiler_options` at
all; only the non-cloud (real hardware / `SimulationConfig`) path still
passes it. A dedicated `CloudQMJob(IQCCJobMixin, QMJob)` class now exists
for IQCC-backed jobs specifically. `qibo_qm_provider`'s own
`pyproject.toml` dependency floor was bumped to
`qiskit-qm-provider>=0.3.4` to match.

One environment note worth flagging, not a code problem: this venv's
editable `qiskit-qm-provider` install still reports version `0.3.3` via
`pip show` (metadata staleness — the editable install points directly at
source files, and the fix/version bump are confirmed live by reading those
files directly), even though `pyproject.toml` inside that checkout already
says `0.3.4`. A plain `pip check`/fresh resolve in this venv could
therefore flag a spurious floor violation against the newly-bumped
`>=0.3.4` constraint until the editable install's dist-info is refreshed
(`pip install -e .` re-run) — worth doing before treating dependency
resolution as clean, not just re-running tests.

Not yet re-verified live against real `"arbel"` cloud execution in this
session (this fix landed and was confirmed by reading source, not by a
fresh live cloud job) — the diagnostic monkeypatch used for the original
live-hardware test above should no longer be necessary, but that specific
claim (monkeypatch removed, cloud job still succeeds) has not been
re-run.

**Bug found #2 — fixed upstream, confirmed.** `add_basic_macros`'s two-qubit
macro installation called `CZGate(flux_pulse_control=...)`, but the installed
`quam_builder`'s `CZGate` dataclass field is named `flux_pulse_qubit`. On
`"arbel"` this surfaced as the single-qubit-macro import warning ("Could not
import single qubit macros from quam_builder... Please upgrade quam-builder")
rather than the `TypeError` seen against the synthetic test fixture, but it
was the same root version mismatch. **Fixed in `qiskit-qm-provider` 0.3.3**
(`CZGate(flux_pulse_qubit=...)`) — re-verified directly against the synthetic
fixture after upgrading the local editable install: `add_basic_macros`
now installs a real `CZGate` instance with no exception and no dangling
`macros["cz"] = None` entry (`pyproject.toml` bumped to
`qiskit-qm-provider>=0.3.3`; a regression test,
`test_cz_macro_installs_cleanly`, now guards this). Two-qubit gate execution
was not re-tried live against `"arbel"` this session, only re-verified
against the synthetic fixture.

**Bug found #3 — new, not previously documented; root cause now pinned down
precisely.** `qibo.gates.I` does not survive the OpenQASM2 → Qiskit →
OpenQASM3 → `qm_qasm` round trip the way `X`/`M` do, and this is **not**
fixable by `Circuit.to_qasm(extended_compatibility=True)` (tested directly,
see below) — the failure happens one step later than initially suspected.
`Circuit.to_qasm()` itself emits a clean `id q[0];` regardless of
`extended_compatibility`. The rewrite happens inside **`qiskit.qasm2.loads`**
(Qiskit's own OQ2 parser), confirmed by direct inspection: parsing a
standalone `id q[0];` produces a Qiskit instruction named `'u'` with params
`[0, 0, 0]`, not `'id'` — Qiskit's OQ2 standard-library definition for `id`
is written in terms of `U(0,0,0)`, and the parser normalizes to that
immediately, before `qibo_qm_provider` or `qm_qasm` ever see the circuit.
`qiskit.qasm3.dumps` then faithfully re-exports that already-normalized `u`
gate as `U(0, 0, 0)`, which `qm_qasm`'s `CodeResolvingPass` cannot resolve
(`SignatureNotFoundException` → `CompilationException`), since no
`OperationIdentifier("u", ...)` is registered — only the Qibo/Qiskit-named
macros `add_basic_macros` installs (`x, sx, sy, sydg, rz, measure, reset,
delay, id, cz`). Since the architecture note's own minimal gate target list
explicitly includes `I`, this is a real, live gap against that target,
distinct from the already-documented `Align` limitation. Two ways to close
it, neither implemented yet: (a) register a macro under the name `"u"` that
special-cases `(0, 0, 0)` params as a no-op (matches how the identity is
actually used), or (b) special-case `gates.I` in circuit conversion to skip
emitting it entirely, mirroring qibolab's own `identity_rule` (`return
PulseSequence()` — a documented no-op, not a lowering target at all).

**Environment note.** `pip install -e .` in `~/venvs/rl_qoc` reports
pre-existing, unrelated dependency conflicts from other packages sharing
this venv (`iqcc-calibration-tools` pins `iqcc-cloud-client==0.17.5` and
`qm-qua==1.2.5`; `quarc` pins `qm-qua==1.2.4`; both are overridden by newer
versions actually installed). Bug #1 above is a direct, confirmed
consequence of that drift for the `iqcc-cloud-client` case specifically —
worth resolving venv-wide before treating live IQCC execution as routinely
available, rather than something that needs a one-off monkeypatch.

## Deviations from the note (explicit, not accidental)

The note made three recommendations that this slice does not follow. Each
was discussed and decided against explicitly during design (see the approved
plan for the full reasoning); recorded here so the deviation reads as a
decision, not drift:

1. **"Avoid converting Qibo to Qiskit solely to discover parameters; that
   would make Qiskit an accidental mandatory dependency."** (note, §4) — This
   slice makes `qiskit` and `qiskit-qm-provider` *hard*, first-class runtime
   dependencies (`pyproject.toml`), and converts every circuit through Qiskit
   as the primary execution path, not just for parameter discovery.
   Rationale: `qm_qasm.Compiler` (QM-maintained, already handles gate-arity
   validation, missing-operation errors, and QUA emission correctly) is
   reused as-is instead of being reimplemented in pure Qibo. Explicitly
   scoped as an interim measure — the plan calls for eventually extracting
   `qiskit-qm-provider`'s SDK-neutral pieces (macro installation, precedence
   semantics, parameter table) into a shared core both providers depend on,
   rather than `qibo-qm-provider` depending on the whole `qiskit-qm-provider`
   package indefinitely.
2. **"Qibolab's pulse-sequence types should not be used as the provider's
   public IR."** (note, §1/§5) — Not violated by what got built (there is no
   pulse-sequence-based IR at all in the core path), but also means the
   note's own recommended alternative (a Qibo-native gate-rule → QuAM-macro
   descriptor, §3) was not built either. The "IR" in the current design is
   just an OpenQASM3 string plus `qiskit_qm_provider`'s existing
   `OperationIdentifier` dict.
3. **"The provider should not be implemented as an unrelated Qibo backend
   that independently recreates all Qibolab topology, channel, native, and
   pulse semantics."** (note, §3) — Consistent in spirit: nothing is
   recreated, but not because Qibolab's `Platform` is delegated to (as the
   note suggests) — instead, QuAM/`qiskit_qm_provider`'s own topology model
   (`Target`, `qubit_dict`, `qubit_pair_dict`) is delegated to, and Qibolab
   is only touched by the optional, secondary `qibolab_bridge` importer.

## Status against the note's §7 suggested implementation order

| # | Note's step | Status |
|---|---|---|
| 1 | Qibolab reference/snapshot schema (IDs, references, units, capabilities, ownership) | **Not done.** No such schema exists; the current design does not route through Qibolab's `Platform` for the core execution path at all. |
| 2 | Read-only Qibolab introspector + diff report | **Not done.** |
| 3 | Provider-neutral operation descriptor + capability registry | **Not done as envisioned.** Superseded by direct reuse of `qiskit_qm_provider`'s `OperationIdentifier`/`Target` registry (accessed via `QiboQMBackend.qiskit_backend`), not a new Qibo-native descriptor. |
| 4 | QuAM synchronizer (lossless/lossy report) before writing a `QuamRoot`, legacy Qibolab driver kept as fallback | **Not done as envisioned.** `qibo_qm_provider.qibolab_bridge.import_qibolab_natives_as_macros` is a narrower, one-shot, best-effort analogue (RX/RX90 fully imported, MZ probe-only, CZ's `VirtualZ` legs skipped) — not the lossless/lossy synchronization report the note describes, and not wired to any legacy-driver fallback logic. |
| 5 | Minimal Qibo backend for `I, X/rotation, Z/RZ, CZ, M, Align`, validation-only tests before live QOP tests | **Partially done, and one gap found live.** `X` and bare `M` are validated end-to-end against real hardware (this session). `Z`/`RZ`/`CZ` are wired through the same mechanism in principle (any macro `qiskit_qm_provider`'s `add_basic_macros`/Target installs is reachable) but were **not** exercised live in this session. `Align` is confirmed to fail explicitly (`Circuit.to_qasm()` raises `NotImplementedError`, wrapped as `UnsupportedGateError`) rather than being silently mishandled — matches the note's "fail explicitly" principle, but means `Align` is unsupported, not deferred-but-working. **`I` does not work** — see New findings; **fixed 2026-08-26**, see the top of this note; this is a live gap against the note's own minimal target list. |
| 6 | Extract SDK-neutral `ParameterTable` core + `from_qibo`/`from_qasm` adapters | **Not done.** `QiboQMBackend.execute_circuit` hardcodes `param_table=None` when calling `quantum_circuit_to_qua`. |
| 7 | `CalibrationBinding` declarations for qua-libs nodes | **Not done**, out of scope for this slice. |
| 8 | One end-to-end Qualibrate adapter (`05_T1` or `03a_qubit_spectroscopy`) | **Not done.** |
| 9 | Expand match registry, explicit partial/no-match records | **Not done.** |

## Status against the note's layered architecture (§ "Executive conclusion")

The note's four-layer picture was:

```text
Qibo Circuit / Backend contract
        v
Qibo-to-QM lowering IR: operation, qubit(s), parameters, timing, measurement
        +--> QuAM operation/macro and generated QUA program
        +--> Qibocal-compatible calibration platform adapter
        +--> shared ParameterTable adapters (Qiskit/Qibo/QASM)
        v
QM/QOP execution and result normalization
```

What exists today collapses most of this into one delegation step:

```text
Qibo Circuit / Backend contract        <- built: QiboQMBackend (qibo.backends.NumpyBackend)
        v
OpenQASM2 -> Qiskit QuantumCircuit     <- built: circuit_conversion.qibo_circuit_to_qiskit
        v
qiskit_qm_provider.QMBackend.run()     <- reused unchanged, not reimplemented
   (OpenQASM3 -> qm_qasm.Compiler ->
    QuAM macro lookup -> QUA -> QOP)
        v
Qiskit Result -> Qibo MeasurementResult <- built: measurement_translation.translate_measurements
```

There is no independent "Qibo-to-QM lowering IR" of the kind the note
describes (an operation descriptor carrying name, qubits, parameters, QuAM
component reference, macro callable, measurement metadata). That role is
played entirely by `qiskit_qm_provider`'s own `OperationIdentifier`-keyed
registry and `Target`, reached through `QiboQMBackend.qiskit_backend`. The
Qibocal-adapter and shared-ParameterTable branches from the note's diagram do
not exist yet at all.

## Executive summary

A working `QiboQMBackend` (and `FluxTunableTransmonBackend` subclass) exists
and has been **run successfully against real hardware**: an `X` gate on qubit
`qA1` produced 174/200 shots measuring `1` (87%); a bare-measurement control
circuit on the same qubit produced 25/200 measuring `1` (12.5%) — a
physically sensible, readout-error-consistent, mirror-image result that
confirms circuit conversion, macro installation, real QUA compilation and
execution, and measurement translation all work correctly together.

The design that got there is **not** the architecture note's originally
described path. It is a translation shim around
`qiskit_qm_provider.backend.qm_backend.QMBackend`, routing every circuit
through OpenQASM2 → Qiskit → OpenQASM3 → `qm_qasm.Compiler` — the exact path
`qiskit-qm-provider` already uses for Qiskit circuits. This was a deliberate,
discussed pivot (see "Deviations from the note"), not an oversight, but
it means several of the note's original architectural goals (a Qibo-native
operation descriptor, no Qiskit dependency, no OpenQASM round-trip) are
**not** what got built, and should be read as superseded for this slice
rather than still-pending in their original form.

Three real, previously-undocumented compatibility bugs in the dependency
stack were found while testing against live hardware (see "New findings").
Bug #1 is now fixed upstream, confirmed directly against the
`qiskit-qm-provider` source (see its updated entry).

A second execution path, `QiboQMPlatformBackend`, was added 2026-08-18 —
a `qibolab._core.backends.QibolabBackend` subclass wrapping a genuine
`qibolab.Platform` built directly from a QuAM object, instead of routing
through `qiskit_qm_provider.QMBackend`/OpenQASM. See the
"`QiboQMPlatformBackend`" section above for what's
built, what's deliberately deferred, and the revised target architecture
(this second path is intended to eventually become the sole/canonical
backend — see that section for the reasoning and what's gating it).

## Appendix: what is `Align`?

Referenced repeatedly throughout this note without being defined. It exists at two layers,
both real, neither redundant with the other:

- **`qibo.gates.Align`** (in Qibo itself, `qibo/gates/gates.py`) is a real,
  circuit-level gate: `Align(q: int, delay: int = 0)`. It is
  **single-qubit**, not variadic — `Align(0, 1)` means "align qubit 0 with a
  10 ns delay" (`q=0, delay=1`), *not* "align qubits 0 and 1" (an earlier
  version of this package's test suite made exactly this mistake; fixed).
  Aligning several qubits means adding one `Align` gate per qubit. It is a
  synchronization/timing primitive, not a unitary operation: "wait `delay`
  ns here, and align whatever comes next on this qubit's timeline to start
  together with the other aligned qubits" — conceptually closer to a
  timing-aware `barrier` than to a logical gate.
- **`qibolab._core.pulses.pulse.Align`** (in Qibolab, a separate package) is
  the pulse-level instruction Qibo's `Align` *compiles to*. Qibolab's
  default compiler rule (`align_rule` in
  `qibolab._core.compilers.default`) lowers each `gates.Align` into
  `Delay` instructions on the relevant qubits' channels within its
  `PulseSequence` IR.

`qibo_qm_provider` does not use Qibolab's compiler at all (see "Deviations
from the note"), so only the first (Qibo) layer is directly relevant
here — and the reason it fails is unrelated to Qibolab: `Circuit.to_qasm()`
raises `NotImplementedError("Align is not supported by OpenQASM")` directly,
because OpenQASM (2 or 3) has no synchronization/barrier-with-delay construct
that maps onto it, regardless of `extended_compatibility`.
