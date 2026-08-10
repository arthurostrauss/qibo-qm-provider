# Integration status vs. `architecture_preliminary_insights.md`

Status: first vertical slice implemented and validated against real hardware,
2026-08-07.

This note tracks what actually exists in `qibo_qm_provider` today against the
objectives laid out in `architecture_preliminary_insights.md`, including
where the implementation deliberately deviated from that note's
recommendations, and what was learned by running the current code against a
real IQCC-hosted QM backend (`"arbel"`).

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
discussed pivot (see "Deviations from the note" below), not an oversight, but
it means several of the note's original architectural goals (a Qibo-native
operation descriptor, no Qiskit dependency, no OpenQASM round-trip) are
**not** what got built, and should be read as superseded for this slice
rather than still-pending in their original form.

Three real, previously-undocumented compatibility bugs in the dependency
stack were found while testing against live hardware (see "New findings").

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

## Status against the note's §7 suggested implementation order

| # | Note's step | Status |
|---|---|---|
| 1 | Qibolab reference/snapshot schema (IDs, references, units, capabilities, ownership) | **Not done.** No such schema exists; the current design does not route through Qibolab's `Platform` for the core execution path at all. |
| 2 | Read-only Qibolab introspector + diff report | **Not done.** |
| 3 | Provider-neutral operation descriptor + capability registry | **Not done as envisioned.** Superseded by direct reuse of `qiskit_qm_provider`'s `OperationIdentifier`/`Target` registry (accessed via `QiboQMBackend.qiskit_backend`), not a new Qibo-native descriptor. |
| 4 | QuAM synchronizer (lossless/lossy report) before writing a `QuamRoot`, legacy Qibolab driver kept as fallback | **Not done as envisioned.** `qibo_qm_provider.qibolab_bridge.import_qibolab_natives_as_macros` is a narrower, one-shot, best-effort analogue (RX/RX90 fully imported, MZ probe-only, CZ's `VirtualZ` legs skipped) — not the lossless/lossy synchronization report the note describes, and not wired to any legacy-driver fallback logic. |
| 5 | Minimal Qibo backend for `I, X/rotation, Z/RZ, CZ, M, Align`, validation-only tests before live QOP tests | **Partially done, and one gap found live.** `X` and bare `M` are validated end-to-end against real hardware (this session). `Z`/`RZ`/`CZ` are wired through the same mechanism in principle (any macro `qiskit_qm_provider`'s `add_basic_macros`/Target installs is reachable) but were **not** exercised live in this session. `Align` is confirmed to fail explicitly (`Circuit.to_qasm()` raises `NotImplementedError`, wrapped as `UnsupportedGateError`) rather than being silently mishandled — matches the note's "fail explicitly" principle, but means `Align` is unsupported, not deferred-but-working. **`I` does not work** — see New findings below; this is a live gap against the note's own minimal target list. |
| 6 | Extract SDK-neutral `ParameterTable` core + `from_qibo`/`from_qasm` adapters | **Not done.** `QiboQMBackend.execute_circuit` hardcodes `param_table=None` when calling `quantum_circuit_to_qua`. |
| 7 | `CalibrationBinding` declarations for qua-libs nodes | **Not done**, out of scope for this slice. |
| 8 | One end-to-end Qualibrate adapter (`05_T1` or `03a_qubit_spectroscopy`) | **Not done.** |
| 9 | Expand match registry, explicit partial/no-match records | **Not done.** |

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

## New findings from live-hardware testing (this session)

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
`IQCCJob` instead and was not tried here. Needs a fix or pin in
`qiskit-qm-provider` (or `iqcc_cloud_client`) upstream either way.

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

## Follow-up additions (this session, after the live-hardware test above)

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

## What's validated vs. still open

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
  the circuit); not yet fixed. Two concrete fix options identified above, not
  implemented.
- Two-qubit (`CZ`) live execution against real hardware — bug #2 (the
  blocker) is now fixed and re-verified against the synthetic fixture, but a
  live `"arbel"` retry has not been done yet.
- The `compiler_options` cloud-execution bug (bug #1) needs an upstream fix
  or dependency pin before live IQCC execution works without a manual patch.
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

## Appendix: what is `Align`?

Referenced repeatedly above without being defined. It exists at two layers,
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
from the note" above), so only the first (Qibo) layer is directly relevant
here — and the reason it fails is unrelated to Qibolab: `Circuit.to_qasm()`
raises `NotImplementedError("Align is not supported by OpenQASM")` directly,
because OpenQASM (2 or 3) has no synchronization/barrier-with-delay construct
that maps onto it, regardless of `extended_compatibility`.
