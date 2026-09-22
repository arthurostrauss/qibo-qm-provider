# qibo-qm-provider
An interface enabling the compilation and execution of Qibo workflows on Quantum Machine's Quantum Orchestration Platform

## Two backends, two jobs

- **`QiboQMBackend`** — Qibo-circuit execution on QM via OpenQASM/`qm_qasm`.
  The only path that supports a gate-level **real-time symbolic parameter**
  (a `sympy` expression as a gate angle, compiled once and re-parameterized
  per shot from QUA, instead of recompiling per value).
- **`QiboQMPlatformBackend`** — the Qibolab/Qibocal-native path, wrapping a
  genuine `qibolab.Platform` built from QuAM. Gate parameters must already
  be concrete; real-time parameterization is available at pulse-field
  (`Sweeper`) granularity instead of circuit-level.

Neither supersedes the other: Qibolab's own compiler does eager numeric
arithmetic on gate parameters, so it structurally cannot accept a symbolic
angle — `QiboQMBackend`'s OpenQASM path is the only place that survives.
`QiboQMPlatformBackend` is what makes Qibocal's existing calibration
protocols work against QM hardware unmodified.

## How the provider is built

Both backends start from the same object — a QuAM machine (typically a
`quam_builder` `FluxTunableQuam`) — but reach QUA/QOP through two unrelated
existing stacks, reused rather than reimplemented. This section says
plainly, for each concern, whether it comes from `qiskit-qm-provider`, from
`qibolab`, or was written in this package to connect the two.

```text
                        Qibo Circuit
                             |
              ,--------------`--------------.
              |                              |
     QiboQMBackend                  QiboQMPlatformBackend
   (qiskit-qm-provider path)          (qibolab-native path)
              |                              |
   Qiskit QuantumCircuit             qibolab native-gate
   (this package's                    Compiler rules
    circuit_conversion.py)            (qibolab, unmodified)
              |                              |
   qiskit_qm_provider.QMBackend        qibolab PulseSequence
   .run() / .quantum_circuit_to_qua          |
   (OpenQASM3 -> qm_qasm.Compiler ->   platform.execute(...)
    installed QuAM macro -> QUA)       (qibolab's own QmController,
   (qiskit-qm-provider, unmodified)     built from QuAM by this
              |                         package's qibolab_bridge)
              v                              v
             QOP                            QOP
```

### `QiboQMBackend`: the `qiskit-qm-provider` path

`QiboQMBackend` wraps a `qiskit_qm_provider.backend.qm_backend.QMBackend`
instance and is a **translation shim**, not an independent reimplementation.
Everything hardware-facing — macro installation, the `Target`/operation
registry, calibration-override precedence, QUA program construction, QM
connection handling, job submission, and result fetching — is inherited
unchanged from that wrapped instance:

```python
from qibo_qm_provider import QiboQMBackend

backend = QiboQMBackend(machine)          # machine: a quam.core.QuamRoot
backend.qiskit_backend                    # the underlying qiskit_qm_provider.QMBackend
backend.qiskit_backend.target             # its qm_qasm Target / operation registry
backend.qiskit_backend.run(...)           # its job submission, unchanged
```

The only code this package adds for this path is the translation at each
end — converting a Qibo `Circuit` into the `QuantumCircuit`
`QMBackend` expects, and converting the resulting Qiskit `Result` back onto
Qibo's per-gate measurement convention:

| Module | What it does |
|---|---|
| [`backend/circuit_conversion.py`](qibo_qm_provider/backend/circuit_conversion.py) | Qibo `Circuit` → Qiskit `QuantumCircuit`, gate by gate (concrete or with symbolic `sympy` parameters alike); resolves `circuit.wire_names`; validates two-qubit connectivity direction |
| [`backend/default_transpile.py`](qibo_qm_provider/backend/default_transpile.py) | `execute_circuit`'s default, opt-out-able (`transpile=False`) step: decomposes any gate not already native to the target into its native gate set (e.g. a plain `H` with no `h` macro installed) |
| [`backend/gate_map.py`](qibo_qm_provider/backend/gate_map.py), [`backend/qibo_qiskit_gates.py`](qibo_qm_provider/backend/qibo_qiskit_gates.py) | The verified Qibo↔Qiskit gate correspondence, plus thin gate subclasses that keep Qibo's own gate name (`prx`, `u1q`, ...) instead of the name Qiskit's standard gate would otherwise emit |
| [`backend/symbolic_parameters.py`](qibo_qm_provider/backend/symbolic_parameters.py) | `sympy` expression → Qiskit `Parameter`, and parameter-name collision checks against the machine's installed operation names |
| [`backend/parameter_table.py`](qibo_qm_provider/backend/parameter_table.py) (`QiboParameterTable`) | A Qibo-circuit source adapter for `qiskit_qm_provider.parameter_table.ParameterTable` — Qibo's counterpart of that class's existing `from_qiskit` |
| [`backend/measurement_translation.py`](qibo_qm_provider/backend/measurement_translation.py) | Qiskit `Result` → Qibo `MeasurementOutcomes` |
| [`quam_macros/superconducting/add_basic_macros.py`](qibo_qm_provider/quam_macros/superconducting/add_basic_macros.py) | Re-exports `qiskit_qm_provider`'s own macro installer unchanged, only unwrapping a `QiboQMBackend`/`QMBackend`/bare `QuamRoot` argument |

`FluxTunableTransmonBackend` is a topology-specific `QiboQMBackend`
subclass, same relationship to its wrapped `QMBackend` subclass.

### `QiboQMPlatformBackend`: the `qibolab`-native path

`QiboQMPlatformBackend` subclasses `qibolab._core.backends.QibolabBackend`
directly. `execute_circuit` adds one step on top of the inherited
implementation — the same default gate-decomposition `QiboQMBackend` gets,
see `backend/default_transpile.py` above (opt out with `transpile=False`) —
then delegates to qibolab's own `Compiler.compile` (native-gate rules →
`PulseSequence`) and `platform.execute(...)`, with no OpenQASM detour at
all. `execute_circuits` is inherited unchanged and has **no** equivalent
decomposition step (callers must pass already-native circuits, or use
`execute_circuit` per item):

```python
from qibo import Circuit, gates
from qibo_qm_provider import QiboQMPlatformBackend

backend = QiboQMPlatformBackend.from_iqcc("arbel")   # fetch QuAM state from IQCC,
                                                      # build a qibolab.Platform
# or: QiboQMPlatformBackend.from_local("quam_state")
# or: QiboQMPlatformBackend.from_machine(machine)

circuit = Circuit(2)
circuit.add(gates.X(0))
circuit.add(gates.CZ(0, 1))
circuit.add(gates.M(0, 1))

result = backend.execute_circuit(circuit, nshots=1000)   # qibolab's Compiler + platform.execute
```

What makes this possible is `qibolab_bridge`: this package's converter that
builds a genuine, populated qibolab `Platform` (`qubits`, `couplers`,
`parameters.native_gates`, `instruments`) **from a QuAM object**, so that
qibolab's own compiler/driver stack — and, transitively, any Qibocal routine
written purely against `platform.execute(...)`/`platform.natives` — runs
against QM hardware unmodified:

| Module | What it does |
|---|---|
| [`qibolab_bridge/quam_platform_conversion.py`](qibo_qm_provider/qibolab_bridge/quam_platform_conversion.py) | QuAM qubits/pairs/macros → qibolab `qubits`/`couplers`/`NativeGates` |
| [`qibolab_bridge/quam_wiring.py`](qibo_qm_provider/qibolab_bridge/quam_wiring.py) | QuAM channel wiring → qibolab `instruments`/`parameters.configs` — OPX1000 MW-FEM (drive/readout) and LF-FEM (flux) only; Octave/IQ-mixer channels fall back to an instruments-less `Platform` with a warning |
| [`qibolab_bridge/quam_pulses.py`](qibo_qm_provider/qibolab_bridge/quam_pulses.py) | QuAM pulse envelope ↔ qibolab pulse envelope conversion (lossless for `Square`/`Gaussian`; sampled-waveform fallback otherwise) |
| [`qibolab_bridge/platform_naming.py`](qibo_qm_provider/qibolab_bridge/platform_naming.py) | The `qibo-qm-local` / `qibo-qm-iqcc-<backend_name>` folder-name grammar `qibolab.create_platform` needs for name-based resolution |
| [`qibolab_bridge/platform_from_quam.py`](qibo_qm_provider/qibolab_bridge/platform_from_quam.py) | `create_iqcc`/`create_local`/`quam_to_qibolab_platform` — the entrypoints `from_iqcc`/`from_local`/`from_machine` and a `$QIBOLAB_PLATFORMS/<folder>/platform.py` stub both call |
| [`qibolab_bridge/qua_macros.py`](qibo_qm_provider/qibolab_bridge/qua_macros.py) | Converts an already-compiled qibolab `PulseSequence` into a reusable QUA macro that plays directly on QuAM channels — an alternative that sidesteps `QmController`/IQCC entirely, still useful when only a standalone macro is needed rather than a full `platform.execute(...)` |
| [`qibolab_bridge/iqcc_controller.py`](qibo_qm_provider/qibolab_bridge/iqcc_controller.py) (`IQCCQmController`) | A `QmController` subclass that connects via `machine.connect()` (reusing QuAM's own network-config-driven manager resolution) and executes via `CloudQuantumMachine.execute(...)` instead of qibolab's local compile/queue sequence — see the status note below |
| [`qibolab_bridge/native_import.py`](qibo_qm_provider/qibolab_bridge/native_import.py) | The reverse direction: importing qibolab platform natives as QuAM macros |
| [`qibolab_bridge/naming.py`](qibo_qm_provider/qibolab_bridge/naming.py) | Shared macro-name/channel-iteration helpers used by both bridge directions |

The top-level `MetaBackend` in [`qibo_qm_provider/__init__.py`](qibo_qm_provider/__init__.py)
registers this package under Qibo's standard backend-discovery mechanism, so
`qibo.set_backend("qibo_qm_provider", platform=...)` resolves to
`QiboQMPlatformBackend` specifically (rather than qibolab's own
`MetaBackend`, which always builds a plain `QibolabBackend`):

```python
import qibo

# Requires a qibo-qm-iqcc-arbel/ folder with a platform.py + quam_source.json
# on $QIBOLAB_PLATFORMS -- create_iqcc/create_local exist and are tested, but
# nothing in this package yet scaffolds that folder pair automatically.
qibo.set_backend("qibo_qm_provider", platform="qibo-qm-iqcc-arbel")
assert type(qibo.backends._Global.backend()).__name__ == "QiboQMPlatformBackend"

# The exact same converter output also works through plain qibolab, with
# zero qibo_qm_provider-specific code:
qibo.set_backend("qibolab", platform="qibo-qm-iqcc-arbel")
assert type(qibo.backends._Global.backend()).__name__ == "QibolabBackend"
```

### Reuse boundary, side by side

| Concern | `QiboQMBackend` | `QiboQMPlatformBackend` |
|---|---|---|
| Circuit lowering | `qiskit-qm-provider`'s `qm_qasm.Compiler`, via OpenQASM3 | `qibolab`'s own `Compiler` (native-gate rules → `PulseSequence`) |
| Machine/operation registry | `qiskit_qm_provider.QMBackend`'s `Target` + macro dict, read from QuAM | a qibolab `Platform` built once from QuAM (this package's `qibolab_bridge`) |
| Execution | `qiskit_qm_provider.QMBackend.run()` → `QMJob`/`CloudQMJob` | qibolab's `QmController` (local) or `IQCCQmController` (IQCC, dispatched automatically from the machine's own `network` config) — hardware-validated for config generation and offline compilation; the IQCC path is offline-verified but **not yet re-run against real IQCC hardware** |
| Native gate set (`.natives`) | Shared Enum∩QuAM-macro set (`gate_map.enum_compatible_quam_natives`) | Same shared set (overrides inherited qibolab list when `machine` is known) |
| Real-time parameters | circuit-level, symbolic (`sympy` → live QUA variable) | pulse-field level only (`Sweeper`-style amplitude/duration/phase) |
| Qibocal/Qualibrate fit | none (no calibration binding exists yet on either path) | designed for it — any routine written against `platform.execute(...)` needs no code change |

## Compiling a symbolic circuit once, real-time parameterized

`circuit_to_qua` does **not** run the default gate-decomposition step that
`execute_circuit` applies. Bind parameters and ensure gates are already
native (or decompose yourself) before calling it.

```python
import sympy
from qibo import Circuit, gates
from qibo_qm_provider import QiboQMBackend, QiboParameterTable, add_basic_macros
from qm.qua import program, for_, declare, fixed

theta = sympy.Symbol("theta")
circuit = Circuit(1)
circuit.add(gates.RZ(0, theta=theta))

add_basic_macros(machine)                    # installs rz, x, sx, measure, ...
backend = QiboQMBackend(machine)

# Build the table up front so it can be declared/assigned inside the sweep,
# then pass it back into circuit_to_qua explicitly (a single call, with no
# table argument, would build and discard a fresh one per invocation).
table = QiboParameterTable.from_qibo_circuit(circuit)

with program() as prog:
    table.declare()
    angle = declare(fixed)
    with for_(angle, 0.0, angle < 0.5, angle + 0.1):
        table.assign_parameters({"theta": angle})
        backend.circuit_to_qua(circuit, param_table=table)
```

## Addressing physical qubits, and two-qubit gate direction

Both `QiboQMBackend` and `QiboQMPlatformBackend` run a default **gate**
decomposition step in `execute_circuit` (`transpile=True`; opt out with
`transpile=False`) — e.g. a plain `H` is rewritten into the machine's
native set before compile. That step never does **placement or routing**:
a circuit's qubit index `i` still addresses whichever QuAM / platform qubit
sits at position `i` (or the corresponding platform qubit list), unless you
set `Circuit(n, wire_names=[...])` to physical qubit names explicitly.

`circuit_to_qua` / `circuit_to_qua_macro` and Platform `execute_circuits`
do **not** auto-decompose; they expect an already-native circuit (or will
fail at compile time the old way). See #8 for transpile parity on those
APIs.

Many QM two-qubit natives (e.g. a flux-tunable `CZ`) are also physically
**asymmetric** — the flux pulse only plays on one qubit of the pair — so
`CZ(a, b)` and `CZ(b, a)` are not interchangeable even though Qibo treats
the gate as order-independent. Writing the wrong direction raises a clear
`UnsupportedConnectivityError` naming the direction that is actually
installed, rather than an opaque compiler failure.

## Status note — 2026-09-03

`architecture_preliminary_insights.md` (2026-08-02) proposed one four-layer
pipeline: Qibo circuit → a Qibo-native lowering IR → QuAM
macros/Qibocal adapter/shared `ParameterTable` → QM execution. What exists
today, after roughly four weeks of implementation and live-hardware testing
against a real 21-qubit IQCC backend (`"arbel"`), is **two independent,
narrower pipelines** instead of that one IR, each collapsing most of the
note's middle layers into a single delegation step (full dated history,
including every bug found and fixed along the way, in
[`INTEGRATION_STATUS.md`](INTEGRATION_STATUS.md)):

- `QiboQMBackend` reuses `qiskit-qm-provider`'s existing `Target`, macro
  registry, and QUA compilation almost entirely as-is, rather than the
  note's proposed Qibo-native operation descriptor. It is the only path
  validated end to end against real hardware (an `X` gate on qubit `qA1`:
  87% vs. 12.5% measured-`1` fraction with/without the gate, mirror-image
  results consistent with real readout error) and the only path offering
  circuit-level real-time symbolic parameters (`circuit_to_qua`).
- `QiboQMPlatformBackend` instead builds a genuine qibolab `Platform`
  directly from QuAM, so qibolab's own compiler/driver stack — and any
  Qibocal routine written against it — runs unmodified. Its topology,
  native-gate, and OPX1000 MW-FEM/LF-FEM instrument conversion are
  hardware-validated against real `"arbel"`. Its cloud execution path
  (`IQCCQmController`, added 2026-09-03) is offline-verified — `qibolab`'s
  `QmController.connect()`/`.play()` assumed a `compile()`/`queue` interface
  IQCC's `CloudQuantumMachinesManager` never exposed (it offers a single
  `execute(program, options)` instead); the new subclass connects via the
  source QuAM object's own `connect()` and executes via that single call —
  but has not yet been re-run against real IQCC hardware.

Neither path implements the note's proposed Qibocal-compatible calibration
adapter or `CalibrationBinding` registry, and no shared, SDK-neutral
`ParameterTable` core exists — `QiboParameterTable` is a Qibo-source adapter
bolted onto `qiskit_qm_provider`'s own `ParameterTable` class, not the
extracted, SDK-neutral core the note called for. Concretely still open,
roughly in the order that unblocks the most:

1. Live-hardware verification of `IQCCQmController` against real IQCC
   execution (built and offline-verified 2026-09-03; no IQCC credentials
   available in this session).
2. ~~A scaffolding tool that writes the `qibo-qm-{local,iqcc-<name>}` folder~~
   **Done (2026-09-06)**: `qibo_qm_provider.qibolab_bridge.scaffold` writes
   the `platform.py`+`quam_source.json` pair (plus a CLI), closing
   `$QIBOLAB_PLATFORMS` name-based resolution. See
   [`INTEGRATION_STATUS.md`](INTEGRATION_STATUS.md)'s "Platform-folder
   scaffolding" section. Note this is a convenience for name-string
   resolution specifically (a runcard's `platform:` field,
   `qibo.set_backend(..., platform=...)`) — it was never required just to
   run a Qibocal node: `Executor.create` accepts an already-built `Platform`
   instance directly (`QiboQMPlatformBackend.from_local(...)`/`.from_iqcc(...)`
   already return one), bypassing name resolution entirely.
3. `CalibrationBinding` declarations against a first qua-libs node (e.g.
   `05_T1` or `03a_qubit_spectroscopy`), and the Qibocal-operation-ID → node
   resolution mechanism the note specifies — lower priority than first
   thought: `QiboQMPlatformBackend`'s native sweeper support (frequency,
   offset, amplitude, duration_interpolated, phase — see `qua_sweep.py`)
   already covers the sweeper shapes most Qibocal nodes use, without this
   layer.
4. Extracting the SDK-neutral `ParameterTable` core the note describes
   (`Qibo`/QASM source adapters against a shared spec, rather than a Qibo
   adapter against `qiskit_qm_provider`'s own concrete class).

See [`INTEGRATION_STATUS.md`](INTEGRATION_STATUS.md) for the full, dated
account of every deviation from the note, bug found and fixed, and
hardware-validation result behind this summary.

---

See [symbolic_circuit_lowering.md](symbolic_circuit_lowering.md) for the full
design write-up (what's supported, what isn't, and why),
[slice2_plan.md](slice2_plan.md) for `circuit_to_qua`'s design,
[qibo_backend_vs_qibolab_platform.md](qibo_backend_vs_qibolab_platform.md) and
[qibolab_platform_from_quam_plan.md](qibolab_platform_from_quam_plan.md) for
the `QiboQMPlatformBackend` design rationale, and
[INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) for the complete, dated
implementation history against `architecture_preliminary_insights.md`.
