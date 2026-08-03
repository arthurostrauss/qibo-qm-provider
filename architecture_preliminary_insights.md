# Preliminary architecture insights for `qibo-qm-provider`

Status: exploratory architecture note, 2026-08-02.

## Scope and evidence

This note is based on the target checkout (currently a packaging scaffold), the
installed sources in `~/venvs/rl_qoc`, and the following local reference
checkouts:

- `qibo` 0.3.3, `qibolab` 0.2.16, and `qibocal` 0.2.5 from
  `~/venvs/rl_qoc/lib/python3.11/site-packages`.
- `qua-libs/qualibration_graphs/superconducting` for the current Qualibrate
  nodes.
- `quam` for the QuAM object, channel, pulse, serialization, and operation
  abstractions.
- `qiskit-qm-provider` for the existing backend, QuAM macro, circuit lowering,
  and parameter-table implementation.

The versions matter. The installed Qibo/Qibolab/Qibocal APIs should be treated
as the initial compatibility target, not as a timeless interface. The
qibocal package installed in the venv is also not the same thing as the
Qualibrate node source in `qua-libs`; those are two distinct calibration
implementations with overlapping experiment names.

## Executive conclusion

The provider should have four explicit layers:

```text
Qibo Circuit / Backend contract
        |
        v
Qibo-to-QM lowering IR: operation, qubit(s), parameters, timing, measurement
        |
        +--> QuAM operation/macro and generated QUA program
        +--> Qibocal-compatible calibration platform adapter
        +--> shared ParameterTable adapters (Qiskit/Qibo/QASM)
        |
        v
QM/QOP execution and result normalization
```

The most important design decision is to avoid making QuAM macros look like
Qibo gates or making Qibocal routines depend on Qibo's execution result type.
Both would couple unrelated contracts. Instead, the provider should expose a
small operation descriptor and an execution/result protocol, then implement
thin adapters at each boundary.

## 1. Qibo, Qibolab, and Qibocal data flow

### Qibo

Qibo's `Circuit` is a symbolic queue of Qibo gate objects. Gates retain their
qubit indices, target qubits, parameters, and parameterized-gate bookkeeping.
`Circuit.set_parameters` and `Circuit.compile` are circuit-level facilities;
the backend is ultimately asked to execute a circuit.

The Qibo `Backend` contract exposes hardware-relevant properties intended for
transpilation: `qubits`, `connectivity`, and `natives`. It also exposes the
execution methods and the numerical backend surface inherited by simulators.
This means a QM backend can start as a Qibo backend implementation without
pretending that QOP is a numerical state-vector engine. Unsupported numerical
methods should fail explicitly, as Qibolab already does for direct gate
application.

Qibo's transpiler has separate placement, routing, unrolling, decomposition,
and optimization components. The provider should not duplicate these. It
should provide truthful `qubits`, connectivity, native operations, and gate
rules, then lower the resulting native circuit into QM operations.

### Qibolab

Qibolab is the closest existing model for the hardware-facing portion. Its
`QibolabBackend` subclasses Qibo's `NumpyBackend`, exposes platform qubits and
pairs, and reports native gates from a `Compiler` plus calibrated two-qubit
natives. It then executes by:

1. compiling a Qibo circuit to a `PulseSequence`;
2. connecting to the platform;
3. executing the sequence for `nshots`; and
4. assigning readout samples back to Qibo measurement gates.

Its `Compiler` is a registry from Qibo gate classes to rules. A rule receives a
gate and a platform-native object and returns a pulse sequence. The default
rules are a useful conceptual reference for the QM provider:

- `I` is omitted;
- `Z` and `RZ` become virtual-Z frame operations;
- `GPI`/`GPI2` become parameterized single-qubit rotations;
- `CZ`, `CNOT`, and `iSWAP` select a calibrated two-qubit native sequence;
- `M` selects readout natives; and
- `Align` becomes channel delays.

Qibolab's `Native`, `SingleQubitNatives`, and `TwoQubitNatives` containers are
particularly reusable as *semantic* inspiration, but their pulse-sequence
types should not be used as the provider's public IR. QM needs to preserve
macro identity, QUA variable parameters, stream declarations, and QuAM object
references, which a plain pulse sequence does not express.

### Qibolab's QM driver and the QuAM boundary

The current Qibolab QM driver makes the boundary concrete. `QmController.play`
receives Qibolab `PulseSequence` objects and a dictionary of Qibolab component
configs. It then dynamically registers QM devices, elements, pulses, waveforms,
integration weights, and a generated QUA program. In other words, the current
driver is configuration-first and pulse-sequence-first; QuAM is not currently
the source of truth for those objects.

This is consistent with Qibolab's general model: a `Platform` exposes
`channels`, `qubits`, `couplers`, `natives`, `parameters.configs`, and
`execute`, while the controller converts those definitions into instrument
configuration. The QM-specific component configs already contain relevant
hardware characteristics such as IQ frequency, LO association, mixer data,
sampling rate, output mode, acquisition delay, kernel, gain, and offsets. The
Qibolab pulse model also carries duration, amplitude, envelope, relative phase,
chirp, readout acquisition, and virtual-Z information.

The integration should therefore be a separate Qibolab↔QuAM path, initially
alongside the legacy Qibolab QM driver. The legacy path remains:

```text
Qibolab Platform -> PulseSequence + configs -> QmController -> QM config + QUA
```

The new path should be:

```text
Qibolab Platform -> synchronized component reference -> QuamRoot -> QuAM/QUA -> QOP
```

The provider should not silently replace the existing driver with QuAM. It
should first expose a stable, dictionary-style reference from Qibolab, then
build a synchronization adapter that can create or update the corresponding
QuAM channels, pulses, qubits, couplers, oscillators, readout components, and
operation names. This makes the two representations comparable and testable
before choosing which one owns execution.

A first reference export could conceptually look like:

```python
{
    "qubits": {"q0": {"drive": "q0/drive", "probe": "q0/probe", ...}},
    "pairs": {"q0-q1": {"drive": "q0-q1/drive", ...}},
    "channels": {"q0/drive": {"kind": "iq", "device": ..., "port": ...}},
    "configs": {"q0/drive": {"frequency": ..., "mixer": ..., ...}},
    "natives": {"q0": {"RX": ..., "MZ": ...}, "q0-q1": {"CZ": ...}},
    "pulses": {"...": {"duration": ..., "amplitude": ..., "envelope": ...}},
}
```

This is a reference/snapshot format, not a second mutable configuration
system. It should include stable IDs, object kinds, references, units, source
version, and ownership (`qibolab`, `quam`, or synchronized). The adapter must
define conflict policy: whether Qibolab updates overwrite QuAM, QuAM updates
flow back to Qibolab, or a field is read-only on one side. Shared LO/mixer
references and acquisition/probe relationships must remain references rather
than duplicated anonymous dictionaries.

The induced architectural change is that the provider needs a dedicated
component synchronization layer, probably with two adapters:

- `QibolabReference`: read-only normalized view of channels, configs, pulses,
  natives, qubits, pairs, and instrument capabilities;
- `QuamSynchronizer`: applies that view to a `QuamRoot`, reports lossless versus
  lossy fields, and emits a synchronization diff.

Only after this exists should a `QuamController` or equivalent execution path
be introduced. It may consume the normalized operation IR directly rather than
reconstructing it from a Qibolab pulse sequence. This avoids forcing QuAM to
imitate the QM driver's transient config-building behavior.

### Qibocal

Qibocal's `Routine` separates acquisition, fit, report, and update. Its
`Executor` resolves a protocol, constructs a typed `Action`, runs a `Task`,
stores history, and optionally applies the platform update. Acquisition
routines generally call `platform.execute(...)`; update functions mutate the
calibration platform through helpers such as `update.drive_frequency`,
`update.drive_amplitude`, and `update.readout_frequency`.

This separation is the best reuse opportunity for calibration integration. The
QM adapter needs to implement the platform-facing acquisition operations and
the calibration-state update operations; it does not need to port Qibocal's
fit/report code. However, direct reuse is only realistic for routines whose
acquisition is expressed in Qibolab pulse sequences and whose update target is
representable in the QM/QuAM state model. Routines that assume a Qibolab
runcard, instrument configuration, or Qibolab-specific result layout require
an adapter or a dedicated Qualibrate implementation.

## 2. Mapping Qibocal calibrations to Qualibrate nodes

The files under `qua-libs/qualibration_graphs/superconducting` are not thin
wrappers around Qibocal. A typical node constructs a `QualibrationNode`, loads
a `Quam`, declares run actions, creates a QUA program, executes or simulates it,
fetches an xarray dataset, fits it, records outcomes, and applies state updates
inside `node.record_state_updates()`.

The mapping should therefore be a registry of *experiment equivalence*, not an
automatic filename importer. The node name, parameters, sweep axes, raw-data
schema, fit results, and state updates must be compared before declaring a
match.

### Confirmed layering for a real-backend calibration

The intended architecture is confirmed at the Qibolab/Qibo boundary:

```text
Qibo circuit and transpiler
        -> Qibolab-backed Qibo backend
        -> Qibolab Platform (qubits, topology, natives, pulse/component model)
        -> qibo-qm-provider's Qibolab↔QuAM synchronization/integration
        -> QuamRoot and QUA
        -> QM/QOP
```

Qibo is above Qibolab when a real backend is used. Qibolab supplies the
hardware/platform definition that the Qibo backend uses for connectivity and
native-gate discovery, and Qibolab's compiler conventions remain relevant to
the initial gate-rule implementation. The provider should therefore not be
implemented as an unrelated Qibo backend that independently recreates all
Qibolab topology, channel, native, and pulse semantics. Its backend facade
should delegate those capabilities to a Qibolab `Platform` or a normalized
view of one.

There are two valid execution modes below that facade:

1. **Legacy pulse mode:** Qibo lowers through Qibolab to `PulseSequence`, and
   the existing Qibolab QM driver generates QM config and QUA dynamically.
2. **QuAM mode:** Qibo/Qibolab capabilities are synchronized into QuAM, and
   the provider lowers operations into QuAM components/macros and QUA.

Legacy pulse mode should be the compatibility fallback. QuAM mode should be
explicit until synchronization, state ownership, and result equivalence are
validated.

### Initial match candidates

| qua-libs node family | likely Qibocal analogue | assessment |
| --- | --- | --- |
| `01a_time_of_flight` | `signal_experiments.time_of_flight_readout` | Strong conceptual match; acquisition/result schemas differ. |
| `02a`/`02b` resonator spectroscopy | `resonator_spectroscopies.resonator_spectroscopy` and related optimization | Strong, but frequency/power sweep and update conventions need normalization. |
| `03a_qubit_spectroscopy` | `qubit_spectroscopies.qubit_spectroscopy` | Strong; both update drive frequency, but platform object paths differ. |
| `04a_rabi_chevron`, `04b_power_rabi`, `13_power_rabi_ef` | `rabi.*` | Strong family match, not necessarily one-to-one parameter match. |
| `05_T1` | `coherence.t1` | Strong fit-level match; acquisition and dataset layout differ. |
| `06a_ramsey`, `06b_echo` | `ramsey.*`, `coherence.spin_echo` | Strong conceptual match; detuning, virtual-Z, and fit-result names need an explicit mapping. |
| `07_iq_blobs`, `15_iq_blobs_gef` | `classification.*` / readout characterization | Partial; state-discrimination and qutrit support must be checked. |
| `08a`/`08b` readout optimization | `readout_optimization.*` | Strong family match; QM state updates target QuAM resonator fields. |
| `10b_drag...` | `drag.*` | Strong family match; pulse parameterization is not identical. |
| `11a`/`11b` single-qubit RB | `randomized_benchmarking.*` | Strong algorithmic match; result representation and circuit generation differ. |
| `18_cryoscope` | `flux_dependence.cryoscope` | Strong conceptual match; flux channel and waveform semantics need a QM-specific adapter. |
| CZ calibration files and two-qubit RB | `two_qubit_interaction.*`, `randomized_benchmarking.standard_rb_2q*` | Partial; several qua-libs nodes have experiment-specific fitting and QuAM update logic. |

The remaining nodes should be classified as no-match, partial-match, or
provider-specific before implementation. In particular, mixer calibration,
MW-FEM variants, JAZZ/Palea variants, calibration-graph bringup/retuning, and
nodes with specialized qutrit or coupler workflows should not be forced into a
generic Qibocal mapping.

### Recommended mapping mechanism

Define a calibration descriptor containing:

- canonical experiment ID and aliases;
- Qibocal routine, if any;
- Qualibrate node module, if any;
- parameter conversion functions;
- acquisition/result conversion functions;
- QuAM state-update functions;
- required channel/component capabilities; and
- match status (`exact`, `adapted`, `partial`, or `none`).

For an exact/adapted match, reuse Qibocal's fit and reporting code where
possible, but let the Qualibrate node own QUA acquisition and QuAM state
updates. This preserves the node's GUI/history/simulation behavior and avoids
pretending that a Qibolab `Platform` and a QuAM `QuamRoot` have the same state
model.

### Explicit mapping in each Qualibrate node

The current superconducting nodes identify themselves through
`QualibrationNode(name=...)` and type their machine as `Quam`; they do not yet
declare a Qibocal operation or a Qibolab backend mapping. That mapping must be
explicit in the node module, rather than inferred from a filename. A proposed
provider-owned declaration is:

```python
CALIBRATION_BINDING = CalibrationBinding(
    node_id="superconducting.05_T1",
    qibocal_operation="coherence.t1",
    backend_layer="qibo.qibolab",
    acquisition="qualibrate_qua",
    fit="qua-libs.calibration_utils.T1.analysis",
    state_update="quam.qubit.T1",
    required_capabilities=("drive", "readout", "relaxation_sweep"),
    status="adapted",
)
```

The exact registration API remains to be designed, but the declaration should
be discoverable from the node module and serializable with the node metadata.
It must state which side owns acquisition, fit, plotting, and updates; the
canonical Qibocal operation ID; the expected Qibolab/QuAM capabilities; and the
parameter, dataset, fit-result, and state-update conversions. A provider can
then resolve a Qibocal request as follows:

```text
qibocal operation ID
    -> unique CalibrationBinding
    -> Qualibrate node module and node_id
    -> QuAM machine + QUA acquisition
    -> declared fit/result/update adapters
```

If there is no binding, more than one binding, or a capability mismatch, the
provider should fail with an actionable error. It should not choose a node by
filename similarity or silently run a different calibration.

### Default behavior for a qibocal calibration script

The safe default is compatibility, not implicit Qualibrate dispatch:

- A qibocal script using a Qibo backend should receive a Qibolab-compatible
  platform/backend facade and execute through the legacy Qibolab acquisition
  contract unless the user selects QuAM mode.
- QuAM/Qualibrate dispatch should require an explicit binding and an explicit
  mode, for example `execution_mode="qualibrate"` or a backend method that
  accepts `node_id`/`CalibrationBinding`.
- A calibration update should be opt-in and transactional. The provider should
  report the proposed Qibolab/QuAM changes before committing them when the
  state-ownership policy is not yet configured.

This avoids a surprising change in behavior for existing Qibocal routines and
prevents a routine from being executed twice: once through Qibolab and once
through a Qualibrate node. Once the binding registry and synchronization layer
are mature, a platform profile may opt into QuAM mode by default, but that
should be a named profile rather than a global implicit behavior.

## 3. Qibo backend expectations and QuAM export

The qiskit provider demonstrates a useful backend pattern. `QMBackend` wraps a
QuAM root, derives a Qiskit `Target` from active qubits and qubit-pair macros,
stores per-operation QUA callables, and supports custom operations by
inspecting macro signatures. The relevant ideas are reusable:

- derive supported operations from the machine, rather than maintaining a
  second hard-coded gate list;
- distinguish single-qubit and pair operations;
- keep connectivity and active-qubit filtering explicit;
- preserve the operation's parameter arity;
- maintain a mapping from the SDK operation identity to the QUA callable; and
- allow target-specific calibrations to override default machine operations.

For Qibo, the equivalent backend should expose Qibo's `qubits`,
`connectivity`, and `natives`, and register a Qibo compiler rule for each
supported gate class. A rule should lower to a provider operation descriptor,
not immediately emit QUA. The descriptor should carry at least:

```text
operation name
Qibo gate class / canonical SDK name
logical and physical qubits
ordered parameters and parameter types
QuAM component/channel or pair reference
macro/operation callable
measurement and stream metadata
duration/alignment metadata when known
```

This gives one place to validate gate arity, qubit ordering, parameter units,
virtual-Z sign conventions, active topology, and missing calibrations. It also
allows the same descriptor to be used by a Qibo circuit path, a QASM path, and
the calibration-node path.

### Gate-rule to QuAM-macro relationship

The relationship should be explicit and directional:

```text
Qibo gate class + parameters + qargs
    -> provider gate rule
    -> QuAM operation/macro on a component
    -> QUA statements / pulse operation
```

Examples are `RZ` to a frame rotation, `X`/`RX` to a QuAM `x180`/rotation
operation, `CZ` to a pair operation, `M` to resonator readout, and `Align` to
an explicit synchronization operation. A QuAM channel's `operations` mapping
is pulse/configuration data; an `OperationsRegistry` callable is a higher-level
QUA operation. The provider must not conflate those two meanings of
“operation”.

QuAM export should mean serializing a complete, valid `QuamRoot` with all
references, channels, pulses, and operation names needed by the generated QUA
program. It should not mean exporting a Qibo backend object. QuAM's
`QuamRoot.to_dict`/`save` and `generate_config` remain the serialization and
QOP-config boundaries.

## 4. Reusing and disentangling `ParameterTable`

`qiskit-qm-provider.parameter_table` is valuable, but the current component is
larger than a circuit-to-table converter. It combines:

- parameter declaration and QUA variable access;
- arrays and typed fields;
- input-stream, IO1/IO2, and OPNIC directions;
- a global `ParameterPool` and scope rules;
- QuARC emission;
- Qiskit circuit discovery; and
- measurement-output restrictions used by circuit compilation.

The reusable core should be extracted conceptually into SDK-neutral pieces:

1. `ParameterSpec`: name, initial value, QUA type, shape, input type, direction,
   and metadata.
2. `ParameterTable`: ordered specs plus declaration/access/emission behavior.
3. `ParameterSource` adapters that discover specs from an input object.
4. `ParameterSink`/emitters for QUA declarations, runtime updates, or QuARC.
5. A separate measurement-output table type; measurement handles should not be
   accepted by an ordinary runtime parameter table.

The existing `ParameterTable.from_qiskit` is the first source adapter. Add
`from_qibo(circuit, ...)` by walking `circuit.queue`, collecting Qibo symbolic
parameters and preserving stable names. Add `from_qasm(qasm, ...)` by parsing
through one canonical QASM representation and then passing the resulting
parameter specs to the core. Avoid converting Qibo to Qiskit solely to discover
parameters; that would make Qiskit an accidental mandatory dependency.

The requested monkey-patching can be supported as an extension mechanism, but
the stable design should be a registered adapter protocol, for example
`ParameterTable.register_source("qibo", from_qibo)`. Monkey patches are useful
for downstream SDKs, but a registry gives validation, discoverability, and
versioned behavior. Keep the provider-specific QuARC/QUA emission layer intact
behind the same table interface.

## 5. What to reuse from `qiskit-qm-provider`

Reuse directly or nearly directly:

- target/machine introspection logic, after replacing Qiskit `Target` with a
  Qibo-facing capability model;
- operation identity and QUA-callable mapping concepts;
- QuAM macro installation and standard/custom operation naming conventions;
- QUA circuit lowering validation, especially parameter arity and operation
  lookup;
- parameter declaration, pooling, scope, array, and QuARC emission machinery;
- result stream assembly concepts where they are SDK-neutral; and
- tests for missing operations, custom operations, parameter redeclaration,
  and calibration override precedence.

Do not copy as-is:

- Qiskit `Target`, `InstructionProperties`, `QuantumCircuit`, or `Var` types
  into the Qibo core;
- Qiskit pulse schedule conversion as the primary Qibo path;
- Qibolab pulse sequences as a substitute for QUA/QuAM operation metadata; or
- Qibocal update helpers that assume a mutable Qibolab runcard.

The existing provider has a useful precedence rule: machine macros establish
the base mapping, while explicit target/circuit calibrations can overwrite a
specific operation. Preserve that behavior in a provider-neutral operation
registry and test it independently of Qiskit.

## 6. Main risks and contract boundaries

- **Logical versus physical qubits.** Qibo circuits use wire names/indices;
  QuAM uses component identities; Qibolab may use named platform qubits. Make
  the mapping explicit and carry both identities through lowering.
- **Native versus calibrated operations.** A gate can be supported by a rule
  but unavailable for a particular qubit or pair. Capability discovery must
  include per-location availability.
- **Parameter units and signs.** Qibolab virtual-Z rules and QUA frame rotations
  can differ in sign, radians/turns, and fixed-point representation. Add
  numerical contract tests for each rule.
- **Compile-time versus runtime parameters.** QUA program structure, loop
  bounds, stream shapes, and branch topology may require compilation. A
  `ParameterTable` can update values at runtime only when the QUA program
  accepts them as runtime variables.
- **Calibration state.** Qibocal mutates a platform/runcard; Qualibrate records
  QuAM updates. Use explicit state translators and record provenance for every
  update.
- **Results.** Qibocal commonly uses structured NumPy data/results, whereas
  Qualibrate nodes use xarray plus dictionaries and figures. Normalize only at
  the adapter boundary; do not force one result model on both ecosystems.
- **Dependency direction.** The Qibo provider should depend on Qibo/Qibolab and
  optionally Qibocal/QuAM integrations. Shared parameter-table code should not
  import Qiskit at module import time.

### Specific challenges for the envisioned Qibolab↔QuAM integration

- **Two configuration owners.** Qibolab currently stores platform parameters
  and the QM driver derives an instrument config from them; QuAM stores a
  hardware object graph and generates config later. A bidirectional sync can
  create cycles or stale writes unless each field has an owner and update
  direction.
- **Different object granularity.** Qibolab has channels, external components,
  `Config` models, pulse objects, natives, and pulse sequences. QuAM has
  components, channels, operations, pulse definitions, references, and root
  serialization. There is no general one-to-one mapping for a Qibolab native
  sequence or a multiplexed readout.
- **Transient versus persistent configuration.** The current QM driver only
  registers elements and pulses needed by the participating sequence and can
  generate a temporary config. QuAM normally represents a persistent machine.
  The integration must decide whether a synchronized QuAM is complete,
  partial, or only a compilation view.
- **Shared hardware references.** Qibolab deliberately represents shared LOs,
  mixers, probe/acquisition relationships, and multiple channels on one wire
  through IDs. Flattening these into QuAM objects can duplicate hardware or
  break updates.
- **Dynamic sweeps.** Qibolab sweepers can cause additional QM pulses,
  waveforms, frequency settings, and program arguments to be registered. QuAM
  operations and QUA programs need an explicit treatment of which sweep axes
  are runtime variables and which require structural recompilation.
- **Driver semantics and limits.** QM-specific constraints such as minimum
  acquisition delay, sampling rate, output voltage, FEM mode, integration
  weights, waveform limits, and mixer calibration are currently enforced in
  the driver/config layer. The synchronized API must preserve these checks or
  surface them before QuAM export.
- **Calibration state divergence.** A Qibocal update may mutate a Qibolab
  `Parameters` object while a Qualibrate node mutates QuAM fields inside a
  recorded state-update block. The provider needs a common update transaction,
  provenance, rollback behavior, and conflict detection.
- **Node identity and versioning.** A node's filename and `name` are not enough
  to establish equivalence with a Qibocal routine. Bindings must be versioned
  with parameter schemas, result schemas, and the QuAM model expected by the
  node.
- **Results and history.** Qibolab returns pulse-ID keyed readout results;
  Qibocal builds typed data/results objects; Qualibrate nodes produce xarray
  datasets, fit dictionaries, figures, outcomes, and history. The adapter must
  preserve axes, qubit IDs, units, and failed-fit semantics.
- **Simulation parity.** The legacy Qibolab dummy/emulator path, QM driver
  simulation path, and Qualibrate `simulate` action do not necessarily exercise
  the same layer. Each mode needs a declared parity target.
- **Partial capability exposure.** A dictionary reference can expose a field
  without making it executable. The provider must distinguish metadata present
  from operation supported, calibrated, addressable, or safely mutable.

## 7. Suggested implementation order

1. Define the Qibolab reference/snapshot schema, including IDs, references,
   units, capabilities, source versions, and field ownership.
2. Implement a read-only Qibolab introspector and a diff report for channels,
   configs, pulses, qubits, pairs, natives, and instrument limits.
3. Implement the provider-neutral operation descriptor and capability registry.
4. Implement a QuAM synchronizer that can report lossless/lossy mappings before
   writing a `QuamRoot`; retain the legacy Qibolab QM driver as the fallback.
5. Implement a minimal Qibo backend for `I`, `X`/rotation, `Z`/`RZ`, `CZ`, `M`,
   and `Align`, with validation-only compilation tests before live QOP tests.
6. Extract the SDK-neutral parameter-table core and add Qibo/QASM source
   adapters, retaining Qiskit compatibility tests.
7. Add `CalibrationBinding` declarations to a small set of qua-libs nodes and
   implement explicit resolution from Qibocal operation ID to node ID.
8. Build one end-to-end Qualibrate adapter for `05_T1` or
   `03a_qubit_spectroscopy`: Qibo/Qibocal descriptor, QUA acquisition, xarray
   result, fit reuse, and QuAM update.
9. Expand the match registry experiment-by-experiment, starting with the
   strong matches in the table above. Add explicit partial/no-match records
   rather than silently presenting incomplete mappings as supported.

The first vertical slice should prove operation discovery, parameter transport,
measurement/result normalization, and a QuAM state update together. A large
gate list without that complete path would not establish the architecture.
