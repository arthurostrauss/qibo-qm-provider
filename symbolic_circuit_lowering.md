# Symbolic Qibo circuits → QUA: how the path works, what it cannot do, and why

Status: implementation notes, 2026-08-26. Covers the slice built that day.
Companion to `INTEGRATION_STATUS.md` (status summary) and `slice2_plan.md`
(what is next). Every claim below was checked by running it against this repo's
own `.venv` — qibo 0.3.3, qiskit 2.5.2, qm-qasm 1.7.7, qm-qua 1.4.0 — not read
off documentation.

---

## 1. For a reader who has not seen this before

### The problem

Qibo lets you put a `sympy` symbol where a gate angle goes, and does not
complain:

```python
import sympy, numpy as np
from qibo import Circuit, gates

theta = sympy.Symbol("theta")
c = Circuit(1)
c.add(gates.RZ(0, theta=theta))          # fine
c.add(gates.RX(0, theta=2 * theta + np.pi / 2))   # also fine — arbitrary expressions
```

Qibo stores whatever you passed, with no numeric coercion. But the moment you
try to lower it, it stops:

```python
c.to_qasm()
# TypeError: Cannot convert expression to float
```

That is not a bug in Qibo. `Circuit.to_qasm()` emits OpenQASM **2**, and
OpenQASM 2 has no way to say "this value arrives later". There is no other
lowering API on a Qibo circuit.

### Why we want it anyway

On QM hardware, a *compile-once, re-parameterize-per-shot* program is the
difference between a usable experiment and an unusable one. Recompiling a QUA
program for every angle in a sweep is prohibitively slow. What you want is one
compiled program holding a live QUA variable that you assign new values into.

`qiskit-qm-provider` already does exactly this for Qiskit circuits:
`ParameterTable.from_qiskit(qc)` declares a QUA variable per free parameter,
and `QMBackend.quantum_circuit_to_qua(qc, param_table)` compiles the circuit
with those variables bound in. The machinery exists — Qibo just had no way to
reach it.

### The fix

For a circuit carrying symbols, skip `to_qasm()` and build the Qiskit
`QuantumCircuit` **directly**, gate by gate, turning each sympy symbol into a
Qiskit `Parameter`. From there the existing pipeline works unchanged:

```text
Qibo Circuit (sympy params)
  └─ build_qiskit_circuit_directly       <- new
       └─ Qiskit QuantumCircuit (Parameter objects)
            └─ qiskit.qasm3 Exporter     <- existing, in qiskit-qm-provider
                 │  emits: input float[64] theta;  rz(theta) q[0];
                 └─ qm_qasm.Compiler.compile(code, inputs={...})
                      └─ QUA program holding a live variable
```

The branch is automatic and one-directional: a circuit *without* symbols still
goes through the original, hardware-validated OpenQASM2 route, unchanged.

### Using it

```python
import sympy
from qibo import Circuit, gates
from qibo_qm_provider import QiboQMBackend, QiboParameterTable, add_basic_macros
from qm.qua import program, for_, declare, fixed

theta = sympy.Symbol("theta")
circuit = Circuit(1)
circuit.add(gates.RZ(0, theta=theta))

add_basic_macros(machine)                 # installs rz, x, sx, measure, ...
backend = QiboQMBackend(machine)

# Build the table once up front, so it can be declared/assigned before the
# sweep loop; pass it back into circuit_to_qua explicitly so it isn't
# rebuilt (and the circuit isn't re-converted) on every call.
table = QiboParameterTable.from_qibo_circuit(circuit)

with program() as prog:
    table.declare()                       # declare the QUA variables
    angle = declare(fixed)
    with for_(angle, 0.0, angle < 0.5, angle + 0.1):
        table.assign_parameters({"theta": angle})
        backend.circuit_to_qua(circuit, param_table=table)
```

which produces exactly the sweep you want — the angle is a live QUA variable
inside the loop, not a recompilation per point:

```python
with for_(v2, 0.0, (v2 < 0.5), (v2 + 0.1)):
    assign(v1, v2)
    assign(v3, v1)
    frame_rotation_2pi(((0.0-v3)*0.15915494309189535), 'q0.xy')
```

(Note `table.declare()`, not `declare_variables()` — the latter is a deprecated
alias in the installed `qiskit-qm-provider`.)

`backend.circuit_to_qua(circuit)` (slice 2) is now the one-call entry point:
it converts the circuit exactly once, builds a table from that same object
unless one is passed explicitly (the multi-circuit-shared-table case above),
and delegates to `quantum_circuit_to_qua`. The manual three-step composition
this section used to require is no longer necessary for the common case.

### Where this lives, and where it does not

This is a **`QiboQMBackend`** capability. `QiboQMPlatformBackend` (the
Qibolab/Qibocal-facing path) cannot do it, and this is structural, not a
missing feature: `qibolab._core.native.rotation()` raises `TypeError` on a
symbolic angle, and every rule in `qibolab._core.compilers.default` does eager
NumPy arithmetic on `gate.parameters`. Qibolab has no symbolic IR node to defer
into. Its real-time story is `Sweeper`, which attaches at *pulse-field*
granularity (amplitude/duration/phase of one already-built pulse), not at gate
level.

So the two backends are not redundant, and this is the concrete reason:

| | `QiboQMBackend` | `QiboQMPlatformBackend` |
|---|---|---|
| Route | OpenQASM/`qm_qasm` (+ this direct builder) | qibolab `Compiler` + QuAM channels |
| Gate-level symbolic parameters | **yes** | no, and cannot be |
| Real-time parameterization | circuit-level, per gate argument | pulse-field-level (`Sweeper` granularity) |
| Qibocal protocol compatibility | no | yes |

---

## 2. What was verified

### The pipeline preserves symbols end to end

| Check | Result |
|---|---|
| Qibo holds a sympy `Symbol` as a gate param | yes; also arbitrary expressions (`2*t + pi/2` survives as a sympy `Add`) |
| `Circuit.to_qasm()` on a symbolic circuit | `TypeError: Cannot convert expression to float` |
| `Exporter(includes=(), basis_gates=…, disable_constants=True)` | emits `input float[64] theta;` + `rz(theta) q[0];` |
| `qm_qasm` given a bare symbol | macro receives a live `QuaVariable` |
| `qm_qasm` given an arithmetic expression | macro receives a `QuaBinaryOperation` — real-time arithmetic |
| Custom Qiskit `Gate` named `gpi`, listed in `basis_gates` | emitted as `gpi(phi) q[0]` with **no** inlined definition; dispatched to the registered `gpi` operation |

That last row is what makes non-Qiskit gates viable at all, and it has a
consequence worth stating plainly: **for the QUA path, a custom gate is just
`(name, num_qubits, params)`.** Its `_define()` body is never used, because
`basis_gates` makes it opaque. Definitions and `__array__` exist here only so
the Qiskit transpiler and `Operator()` still work — which is why adding `MS`
was cheap despite its awkward decomposition.

### The gate correspondence

Compared as 2-qubit **circuit** unitaries with `reverse_bits()`, so what is
tested is the argument-role mapping rather than matrix basis ordering. Comparing
bare `gate.matrix()` arrays instead reports false differences for every
asymmetric 2-qubit gate, since Qibo is big-endian and Qiskit little-endian.
This is a real trap when extending the table.

Exact, no wrapper needed: `RX RY RZ`, `U1`→`PhaseGate`, `U2`→`UGate(π/2,φ,λ)`,
`U3`→`UGate`, `PRX`/`U1q`→`RGate`, `CRX CRY CRZ`, `CU1`→`CPhaseGate`,
`RXX RYY RZZ RZX`, `RXXYY`→`XXPlusYYGate`, plus the non-parametric gates.

Three findings that a reasonable person would get wrong:

- **`qibo.CU3(θ,φ,λ) == CUGate(θ, φ, λ, γ=-(φ+λ)/2)`**, and
  **`qibo.CU2(φ,λ) == CUGate(π/2, φ, λ, γ=-(φ+λ)/2)`**. Both `CU3Gate` and
  `CUGate(…, γ=0)` are wrong. Qibo's controlled-U gates carry the phase
  convention of its own `U3`, and controlling a gate makes that global phase
  physical.
- **`GIVENS` and `RBS` are not `XXPlusYYGate`** at any angle scaling. They are
  *real* rotations; `XXPlusYY` has the `-i sin` factors. They are left unmapped
  rather than mapped approximately.
- **Qiskit's `MSGate`/`GMS` cannot express Qibo's `MS`.** `qibo.MS(q0,q1,0,0,θ)`
  *does* equal `MSGate(2,θ)` (itself `RXXGate(θ)`), but Qiskit's version takes a
  symmetric θ *matrix* over `num_qubits` and has nowhere to put the two
  individual phases `φ0`/`φ1`. Hence a 3-parameter custom gate. Its
  decomposition is `MS = (RZ(φ0)⊗RZ(φ1))·RXX(θ)·(RZ(-φ0)⊗RZ(-φ1))`, derived from
  `X_φ = RZ(φ) X RZ(φ)†` and verified numerically.

`fSim` needed nothing new: `qiskit_qm_provider.additional_gates.FSimGate` is
already numerically equal to `qibo.fSim`, so it is re-exported rather than
redefined. `GPI`/`GPI2` reuse Qibo's own `qasm_label` decompositions verbatim
(`u3(π, φ-π/2, π/2-φ)` and `u3(π/2, φ-π/2, π/2-φ)`).

All 42 mapped gates, and both the `__array__` and `_define()` bodies of every
custom or renamed gate class (thirteen, after §3's renaming fix), are
unit-verified against Qibo's own `matrix()`.

---

## 3. Limits, and where they come from

### `qm_qasm` accepts only `+ - * /` on a gate argument

This is the single most important constraint, and it is **not** sympy's limit or
Qiskit's. Qiskit's `ParameterExpression` handles `sin`/`cos`/`**` happily, and
`qiskit.qasm3` exports them without complaint. `qm_qasm` then refuses them:

```text
rz(sin(theta))  -> DisallowedNodeTypeException:
                   Nodes of type <class 'openqasm3.ast.FunctionCall'> are not allowed
rz(theta**2)    -> NotImplementedError: Binary expression undefined: **
```

So the supported sympy grammar is `Symbol`, `Add`, `Mul`, and numbers
(`Float`, `Integer`, `Rational`, `pi`). Anything else raises
`UnsupportedParameterError` naming the node — rejected early, where the message
can mention the gate, rather than surfacing from inside the compiler.

Two details:

- Use `expr.is_number` (lowercase), not the class check `is_Number`. The former
  is `True` for `pi` and `Rational`; the latter is `False` for `pi`, which would
  send it down the unsupported branch.
- sympy folds `t*t` into `Pow(t, 2)`, so squares are *unreachable*, not merely
  unimplemented. There is no way to write one within what `qm_qasm` accepts.

**If `sin`/`cos` are ever needed**, the fix is not in this lowering. It is a QUA
prologue that computes the expression with `qiskit_qm_provider.pulse.
sympy_to_qua` (which translates sympy straight to `qua.Math.*`) and assigns the
result into the input variable. Noted, not built.

On that module: it exists and does translate sympy, but to a **different
target** — sympy → QUA, not sympy → Qiskit `ParameterExpression`. It cannot be
reused on this path, because here the arithmetic is compiled by `qm_qasm` from
OpenQASM3 text. Its `sympy_to_qua_dict`/`match_expr` dispatch shape was the
precedent followed.

### Parameter names can be silently renamed

The failure mode here is not a loud error. Qiskit's exporter registers circuit
parameters with `allow_rename=True`, so a name colliding with a reserved word or
a gate in scope is quietly re-emitted as `x_0`. The `inputs` dict is still keyed
by the original `parameter.name`, so the compile fails with:

```text
MissingInputException: The input x_0 was not provided to the compiler
```

— naming neither the gate nor the cause. Rejected up front instead:

- **Reserved words** — using `qiskit.qasm3.exporter._RESERVED_KEYWORDS`, Qiskit's
  own set, rather than a hand-maintained copy that could drift from what the
  exporter actually tests against.
- **OpenQASM3 built-in constants** (`pi`, `tau`, `euler`, `im`) — *not* in
  Qiskit's reserved set, so the exporter emits `input float[64] pi;` happily and
  `qm_qasm` then raises `RedeclarationException`.
- **Gate-name collisions** — scoped to the operations *this circuit* emits.

That last scoping is a deliberate compromise. `t`, `x`, `s`, `p` are all gate
names and also perfectly reasonable parameter names, so rejecting them
unconditionally would be hostile. But `Exporter`'s `basis_gates` comes from the
backend's whole `Target`, so a parameter named after a macro installed on the
machine but *not* used in this circuit still collides and is **not** caught.
The check is a lower bound. Closing it needs the machine, so it belongs in the
backend-level compile step (slice 2).

Verified as genuinely fine: `theta`, `t`, `q`, `sin`, `my_angle`. (`sin` is not
reserved by Qiskit and compiles.)

### `fixed` overflow on angles — resolved in slice 2, correcting a figure from this section

`ParameterTable` types every circuit parameter as QUA `fixed`, a signed 4.28
fixed-point number covering `[-8, 8)` — this section originally said
`[-2, 2)`, which is actually the valid `amplitude_scale` range for
`play`/`measure` (a different, narrower QUA convention), not the type's
range (verified against `qm.qua.declare`'s own docstring and
`quam.components.channels.Channel.frame_rotation`'s). A single Qibo angle
sweep over one full turn (`0` to `2π ≈ 6.28` rad) fits inside `[-8, 8)`
without pre-scaling. The real risk is a **composite** expression (e.g.
`2*theta + phi`, or a sweep spanning more than ~1.27 turns) whose value
exceeds `8` even when each symbol individually stays in range. Worse,
`VirtualZMacro` calls `frame_rotation` (not `frame_rotation_2pi` directly),
which divides by `2π` *inside* QUA:

```python
qubit.xy.frame_rotation(-angle)      # -> frame_rotation_2pi(-angle / (2*pi))
```

so the overflow happens *before* the division, not after —
`frame_rotation_2pi`'s own automatic `2π` wrap-around never gets a chance to
apply to a radian value that already overflowed `[-8, 8)`.

This is inherited from `from_qiskit`, not introduced here, but it bites harder
for angles than for the amplitudes `fixed` was chosen for. Slice 2
(`slice2_plan.md`) resolved this: fix the documentation to the corrected
bound (done here and on `QiboParameterTable`), and defer a
turns-valued-parameter flag to slice 3, since the corrected bound removes
the urgency for the common single-turn case and there is no live circuit
yet whose composite expression actually needs it.

### Renaming: nine gates, not three — fixed at the source, not aliased

**This section originally described a limitation; it now describes a closed
one.** Mapping a Qibo gate onto a *bare* Qiskit standard gate renames the
operation, and the Qiskit name is what `qm_qasm` looks up. Enumerated by
comparing each mapped gate's Qibo `gate.name` against the built Qiskit gate's
`.name`:

| Qibo | would-be Qiskit name | | Qibo | would-be Qiskit name |
|---|---|---|---|---|
| `u1` | `p` | | `cu1` | `cp` |
| `u2` | `u` | | `cu2` | `cu` |
| `u3` | `u` | | `cu3` | `cu` |
| `prx` | `r` | | `rxxyy` | `xx_plus_yy` |
| `u1q` | `r` | | | |

A user who installed a macro named `prx` — after the gate they actually wrote —
would get a missing-operation error for `r`.

**First fix, tried and superseded:** `QiboQMBackend.install_qibo_gate_aliases()`
walked the machine and, wherever a macro existed under the Qibo name but not
the emitted name, installed the latter as a QuAM reference alias
(`macros["r"] = macros["prx"].get_reference()` — a reference string, not a
second binding of the same object, since a QuAM component may only have one
parent). It worked, but it was a per-machine patch for a problem that exists
for every machine, and it had a real, structural sharp edge: three emitted
names have two Qibo sources each (`u` ← u2/u3, `r` ← prx/u1q, `cu` ← cu2/cu3),
so with both sources installed on one machine, which one the alias should
point at is genuinely undecidable — it had to refuse rather than guess.

**Fix that replaced it:** eliminate the rename at the source instead of
patching around it on every machine. `qibo_qiskit_gates.py` gained nine thin
subclasses of the real standard gate — `QiboU1Gate(PhaseGate)`,
`QiboU2Gate(UGate)`, `QiboU3Gate(UGate)`, `QiboPRXGate(RGate)`,
`QiboU1qGate(RGate)`, `QiboCU1Gate(CPhaseGate)`, `QiboCU2Gate(CUGate)`,
`QiboCU3Gate(CUGate)`, `QiboRXXYYGate(XXPlusYYGate)` — each overriding only
`.name` to Qibo's own name in `__init__`. Verified this is not a shortcut:

- `.name` is a plain mutable instance attribute on a Qiskit `Gate`, not a
  property, and renaming this way survives export + `qm_qasm` dispatch
  unchanged (checked end to end, not just at the Python-object level).
- Subclassing rather than "build a bare `PhaseGate` and mutate `.name`
  afterwards" keeps `isinstance(gate, PhaseGate)` true, so a transpiler pass or
  calibration rule that pattern-matches on gate type is unaffected.
- There is nothing to get wrong about the unitary, since `_define()`/
  `__array__` are simply inherited from the real gate.

Net effect: `install_qibo_gate_aliases`, `gate_map.QIBO_NAME_TO_EMITTED`, and
their six tests are gone. `QIBO_TO_OPERATION_NAME[name] == gate.name` now holds
for **every** entry in `QIBO_TO_QISKIT`, pinned directly by
`test_no_gate_is_ever_renamed` — there is no rename left to alias, on any
machine, ever. The two ambiguous-collision cases (`u2`/`u3`, `prx`/`u1q`,
`cu2`/`cu3`) are not merely handled now; they cannot arise, because each pair
was never the same operation name to begin with.

---

## 4. Bugs found, and what was done about them

### Fixed: bug #3 (`I` → `u(0,0,0)`), open since 2026-08-07

`INTEGRATION_STATUS.md` recorded this as rooted "inside Qiskit's own OQ2 parser
normalization, unaffected by anything Qibo emits" — and therefore not ours. The
cause was right; the conclusion was not.

`Circuit.to_qasm()` always emits `include "qelib1.inc";`, and
`qiskit.qasm2.loads` does not build that whole library in — only a minimal core.
Passing `custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS` supplies the
rest, which is simply honouring the include Qibo asked for. `id` then stays
`id`. One line.

### Fixed: a larger, previously unrecorded gap in the same place

The same missing definitions meant `swap`, `crx`, `cry`, `crz`, `rzz`, `rxx`,
`ryy` — ordinary `qelib1.inc` gates — **all failed the OpenQASM2 route** with
`QASM2ParseError: '<gate>' is not defined in this scope`. This had never been
recorded as a limitation because nothing exercised those gates. The equivalence
test between the two lowering routes found it immediately.

No regressions from enabling the legacy definitions: `x`, `rz`, `cz`, `u3`,
`ccx`, `cu1`, `gpi`, `measure` all parse identically either way, checked gate by
gate. `iSWAP` still fails there — genuinely absent from `qelib1.inc` — and is
supported on the symbolic route instead.

### `register_gate`'s contract was wrong in two ways — one documented, one root-caused and fixed upstream

Both corrected in its docstring, both pinned by tests:

1. **A bare callable is not accepted.** `QMBackend._populate_target` reads
   `macro.apply` unconditionally, so a plain function raises
   `AttributeError: 'function' object has no attribute 'apply'`. The macro must
   be a `QuamMacro` (e.g. a `QubitMacro` subclass). Documented, not fixed —
   this is a real constraint on the macro shape, not a bug.
2. **Registering an existing name did not reach the compiler — and the reason
   turned out to be a genuine, previously-unexplained bug**, not merely an
   `update_target()` limitation. Traced precisely: `qm_qasm.OperationIdentifier`
   defines neither `__eq__` nor `__hash__`, so it falls back to Python's
   default — object identity. Verified directly against qm-qasm 1.7.7:
   `OperationIdentifier("rz", 1, (0,)) == OperationIdentifier("rz", 1, (0,))`
   is `False`. `qiskit-qm-provider` keyed its internal QUA-operation cache
   directly by these objects, so every re-population of an operation that
   already had an entry silently added a duplicate rather than overwriting —
   confirmed by dumping the dict live: after one `register_gate("rz", ...)`
   call, three identical-looking "rz" entries coexist. Whatever matching
   `qm_qasm.Compiler` does at compile time resolves to the *first* one ever
   inserted, which is exactly the observed symptom (`.natives` updates,
   `quantum_circuit_to_qua` keeps calling the original macro).

   **Same defect, second bug, found in passing:**
   `backend_utils.has_conflicting_calibrations` built a `set()` of
   `OperationIdentifier` objects to detect a repeated calibration; with no
   value equality, a real conflict could never be detected — the function
   always silently returned `False`.

   **Fixed upstream, in source** (the real `qiskit-qm-provider` git checkout,
   not the installed copy): `backend_utils.operation_key(name,
   number_of_params, qubits) -> tuple`, a plain value-hashable stand-in, used
   throughout `_populate_target`/`update_target`/`update_calibrations`/
   `has_conflicting_calibrations`. Real `OperationIdentifier` objects are
   materialized fresh, once, only in the `compiler` property — after the
   canonical mapping is known to be correctly deduplicated. Regression tests
   added in that repo's own suite. **Not yet released** — this package's
   dependency floor (`qiskit-qm-provider>=0.3.4`) predates the fix, so a plain
   `pip install` still has the bug; verified the fix works by mirroring the
   two patched files into this repo's `.venv` directly, which flips
   `test_register_gate_cannot_override_an_existing_operation` from `xfail` to
   `XPASS` with no other regressions. Installing the macro on the machine
   *before* constructing the backend remains the documented workaround either
   way. Not yet filed as an upstream `qm_qasm` issue.

### Reported separately: `sequence_to_qua_macro`'s `phase` hook

Not part of this path, found while reading it. `qua_macros.py:216` guards the
frame rotation with `if phase:`, and `bool()` on a QUA variable raises
`QmQuaException: Attempted to use a Python logical operator on a QUA variable`.
So `parameters={name: (pulse_id, "phase")}` — an advertised feature — crashes for
`Pulse` instructions. The `VirtualZ` branch has no such guard and is fine. Fix
is to track "is this overridden" as a Python-level boolean instead of testing
the value's truthiness.

---

## 5. Compromises taken, stated plainly

1. **`from_qibo_circuit` delegates to `from_qiskit`** rather than walking sympy
   itself. `from_qiskit` reads `qc.parameters`, which Qiskit sorts **by name**; a
   circuit-order sympy walk would silently diverge from the order the compiled
   program expects its inputs in, and nothing would raise. Delegating makes that
   class of bug unreachable rather than merely tested for. The cost is a double
   conversion when the caller also wants the circuit — documented, with the
   one-conversion recipe given.
2. **Gate-name collision checking is a lower bound** (see §3). Precise checking
   needs the machine's `Target`.
3. **`_define()` bodies are best-effort.** They are unused on the QUA path. They
   are still verified against Qibo, so transpiling or simulating the same circuit
   cannot silently mean something else.
4. **`GIVENS`, `RBS`, `GeneralizedRBS`, `GeneralizedfSim`, `DEUTSCH` are
   unmapped**, raising with the reason. Mapping `GIVENS`/`RBS` onto `XXPlusYY`
   would have been wrong.
5. **`M` is refused unless plain.** `collapse=True`, a non-`Z` basis, and the
   `p0`/`p1` readout-error model each change what the gate means and have no
   representation here. Silently dropping them would produce results that look
   fine and are wrong.
6. **`.controlled_by(...)` is refused**, detected by comparing the mapped Qiskit
   gate's arity against the Qibo gate's qubit count.
7. **`MS` cannot take a symbolic `θ`** — Qibo's own `MS.__init__` validates
   `0 <= theta <= pi/2` numerically and raises `TypeError: cannot determine truth
   value of Relational`. Its phases may be symbolic. This is upstream of us and
   not worked around.

---

## 6. Verification

```bash
.venv/bin/python -m pytest test/ -q
```

At the time of writing: **213 passed, 5 deselected, 1 xpassed** (the deselected
are `-m iqcc`, needing real cloud credentials; the xpass is
`test_register_gate_cannot_override_an_existing_operation` — see §3/§4, this
`.venv` currently carries the mirrored upstream fix, so it flips from its
documented `xfail`). `test_symbolic_lowering.py` contributes 112 passing tests
plus that one.

Nothing in the new suite needs hardware or credentials: both
`qm_qasm.Compiler.compile` and `qm.generate_qua_script` run fully offline.

The headline end-to-end assertion is worth knowing about, because it does a lot
of work in one line. `add_basic_macros` installs `rz` as `VirtualZMacro`, whose
`apply` is `qubit.xy.frame_rotation(-angle)`. Compiling a symbolic `RZ` and
reading back `generate_qua_script(prog)` gives:

```python
frame_rotation_2pi(((0.0-v2)*0.15915494309189535), 'q0.xy')
```

which simultaneously confirms the symbol survived as a live QUA **variable**
(`v2 = declare(fixed, …)`, not a baked-in float), that the **minus sign** of the
virtual-Z was applied (`0.0-v2`), that the `1/(2π)` conversion to turns
happened, and that it landed on the right qubit's `xy` element.

**Not verified:** anything on real hardware. This path has never been run
against `"arbel"` or any live QOP.
