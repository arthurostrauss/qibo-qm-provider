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

### Compiling a symbolic circuit once, real-time parameterized

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

### Addressing physical qubits, and two-qubit gate direction

`QiboQMBackend` (unlike `QiboQMPlatformBackend`) has no default transpiler
step: a circuit's qubit index `i` addresses whichever QuAM qubit sits at
position `i` of the machine's `active_qubit_names` list, unless you set
`Circuit(n, wire_names=[...])` to physical qubit names explicitly. Many QM
two-qubit natives (e.g. a flux-tunable `CZ`) are also physically
**asymmetric** — the flux pulse only plays on one qubit of the pair — so
`CZ(a, b)` and `CZ(b, a)` are not interchangeable even though Qibo treats
the gate as order-independent. Writing the wrong direction raises a clear
`UnsupportedConnectivityError` naming the direction that is actually
installed, rather than an opaque compiler failure.

See [symbolic_circuit_lowering.md](symbolic_circuit_lowering.md) for the full
design write-up (what's supported, what isn't, and why) and
[slice2_plan.md](slice2_plan.md) for `circuit_to_qua`'s design.
