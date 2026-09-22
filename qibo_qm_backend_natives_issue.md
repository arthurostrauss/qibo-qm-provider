# `QiboQMBackend.natives` / shared Enum∩QuAM natives

Status: Route A (below) implemented 2026-09-18. Issue #6 (2026-09-22) further
unified both backends on the **most restrictive** set that is (a) backed by
QuAM macros and (b) accepted by both ``qibo.transpiler.unroller.NativeGates``
and qibolab's default ``Compiler`` rule table -- see
``gate_map.ENUM_COMPATIBLE_NATIVE_GATES`` /
``gate_map.enum_compatible_quam_natives``. ``U3`` is excluded (qibolab has no
default compiler rule); ``GPI``/``Align`` are excluded (not in Qibo's
``NativeGates``). Both backends' ``.natives`` and ``execute_circuit``
already-native / decomposition-target sets use this shared helper; the PR #5
special case that widened ``QiboQMBackend``'s already-native set to full
``target.operation_names`` is dropped.

---

Original investigation note follows.


Status: investigation note, 2026-09-18. Route A (below) implemented the same
day: `gate_map.OPERATION_NAME_TO_NATIVE_GATE` plus the filtered `natives`
property in `qibo_qm_backend.py`, with the pinned tests updated to match
(`test/test_qibo_qm_backend.py`, `test/test_symbolic_lowering.py`).

## What it returns today

```python
@property
def natives(self) -> List[str]:
    return list(self._qiskit_backend.target.operation_names)
```
([qibo_qm_backend.py:102-104](qibo_qm_provider/backend/qibo_qm_backend.py:102))

`target.operation_names` is populated by `QMBackend._populate_target()`
(`qiskit_qm_provider/backend/qm_backend.py:423-528`) from three sources:

1. Machine macros whose name matches Qiskit's own gate-name table
   (`get_extended_gate_name_mapping()`, i.e. `get_standard_gate_name_mapping()`
   plus a handful of extras) — the intended case.
2. Machine macros whose name does **not** match that table: a bespoke
   `Instruction(op_, ...)` is synthesized on the spot and added under
   whatever name the macro happens to have (`qm_backend.py:452-462`) — this
   is how the existing tests get `"foo"`/`"bar"` into `natives`
   ([test_qibo_qm_backend.py:256-282](test/test_qibo_qm_backend.py:256)).
3. Qiskit control-flow ops, added unconditionally regardless of what's on
   the machine: `for op, cls in control_flow_name_mapping.items(): ... target.add_instruction(...)` (`qm_backend.py:525-527`), currently
   `box`, `for_loop`, `if_else`, `switch_case`, `while_loop`.

So the user's read is right, and more precisely: it's not "unsorted", it's
that (2) and (3) have no Qibo-gate meaning at all, and even (1) is keyed to
Qiskit's gate vocabulary, which is broader than Qibo's.

## It's not just "too broad" — it doesn't work with Qibo's own contract

`natives` isn't cosmetic; it's a real hook in `qibo.backends.abstract.Backend`:

> Return the native gates of the backend. List[str] or None: for hardware
> backends, return the native gates of the backend.
([abstract.py:79-87](.venv/lib/python3.12/site-packages/qibo/backends/abstract.py:79))

and it's consumed here, when Qibo builds a backend's default transpiler:

```python
natives = cls.backend().natives
...
Unroller(NativeGates[natives]),
```
([qibo/backends/__init__.py:151,166](.venv/lib/python3.12/site-packages/qibo/backends/__init__.py:151))

`NativeGates` is a plain `enum.Flag` with exactly nine members:
`I, Z, RZ, M, GPI2, U3, CZ, iSWAP, CNOT`
([unroller.py:34-51](.venv/lib/python3.12/site-packages/qibo/transpiler/unroller.py:34)).
Lookup is by exact member name, and `FlagMeta.__getitem__` swallows an
unknown key into `NativeGates.NONE` rather than raising
([unroller.py:22-30](.venv/lib/python3.12/site-packages/qibo/transpiler/unroller.py:22)).
Verified directly:

```python
>>> NativeGates[['x', 'cx', 'rz']]
<NativeGates.NONE: 0>
>>> NativeGates[['CZ', 'U3']]
<NativeGates.U3|CZ: 96>
```

`target.operation_names` returns lowercase Qiskit-style strings (`"x"`,
`"cx"`, `"rz"`, ...), which never match any `NativeGates` member name. So
today, `Unroller(NativeGates[backend.natives])` silently resolves to
`NativeGates.NONE` for `QiboQMBackend` regardless of what's actually
installed — the property isn't just noisy, it can't currently do the one
thing Qibo's own API defines it for.

## Two ways to fix it

**Route A — reverse-lookup filter, no Qibo changes needed.**
`qibo_qm_provider/backend/gate_map.py` already has a verified, one-to-one
`QIBO_TO_OPERATION_NAME` table (Qibo gate class name → emitted operation
name; see its module docstring for why the mapping is exact and unrenamed).
Inverting it gives operation name → Qibo gate name for everything this
package can lower. Filtering `target.operation_names` through that inverse
(plus `"measure" -> "M"`) drops control-flow ops and unmapped custom
macros automatically, because they simply have no entry. Restricting the
result further to `NativeGates`'s own nine member names is what actually
matters for the transpiler hook to work: `I, Z, RZ, M, GPI2, U3, CZ, iSWAP,
CNOT` map cleanly today; gates this package supports but `NativeGates`
doesn't know about (`GPI`, `PRX`, `MS`, `fSim`, `U1q`, ...) would still be
absent from `natives`, but they're absent from usefully driving Qibo's
transpiler either way, so nothing is lost that Route A could have kept.

**Route B — register new Qibo native gates.** Checked whether `NativeGates`
supports this: it's a bare `enum.Flag` subclass with no register/extend
hook (`unroller.py:34-91`), and stdlib `Flag` doesn't support adding
members after class creation without private-API surgery
(`aenum.extend_enum` or similar). This is a Qibo-side limitation, not
something `qibo_qm_provider` can address locally — an upstream feature
request, not a local fix. Worth floating with the qibo maintainers if
GPI/MS/fSim-class hardware natives matter, but per the request, not worth
blocking on.

**Recommendation:** Route A locally now. It's a pure filter over an
already-verified table, requires no upstream coordination, and fixes the
concrete bug (natives never reaching `NativeGates` correctly) rather than
just narrowing the list.

## What it breaks

Existing tests currently pin today's raw pass-through behavior and would
need to change with Route A:

- `test_natives_reflect_wrapped_target_after_add_basic_macros`
  (`test/test_qibo_qm_backend.py:40-45`) asserts
  `set(backend.natives) == set(target.operation_names)` exactly.
- `test/test_qibo_qm_backend.py:256,261,267,272,282` assert that custom
  macro names (`"foo"`, `"bar"`) appear directly in `backend.natives`.

Both would need to switch to asserting on the filtered/mapped Qibo names
instead (or a separate accessor exposing the raw operation names, if that
raw view is still wanted for introspection).
