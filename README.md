# qibo-qm-provider
An interface enabling the compilation and execution of Qibo workflows on Quantum Machine's Quantum Orchestration Platform.

## Current bridge scope
This repository now includes a minimal bridge scaffold for the three planned integration tracks:

1. **Qibocal ↔ Qualibrate node mapping** via `QibocalQualibrateMapper`.
2. **Qibo circuit lowering path** from OpenQASM2 to OpenQASM3 through Qiskit APIs (`qiskit.qasm2.loads` and `qiskit.qasm3.dumps`) via `qasm2_to_qasm3`.
3. **Qibolab pulse lowering** into QUA-like play instructions via `QibolabPulseLowerer`.

It also contains `QiboDeviceSpec` and `QmBackendRepresentation` as an initial backend/device abstraction aligned with Qibo device-class concepts and QM-facing payload generation.
