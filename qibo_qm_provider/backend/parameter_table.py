"""A ``ParameterTable`` that can be built straight from a Qibo circuit.

``qiskit_qm_provider.parameter_table.ParameterTable`` already does everything
needed to declare QUA variables for a circuit's free parameters and stream new
values into them across shots. The only thing missing for Qibo is a source
adapter, which is what :meth:`QiboParameterTable.from_qibo_circuit` adds --
the Qibo counterpart of the existing ``ParameterTable.from_qiskit``.

Why this delegates instead of walking sympy itself
--------------------------------------------------
``from_qiskit`` builds its entries from ``qc.parameters``, and Qiskit sorts
that collection **by name**. A hand-rolled sympy walk over ``circuit.queue``
would naturally collect symbols in *circuit* order, which for a circuit like
``RZ(0, b)`` then ``RX(0, a)`` diverges from the order the compiled program
expects its inputs in. Nothing would raise -- the values would simply be
streamed into the wrong variables.

Delegating makes that class of bug unreachable rather than merely tested for,
and it inherits ``from_qiskit``'s other behaviours for free: the
``ParameterVectorElement`` name-mangling rule (``p[i]`` -> ``_p_i_``), the
classical-``Var`` handling for ``iter_input_vars()``, and whatever upstream
adds later.

The cost is that ``from_qibo_circuit`` converts the circuit internally. A
caller who needs *both* the table and the Qiskit circuit should convert once
and call ``from_qiskit`` directly rather than converting twice::

    qc = qibo_circuit_to_qiskit(circuit)
    table = QiboParameterTable.from_qiskit(qc)

That is exactly what a backend-level entry point should do, and is why this
class stays a thin convenience rather than the primary API.
"""

from __future__ import annotations

from typing import Callable, Optional

from qibo.models import Circuit as QiboCircuit
from qiskit_qm_provider.parameter_table import ParameterTable

from .circuit_conversion import qibo_circuit_to_qiskit

__all__ = ["QiboParameterTable"]


class QiboParameterTable(ParameterTable):
    """``ParameterTable`` with a Qibo-circuit source adapter.

    Every symbolic gate parameter becomes a QUA ``fixed`` variable, matching
    ``from_qiskit``'s own typing choice for circuit parameters.

    .. warning::
       QUA's ``fixed`` type covers roughly ``[-2, 2)``. Qibo gate angles are in
       **radians**, so any angle beyond about 2 rad overflows. This is not
       introduced here -- ``from_qiskit`` types circuit parameters the same way
       -- but it bites harder for angles than for the amplitudes ``fixed`` was
       chosen for. Note in particular that
       ``qiskit_qm_provider``'s ``VirtualZMacro`` divides by ``2*pi`` *inside*
       QUA, so the overflow happens before the division, not after. Sweeping an
       angle over a full turn needs the value pre-scaled to turns on the host
       side.
    """

    @classmethod
    def from_qibo_circuit(
        cls,
        circuit: QiboCircuit,
        input_type=None,
        filter_function: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> Optional["QiboParameterTable"]:
        """Build a parameter table from a Qibo circuit's symbolic parameters.

        Args:
            circuit: The Qibo circuit. Its sympy symbols are converted to
                Qiskit ``Parameter`` objects by
                :func:`~qibo_qm_provider.backend.circuit_conversion.
                qibo_circuit_to_qiskit`, so the names in the resulting table
                are the sympy symbol names.
            input_type: Forwarded to ``from_qiskit`` -- one of
                ``"INPUT_STREAM"``, ``"OPNIC"``, ``"IO1"``, ``"IO2"``, an
                ``InputType``, or ``None``.
            filter_function: Forwarded to ``from_qiskit``; receives each Qiskit
                ``Parameter``/``Var`` and returns whether to include it.
            name: Table name. Defaults to the converted circuit's name.

        Returns:
            A :class:`QiboParameterTable`, or ``None`` if the circuit has no
            symbolic parameters at all. The ``None`` return is ``from_qiskit``'s
            own convention and is preserved deliberately, since callers
            already branch on it.
        """
        return cls.from_qiskit(
            qibo_circuit_to_qiskit(circuit),
            input_type=input_type,
            filter_function=filter_function,
            name=name,
        )
