# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import string
from typing import Callable, Dict, Set, Tuple, Union, Any
import numpy as np
import cirq
from cirq import protocols, value, ops


def to_quil_complex_format(num) -> str:
    """A function for outputting a number to a complex string in QUIL format."""
    cnum = complex(str(num))
    return f"{cnum.real}+{cnum.imag}i"

class QuilFormatter(string.Formatter):
    """A unique formatter to correctly output values to QUIL."""

    def __init__(
        self, qubit_id_map: Dict['cirq.Qid', str], measurement_id_map: Dict[str, str]
    ) -> None:
        """Inits QuilFormatter.

        Args:
            qubit_id_map: A dictionary {qubit, quil_output_string} for
            the proper QUIL output for each qubit.
            measurement_id_map: A dictionary {measurement_key,
            quil_output_string} for the proper QUIL output for each
            measurement key.
        """
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.measurement_id_map = {} if measurement_id_map is None else measurement_id_map

    def format_field(self, value: Any, spec: str) -> str:
        if isinstance(value, cirq.ops.Qid):
            value = self.qubit_id_map[value]
        if isinstance(value, str) and spec == 'meas':
            value = self.measurement_id_map[value]
            spec = ''
        return super().format_field(value, spec)

@value.value_equality(approximate=True)
class QuilOneQubitGate(ops.Gate):
    """A QUIL gate representing any single qubit unitary with a DEFGATE and
    2x2 matrix in QUIL.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilOneQubitGate.

        Args:
            matrix: The 2x2 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 1

    def _quil_(self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter') -> str:
        return (
            f'DEFGATE USERGATE:\n    '
            f'{to_quil_complex_format(self.matrix[0, 0])}, '
            f'{to_quil_complex_format(self.matrix[0, 1])}\n    '
            f'{to_quil_complex_format(self.matrix[1, 0])}, '
            f'{to_quil_complex_format(self.matrix[1, 1])}\n'
            f'{formatter.format("USERGATE {0}", qubits[0])}\n'
        )

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilOneQubitGate(matrix=\n{self.matrix}\n)'

    def _value_equality_values_(self):
        return self.matrix


@value.value_equality(approximate=True)
class QuilTwoQubitGate(ops.Gate):
    """A two qubit gate represented in QUIL with a DEFGATE and it's 4x4
    unitary matrix.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilTwoQubitGate.

        Args:
            matrix: The 4x4 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self):
        return self.matrix

    def _quil_(self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter') -> str:
        return (
            f'DEFGATE USERGATE:\n    '
            f'{to_quil_complex_format(self.matrix[0, 0])}, '
            f'{to_quil_complex_format(self.matrix[0, 1])}, '
            f'{to_quil_complex_format(self.matrix[0, 2])}, '
            f'{to_quil_complex_format(self.matrix[0, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[1, 0])}, '
            f'{to_quil_complex_format(self.matrix[1, 1])}, '
            f'{to_quil_complex_format(self.matrix[1, 2])}, '
            f'{to_quil_complex_format(self.matrix[1, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[2, 0])}, '
            f'{to_quil_complex_format(self.matrix[2, 1])}, '
            f'{to_quil_complex_format(self.matrix[2, 2])}, '
            f'{to_quil_complex_format(self.matrix[2, 3])}\n    '
            f'{to_quil_complex_format(self.matrix[3, 0])}, '
            f'{to_quil_complex_format(self.matrix[3, 1])}, '
            f'{to_quil_complex_format(self.matrix[3, 2])}, '
            f'{to_quil_complex_format(self.matrix[3, 3])}\n'
            f'{formatter.format("USERGATE {0} {1}", qubits[0], qubits[1])}\n'
        )

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilTwoQubitGate(matrix=\n{self.matrix}\n)'


def cphase(param: float) -> ops.CZPowGate:
    """Returns a controlled-phase gate as a Cirq CZPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's CPHASE
    gate and the exponent of Cirq's CZPowGate differ by a factor of pi.
    Args:
        param: Gate parameter (in radians).
    Returns:
        A CZPowGate equivalent to a CPHASE gate of given angle.
    """
    return ops.CZPowGate(exponent=param / np.pi)


def cphase00(phi: float) -> ops.TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE00 gate.
    In pyQuil, CPHASE00(phi) = diag([exp(1j * phi), 1, 1, 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [phi, 0, 0, 0].
    Args:
        phi: Gate parameter (in radians).
    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE00 gate of given angle.
    """
    return ops.TwoQubitDiagonalGate([phi, 0, 0, 0])


def cphase01(phi: float) -> ops.TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE01 gate.
    In pyQuil, CPHASE01(phi) = diag(1, [exp(1j * phi), 1, 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [0, phi, 0, 0].
    Args:
        phi: Gate parameter (in radians).
    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE01 gate of given angle.
    """
    return ops.TwoQubitDiagonalGate([0, phi, 0, 0])


def cphase10(phi: float) -> ops.TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE10 gate.
    In pyQuil, CPHASE10(phi) = diag(1, 1, [exp(1j * phi), 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [0, 0, phi, 0].
    Args:
        phi: Gate parameter (in radians).
    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE10 gate of given angle.
    """
    return ops.TwoQubitDiagonalGate([0, 0, phi, 0])


def phase(param: float) -> ops.ZPowGate:
    """Returns a single-qubit phase gate as a Cirq ZPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's PHASE
    gate and the exponent of Cirq's ZPowGate differ by a factor of pi.
    Args:
        param: Gate parameter (in radians).
    Returns:
        A ZPowGate equivalent to a PHASE gate of given angle.
    """
    return ops.ZPowGate(exponent=param / np.pi)


def pswap(phi: float) -> ops.MatrixGate:
    """Returns a Cirq MatrixGate for pyQuil's PSWAP gate.
    Args:
        phi: Gate parameter (in radians).
    Returns:
        A MatrixGate equivalent to a PSWAP gate of given angle.
    """
    # fmt: off
    pswap_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, np.exp(1j * phi), 0],
            [0, np.exp(1j * phi), 0, 0],
            [0, 0, 0, 1]
        ],
        dtype=complex,
    )
    # fmt: on
    return ops.MatrixGate(pswap_matrix)


def xy(param: float) -> ops.ISwapPowGate:
    """Returns an ISWAP-family gate as a Cirq ISwapPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's XY gate
    and the exponent of Cirq's ISwapPowGate differ by a factor of pi.
    Args:
        param: Gate parameter (in radians).
    Returns:
        An ISwapPowGate equivalent to an XY gate of given angle.
    """
    return ops.ISwapPowGate(exponent=param / np.pi)


def xpow(op, formatter):
    if op.gate._exponent == 1 and op.gate._global_shift != -0.5:
        return formatter.format('X {0}\n', op.qubits[0])
    return formatter.format('RX({0}) {1}\n', op.gate._exponent * np.pi,
                                             op.qubits[0])
def ypow(op, formatter):
    if op.gate._exponent == 1 and op.global_shift != -0.5:
        return formatter.format('Y {0}\n', op.qubits[0])
    return formatter.format('RY({0}) {1}\n', op.gate._exponent * np.pi, op.qubits[0])
def zpow(op, formatter):
    if op.gate._exponent == 1 and op.gate.global_shift != -0.5:
        return formatter.format('Z {0}\n', op.qubits[0])
    return formatter.format('RZ({0}) {1}\n', op.gate._exponent * np.pi, op.qubits[0])
def hpow(op, formatter):
    if op.gate._exponent == 1:
        return formatter.format('H {0}\n', op.qubits[0])
    return formatter.format(
            'RY({0}) {3}\nRX({1}) {3}\nRY({2}) {3}\n',
            0.25 * np.pi,
            op.gate._exponent * np.pi,
            -0.25 * np.pi,
            op.qubits[0],
        )

def cz(op, formatter):
    if op.gate._exponent == 1:
        return formatter.format('CZ {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format(
        'CPHASE({0}) {1} {2}\n', op.gate._exponent * np.pi, op.qubits[0], op.qubits[1]
    )

def iswap(op, formatter):
    if op.gate._exponent == 1:
        return formatter.format('ISWAP {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('XY({0}) {1} {2}\n', op.gate._exponent * np.pi, op.qubits[0], op.qubits[1])

def measure(op, formatter):
    if not all(d == 2 for d in op.gate._qid_shape):
        return NotImplemented
    invert_mask = op.gate.invert_mask
    if len(invert_mask) < len(op.qubits):
        invert_mask = invert_mask + (False,) * (len(op.qubits) - len(invert_mask))
    lines = []
    for i, (qubit, inv) in enumerate(zip(op.qubits, invert_mask)):
        if inv:
            lines.append(
                formatter.format('X {0} # Inverting for following measurement\n', qubit)
            )
        lines.append(formatter.format('MEASURE {0} {1:meas}[{2}]\n', qubit, op.gate.key, i))
    return ''.join(lines)

def quilonequbit(op, formatter):
    return (
        f'DEFGATE USERGATE:\n    '
        f'{to_quil_complex_format(op.gate.matrix[0, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[0, 1])}\n    '
        f'{to_quil_complex_format(op.gate.matrix[1, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[1, 1])}\n'
        f'{formatter.format("USERGATE {0}", op.qubits[0])}\n'
    )

def quiltwoqubit(op, formatter):
    return (
        f'DEFGATE USERGATE:\n    '
        f'{to_quil_complex_format(op.gate.matrix[0, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[0, 1])}, '
        f'{to_quil_complex_format(op.gate.matrix[0, 2])}, '
        f'{to_quil_complex_format(op.gate.matrix[0, 3])}\n    '
        f'{to_quil_complex_format(op.gate.matrix[1, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[1, 1])}, '
        f'{to_quil_complex_format(op.gate.matrix[1, 2])}, '
        f'{to_quil_complex_format(op.gate.matrix[1, 3])}\n    '
        f'{to_quil_complex_format(op.gate.matrix[2, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[2, 1])}, '
        f'{to_quil_complex_format(op.gate.matrix[2, 2])}, '
        f'{to_quil_complex_format(op.gate.matrix[2, 3])}\n    '
        f'{to_quil_complex_format(op.gate.matrix[3, 0])}, '
        f'{to_quil_complex_format(op.gate.matrix[3, 1])}, '
        f'{to_quil_complex_format(op.gate.matrix[3, 2])}, '
        f'{to_quil_complex_format(op.gate.matrix[3, 3])}\n'
        f'{formatter.format("USERGATE {0} {1}", op.qubits[0], op.qubits[1])}\n'
    )


class QuilOutput:
    """An object for passing operations and qubits then outputting them to
    QUIL format. The string representation returns the QUIL output for the
    circuit.
    """

    def __init__(self, operations: 'cirq.OP_TREE', qubits: Tuple['cirq.Qid', ...]) -> None:
        """Inits QuilOutput.

        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
        """
        self.qubits = qubits
        self.operations = tuple(cirq.ops.flatten_to_ops(operations))
        self.measurements = tuple(
            op for op in self.operations if isinstance(op.gate, ops.MeasurementGate)
        )
        self.qubit_id_map = self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()
        self.formatter = QuilFormatter(
            qubit_id_map=self.qubit_id_map, measurement_id_map=self.measurement_id_map
        )

    def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
        return {qubit: str(i) for i, qubit in enumerate(self.qubits)}

    def _generate_measurement_ids(self) -> Dict[str, str]:
        index = 0
        measurement_id_map: Dict[str, str] = {}
        for op in self.operations:
            if isinstance(op.gate, ops.MeasurementGate):
                key = protocols.measurement_key_name(op)
                if key in measurement_id_map:
                    continue
                measurement_id_map[key] = f'm{index}'
                index += 1
        return measurement_id_map

    def save_to_file(self, path: Union[str, bytes, int]) -> None:
        """Write QUIL output to a file specified by path."""
        with open(path, 'w') as f:
            f.write(str(self))

    def __str__(self) -> str:
        output = []
        self._write_quil(lambda s: output.append(s))
        return self.rename_defgates(''.join(output))

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        output_func('# Created using Cirq.\n\n')
        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = protocols.measurement_key_name(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func(f'DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n')
            output_func('\n')

        def keep(op: 'cirq.Operation') -> bool:
            if isinstance(op.gate, (ops.XPowGate, ops.YPowGate, ops.ZPowGate, ops.CZPowGate,
                                    ops.HPowGate, ops.MeasurementGate, ops.ISwapPowGate,
                                    QuilOneQubitGate, QuilTwoQubitGate)):
                return True
            return False

        def fallback(op):
            if len(op.qubits) not in [1, 2]:
                return NotImplemented

            mat = protocols.unitary(op, None)
            if mat is None:
                return NotImplemented

            # Following code is a safety measure
            # Could not find a gate that doesn't decompose into a gate
            # with a _quil_ implementation
            # coverage: ignore
            if len(op.qubits) == 1:
                return QuilOneQubitGate(mat).on(*op.qubits)
            return QuilTwoQubitGate(mat).on(*op.qubits)

        def on_stuck(bad_op):
            return ValueError(f'Cannot output operation as QUIL: {bad_op!r}')

        for main_op in self.operations:
            decomposed = protocols.decompose(
                main_op, keep=keep, fallback_decomposer=fallback, on_stuck_raise=on_stuck
            )

            for decomposed_op in decomposed:
                if isinstance(decomposed_op.gate, ops.XPowGate):
                    formated_str = xpow(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.YPowGate):
                    formated_str = ypow(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.ZPowGate):
                    formated_str = zpow(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.CZPowGate):
                    formated_str = cz(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.HPowGate):
                   formated_str = hpow(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.ISwapPowGate):
                    formated_str = iswap(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, ops.MeasurementGate):
                    formated_str = measure(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, QuilOneQubitGate):
                    formated_str = quilonequbit(decomposed_op, self.formatter)
                elif isinstance(decomposed_op.gate, QuilTwoQubitGate):
                    formated_str = quiltwoqubit(decomposed_op, self.formatter)


                output_func(formated_str)

    def rename_defgates(self, output: str) -> str:
        """A function for renaming the DEFGATEs within the QUIL output. This
        utilizes a second pass to find each DEFGATE and rename it based on
        a counter.
        """
        result = output
        defString = "DEFGATE"
        nameString = "USERGATE"
        defIdx = 0
        nameIdx = 0
        gateNum = 0
        i = 0
        while i < len(output):
            if result[i] == defString[defIdx]:
                defIdx += 1
            else:
                defIdx = 0
            if result[i] == nameString[nameIdx]:
                nameIdx += 1
            else:
                nameIdx = 0
            if defIdx == len(defString):
                gateNum += 1
                defIdx = 0
            if nameIdx == len(nameString):
                result = result[: i + 1] + str(gateNum) + result[i + 1 :]
                nameIdx = 0
                i += 1
            i += 1
        return result
