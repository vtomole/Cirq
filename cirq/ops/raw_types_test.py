# Copyright 2018 The Cirq Developers
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

import pytest

import cirq


class ValiGate(cirq.Gate):
    def num_qubits(self):
        return 2

    def validate_args(self, qubits):
        if len(qubits) == 3:
            raise ValueError()


def test_gate():
    a, b, c = cirq.LineQubit.range(3)

    g = ValiGate()
    assert g.num_qubits() == 2

    _ = g.on(a, c)
    with pytest.raises(ValueError):
        _ = g.on(a, c, b)

    _ = g(a)
    _ = g(a, c)
    with pytest.raises(ValueError):
        _ = g(c, b, a)


def test_op():
    a, b, c = cirq.LineQubit.range(3)
    g = ValiGate()
    op = g(a)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(b, c)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (b, c)


def test_default_validation_and_inverse():
    class TestGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            a, b = qubits
            yield cirq.Z(a)
            yield cirq.S(b)
            yield cirq.X(a)

        def __eq__(self, other):
            return isinstance(other, TestGate)

        def __repr__(self):
            return 'TestGate()'

    a, b = cirq.LineQubit.range(2)

    with pytest.raises(ValueError, match='number of qubits'):
        TestGate().on(a)

    t = TestGate().on(a, b)
    i = t**-1
    assert i**-1 == t
    assert t**-1 == i
    assert cirq.decompose(i) == [cirq.X(a), cirq.S(b)**-1, cirq.Z(a)]
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(i),
        cirq.unitary(t).conj().T,
        atol=1e-8)

    cirq.testing.assert_implements_consistent_protocols(
        i,
        local_vals={'TestGate': TestGate})


def test_no_inverse_if_not_unitary():
    class TestGate(cirq.Gate):
        def num_qubits(self):
            return 1

        def _decompose_(self, qubits):
            return cirq.amplitude_damp(0.5).on(qubits[0])

    assert cirq.inverse(TestGate(), None) is None


@pytest.mark.parametrize('expression, expected_result', (
    (cirq.X * 2, 2 * cirq.X),
    (cirq.Y * 2, cirq.Y + cirq.Y),
    (cirq.Z - cirq.Z + cirq.Z, cirq.Z.wrap_in_linear_combination()),
    (1j * cirq.S * 1j, -cirq.S),
    (cirq.CZ * 1, cirq.CZ / 1),
    (-cirq.CSWAP * 1j, cirq.CSWAP / 1j),
    (cirq.TOFFOLI * 0.5, cirq.TOFFOLI / 2),
))
def test_gate_algebra(expression, expected_result):
    assert expression == expected_result
