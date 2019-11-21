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

import numpy as np
import pytest
import sympy

import cirq

def test_cz00_str():
    assert str(cirq.CZ00) == 'CZ00'
    assert str(cirq.CZ00**0.5) == 'CZ00**0.5'
    assert str(cirq.CZ00**-0.25) == 'CZ00**-0.25'


def test_cz00_repr():
    assert repr(cirq.CZ00) == 'cirq.CZ00'
    assert repr(cirq.CZ00**0.5) == '(cirq.CZ00**0.5)'
    assert repr(cirq.CZ00**-0.25) == '(cirq.CZ00**-0.25)'


def test_cz00_unitary():
    assert np.allclose(cirq.unitary(cirq.CZ00),
                       np.array([[-1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ00**0.5),
                       np.array([[1j, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ00**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ00**-0.5),
                       np.array([[-1j, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))


def test_cz01_str():
    assert str(cirq.CZ01) == 'CZ01'
    assert str(cirq.CZ01**0.5) == 'CZ01**0.5'
    assert str(cirq.CZ01**-0.25) == 'CZ01**-0.25'


def test_cz00_repr():
    assert repr(cirq.CZ01) == 'cirq.CZ01'
    assert repr(cirq.CZ01**0.5) == '(cirq.CZ01**0.5)'
    assert repr(cirq.CZ01**-0.25) == '(cirq.CZ01**-0.25)'


def test_cz01_unitary():
    assert np.allclose(cirq.unitary(cirq.CZ01),
                       np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ01**0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1j, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ01**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ01**-0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, -1j, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))


def test_cz10_str():
    assert str(cirq.CZ10) == 'CZ10'
    assert str(cirq.CZ10**0.5) == 'CZ10**0.5'
    assert str(cirq.CZ10**-0.25) == 'CZ10**-0.25'


def test_cz10_repr():
    assert repr(cirq.CZ10) == 'cirq.CZ10'
    assert repr(cirq.CZ10**0.5) == '(cirq.CZ10**0.5)'
    assert repr(cirq.CZ10**-0.25) == '(cirq.CZ10**-0.25)'


def test_cz10_unitary():
    assert np.allclose(cirq.unitary(cirq.CZ10),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ10**0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1j, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ10**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ10**-0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, -1j, 0],
                                 [0, 0, 0, 1]]))

def test_trace_distance():
    foo = sympy.Symbol('foo')
    scz00 = cirq.CZ00**foo
    scz01 = cirq.CZ01 ** foo
    scz10 = cirq.CZ10 ** foo
    assert cirq.trace_distance_bound(scz00) == 1.0
    assert cirq.trace_distance_bound(scz01) == 1.0
    assert cirq.trace_distance_bound(scz10) == 1.0
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ00 ** (1 / 9)),
                          np.sin(np.pi / 18))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ01 ** (1 / 9)),
                          np.sin(np.pi / 18))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ10 ** (1 / 9)),
                          np.sin(np.pi / 18))

def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.X(a), cirq.Y(a), cirq.Z(a),
                           cirq.Z(a) ** sympy.Symbol('x'),
                           cirq.Rx(sympy.Symbol('x')).on(a), cirq.CZ00(a, b),
                           cirq.CZ01(a, b), cirq.CZ10(b, a),
                           cirq.H(a) ** 0.5, cirq.I(a),
                           cirq.IdentityGate(2)(a, b))

    cirq.testing.assert_has_diagram(
        circuit, """
a: ───X───Y───Z───Z^x───Rx(x)───@─────@─────@10───H^0.5───I───I───
                                │     │     │                 │
b: ─────────────────────────────@00───@01───@─────────────────I───
""")

    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ---X---Y---Z---Z^x---Rx(x)---@-----@-----@10---H^0.5---I---I---
                                |     |     |                 |
b: -----------------------------@00---@01---@-----------------I---

""",
                                    use_unicode_characters=False)


def test_gates():
    # Pick a qubit.
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    # Create a circuit
    circuit = cirq.Circuit(
        cirq.CZ00(q0, q1) ** 0.5,  # Square root of NOT.
        cirq.CZ01(q0, q1) ** 0.5,  # Square root of NOT.
        cirq.CZ10(q0, q1) ** 0.5,  # Square root of NOT.
        cirq.measure(q1, key='m')  # Measurement.
    )
    print("Circuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)