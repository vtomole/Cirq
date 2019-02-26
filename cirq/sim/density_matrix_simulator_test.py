# Copyright 2019 The Cirq Developers
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
from typing import Tuple

import numpy as np
import pytest
import sympy

import cirq


def test_invalid_dtype():
    with pytest.raises(ValueError, match='complex'):
        cirq.DensityMatrixSimulator(dtype=np.int32)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_measurements(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    result = simulator.run(circuit)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_results(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    result = simulator.run(circuit)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_empty_circuit(dtype):
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.run(cirq.Circuit())
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                            cirq.measure(q0), cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_not_channel_op(dtype):
    class BadOp(cirq.Operation):
        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            # coverage: ignore
            return BadOp(self._qubits)

    q0 = cirq.LineQubit(0)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops([BadOp([q0])])
    with pytest.raises(TypeError):
        simulator.simulate(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0),
                                    cirq.measure(q0), cirq.measure(q1))
    simulator = cirq.DensityMatrixSimulator(dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is 2 ** (-99).
    q0_measurements = set(x[0] for x in result.measurements['0'].tolist())
    assert q0_measurements == {0, 1}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_channel(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.amplitude_damp(0.5)(q0),
                                    cirq.measure(q0), cirq.measure(q1))

    simulator = cirq.DensityMatrixSimulator(dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is 2 ** (-99).
    q0_measurements = set(x[0] for x in result.measurements['0'].tolist())
    assert q0_measurements == {0, 1}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measure_at_end(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]] * 3, '1': [[b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measurement_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1),
                                            cirq.H(q0),
                                            cirq.H(q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]] * 3, '1': [[b1]] * 3})
            assert result.repetitions == 3


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            param_resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
            result = simulator.run(circuit, param_resolver=param_resolver)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(result.params, param_resolver)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_ignore_displays(dtype):
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    q0 = cirq.LineQubit(0)
    display = cirq.ApproxPauliStringExpectation(
            cirq.PauliString({q0: cirq.Z}),
            num_samples=1
    )
    circuit = cirq.Circuit.from_ops(cirq.X(q0), display, cirq.measure(q0))
    result = simulator.run(circuit)
    assert result.measurements['0'] == [[True]]

@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [[b0, b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_sweeps_param_resolvers(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.run_sweep(circuit, params=params)

            assert len(results) == 2
            np.testing.assert_equal(results[0].measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(results[1].measurements,
                                    {'0': [[b1]], '1': [[b0]] })
            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_no_circuit(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    expected = np.zeros((4, 4))
    expected[0, 0] = 1.0
    np.testing.assert_almost_equal(result.final_density_matrix, expected)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_density_matrix,
                                   np.ones((4, 4)) * 0.25)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_compare_to_wave_function_simulator(dtype):
    for _ in range(20):
        qubits = cirq.LineQubit.range(4)
        circuit = cirq.testing.random_circuit(qubits, 5, 0.9)
        pure_result = (cirq.Simulator(dtype=dtype)
                       .simulate(circuit,qubit_order=qubits)
                       .density_matrix_of())
        mixed_result = (cirq.DensityMatrixSimulator(dtype=dtype)
                        .simulate(circuit,qubit_order=qubits)
                        .final_density_matrix)
        np.testing.assert_almost_equal(mixed_result, pure_result)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'0': [b0], '1': [b1]})
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + b1, b0 * 2 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_initial_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, initial_state=1)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + 1 - b1, b0 * 2 + 1 - b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)

@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_qubit_order(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, qubit_order=[q1, q0])
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b1 + b0, 2 * b1 + b0] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1))
            resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
            result = simulator.simulate(circuit, param_resolver=resolver)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b0 + b1, 2 * b0 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)
            assert result.params == resolver
            assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0, q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [b0, b1]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_sweeps_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.simulate_sweep(circuit, params=params)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b0 + b1, 2 * b0 + b1] = 1.0
            np.testing.assert_equal(results[0].final_density_matrix,
                                    expected_density_matrix)

            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b1 + b0, 2 * b1 + b0] = 1.0
            np.testing.assert_equal(results[1].final_density_matrix,
                                    expected_density_matrix)

            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1), cirq.H(q0),
                                    cirq.H(q1))
    simulator = cirq.Simulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step.state_vector(),
                                           np.array([0.5] * 4))
        else:
            np.testing.assert_almost_equal(step.state_vector(),
                                           np.array([1, 0, 0, 0]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_empty_circuit(dtype):
    circuit = cirq.Circuit()
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    step = None
    for step in simulator.simulate_moment_steps(circuit):
        pass
    print(step.simulator_state().density_matrix)
    assert step.simulator_state() == cirq.DensityMatrixSimulatorState(
        density_matrix=np.array([[1]]), qubit_map={})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_set_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1), cirq.H(q0),
                                    cirq.H(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        np.testing.assert_almost_equal(step.density_matrix(),
                                       np.ones((4, 4)) * 0.25)
        if i == 0:
            zero_zero = np.zeros((4, 4), dtype=dtype)
            zero_zero[0, 0] = 1
            step.set_density_matrix(zero_zero)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_sample(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, False])
                        or np.array_equal(sample, [False, False]))
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, True])
                        or np.array_equal(sample, [False, False]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_intermediate_measurement(dtype):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.measure(q0), cirq.H(q0))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 1:
            result = int(step.measurements['0'][0])
            expected = np.zeros((2, 2))
            expected[result, result] = 1
            np.testing.assert_almost_equal(step.density_matrix(), expected)
        if i == 2:
            expected = np.array([[0.5, 0.5 * (-1) ** result],
                                 [0.5 * (-1) ** result, 0.5]])
            np.testing.assert_almost_equal(step.density_matrix(), expected)


def test_density_matrix_simulator_state_eq():
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((2, 2)) * 0.5,
                                         qubit_map={q0: 0}),
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((2, 2)) * 0.5,
                                         qubit_map={q0: 0}))
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.eye(2) * 0.5,
                                         qubit_map={q0: 0}))
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.eye(2) * 0.5,
                                         qubit_map={q0: 0, q1: 1}))


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_density_matrix_simulator_state_repr():
    q0 = cirq.LineQubit(0)
    assert (repr(cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5, qubit_map={q0: 0}))
            == "cirq.DensityMatrixSimulatorState(density_matrix="
               "np.array([[0.5, 0.5], [0.5, 0.5]]), "
               "qubit_map={cirq.LineQubit(0): 0})")


def test_density_matrix_trial_result_eq():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5,
        qubit_map={q0: 0})
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state),
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state))
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_simulator_state=final_simulator_state))
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=final_simulator_state))


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_density_matrix_trial_result_repr():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5,
        qubit_map={q0: 0})
    assert (repr(cirq.DensityMatrixTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state)) ==
            "cirq.DensityMatrixTrialResult("
            "params=cirq.ParamResolver({'s': 1}), "
            "measurements={'m': array([[1]])}, "
            "final_simulator_state=cirq.DensityMatrixSimulatorState("
                "density_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]), "
                "qubit_map={cirq.LineQubit(0): 0}))""")
