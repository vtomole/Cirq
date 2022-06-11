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

from unittest import mock

import pytest

import cirq
import cirq_google as cg
import cirq_google.cloud.quantum


@pytest.mark.parametrize('circuit', [cirq.Circuit(), cirq.FrozenCircuit()])
def test_run_circuit(circuit):
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(circuit, params, 5)
    engine.run_sweep.assert_called_with(
        params=params, processor_ids=['tmp'], program=circuit, repetitions=5
    )


def test_run_engine_program():
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    program = mock.Mock(spec=cg.EngineProgram)
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(program, params, 5)
    program.run_sweep.assert_called_with(params=params, processor_ids=['tmp'], repetitions=5)
    engine.run_sweep.assert_not_called()


def test_run_batch():
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    sampler.run_batch(circuits, params_list, 5)
    engine.run_batch.assert_called_with(
        params_list=params_list, processor_ids=['tmp'], programs=circuits, repetitions=5
    )


def test_run_batch_identical_repetitions():
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    sampler.run_batch(circuits, params_list, [5, 5])
    engine.run_batch.assert_called_with(
        params_list=params_list, processor_ids=['tmp'], programs=circuits, repetitions=5
    )


def test_run_batch_bad_number_of_repetitions():
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    with pytest.raises(ValueError, match='2 and 3'):
        sampler.run_batch(circuits, params_list, [5, 5, 5])


def test_run_batch_differing_repetitions():
    engine = mock.Mock()
    job = mock.Mock()
    job.results.return_value = []
    engine.run_sweep.return_value = job
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    repetitions = [1, 2]
    sampler.run_batch(circuits, params_list, repetitions)
    engine.run_sweep.assert_called_with(
        params=params2, processor_ids=['tmp'], program=circuit2, repetitions=2
    )
    engine.run_batch.assert_not_called()


def test_engine_sampler_engine_property():
    engine = mock.Mock()
    with cirq.testing.assert_deprecated(
        'Use cirq_google.ProcessorSampler instead.', deadline='v0.16'
    ):
        sampler = cg.QuantumEngineSampler(engine=engine, processor_id='tmp')
    assert sampler.engine is engine


def test_get_engine_sampler_explicit_project_id():
    with mock.patch.object(cirq_google.cloud.quantum, 'QuantumEngineServiceClient', autospec=True):
        sampler = cg.get_engine_sampler(processor_id='hi mom', project_id='myproj')
    assert hasattr(sampler, 'run_sweep')


def test_get_engine_sampler():
    with mock.patch.object(cirq_google.cloud.quantum, 'QuantumEngineServiceClient', autospec=True):
        with mock.patch('google.auth.default', lambda: (None, 'myproj')):
            sampler = cg.get_engine_sampler(processor_id='hi mom')
    assert hasattr(sampler, 'run_sweep')
