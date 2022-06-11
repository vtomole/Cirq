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

from typing import List, Optional, Sequence, TYPE_CHECKING, Union

import cirq
from cirq_google import engine
from cirq_google.engine import util

if TYPE_CHECKING:
    import cirq_google


@cirq._compat.deprecated_class(deadline='v0.16', fix='Use cirq_google.ProcessorSampler instead.')
class QuantumEngineSampler(cirq.Sampler):
    """A sampler that samples from processors managed by the Quantum Engine.

    Exposes a `cirq_google.Engine` instance as a `cirq.Sampler`.
    """

    @util.deprecated_gate_set_parameter
    def __init__(
        self,
        *,
        engine: 'cirq_google.Engine',
        processor_id: Union[str, List[str]],
        gate_set: Optional['cirq_google.serialization.Serializer'] = None,
    ):
        """Inits QuantumEngineSampler.

        Args:
            engine: Quantum engine instance to use.
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
        self._processor_ids = [processor_id] if isinstance(processor_id, str) else processor_id
        self._engine = engine

    def run_sweep(
        self,
        program: Union[cirq.AbstractCircuit, 'cirq_google.EngineProgram'],
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> Sequence[cirq.Result]:
        if isinstance(program, engine.EngineProgram):
            job = program.run_sweep(
                params=params, repetitions=repetitions, processor_ids=self._processor_ids
            )
        else:
            job = self._engine.run_sweep(
                program=program,
                params=params,
                repetitions=repetitions,
                processor_ids=self._processor_ids,
            )
        return job.results()

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[List[cirq.Sweepable]] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> Sequence[Sequence[cirq.Result]]:
        """Runs the supplied circuits.

        In order to gain a speedup from using this method instead of other run
        methods, the following conditions must be satisfied:
            1. All circuits must measure the same set of qubits.
            2. The number of circuit repetitions must be the same for all
               circuits. That is, the `repetitions` argument must be an integer,
               or else a list with identical values.
        """
        if isinstance(repetitions, List) and len(programs) != len(repetitions):
            raise ValueError(
                'len(programs) and len(repetitions) must match. '
                f'Got {len(programs)} and {len(repetitions)}.'
            )
        if isinstance(repetitions, int) or len(set(repetitions)) == 1:
            # All repetitions are the same so batching can be done efficiently
            if isinstance(repetitions, List):
                repetitions = repetitions[0]
            job = self._engine.run_batch(
                programs=programs,
                params_list=params_list,
                repetitions=repetitions,
                processor_ids=self._processor_ids,
            )
            return job.batched_results()
        # Varying number of repetitions so no speedup
        return super().run_batch(programs, params_list, repetitions)

    @property
    def engine(self) -> 'cirq_google.Engine':
        return self._engine


@cirq._compat.deprecated(deadline='v0.16', fix='Use cirq_google.get_engine_sampler()')
def get_engine_sampler(
    processor_id: str, gate_set_name: str = '', project_id: Optional[str] = None
) -> cirq.Sampler:
    """Get an EngineSampler assuming some sensible defaults.

    This uses the environment variable GOOGLE_CLOUD_PROJECT for the Engine
    project_id, unless set explicitly.

    Args:
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        gate_set_name: One of ['sqrt_iswap', 'sycamore'].
            See `cirq_google.NAMED_GATESETS`.
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.

    Returns:
        A `QuantumEngineSampler` instance.

    Raises:
         ValueError: If the supplied gate set is not a supported gate set name.
         EnvironmentError: If no project_id is specified and the environment
            variable GOOGLE_CLOUD_PROJECT is not set.
    """
    return engine.get_engine(project_id).get_sampler(processor_id=processor_id)
