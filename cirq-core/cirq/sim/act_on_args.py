# Copyright 2021 The Cirq Developers
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
"""Objects and methods for acting efficiently on a state tensor."""
import abc
import copy
from typing import (
    Any,
    Iterable,
    Dict,
    List,
    TypeVar,
    TYPE_CHECKING,
    Sequence,
    Tuple,
    cast,
    Optional,
    Iterator,
)

import numpy as np

from cirq import protocols
from cirq._compat import deprecated
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.sim.operation_target import OperationTarget

TSelf = TypeVar('TSelf', bound='ActOnArgs')

if TYPE_CHECKING:
    import cirq


class ActOnArgs(OperationTarget[TSelf]):
    """State and context for an operation acting on a state tensor."""

    def __init__(
        self,
        prng: np.random.RandomState = None,
        qubits: Sequence['cirq.Qid'] = None,
        axes: Iterable[int] = None,
        log_of_measurement_results: Dict[str, Any] = None,
    ):
        """Inits ActOnArgs.

        Args:
            prng: The pseudo random number generator to use for probabilistic
                effects.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
        """
        if prng is None:
            prng = cast(np.random.RandomState, np.random)
        if qubits is None:
            qubits = ()
        if axes is None:
            axes = ()
        if log_of_measurement_results is None:
            log_of_measurement_results = {}
        self._qubits = tuple(qubits)
        self.qubit_map = {q: i for i, q in enumerate(qubits)}
        self._axes = tuple(axes)
        self.prng = prng
        self._log_of_measurement_results = log_of_measurement_results
        self._all_measurements: Dict[str, bool] = {}

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def measure(self, qubits: Sequence['cirq.Qid'], key: str, invert_mask: Sequence[bool]):
        """Adds a measurement result to the log.

        Args:
            qubits: The qubits to measure.
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_key_names_` method.
            invert_mask: The invert mask for the measurement.
        """
        bits = self._perform_measurement(qubits)
        corrected = [bit ^ (bit < 2 and mask) for bit, mask in zip(bits, invert_mask)]
        if key in self._log_of_measurement_results:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self._log_of_measurement_results[key] = corrected
        self._all_measurements[key] = any(corrected)

    # pylint: enable=missing-raises-doc
    def get_axes(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return [self.qubit_map[q] for q in qubits]

    @abc.abstractmethod
    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Child classes that perform measurements should implement this with
        the implementation."""

    @abc.abstractmethod
    def copy(self: TSelf) -> TSelf:
        """Creates a copy of the object."""

    def create_merged_state(self: TSelf) -> TSelf:
        """Creates a final merged state."""
        return self

    def kronecker_product(self: TSelf, other: TSelf) -> TSelf:
        """Joins two state spaces together."""
        raise NotImplementedError()

    def factor(
        self: TSelf,
        qubits: Sequence['cirq.Qid'],
        *,
        validate=True,
        atol=1e-07,
    ) -> Tuple[TSelf, TSelf]:
        """Splits two state spaces after a measurement or reset."""
        raise NotImplementedError()

    def transpose_to_qubit_order(self: TSelf, qubits: Sequence['cirq.Qid']) -> TSelf:
        """Physically reindexes the state by the new basis."""
        raise NotImplementedError()

    @property
    def log_of_measurement_results(self) -> Dict[str, Any]:
        return self._log_of_measurement_results

    @property
    def all_measurements(self) -> Dict[str, bool]:
        return self._all_measurements

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._qubits

    def swap(self, q1: 'cirq.Qid', q2: 'cirq.Qid', *, inplace=False):
        """Swaps two qubits.

        This only affects the index, and does not modify the underlying
        state.

        Args:
            q1: The first qubit to swap.
            q2: The second qubit to swap.
            inplace: True to swap the qubits in the current object, False to
                create a copy with the qubits swapped.

        Returns:
            The original object with the qubits swapped if inplace is
            requested, or a copy of the original object with the qubits swapped
            otherwise.

        Raises:
            ValueError: If the qubits are of different dimensionality.
        """
        if q1.dimension != q2.dimension:
            raise ValueError(f'Cannot swap different dimensions: q1={q1}, q2={q2}')

        args = self if inplace else copy.copy(self)
        i1 = self.qubits.index(q1)
        i2 = self.qubits.index(q2)
        qubits = list(args.qubits)
        qubits[i1], qubits[i2] = qubits[i2], qubits[i1]
        args._qubits = tuple(qubits)
        args.qubit_map = {q: i for i, q in enumerate(qubits)}
        return args

    def rename(self, q1: 'cirq.Qid', q2: 'cirq.Qid', *, inplace=False):
        """Renames `q1` to `q2`.

        Args:
            q1: The qubit to rename.
            q2: The new name.
            inplace: True to rename the qubit in the current object, False to
                create a copy with the qubit renamed.

        Returns:
            The original object with the qubits renamed if inplace is
            requested, or a copy of the original object with the qubits renamed
            otherwise.

        Raises:
            ValueError: If the qubits are of different dimensionality.
        """
        if q1.dimension != q2.dimension:
            raise ValueError(f'Cannot rename to different dimensions: q1={q1}, q2={q2}')

        args = self if inplace else copy.copy(self)
        i1 = self.qubits.index(q1)
        qubits = list(args.qubits)
        qubits[i1] = q2
        args._qubits = tuple(qubits)
        args.qubit_map = {q: i for i, q in enumerate(qubits)}
        return args

    def __getitem__(self: TSelf, item: Optional['cirq.Qid']) -> TSelf:
        if item not in self.qubit_map:
            raise IndexError(f'{item} not in {self.qubits}')
        return self

    def __len__(self) -> int:
        return len(self.qubits)

    def __iter__(self) -> Iterator[Optional['cirq.Qid']]:
        return iter(self.qubits)

    @property  # type: ignore
    @deprecated(
        deadline="v0.13",
        fix="Use `protocols.act_on` instead.",
    )
    def axes(self) -> Tuple[int, ...]:
        return self._axes

    @axes.setter  # type: ignore
    @deprecated(
        deadline="v0.13",
        fix="Use `protocols.act_on` instead.",
    )
    def axes(self, value: Iterable[int]):
        self._axes = tuple(value)


def strat_act_on_from_apply_decompose(
    val: Any,
    args: ActOnArgs,
    qubits: Sequence['cirq.Qid'],
) -> bool:
    operations, qubits1, _ = _try_decompose_into_operations_and_qubits(val)
    assert len(qubits1) == len(qubits)
    qubit_map = {q: qubits[i] for i, q in enumerate(qubits1)}
    if operations is None:
        return NotImplemented
    for operation in operations:
        operation = operation.with_qubits(*[qubit_map[q] for q in operation.qubits])
        protocols.act_on(operation, args)
    return True
