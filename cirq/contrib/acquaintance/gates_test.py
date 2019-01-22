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

from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Sequence, Tuple, TYPE_CHECKING

from numpy.random import poisson
import pytest

import cirq
import cirq.contrib.acquaintance as cca

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Optional


def test_acquaintance_gate_repr():
    assert repr(cca.ACQUAINT) == 'Acq'


def test_acquaintance_gate_text_diagram_info():
    qubits = [cirq.NamedQubit(s) for s in 'xyz']
    circuit = cirq.Circuit([cirq.Moment([cca.ACQUAINT(*qubits)])])
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───█───
      │
y: ───█───
      │
z: ───█───
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_acquaintance_gate_unknown_qubit_count():
    assert cirq.circuit_diagram_info(cca.ACQUAINT, default=None) is None


def test_swap_network_gate():
    qubits = tuple(cirq.NamedQubit(s) for s in alphabet)

    acquaintance_size = 3
    n_parts = 3
    part_lens = (acquaintance_size - 1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = cca.SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit.from_ops(swap_network_op)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───×(0,0)───
      │
b: ───×(0,1)───
      │
c: ───×(1,0)───
      │
d: ───×(1,1)───
      │
e: ───×(2,0)───
      │
f: ───×(2,1)───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    no_decomp = lambda op: isinstance(op.gate,
            (cca.CircularShiftGate, cca.LinearPermutationGate))
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───█───────╲0╱───█─────────────────█───────────╲0╱───█───────0↦1───
      │       │     │                 │           │     │       │
b: ───█───█───╲1╱───█───█─────────────█───█───────╲1╱───█───█───1↦0───
      │   │   │     │   │             │   │       │     │   │
c: ───█───█───╱2╲───█───█───█───╲0╱───█───█───█───╱2╲───█───█───0↦1───
          │   │         │   │   │         │   │   │         │   │
d: ───────█───╱3╲───█───█───█───╲1╱───█───█───█───╱3╲───────█───1↦0───
                    │       │   │     │       │
e: ─────────────────█───────█───╱2╲───█───────█───0↦1─────────────────
                    │           │     │           │
f: ─────────────────█───────────╱3╲───█───────────1↦0─────────────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    no_decomp = lambda op: isinstance(op.gate, cca.CircularShiftGate)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)

    acquaintance_size = 3
    n_parts = 6
    part_lens = (1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = cca.SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit.from_ops(swap_network_op)

    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───╲0╱─────────╲0╱─────────╲0╱─────────
      │           │           │
b: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───
            │           │           │
c: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───
      │           │           │
d: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───
            │           │           │
e: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───
      │           │           │
f: ───╱1╲─────────╱1╲─────────╱1╲─────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

@pytest.mark.parametrize('part_lens',
    [tuple(randint(1, 3) for _ in range(randint(2, 10))) for _ in range(3)])
def test_acquaint_part_pairs(part_lens):
    parts = []
    n_qubits = 0
    for part_len in part_lens:
        parts.append(tuple(range(n_qubits, n_qubits + part_len)))
        n_qubits += part_len
    qubits = tuple(cirq.NamedQubit(s) for s in alphabet)[:n_qubits]
    swap_network_op = cca.SwapNetworkGate(
        part_lens, acquaintance_size=None)(*qubits)
    swap_network = cirq.Circuit.from_ops(
            swap_network_op, device=cca.UnconstrainedAcquaintanceDevice)
    initial_mapping = {q: i for i, q in enumerate(qubits)}

    actual_opps = cca.get_logical_acquaintance_opportunities(
            swap_network, initial_mapping)
    expected_opps = set(frozenset(s + t) for s, t in combinations(parts, 2))
    assert expected_opps == actual_opps

acquaintance_sizes = (None,) # type: Tuple[Optional[int], ...]
acquaintance_sizes += tuple(range(5))
@pytest.mark.parametrize('part_lens, acquaintance_size',
    list(((part_len,) * n_parts, acquaintance_size) for
         part_len, acquaintance_size, n_parts in
         product(range(1, 5),
                 acquaintance_sizes,
                 range(2, 5)))
    )
def test_swap_network_gate_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    qubits = cirq.LineQubit.range(n_qubits)
    swap_network_gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    operations = cirq.decompose_once_with_qubits(swap_network_gate, qubits)
    operations = list(cirq.flatten_op_tree(operations))
    mapping = {q: i for i, q in enumerate(qubits)}
    cca.update_mapping(mapping, operations)
    assert mapping == {q: i for i, q in enumerate(reversed(qubits))}

def test_swap_network_gate_from_ops():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    part_lens = (1, 2, 1, 3, 3)
    operations = [cirq.Z(qubits[0]),
                  cirq.CZ(*qubits[1:3]),
                  cirq.CCZ(*qubits[4:7]),
                  cirq.CCZ(*qubits[7:])]
    acquaintance_size = 3
    swap_network = cca.SwapNetworkGate.from_operations(
            qubits, operations, acquaintance_size)
    assert swap_network.acquaintance_size == acquaintance_size
    assert swap_network.part_lens == part_lens


def test_swap_network_decomposition():
    qubits = cirq.LineQubit.range(8)
    swap_network_gate = cca.SwapNetworkGate((4, 4), 5)
    operations = cirq.decompose_once_with_qubits(swap_network_gate, qubits)
    circuit = cirq.Circuit.from_ops(operations)
    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
0: ───█─────────────█─────────────╲0╱─────────────█─────────█───────0↦2───
      │             │             │               │         │       │
1: ───█─────────────█─────────────╲1╱─────────────█─────────█───────1↦3───
      │             │             │               │         │       │
2: ───█─────────────█───1↦0───────╲2╱───────1↦0───█─────────█───────2↦0───
      │             │   │         │         │     │         │       │
3: ───█───█─────────█───0↦1───█───╲3╱───█───0↦1───█─────────█───█───3↦1───
      │   │         │         │   │     │         │         │   │
4: ───█───█───0↦1───█─────────█───╱4╲───█─────────█───0↦1───█───█───0↦2───
          │   │               │   │     │             │         │   │
5: ───────█───1↦0─────────────█───╱5╲───█─────────────1↦0───────█───1↦3───
          │                   │   │     │                       │   │
6: ───────█───────────────────█───╱6╲───█───────────────────────█───2↦0───
          │                   │   │     │                       │   │
7: ───────█───────────────────█───╱7╲───█───────────────────────█───3↦1───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

def test_swap_network_init_error():
    with pytest.raises(ValueError):
        cca.SwapNetworkGate(())
    with pytest.raises(ValueError):
        cca.SwapNetworkGate((3,))

@pytest.mark.parametrize('part_lens, acquaintance_size', [
    [[l + 1 for l in poisson(size=n_parts, lam=lam)], poisson(4)]
    for n_parts, lam in product(range(2, 20, 3), range(1, 4))
     ])
def test_swap_network_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)

    expected_permutation = {i: j for i, j in
            zip(range(n_qubits), reversed(range(n_qubits)))}
    assert gate.permutation(n_qubits) == expected_permutation

def test_swap_network_permutation_error():
    gate = cca.SwapNetworkGate((1, 1))
    with pytest.raises(ValueError):
        gate.permutation(1)

class OtherOperation(cirq.Operation):
    def __init__(self, qubits: Sequence[cirq.QuditId]) -> None:
        self._qubits = tuple(qubits)

    @property
    def qubits(self) -> Tuple[cirq.QuditId, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.QuditId) -> 'OtherOperation':
        return type(self)(self._qubits)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.qubits == other.qubits)

def test_get_acquaintance_size():
    qubits = cirq.LineQubit.range(5)
    op = OtherOperation(qubits)
    assert op.with_qubits(qubits) == op
    assert cca.get_acquaintance_size(op) == 0

    for s, _ in enumerate(qubits):
        op = cca.ACQUAINT(*qubits[:s + 1])
        assert cca.get_acquaintance_size(op) == s + 1

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 3
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 3

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 4
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0
