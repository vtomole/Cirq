import cirq
import sympy
import numpy as np 
from typing import Dict, Union

class IfElse(cirq.Operation):
    import sympy.parsing.sympy_parser

    def __init__(self, key: Union[str, sympy.Expr], fst: cirq.Operation, snd: cirq.Operation):
        self.key = sympy.parsing.sympy_parser.parse_expr(key) if isinstance(key, str) else key
        self.fst = fst
        self.snd = snd

    def _act_on_(self, args):
        opt = self.fst if self.key.subs(args.all_measurements) else self.snd
        cirq.act_on(opt, args)
        return True

    @property
    def qubits(self):
        return sorted(set(self.fst.qubits + self.snd.qubits))

    def with_qubits(self, qubits):
        return IfElse(
            self.key,
            self.fst.with_qubits(qubits),
            self.snd.with_qubits(qubits),
        )

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        return IfElse(
            self.key.subs(key_map),
            self.fst,
            self.snd,
        )

    def _measurement_keys_(self):
        return [str(s) for s in self.key.free_symbols]

qubits = cirq.LineQubit.range(10)

g1_circuit = cirq.FrozenCircuit(
        cirq.Z(qubits[0]),
        cirq.Z(qubits[1]),
        cirq.I(qubits[2]),
    )

g2_circuit = cirq.FrozenCircuit(
        cirq.Z(qubits[0]),
        cirq.I(qubits[1]),
        cirq.Z(qubits[2]),
    )

g1 = cirq.CircuitOperation(g1_circuit)
g2 = cirq.CircuitOperation(g2_circuit)

circuit = cirq.Circuit(
    # cirq.X(qubits[0]),
    # cirq.X(qubits[1]),
    # cirq.X(qubits[2]),

    cirq.H(qubits[5]),
    cirq.H(qubits[6]),

    g1.controlled_by(qubits[5]), 
    g2.controlled_by(qubits[6]), 

    cirq.H(qubits[5]),
    cirq.H(qubits[6]),

    cirq.measure(qubits[5], key='a'),
    cirq.measure(qubits[6], key='b'),

    cirq.I(qubits[0]),
    
    IfElse('a&b', cirq.X(qubits[0]), cirq.I(qubits[0])),
    IfElse('a&~b', cirq.X(qubits[1]), cirq.I(qubits[0])),
    IfElse('~a&b', cirq.X(qubits[2]), cirq.I(qubits[0])),


    cirq.reset(qubits[5]),
    cirq.reset(qubits[6]),

    cirq.H(qubits[5]),
    cirq.H(qubits[6]),

    g1.controlled_by(qubits[5]), 
    g2.controlled_by(qubits[6]), 

    cirq.H(qubits[5]),
    cirq.H(qubits[6]),

    cirq.measure(qubits[5], key = 'c'),
    cirq.measure(qubits[6], key = 'd'),

    )
# print(circuit)
simulator = cirq.Simulator(split_untangled_states=False)
result = simulator.simulate(circuit)
print("Results:")
print(result)