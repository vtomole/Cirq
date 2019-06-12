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

from typing import Callable, cast, Dict, List, NamedTuple, Type, TypeVar, Union

import numpy as np
import sympy

from cirq import devices, ops
from cirq.google import arg_func_langs

# Type for variables that are subclasses of ops.Gate.
Gate = TypeVar('Gate', bound=ops.Gate)


class SerializingArg(
        NamedTuple('SerializingArg',
                   [('serialized_name', str),
                    ('serialized_type', Type[arg_func_langs.ArgValue]),
                    ('gate_getter',
                     Union[str, Callable[[ops.Gate], arg_func_langs.ArgValue]]),
                    ('required', bool)])):
    """Specification of the arguments for a Gate and its serialization.

    Attributes:
        serialized_name: The name of the argument when it is serialized.
        serialized_type: The type of the argument when it is serialized.
        gate_getter: The name of the property or attribute for getting the
            value of this argument from a gate, or a function that takes a
            gate and returns this value. The later can be used to supply
            a value of the serialized arg by supplying a lambda that
            returns this value (i.e. `lambda x: default_value`)
        required: Whether this argument is a required argument for the
            serialized form.
    """

    def __new__(cls,
                serialized_name,
                serialized_type,
                gate_getter,
                required=True):
        return super(SerializingArg,
                     cls).__new__(cls, serialized_name, serialized_type,
                                  gate_getter, required)


class GateOpSerializer:
    """Describes how to serialize a GateOperation for a given Gate type.

    Attributes:
        gate_type: The type of the gate that can be serialized.
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(self, *, gate_type: Type[Gate], serialized_gate_id: str,
                 args: List[SerializingArg]):
        """Construct the serializer.

        Args:
            gate_type: The type of the gate that is being serialized.
            serialized_gate_id: The string id of the gate when serialized.
            args: A list of specification of the arguments to the gate when
                serializing, including how to get this information from the
                gate of the given gate type.
        """
        self.gate_type = gate_type
        self.serialized_gate_id = serialized_gate_id
        self.args = args

    def to_proto_dict(self, op: ops.GateOperation) -> Dict:
        """Returns the cirq.api.google.v2.Operation message as a proto dict."""
        if not all(isinstance(qubit, devices.GridQubit) for qubit in op.qubits):
            raise ValueError('All qubits must be GridQubits')
        proto_dict = {
            'gate': {
                'id': self.serialized_gate_id
            },
            'qubits': [{
                'id': cast(devices.GridQubit, qubit).proto_id()
            } for qubit in op.qubits]
        }  # type: Dict
        gate = op.gate
        if not isinstance(gate, self.gate_type):
            raise ValueError(
                'Gate of type {} but serializer expected type {}'.format(
                    type(gate), self.gate_type))

        for arg in self.args:
            value = self._value_from_gate(gate, arg)
            if value is not None:
                arg_proto = self._arg_value_to_proto(value)
                if 'args' not in proto_dict:
                    proto_dict['args'] = {}
                proto_dict['args'][arg.serialized_name] = arg_proto

        return proto_dict

    def _value_from_gate(self, gate: ops.Gate,
                         arg: SerializingArg) -> arg_func_langs.ArgValue:
        value = None
        gate_getter = arg.gate_getter

        if isinstance(gate_getter, str):
            value = getattr(gate, gate_getter, None)
            if value is None and arg.required:
                raise ValueError(
                    'Gate {!r} does not have attribute or property {}'.format(
                        gate, gate_getter))
        elif callable(gate_getter):
            value = gate_getter(gate)

        if arg.required and value is None:
            raise ValueError(
                'Argument {} is required, but could not get from gate {!r}'.
                format(arg.serialized_name, gate))

        if isinstance(value, sympy.Symbol):
            return value

        if value is not None:
            self._check_type(value, arg)

        return value

    def _check_type(self, value: arg_func_langs.ArgValue,
                    arg: SerializingArg) -> None:
        if arg.serialized_type == List[bool]:
            if (not isinstance(value, (list, tuple, np.ndarray)) or
                    not all(isinstance(x, (bool, np.bool_)) for x in value)):
                raise ValueError('Expected type List[bool] but was {}'.format(
                    type(value)))
        elif arg.serialized_type == float:
            if not isinstance(value, (float, int)):
                raise ValueError(
                    'Expected type convertible to float but was {}'.format(
                        type(value)))
        elif value is not None and not isinstance(value, arg.serialized_type):
            raise ValueError(
                'Argument {} had type {} but gate returned type {}'.format(
                    arg.serialized_name, arg.serialized_type, type(value)))

    def _arg_value_to_proto(self, value: arg_func_langs.ArgValue) -> Dict:
        arg_value = lambda x: {'arg_value': x}
        if isinstance(value, (float, int)):
            return arg_value({'float_value': float(value)})
        if isinstance(value, str):
            return arg_value({'string_value': str(value)})
        if (isinstance(value, (list, tuple, np.ndarray)) and
                all(isinstance(x, (bool, np.bool_)) for x in value)):
            return arg_value({'bool_values': {'values': list(value)}})
        if isinstance(value, sympy.Symbol):
            return {'symbol': str(value.free_symbols.pop())}
        raise ValueError('Unsupported type of arg value: {}'.format(
            type(value)))
