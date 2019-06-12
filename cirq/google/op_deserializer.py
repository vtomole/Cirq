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

from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence
import sympy

from cirq import devices, ops
from cirq.google import arg_func_langs


class DeserializingArg(
        NamedTuple('DeserializingArg', [
            ('serialized_name', str),
            ('constructor_arg_name', str),
            ('value_func', Optional[Callable[[arg_func_langs.ArgValue], Any]]),
            ('required', bool),
        ])):
    """Specification of the arguments to deserialize an argument to a gate.

    Attributes:
        serialized_name: The serialized name of the gate that is being
            deserialized.
        constructor_arg_name: The name of the argument in the constructor of
            the gate corresponding to this serialized argument.
        value_func: Sometimes a value from the serialized proto needs to
            converted to an appropriate type or form. This function takes the
            serialized value and returns the appropriate type. Defaults to
            None.
        required: Whether a value must be specified when constructing the
            deserialized gate. Defaults to True.
    """

    def __new__(cls,
                serialized_name,
                constructor_arg_name,
                value_func=None,
                required=True):
        return super(DeserializingArg,
                     cls).__new__(cls, serialized_name, constructor_arg_name,
                                  value_func, required)


class GateOpDeserializer:
    """Describes how to deserialize a proto to a given Gate type.

    Attributes:
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(self,
                 serialized_gate_id: str,
                 gate_constructor: type,
                 args: Sequence[DeserializingArg],
                 num_qubits_param: Optional[str] = None):
        """Constructs a deserializer.

        Args:
            serialized_gate_id: The serialized id of the gate that is being
                deserialized.
            gate_constructor: The constructor for the deserialized gate.
            args: A list of the arguments to be read from the serialized
                gate and the information required to use this to construct
                the gate using the gate_constructor above.
            num_qubits_param: Some gate constructors require that the number
                of qubits be passed to their constructor. This is the name
                of the parameter in the constructor for this value. If None,
                no number of qubits is passed to the constructor.
        """
        self.serialized_gate_id = serialized_gate_id
        self.gate_constructor = gate_constructor
        self.args = args
        self.num_qubits_param = num_qubits_param

    def from_proto_dict(self, proto: Dict) -> ops.GateOperation:
        """Turns a cirq.api.google.v2.Operation proto into a GateOperation."""
        qubits = [devices.GridQubit.from_proto_dict(x) for x in proto['qubits']]
        args = self._args_from_proto(proto['args']) if 'args' in proto else {}
        if self.num_qubits_param is not None:
            args[self.num_qubits_param] = len(qubits)
        gate = self.gate_constructor(**args)
        return gate.on(*qubits)

    def _args_from_proto(self, args_proto_dict: Dict
                        ) -> Dict[str, arg_func_langs.ArgValue]:
        return_args = {}
        for arg in self.args:
            if arg.serialized_name not in args_proto_dict and arg.required:
                raise ValueError(
                    'Argument {} not in deserializing args, but is required.'.
                    format(arg.serialized_name))

            value = None  # type: Optional[arg_func_langs.ArgValue]
            if arg.serialized_name in args_proto_dict:
                if 'arg_value' in args_proto_dict[arg.serialized_name]:
                    arg_value = args_proto_dict[
                        arg.serialized_name]['arg_value']
                    if 'float_value' in arg_value:
                        value = float(arg_value['float_value'])
                    elif 'bool_values' in arg_value:
                        value = arg_value['bool_values']['values']
                    elif 'string_value' in arg_value:
                        value = str(arg_value['string_value'])

                if 'symbol' in args_proto_dict[arg.serialized_name]:
                    value = sympy.Symbol(
                        args_proto_dict[arg.serialized_name]['symbol'])

            if value is None and arg.required:
                raise ValueError(
                    'Could not get arg {} from arg_proto {}'.format(
                        arg.serialized_name, args_proto_dict))

            if arg.value_func is not None:
                value = arg.value_func(value)

            if value is not None:
                return_args[arg.constructor_arg_name] = value
        return return_args
