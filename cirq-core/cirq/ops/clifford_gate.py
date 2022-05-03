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

from typing import (
    Any,
    cast,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np

from cirq import protocols, value, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import common_gates, identity, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

# Lazy imports to break circular dependencies.
devices = LazyLoader("devices", globals(), "cirq.devices")
sim = LazyLoader("sim", globals(), "cirq.sim")
transformers = LazyLoader("transformers", globals(), "cirq.transformers")

PauliTransform = NamedTuple('PauliTransform', [('to', Pauli), ('flip', bool)])
document(PauliTransform, """+X, -X, +Y, -Y, +Z, or -Z.""")


def _to_pauli_transform(matrix: np.ndarray) -> Optional[PauliTransform]:
    """Converts matrix to PauliTransform.

    If matrix is not ±Pauli matrix, returns None.
    """
    for pauli in Pauli._XYZ:
        p = protocols.unitary(pauli)
        if np.allclose(matrix, p):
            return PauliTransform(pauli, False)
        if np.allclose(matrix, -p):
            return PauliTransform(pauli, True)
    return None


def _to_clifford_tableau(
    rotation_map: Optional[Dict[Pauli, PauliTransform]] = None,
    *,
    x_to: Optional[PauliTransform] = None,
    z_to: Optional[PauliTransform] = None,
) -> qis.CliffordTableau:
    """Transfer the rotation map to clifford tableau representation"""
    if x_to is None and z_to is None and rotation_map is None:
        raise ValueError(
            "The function either takes rotation_map or a combination "
            ' of x_to and z_to but none were given.'
        )
    elif rotation_map is not None:
        x_to = rotation_map[pauli_gates.X]
        z_to = rotation_map[pauli_gates.Z]
    else:
        assert x_to is not None and z_to is not None, "Both x_to and z_to have to be provided."

    clifford_tableau = qis.CliffordTableau(num_qubits=1)
    clifford_tableau.xs[0, 0] = x_to.to in (pauli_gates.X, pauli_gates.Y)
    clifford_tableau.zs[0, 0] = x_to.to in (pauli_gates.Y, pauli_gates.Z)

    clifford_tableau.xs[1, 0] = z_to.to in (pauli_gates.X, pauli_gates.Y)
    clifford_tableau.zs[1, 0] = z_to.to in (pauli_gates.Y, pauli_gates.Z)

    clifford_tableau.rs = (x_to.flip, z_to.flip)
    return clifford_tableau


def _pretend_initialized() -> 'SingleQubitCliffordGate':
    # HACK: This is a workaround to fool mypy and pylint into correctly handling
    # class fields that can't be initialized until after the class is defined.
    pass


def _validate_map_input(
    required_transform_count: int,
    pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]],
    x_to: Optional[Tuple[Pauli, bool]],
    y_to: Optional[Tuple[Pauli, bool]],
    z_to: Optional[Tuple[Pauli, bool]],
) -> Dict[Pauli, PauliTransform]:
    if pauli_map_to is None:
        xyz_to = {pauli_gates.X: x_to, pauli_gates.Y: y_to, pauli_gates.Z: z_to}
        pauli_map_to = {cast(Pauli, p): trans for p, trans in xyz_to.items() if trans is not None}
    elif x_to is not None or y_to is not None or z_to is not None:
        raise ValueError(
            '{} can take either pauli_map_to or a combination'
            ' of x_to, y_to, and z_to but both were given'
        )
    if len(pauli_map_to) != required_transform_count:
        raise ValueError(
            'Method takes {} transform{} but {} {} given'.format(
                required_transform_count,
                '' if required_transform_count == 1 else 's',
                len(pauli_map_to),
                'was' if len(pauli_map_to) == 1 else 'were',
            )
        )
    if len(set((to for to, _ in pauli_map_to.values()))) != len(pauli_map_to):
        raise ValueError('A rotation cannot map two Paulis to the same')
    return {frm: PauliTransform(to, flip) for frm, (to, flip) in pauli_map_to.items()}


def _pad_tableau(
    clifford_tableau: qis.CliffordTableau, num_qubits_after_padding: int, axes: List[int]
) -> qis.CliffordTableau:
    """Roughly, this function copies self.tableau into the "identity" matrix."""
    # Sanity check
    if len(set(axes)) != clifford_tableau.n:
        raise ValueError(
            "Input axes of padding should match with the number of qubits in the input tableau."
        )
    if clifford_tableau.n > num_qubits_after_padding:
        raise ValueError(
            "The number of qubits in the input tableau should not be larger than "
            "num_qubits_after_padding."
        )
    padded_tableau = qis.CliffordTableau(num_qubits_after_padding)
    v_index = np.concatenate((np.asarray(axes), num_qubits_after_padding + np.asarray(axes)))

    padded_tableau.xs[np.ix_(v_index, axes)] = clifford_tableau.xs
    padded_tableau.zs[np.ix_(v_index, axes)] = clifford_tableau.zs
    padded_tableau.rs[v_index] = clifford_tableau.rs
    return padded_tableau


class CommonCliffordGateMetaClass(value.ABCMetaImplementAnyOneOf):
    """A metaclass used to lazy initialize several common Clifford Gate as class attributes."""

    @property
    def I(cls):
        if getattr(cls, '_I', None) is None:
            cls._I = cls._generate_clifford_from_known_gate(1, identity.I)
        return cls._I

    @property
    def X(cls):
        if getattr(cls, '_X', None) is None:
            cls._X = cls._generate_clifford_from_known_gate(1, pauli_gates.X)
        return cls._X

    @property
    def Y(cls):
        if getattr(cls, '_Y', None) is None:
            cls._Y = cls._generate_clifford_from_known_gate(1, pauli_gates.Y)
        return cls._Y

    @property
    def Z(cls):
        if getattr(cls, '_Z', None) is None:
            cls._Z = cls._generate_clifford_from_known_gate(1, pauli_gates.Z)
        return cls._Z

    @property
    def H(cls):
        if getattr(cls, '_H', None) is None:
            cls._H = cls._generate_clifford_from_known_gate(1, common_gates.H)
        return cls._H

    @property
    def S(cls):
        if getattr(cls, '_S', None) is None:
            cls._S = cls._generate_clifford_from_known_gate(1, common_gates.S)
        return cls._S

    @property
    def CNOT(cls):
        if getattr(cls, '_CNOT', None) is None:
            cls._CNOT = cls._generate_clifford_from_known_gate(2, common_gates.CNOT)
        return cls._CNOT

    @property
    def CZ(cls):
        if getattr(cls, '_CZ', None) is None:
            cls._CZ = cls._generate_clifford_from_known_gate(2, common_gates.CZ)
        return cls._CZ

    @property
    def SWAP(cls):
        if getattr(cls, '_SWAP', None) is None:
            cls._SWAP = cls._generate_clifford_from_known_gate(2, common_gates.SWAP)
        return cls._SWAP

    @property
    def X_sqrt(cls):
        if getattr(cls, '_X_sqrt', None) is None:
            # Unfortunately, due the code style, the matrix should be viewed transposed.
            # Note xs, zs, and rs are column vector.
            # Transformation: X -> X, Z -> -Y
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[0, 1], xs=[[1], [1]], zs=[[0], [1]]
            )
            cls._X_sqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._X_sqrt

    @property
    def X_nsqrt(cls):
        if getattr(cls, '_X_nsqrt', None) is None:
            # Transformation: X->X, Z->Y
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[0, 0], xs=[[1], [1]], zs=[[0], [1]]
            )
            cls._X_nsqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._X_nsqrt

    @property
    def Y_sqrt(cls):
        if getattr(cls, '_Y_sqrt', None) is None:
            # Transformation: X -> -Z,  Z -> X
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[1, 0], xs=[[0], [1]], zs=[[1], [0]]
            )
            cls._Y_sqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._Y_sqrt

    @property
    def Y_nsqrt(cls):
        if getattr(cls, '_Y_nsqrt', None) is None:
            # Transformation: X -> Z, Z -> -X
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[0, 1], xs=[[0], [1]], zs=[[1], [0]]
            )
            cls._Y_nsqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._Y_nsqrt

    @property
    def Z_sqrt(cls):
        if getattr(cls, '_Z_sqrt', None) is None:
            # Transformation: X -> Y, Z -> Z
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[0, 0], xs=[[1], [0]], zs=[[1], [1]]
            )
            cls._Z_sqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._Z_sqrt

    @property
    def Z_nsqrt(cls):
        if getattr(cls, '_Z_nsqrt', None) is None:
            # Transformation: X -> -Y,  Z -> Z
            _clifford_tableau = qis.CliffordTableau._from_json_dict_(
                n=1, rs=[1, 0], xs=[[1], [0]], zs=[[1], [1]]
            )
            cls._Z_nsqrt = cls.from_clifford_tableau(_clifford_tableau)
        return cls._Z_nsqrt


class CommonCliffordGates(metaclass=CommonCliffordGateMetaClass):

    # We need to use the lazy initialization of these common gates since they need to use
    # cirq.sim, which can not be imported when
    @classmethod
    def _generate_clifford_from_known_gate(
        cls, num_qubits: int, gate: raw_types.Gate
    ) -> Union['SingleQubitCliffordGate', 'CliffordGate']:
        qubits = devices.LineQubit.range(num_qubits)
        t = qis.CliffordTableau(num_qubits=num_qubits)
        args = sim.CliffordTableauSimulationState(
            tableau=t, qubits=qubits, prng=np.random.RandomState()
        )

        protocols.act_on(gate, args, qubits, allow_decompose=False)
        if num_qubits == 1:
            return SingleQubitCliffordGate.from_clifford_tableau(args.tableau)
        return CliffordGate.from_clifford_tableau(args.tableau)

    @classmethod
    def from_clifford_tableau(cls, tableau: qis.CliffordTableau) -> 'CliffordGate':
        """Create the CliffordGate instance from Clifford Tableau.

        Args:
            tableau: A CliffordTableau to define the effect of Clifford Gate applying on
            the stabilizer state or Pauli group. The meaning of tableau here is
                    To  X   Z    sign
            from  X  [ X_x Z_x | r_x ]
            from  Z  [ X_z Z_z | r_z ]
            Each row in the Clifford tableau indicates how the transformation of original
            Pauli gates to the new gates after applying this Clifford Gate.

        Returns:
            A CliffordGate instance, which has the transformation defined by
            the input tableau.

        Raises:
            ValueError: When input tableau is wrong type or the tableau does not
            satisfy the symplectic property.
        """
        if not isinstance(tableau, qis.CliffordTableau):
            raise ValueError('Input argument has to be a CliffordTableau instance.')
        if not tableau._validate():
            raise ValueError('It is not a valid Clifford tableau.')
        return CliffordGate(_clifford_tableau=tableau)

    @classmethod
    def from_op_list(
        cls, operations: Sequence[raw_types.Operation], qubit_order: Sequence[raw_types.Qid]
    ) -> 'CliffordGate':
        """Construct a new Clifford gates from several known operations.

        Args:
            operations: A list of cirq operations to construct the Clifford gate.
                The combination order is the first element in the list applies the transformation
                on the stabilizer state first.
            qubit_order: Determines how qubits are ordered when decomposite the operations.

        Returns:
            A CliffordGate instance, which has the transformation on the stabilizer
            state equivalent to the composition of operations.

        Raises:
            ValueError: When one or more operations do not have stabilizer effect.
        """
        for op in operations:
            if op.gate and op.gate._has_stabilizer_effect_():
                continue
            raise ValueError(
                "Clifford Gate can only be constructed from the "
                "operations that has stabilizer effect."
            )

        base_tableau = qis.CliffordTableau(len(qubit_order))
        args = sim.clifford.CliffordTableauSimulationState(
            tableau=base_tableau, qubits=qubit_order, prng=np.random.RandomState(0)  # unused
        )
        for op in operations:
            protocols.act_on(op, args, allow_decompose=True)

        return CliffordGate.from_clifford_tableau(args.tableau)

    @classmethod
    def _from_json_dict_(cls, n, rs, xs, zs, **kwargs):
        _clifford_tableau = qis.CliffordTableau._from_json_dict_(n, rs, xs, zs)
        return cls(_clifford_tableau=_clifford_tableau)

    @classmethod
    def _get_sqrt_map(
        cls,
    ) -> Dict[float, Dict['SingleQubitCliffordGate', 'SingleQubitCliffordGate']]:
        """Returns a map containing two keys 0.5 and -0.5 for the sqrt mapping of Pauli gates."""
        return {
            0.5: {cls.X: cls.X_sqrt, cls.Y: cls.Y_sqrt, cls.Z: cls.Z_sqrt},
            -0.5: {cls.X: cls.X_nsqrt, cls.Y: cls.Y_nsqrt, cls.Z: cls.Z_nsqrt},
        }


@value.value_equality
class CliffordGate(raw_types.Gate, CommonCliffordGates):
    """Clifford rotation for N-qubit."""

    def __init__(self, *, _clifford_tableau: qis.CliffordTableau) -> None:
        # We use the Clifford tableau to represent a Clifford gate.
        # It is crucial to note that the meaning of tableau here is different
        # from the one used to represent a Clifford state (Of course, they are related).
        # A) We have to use the full 2n * (2n + 1) matrix
        # B) The meaning of tableau here is
        #                 X   Z    sign
        #     from  X  [ X_x Z_x | r_x ]
        #     from  Z  [ X_z Z_z | r_z ]
        # Each row in the Clifford tableau means the transformation of original Pauli gates.
        # For example, take a 2 * (2+1) tableau as example:
        #         X       Z     r
        #  XI  [ 1  0 | 1  0  | 0 ]
        #  IX  [ 0  0 | 1  1  | 0 ]
        #  ZI  [ 0  0 | 1  0  | 1 ]
        #  IZ  [ 1  0 | 1  1  | 0 ]
        # Take the third row as example: this means the ZI gate after the this gate,
        # more precisely the conjugate transformation of ZI by this gate, becomes -ZI.
        # (Note the real clifford tableau has to satify the Symplectic property.
        # here is just for illustration)
        self._clifford_tableau = _clifford_tableau.copy()

    @property
    def clifford_tableau(self):
        return self._clifford_tableau

    def _json_dict_(self) -> Dict[str, Any]:
        json_dict = self._clifford_tableau._json_dict_()
        return json_dict

    def _value_equality_values_(self):
        return self._clifford_tableau.matrix().tobytes() + self._clifford_tableau.rs.tobytes()

    def _num_qubits_(self):
        return self.clifford_tableau.n

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        # By definition, Clifford Gate should always return True.
        return True

    def __pow__(self, exponent) -> 'CliffordGate':
        if exponent == -1:
            return CliffordGate.from_clifford_tableau(self.clifford_tableau.inverse())
        if exponent > 0 and int(exponent) == exponent:
            base_tableau = self.clifford_tableau.copy()
            for _ in range(int(exponent) - 1):
                base_tableau = base_tableau.then(self.clifford_tableau)
            return CliffordGate.from_clifford_tableau(base_tableau)
        if exponent < 0 and int(exponent) == exponent:
            base_tableau = self.clifford_tableau.copy()
            for _ in range(int(-exponent) - 1):
                base_tableau = base_tableau.then(self.clifford_tableau)
            return CliffordGate.from_clifford_tableau(base_tableau.inverse())

        return NotImplemented

    def __repr__(self) -> str:
        return f"Clifford Gate with Tableau:\n {self.clifford_tableau._str_full_()}"

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        # Note even if we assume two gates define the tabluea based on the same qubit order,
        # the following approach cannot judge it:
        # self.clifford_tableau.then(other.clifford_tableau) == other.clifford_tableau.then(
        #     self.clifford_tableau
        # )
        # For example: X.then(Z) and Z.then(X) both return same tableau
        # it is because Clifford tableau ignores the global phase information.
        return NotImplemented

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return transformers.analytical_decompositions.decompose_clifford_tableau_to_operations(
            list(qubits), self.clifford_tableau
        )

    def _act_on_(
        self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']
    ) -> Union[NotImplementedType, bool]:

        # Note the computation complexity difference between _decompose_ and _act_on_.
        # Suppose this Gate has `m` qubits, args has `n` qubits, and the decomposition of
        # this operation into `k` operations:
        #   1. Direct act_on is O(n^3) -- two matrices multiplication
        #   2. Decomposition is O(m^3)+O(k*n^2) -- Decomposition complexity + k * One/two-qubits Ops
        # So when m << n, the decomposition is more efficient.
        if isinstance(sim_state, sim.clifford.CliffordTableauSimulationState):
            axes = sim_state.get_axes(qubits)
            # This padding is important and cannot be omitted.
            padded_tableau = _pad_tableau(self._clifford_tableau, len(sim_state.qubits), axes)
            sim_state._state = sim_state.tableau.then(padded_tableau)
            return True

        if isinstance(sim_state, sim.clifford.StabilizerChFormSimulationState):  # coverage: ignore
            # Do we know how to apply CliffordTableau on StabilizerChFormSimulationState?
            # It should be unlike because CliffordTableau ignores the global phase but CHForm
            # is aimed to fix that.
            return NotImplemented

        return NotImplemented


@value.value_equality(manual_cls=True)
class SingleQubitCliffordGate(CliffordGate):
    """Any single qubit Clifford rotation."""

    def __init__(self, *, _clifford_tableau: qis.CliffordTableau) -> None:
        super().__init__(_clifford_tableau=_clifford_tableau)

    def _num_qubits_(self):
        return 1

    @staticmethod
    def from_clifford_tableau(tableau: qis.CliffordTableau) -> 'SingleQubitCliffordGate':
        if not isinstance(tableau, qis.CliffordTableau):
            raise ValueError('Input argument has to be a CliffordTableau instance.')
        if not tableau._validate():
            raise ValueError('Input tableau is not a valid Clifford tableau.')
        if tableau.n != 1:
            raise ValueError(
                'The number of qubit of input tableau should be 1 for SingleQubitCliffordGate.'
            )
        return SingleQubitCliffordGate(_clifford_tableau=tableau)

    @staticmethod
    def from_xz_map(
        x_to: Tuple[Pauli, bool], z_to: Tuple[Pauli, bool]
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the specified transforms.
        The Y transform is derived from the X and Z.

        Args:
            x_to: Which Pauli to transform X to and if it should negate.
            z_to: Which Pauli to transform Z to and if it should negate.
        """
        return SingleQubitCliffordGate.from_clifford_tableau(
            _to_clifford_tableau(x_to=PauliTransform(*x_to), z_to=PauliTransform(*z_to))
        )

    @staticmethod
    def from_single_map(
        pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]] = None,
        *,
        x_to: Optional[Tuple[Pauli, bool]] = None,
        y_to: Optional[Tuple[Pauli, bool]] = None,
        z_to: Optional[Tuple[Pauli, bool]] = None,
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        The arguments are exclusive, only one may be specified.

        Args:
            pauli_map_to: A dictionary with a single key value pair describing
                the transform.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(1, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        ((trans_from, (trans_to, flip)),) = tuple(rotation_map.items())
        if trans_from == trans_to:
            trans_from2 = Pauli.by_relative_index(trans_to, 1)  # 1 or 2 work
            trans_to2 = Pauli.by_relative_index(trans_from, 1)
            flip2 = False
        else:
            trans_from2 = trans_to
            trans_to2 = trans_from
            flip2 = not flip
        rotation_map[trans_from2] = PauliTransform(trans_to2, flip2)
        return SingleQubitCliffordGate.from_double_map(
            cast(Dict[Pauli, Tuple[Pauli, bool]], rotation_map)
        )

    @staticmethod
    def from_double_map(
        pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]] = None,
        *,
        x_to: Optional[Tuple[Pauli, bool]] = None,
        y_to: Optional[Tuple[Pauli, bool]] = None,
        z_to: Optional[Tuple[Pauli, bool]] = None,
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

        Args:
            pauli_map_to: A dictionary with two key value pairs describing
                two transforms.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(2, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        (from1, trans1), (from2, trans2) = tuple(rotation_map.items())
        from3 = from1.third(from2)
        to3 = trans1.to.third(trans2.to)
        flip3 = trans1.flip ^ trans2.flip ^ ((from1 < from2) != (trans1.to < trans2.to))
        rotation_map[from3] = PauliTransform(to3, flip3)

        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))

    @staticmethod
    def from_pauli(pauli: Pauli, sqrt: bool = False) -> 'SingleQubitCliffordGate':
        prev_pauli = Pauli.by_relative_index(pauli, -1)
        next_pauli = Pauli.by_relative_index(pauli, 1)
        if sqrt:
            rotation_map = {
                prev_pauli: PauliTransform(next_pauli, True),
                pauli: PauliTransform(pauli, False),
                next_pauli: PauliTransform(prev_pauli, False),
            }
        else:
            rotation_map = {
                prev_pauli: PauliTransform(prev_pauli, True),
                pauli: PauliTransform(pauli, False),
                next_pauli: PauliTransform(next_pauli, True),
            }
        return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))

    @staticmethod
    def from_quarter_turns(pauli: Pauli, quarter_turns: int) -> 'SingleQubitCliffordGate':
        quarter_turns = quarter_turns % 4
        if quarter_turns == 0:
            return SingleQubitCliffordGate.I
        if quarter_turns == 1:
            return SingleQubitCliffordGate.from_pauli(pauli, True)
        if quarter_turns == 2:
            return SingleQubitCliffordGate.from_pauli(pauli)

        return SingleQubitCliffordGate.from_pauli(pauli, True) ** -1

    @staticmethod
    def from_unitary(u: np.ndarray) -> Optional['SingleQubitCliffordGate']:
        """Creates Clifford gate with given unitary (up to global phase).

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            SingleQubitCliffordGate, whose matrix is equal to given matrix (up
            to global phase), or `None` if `u` is not a matrix of a single-qubit
            Clifford gate.
        """
        if u.shape != (2, 2) or not linalg.is_unitary(u):
            return None
        x = protocols.unitary(pauli_gates.X)
        z = protocols.unitary(pauli_gates.Z)
        x_to = _to_pauli_transform(u @ x @ u.conj().T)
        z_to = _to_pauli_transform(u @ z @ u.conj().T)
        if x_to is None or z_to is None:
            return None
        return SingleQubitCliffordGate.from_clifford_tableau(
            _to_clifford_tableau(x_to=x_to, z_to=z_to)
        )

    def transform(self, pauli: Pauli) -> PauliTransform:
        x_to = self._clifford_tableau.destabilizers()[0]
        z_to = self._clifford_tableau.stabilizers()[0]
        if pauli == pauli_gates.X:
            to = x_to
        elif pauli == pauli_gates.Z:
            to = z_to
        else:
            to = x_to * z_to  # Y = iXZ
            to._coefficient *= 1j
        # pauli_mask returns a value between 0 and 4 for [I, X, Y, Z].
        to_gate = Pauli._XYZ[to.pauli_mask[0] - 1]
        return PauliTransform(to=to_gate, flip=bool(to.coefficient != 1.0))

    def to_phased_xz_gate(self) -> phased_x_z_gate.PhasedXZGate:
        """Convert this gate to a PhasedXZGate instance.

        The rotation can be categorized by {axis} * {degree}:
            * Identity: I
            * {x, y, z} * {90, 180, 270}  --- {X, Y, Z} + 6 Quarter turn gates
            * {+/-xy, +/-yz, +/-zx} * 180  --- 6 Hadamard-like gates
            * {middle point of xyz in 4 Quadrant} * {120, 240} --- swapping axis
        note 1 + 9 + 6 + 8 = 24 in total.

        To associate with Clifford Tableau, it can also be grouped by 4:
            * {I,X,Y,Z} is [[1 0], [0, 1]]
            * {+/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane} is [[1, 0], [1, 1]]
            * {+/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane} is [[1, 1], [0, 1]]
            * {+/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane} is [[0, 1], [1, 0]]
            * {middle point of xyz in 4 Quadrant} * 120 is [[0, 1], [1, 1]]
            * {middle point of xyz in 4 Quadrant} * 240 is [[1, 1], [1, 0]]
        """
        x_to_flip, z_to_flip = self.clifford_tableau.rs
        flip_index = int(z_to_flip) * 2 + x_to_flip
        a, x, z = 0.0, 0.0, 0.0

        if np.array_equal(self.clifford_tableau.matrix(), [[1, 0], [0, 1]]):
            # I, Z, X, Y cases
            to_phased_xz = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif np.array_equal(self.clifford_tableau.matrix(), [[1, 0], [1, 1]]):
            # +/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane
            a = 0.0
            x = 0.5 if x_to_flip ^ z_to_flip else -0.5
            z = 1.0 if x_to_flip else 0.0
        elif np.array_equal(self.clifford_tableau.matrix(), [[0, 1], [1, 0]]):
            # +/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane
            a = 0.5
            x = 0.5 if x_to_flip else -0.5
            z = 0.0 if x_to_flip ^ z_to_flip else 1.0
        elif np.array_equal(self.clifford_tableau.matrix(), [[1, 1], [0, 1]]):
            # +/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane
            to_phased_xz = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.25, 1.0, 0.0), (-0.25, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif np.array_equal(self.clifford_tableau.matrix(), [[0, 1], [1, 1]]):
            # axis swapping rotation -- (312) permutation
            a = 0.5
            x = 0.5 if x_to_flip else -0.5
            z = 0.5 if x_to_flip ^ z_to_flip else -0.5
        else:
            # axis swapping rotation -- (231) permutation.
            # This should be the only cases left.
            assert np.array_equal(self.clifford_tableau.matrix(), [[1, 1], [1, 0]])
            a = 0.0
            x = -0.5 if x_to_flip ^ z_to_flip else 0.5
            z = -0.5 if x_to_flip else 0.5
        return phased_x_z_gate.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    def __pow__(self, exponent) -> 'SingleQubitCliffordGate':
        # First to check if we can get the sqrt and negative sqrt Clifford.
        if self._get_sqrt_map().get(exponent, None):
            pow_gate = self._get_sqrt_map()[exponent].get(self, None)
            if pow_gate:
                return pow_gate
        # If not, we try the Clifford Tableau based method.
        ret_gate = super().__pow__(exponent)
        if ret_gate is NotImplemented:
            return NotImplemented
        return SingleQubitCliffordGate.from_clifford_tableau(ret_gate.clifford_tableau)

    def _act_on_(
        self,
        sim_state: 'cirq.SimulationStateBase',  # pylint: disable=unused-argument
        qubits: Sequence['cirq.Qid'],  # pylint: disable=unused-argument
    ):
        # TODO(#5256) Add the implementation of _act_on_ with CliffordTableauSimulationState.
        return NotImplemented

    # Single Clifford Gate decomposition is more efficient than the general Tableau decomposition.
    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        (qubit,) = qubits
        if self == SingleQubitCliffordGate.H:
            return (common_gates.H(qubit),)
        rotations = self.decompose_rotation()
        return tuple(r.on(qubit) ** (qt / 2) for r, qt in rotations)

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        if isinstance(other, SingleQubitCliffordGate):
            return self.commutes_with_single_qubit_gate(other)
        if isinstance(other, Pauli):
            return self.commutes_with_pauli(other)
        return NotImplemented

    def commutes_with_single_qubit_gate(self, gate: 'SingleQubitCliffordGate') -> bool:
        """Tests if the two circuits would be equivalent up to global phase:
        --self--gate-- and --gate--self--"""
        self_then_gate = self.clifford_tableau.then(gate.clifford_tableau)
        gate_then_self = gate.clifford_tableau.then(self.clifford_tableau)
        return self_then_gate == gate_then_self

    def commutes_with_pauli(self, pauli: Pauli) -> bool:
        to, flip = self.transform(pauli)
        return to == pauli and not flip

    def merged_with(self, second: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
        return SingleQubitCliffordGate.from_clifford_tableau(
            self.clifford_tableau.then(second.clifford_tableau)
        )

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        mat = np.eye(2)
        qubit = named_qubit.NamedQubit('arbitrary')
        for op in protocols.decompose_once_with_qubits(self, (qubit,)):
            mat = protocols.unitary(op).dot(mat)
        return mat

    def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
        """Returns ((first_rotation_axis, first_rotation_quarter_turns), ...)

        This is a sequence of zero, one, or two rotations."""
        x_rot = self.transform(pauli_gates.X)
        y_rot = self.transform(pauli_gates.Y)
        z_rot = self.transform(pauli_gates.Z)
        whole_arr = (
            x_rot.to == pauli_gates.X,
            y_rot.to == pauli_gates.Y,
            z_rot.to == pauli_gates.Z,
        )
        num_whole = sum(whole_arr)
        flip_arr = (x_rot.flip, y_rot.flip, z_rot.flip)
        num_flip = sum(flip_arr)
        if num_whole == 3:
            if num_flip == 0:
                # Gate is identity
                return []

            # 180 rotation about some axis
            pauli = Pauli.by_index(flip_arr.index(False))
            return [(pauli, 2)]
        if num_whole == 1:
            index = whole_arr.index(True)
            pauli = Pauli.by_index(index)
            next_pauli = Pauli.by_index(index + 1)
            flip = flip_arr[index]
            output = []
            if flip:
                # 180 degree rotation
                output.append((next_pauli, 2))
            # 90 degree rotation about some axis
            if self.transform(next_pauli).flip:
                # Negative 90 degree rotation
                output.append((pauli, -1))
            else:
                # Positive 90 degree rotation
                output.append((pauli, 1))
            return output
        elif num_whole == 0:
            # Gate is a 120 degree rotation
            if x_rot.to == pauli_gates.Y:
                return [
                    (pauli_gates.X, -1 if y_rot.flip else 1),
                    (pauli_gates.Z, -1 if x_rot.flip else 1),
                ]

            return [
                (pauli_gates.Z, 1 if y_rot.flip else -1),
                (pauli_gates.X, 1 if z_rot.flip else -1),
            ]
        # coverage: ignore
        assert (
            False
        ), 'Impossible condition where this gate only rotates one Pauli to a different Pauli.'

    def equivalent_gate_before(self, after: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
        return self.merged_with(after).merged_with(self**-1)

    def __repr__(self) -> str:
        x = self.transform(pauli_gates.X)
        y = self.transform(pauli_gates.Y)
        z = self.transform(pauli_gates.Z)
        x_sign = '-' if x.flip else '+'
        y_sign = '-' if y.flip else '+'
        z_sign = '-' if z.flip else '+'
        return (
            f'cirq.SingleQubitCliffordGate(X:{x_sign}{x.to!s}, '
            f'Y:{y_sign}{y.to!s}, Z:{z_sign}{z.to!s})'
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        well_known_map = {
            SingleQubitCliffordGate.I: 'I',
            SingleQubitCliffordGate.H: 'H',
            SingleQubitCliffordGate.X: 'X',
            SingleQubitCliffordGate.Y: 'Y',
            SingleQubitCliffordGate.Z: 'Z',
            SingleQubitCliffordGate.X_sqrt: 'X',
            SingleQubitCliffordGate.Y_sqrt: 'Y',
            SingleQubitCliffordGate.Z_sqrt: 'Z',
            SingleQubitCliffordGate.X_nsqrt: 'X',
            SingleQubitCliffordGate.Y_nsqrt: 'Y',
            SingleQubitCliffordGate.Z_nsqrt: 'Z',
        }
        if self in well_known_map:
            symbol = well_known_map[self]
        else:
            rotations = self.decompose_rotation()
            symbol = '-'.join(str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2) for r, qt in rotations)
            symbol = f'({symbol})'
        return protocols.CircuitDiagramInfo(
            wire_symbols=(symbol,),
            exponent={
                SingleQubitCliffordGate.X_sqrt: 0.5,
                SingleQubitCliffordGate.Y_sqrt: 0.5,
                SingleQubitCliffordGate.Z_sqrt: 0.5,
                SingleQubitCliffordGate.X_nsqrt: -0.5,
                SingleQubitCliffordGate.Y_nsqrt: -0.5,
                SingleQubitCliffordGate.Z_nsqrt: -0.5,
            }.get(self, 1),
        )

    def _value_equality_values_(self):
        return self._clifford_tableau.matrix().tobytes() + self._clifford_tableau.rs.tobytes()

    def _value_equality_values_cls_(self):
        """To make it with compatible to compare with clifford gate."""
        return CliffordGate
