# Copyright 2022 The Cirq Developers
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

"""Device object representing Google devices with a grid qubit layout."""

from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
import re
import warnings

import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops


SYC_GATE_FAMILY = cirq.GateFamily(ops.SYC)
SQRT_ISWAP_GATE_FAMILY = cirq.GateFamily(cirq.SQRT_ISWAP)
SQRT_ISWAP_INV_GATE_FAMILY = cirq.GateFamily(cirq.SQRT_ISWAP_INV)
CZ_GATE_FAMILY = cirq.GateFamily(cirq.CZ)
PHASED_XZ_GATE_FAMILY = cirq.GateFamily(cirq.PhasedXZGate)
VIRTUAL_ZPOW_GATE_FAMILY = cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[ops.PhysicalZTag()])
PHYSICAL_ZPOW_GATE_FAMILY = cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[ops.PhysicalZTag()])
COUPLER_PULSE_GATE_FAMILY = cirq.GateFamily(experimental_ops.CouplerPulse)
MEASUREMENT_GATE_FAMILY = cirq.GateFamily(cirq.MeasurementGate)
WAIT_GATE_FAMILY = cirq.GateFamily(cirq.WaitGate)


def _validate_device_specification(proto: v2.device_pb2.DeviceSpecification) -> None:
    """Raises a ValueError if the `DeviceSpecification` proto is invalid."""

    qubit_set = set()
    for q_name in proto.valid_qubits:
        # Qubit names must be unique.
        if q_name in qubit_set:
            raise ValueError(
                f"Invalid DeviceSpecification: valid_qubits contains duplicate qubit '{q_name}'."
            )
        # Qubit names must be in the form <int>_<int> to be parsed as cirq.GridQubits.
        if re.match(r'^[0-9]+\_[0-9]+$', q_name) is None:
            raise ValueError(
                f"Invalid DeviceSpecification: valid_qubits contains the qubit '{q_name}' which is"
                " not in the GridQubit form '<int>_<int>."
            )
        qubit_set.add(q_name)

    for target_set in proto.valid_targets:

        # Check for unknown qubits in targets.
        for target in target_set.targets:
            for target_id in target.ids:
                if target_id not in proto.valid_qubits:
                    raise ValueError(
                        f"Invalid DeviceSpecification: valid_targets contain qubit '{target_id}'"
                        " which is not in valid_qubits."
                    )

        # Symmetric targets should not have repeated qubits.
        if target_set.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC:
            for target in target_set.targets:
                if len(target.ids) > len(set(target.ids)):
                    raise ValueError(
                        f"Invalid DeviceSpecification: the target set '{target_set.name}' is"
                        " SYMMETRIC but has a target which contains repeated qubits:"
                        f" {target.ids}."
                    )

        # Asymmetric target set type is not expected.
        # While this is allowed by the proto, it has never been set, so it's safe to raise an
        # exception if this is set unexpectedly.
        if target_set.target_ordering == v2.device_pb2.TargetSet.ASYMMETRIC:
            raise ValueError("Invalid DeviceSpecification: target_ordering cannot be ASYMMETRIC.")


def _build_gateset_and_gate_durations(
    proto: v2.device_pb2.DeviceSpecification,
) -> Tuple[cirq.Gateset, Dict[cirq.GateFamily, cirq.Duration]]:
    """Extracts gate set and gate duration information from the given DeviceSpecification proto."""

    gates_list: List[Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]] = []
    gate_durations: Dict[cirq.GateFamily, cirq.Duration] = {}

    # TODO(#5050) Describe how to add/remove gates.

    for gate_spec in proto.valid_gates:
        gate_name = gate_spec.WhichOneof('gate')
        cirq_gates: List[Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]] = []

        if gate_name == 'syc':
            cirq_gates = [ops.FSimGateFamily(gates_to_accept=[ops.SYC])]
        elif gate_name == 'sqrt_iswap':
            cirq_gates = [ops.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP])]
        elif gate_name == 'sqrt_iswap_inv':
            cirq_gates = [ops.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV])]
        elif gate_name == 'cz':
            cirq_gates = [ops.FSimGateFamily(gates_to_accept=[cirq.CZ])]
        elif gate_name == 'phased_xz':
            cirq_gates = [cirq.PhasedXZGate, cirq.XPowGate, cirq.YPowGate, cirq.PhasedXPowGate]
        elif gate_name == 'virtual_zpow':
            cirq_gates = [cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[ops.PhysicalZTag()])]
        elif gate_name == 'physical_zpow':
            cirq_gates = [cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[ops.PhysicalZTag()])]
        elif gate_name == 'coupler_pulse':
            cirq_gates = [experimental_ops.CouplerPulse]
        elif gate_name == 'meas':
            cirq_gates = [cirq.MeasurementGate]
        elif gate_name == 'wait':
            cirq_gates = [cirq.WaitGate]
        else:
            # coverage: ignore
            warnings.warn(
                f"The DeviceSpecification contains the gate '{gate_name}' which is not recognized"
                " by Cirq and will be ignored. This may be due to an out-of-date Cirq version.",
                UserWarning,
            )
            continue

        gates_list.extend(cirq_gates)

        # TODO(#5050) Allow different gate representations of the same gate to be looked up in
        # gate_durations.
        for g in cirq_gates:
            if not isinstance(g, cirq.GateFamily):
                g = cirq.GateFamily(g)
            gate_durations[g] = cirq.Duration(picos=gate_spec.gate_duration_picos)

    # TODO(#4833) Add identity gate support
    # TODO(#5050) Add GlobalPhaseGate support

    return cirq.Gateset(*gates_list), gate_durations


def _build_compilation_target_gatesets(
    gateset: cirq.Gateset,
) -> Sequence[cirq.CompilationTargetGateset]:
    """Detects compilation target gatesets based on what gates are inside the gateset.

    If a device contains gates which yield multiple compilation target gatesets, the user can only
    choose one target gateset to compile to. For example, a device may contain both SYC and
    SQRT_ISWAP gates which yield two separate target gatesets, but a circuit can only be compiled to
    either SYC or SQRT_ISWAP for its two-qubit gates, not both.

    TODO(#5050) when cirq-google CompilationTargetGateset subclasses are implemented, mention that
    gates which are part of the gateset but not the compilation target gateset are untouched when
    compiled.
    """

    # TODO(#5050) Subclass core CompilationTargetGatesets in cirq-google.

    target_gatesets: List[cirq.CompilationTargetGateset] = []
    if cirq.CZ in gateset:
        target_gatesets.append(cirq.CZTargetGateset())
    if ops.SYC in gateset:
        target_gatesets.append(transformers.SycamoreTargetGateset())
    if cirq.SQRT_ISWAP in gateset:
        target_gatesets.append(
            cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=cirq.SQRT_ISWAP_INV in gateset)
        )

    return tuple(target_gatesets)


@cirq.value_equality
class GridDevice(cirq.Device):
    """Device object representing Google devices with a grid qubit layout.

    For end users, instances of this class are typically accessed via
    `Engine.get_processor('processor_name').get_device()`.

    This class is compliant with the core `cirq.Device` abstraction. In particular:
        * Device information is captured in the `metadata` property.
        * An instance of `GridDevice` can be used to validate circuits, moments, and operations.

    Example use cases:

        * Get an instance of a Google grid device.
        >>> device = cirq_google.get_engine().get_processor('processor_name').get_device()

        * Print the grid layout of the device.
        >>> print(device)

        * Determine whether a circuit can be run on the device.
        >>> device.validate_circuit(circuit)  # Raises a ValueError if the circuit is invalid.

        * Determine whether an operation can be run on the device.
        >>> device.validate_operation(operation)  # Raises a ValueError if the operation is invalid.

        * Get the `cirq.Gateset` containing valid gates for the device, and inspect the full list
          of valid gates.
        >>> gateset = device.metadata.gateset
        >>> print(gateset)

        * Determine whether a gate is available on the device.
        >>> gate in device.metadata.gateset

        * Get a collection of valid qubits on the device.
        >>> device.metadata.qubit_set

        * Get a collection of valid qubit pairs for two-qubit gates.
        >>> device.metadata.qubit_pairs

        * Get a collection of isolated qubits, i.e. qubits which are not part of any qubit pair.
        >>> device.metadata.isolated_qubits

        * Get a collection of approximate gate durations for every gate supported by the device.
        >>> device.metadata.gate_durations

        * Get a collection of valid CompilationTargetGatesets for the device, which can be used to
          transform a circuit to one which only contains gates from a native target gateset
          supported by the device.
        >>> device.metadata.compilation_target_gatesets

        * Assuming valid CompilationTargetGatesets exist for the device, select the first one and
          use it to transform a circuit to one which only contains gates from a native target
          gateset supported by the device.
        >>> cirq.optimize_for_target_gateset(
                circuit,
                gateset=device.metadata.compilation_target_gatesets[0]
            )

    A note about CompilationTargetGatesets:

    A circuit which contains `cirq.WaitGate`s will be dropped if it is transformed using
    CompilationTargetGatesets generated by GridDevice. To better control circuit timing, insert
    WaitGates after the circuit has been transformed.

    Notes for cirq_google internal implementation:

    For Google devices, the
    [DeviceSpecification proto](
        https://github.com/quantumlib/Cirq/blob/master/cirq-google/cirq_google/api/v2/device.proto
    )
    is the main specification for device information surfaced by the Quantum Computing Service.
    Thus, this class is should be instantiated using a `DeviceSpecification` proto via the
    `from_proto()` class method.
    """

    def __init__(self, metadata: cirq.GridDeviceMetadata):
        """Creates a GridDevice object.

        This constructor typically should not be used directly. Use `from_proto()` instead.
        """
        self._metadata = metadata

    @classmethod
    def from_proto(cls, proto: v2.device_pb2.DeviceSpecification) -> 'GridDevice':
        """Create a `GridDevice` from a `DeviceSpecification` proto.

        Args:
            proto: The `DeviceSpecification` proto describing a Google device.

        Raises:
            ValueError: If the given `DeviceSpecification` is invalid. It is invalid if:
                * A `DeviceSpecification.valid_qubits` string is not in the form `<int>_<int>`, thus
                  cannot be parsed as a `cirq.GridQubit`.
                * `DeviceSpecification.valid_targets` refer to qubits which are not in
                  `DeviceSpecification.valid_qubits`.
                * A target set in `DeviceSpecification.valid_targets` has type `SYMMETRIC` but
                  contains targets with repeated qubits, e.g. a qubit pair with a self loop.
        """

        _validate_device_specification(proto)

        # Create qubit set
        all_qubits = {v2.grid_qubit_from_proto_id(q) for q in proto.valid_qubits}

        # Create qubit pair set
        qubit_pairs = [
            (v2.grid_qubit_from_proto_id(target.ids[0]), v2.grid_qubit_from_proto_id(target.ids[1]))
            for ts in proto.valid_targets
            for target in ts.targets
            if len(target.ids) == 2 and ts.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC
        ]

        gateset, gate_durations = _build_gateset_and_gate_durations(proto)

        try:
            metadata = cirq.GridDeviceMetadata(
                qubit_pairs=qubit_pairs,
                gateset=gateset,
                gate_durations=gate_durations if len(gate_durations) > 0 else None,
                all_qubits=all_qubits,
                compilation_target_gatesets=_build_compilation_target_gatesets(gateset),
            )
        except ValueError as ve:  # coverage: ignore
            # Spec errors should have been caught in validation above.
            raise ValueError("DeviceSpecification is invalid.") from ve  # coverage: ignore

        return GridDevice(metadata)

    @property
    def metadata(self) -> cirq.GridDeviceMetadata:
        """Get metadata information for the device."""
        return self._metadata

    def validate_operation(self, operation: cirq.Operation) -> None:
        """Raises an exception if an operation is not valid.

        An operation is valid if
            * The operation is in the device gateset.
            * The operation targets a valid qubit
            * The operation targets a valid qubit pair, if it is a two-qubit operation.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """

        if operation not in self._metadata.gateset:
            raise ValueError(f'Operation {operation} contains a gate which is not supported.')

        for q in operation.qubits:
            if q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}.')

        if (
            len(operation.qubits) == 2
            and frozenset(operation.qubits) not in self._metadata.qubit_pairs
        ):
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')

    def __str__(self) -> str:
        diagram = cirq.TextDiagramDrawer()

        qubits = cast(Set[cirq.GridQubit], self._metadata.qubit_set)

        # Don't print out extras newlines if the row/col doesn't start at 0
        min_col = min(q.col for q in qubits)
        min_row = min(q.row for q in qubits)

        for q in qubits:
            info = cirq.circuit_diagram_info(q, default=None)
            qubit_name = info.wire_symbols[0] if info else str(q)
            diagram.write(q.col - min_col, q.row - min_row, qubit_name)

        # Find pairs that are connected by two-qubit gates.
        Pair = Tuple[cirq.GridQubit, cirq.GridQubit]
        pairs = sorted({cast(Pair, tuple(pair)) for pair in self._metadata.qubit_pairs})

        # Draw lines between connected pairs. Limit to horizontal/vertical
        # lines since that is all the diagram drawer can handle.
        for q1, q2 in pairs:
            if q1.row == q2.row or q1.col == q2.col:
                diagram.grid_line(
                    q1.col - min_col, q1.row - min_row, q2.col - min_col, q2.row - min_row
                )

        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Creates ASCII diagram for Jupyter, IPython, etc."""
        # There should never be a cycle, but just in case use the default repr.
        p.text(repr(self) if cycle else str(self))

    def __repr__(self) -> str:
        return f'cirq_google.GridDevice({repr(self._metadata)})'

    def _json_dict_(self):
        return {'metadata': self._metadata}

    @classmethod
    def _from_json_dict_(cls, metadata, **kwargs):
        return cls(metadata)

    def _value_equality_values_(self):
        return self._metadata


def _set_gate_in_gate_spec(
    gate_spec: v2.device_pb2.GateSpecification, gate_family: cirq.GateFamily
) -> None:
    if gate_family == SYC_GATE_FAMILY:
        gate_spec.syc.SetInParent()
    elif gate_family == SQRT_ISWAP_GATE_FAMILY:
        gate_spec.sqrt_iswap.SetInParent()
    elif gate_family == SQRT_ISWAP_INV_GATE_FAMILY:
        gate_spec.sqrt_iswap_inv.SetInParent()
    elif gate_family == CZ_GATE_FAMILY:
        gate_spec.cz.SetInParent()
    elif gate_family == PHASED_XZ_GATE_FAMILY:
        gate_spec.phased_xz.SetInParent()
    elif gate_family == VIRTUAL_ZPOW_GATE_FAMILY:
        gate_spec.virtual_zpow.SetInParent()
    elif gate_family == PHYSICAL_ZPOW_GATE_FAMILY:
        gate_spec.physical_zpow.SetInParent()
    elif gate_family == COUPLER_PULSE_GATE_FAMILY:
        gate_spec.coupler_pulse.SetInParent()
    elif gate_family == MEASUREMENT_GATE_FAMILY:
        gate_spec.meas.SetInParent()
    elif gate_family == WAIT_GATE_FAMILY:
        gate_spec.wait.SetInParent()
    else:
        raise ValueError(f'Unrecognized gate {gate_family}.')


def create_device_specification_proto(
    *,
    qubits: Collection[cirq.GridQubit],
    pairs: Collection[Tuple[cirq.GridQubit, cirq.GridQubit]],
    gateset: cirq.Gateset,
    gate_durations: Optional[Dict['cirq.GateFamily', 'cirq.Duration']] = None,
    out: Optional[v2.device_pb2.DeviceSpecification] = None,
) -> v2.device_pb2.DeviceSpecification:
    """Serializes the given device information into a DeviceSpecification proto.

    Args:
        qubits: Collection of qubits available on the device.
        pairs: Collection of bidirectional qubit couplings available on the device.
        gateset: The gate set supported by the device.
        gate_durations: Optional mapping from gates supported by the device to their timing
            estimates. Not every gate is required to have an associated duration.
        out: If set, device information will be serialized into this DeviceSpecification.

    Raises:
        ValueError: If a qubit in `pairs` is not part of `qubits`.
        ValueError: If a pair contains two identical qubits.
        ValueError: If `gate_durations` contains keys which are not in `gateset`.
        ValueError: If `gateset` contains a gate which is not recognized by DeviceSpecification.
    """

    if gate_durations is not None:
        extra_gate_families = (gate_durations.keys() | gateset.gates) - gateset.gates
        if extra_gate_families:
            raise ValueError(
                'Gate durations contain keys which are not part of the gateset:'
                f' {extra_gate_families}'
            )

    if out is None:
        out = v2.device_pb2.DeviceSpecification()

    # If fields are already filled (i.e. as part of the old DeviceSpecification format), leave them
    # as is. Fields populated in the new format do not conflict with how they were populated in the
    # old format.
    # TODO(#5050) remove empty checks below once deprecated fields in DeviceSpecification are
    # removed.

    if len(out.valid_qubits) == 0:
        known_devices.populate_qubits_in_device_proto(qubits, out)

    if len(out.valid_targets) == 0:
        known_devices.populate_qubit_pairs_in_device_proto(pairs, out)

    gate_specs = []
    for gate_family in gateset.gates:
        gate_spec = v2.device_pb2.GateSpecification()
        _set_gate_in_gate_spec(gate_spec, gate_family)
        if gate_durations is not None and gate_family in gate_durations:
            gate_spec.gate_duration_picos = int(gate_durations[gate_family].total_picos())
        gate_specs.append(gate_spec)

    # Sort by gate name to keep valid_gates stable.
    out.valid_gates.extend(sorted(gate_specs, key=lambda s: s.WhichOneof('gate')))

    _validate_device_specification(out)

    return out
