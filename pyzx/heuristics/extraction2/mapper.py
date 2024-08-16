from pyzx.circuit import Circuit
from pyzx.routing.architecture import Architecture
from pyzx.circuit.qasmparser import QASMParser

from mqt.qmap import HybridSynthesisMapper

from qiskit import QuantumCircuit

import numpy as np
from fractions import Fraction
from typing import Tuple


def gate_mapper(mapper:HybridSynthesisMapper, circuit: Circuit | list[Circuit]) -> Tuple[Architecture, int]:
    """Map the gates in the given circuit to the architecture and return the new architecture and the index of the mapped circuit.
    If multiple circuits are given, the mapper will choose the optimal one to map and return the index of that circuit."""

    if not isinstance(circuit, list):
        circuit = [circuit]

    qiskit_circuits = []

    for circ in circuit:
        qiskit_circuits.append(QuantumCircuit().from_qasm_str(circ.to_qasm()))

    index = mapper.evaluate_synthesis_steps(qiskit_circuits, also_map=False)

    # append a circuit to the mapper by mapping it to the architecture and adding it to the circuit
    mapper.append_with_mapping(qiskit_circuits[index])
    
    adjacency_matrix = np.array(mapper.get_circuit_adjacency_matrix())

    return Architecture("new_coupling", coupling_matrix=adjacency_matrix, qubit_map=list(range(qiskit_circuits[index].num_qubits))), index

def get_circuit_from_mapper(mapper:HybridSynthesisMapper, get_exact_phases:bool=False) -> Tuple[Circuit, Architecture]:
    """Get the circuit from the mapper and return the circuit and the new architecture.
    If get_exact_phases is True, the exact phases of the gates will be returned, otherwise the phases will be rounded to the nearest quarter of pi."""
    
    #TODO: The mapper should return the mapped circuit ``.get_mapped_qc()`` but move operations are not yet supported
    synthesized_circuit_qasm = mapper.get_synthesized_qc()
    circuit = QASMParser().parse(synthesized_circuit_qasm)
    for gate in circuit.gates:
        if hasattr(gate, "phase"):
            exact_phase = gate.phase * np.pi

            if not get_exact_phases:
                rounded_phase = Fraction().from_float(round(exact_phase / 0.25) * 0.25)
                gate.phase = rounded_phase
            else:
                gate.phase = Fraction().from_float(exact_phase)
    adjacency_matrix = np.array(mapper.get_circuit_adjacency_matrix())
    
    new_arch = Architecture("new_coupling", coupling_matrix=adjacency_matrix)

    return circuit, new_arch