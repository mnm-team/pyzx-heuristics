from fractions import Fraction
from pathlib import Path
from typing import Tuple
import numpy as np
from qiskit import QuantumCircuit

from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping

from pyzx.circuit import Circuit
from pyzx.circuit.qasmparser import QASMParser
from pyzx.routing.architecture import Architecture


def create_mapper(architecture_name: str = "rubidium.json") -> HybridSynthesisMapper:
    """Create a mapper object with the given mapper_architecture file."""

    path = Path(__file__).parent.parent.resolve()   
    # create a neutral atom hybrid architecture
    architecture = NeutralAtomHybridArchitecture(str(path)+"/mapper_files/"+architecture_name)

    # set mapper parameters (skip to use default values)
    params = HybridMapperParameters()
    # mapper should use SWAP gates or shuttling operations
    params.gate_weight = 0
    params.shuttling_weight = 1
    # look-ahead weights
    params.lookahead_weight_moves = 0.1
    params.lookahead_weight_swaps = 0.1
    # The initial mapping between atoms and hardware
    params.initial_mapping = InitialCoordinateMapping.trivial
    # If mapper should print debug information
    params.verbose = True

    # create mapper
    synthesis_mapper = HybridSynthesisMapper(arch=architecture, params=params)
    # alternatively
    # synthesis_mapper = HybridSynthesisMapper(arch=architecture)
    # synthesis_mapper.set_parameters(params)

    return synthesis_mapper


def init_mapper(mapper:HybridSynthesisMapper, num_qubits:int, initial_mapping:InitialCircuitMapping = InitialCircuitMapping.identity) -> None:
    """Initialize the mapper with the given number of qubits and initial mapping."""
    # set the initial circuit mapping and the size of the mapping(number of qubits)
    mapper.init_mapping(n_qubits=num_qubits, initial_mapping=initial_mapping)


def gate_mapper(mapper:HybridSynthesisMapper, circuit: Circuit | list[Circuit]) -> Tuple[Architecture, int]:
    """Map the gates in the given circuit to the architecture and return the new architecture and the index of the mapped circuit.
    If multiple circuits are given, the mapper will choose the optimal one to map and return the index of that circuit."""

    if not isinstance(circuit, list):
        circuit = [circuit]

    qiskit_circuits = []

    for circ in circuit:
        if isinstance(circ, QuantumCircuit):
            qiskit_circuits.append(circ)
        else:
            qiskit_circuits.append(QuantumCircuit().from_qasm_str(circ.to_qasm()))

    index = mapper.evaluate_synthesis_steps(qiskit_circuits, also_map=False)

    #TODO: Mapper does not map the gates correctly, it just takes the given gates while changing the adjacency matrix
    # append a circuit to the mapper by mapping it to the architecture and adding it to the circuit
    mapper.append_with_mapping(qiskit_circuits[index])

    #mapper.append_without_mapping(circuit[index])
    
    adjacency_matrix = np.array(mapper.get_circuit_adjacency_matrix())

    # m = np.zeros((num_qubits, num_qubits), dtype=int)
    
    # for i in range(num_qubits):
    #     if i > 0:
    #         m[i, i-1] = 1
    #     if i < num_qubits - 1:
    #         m[i, i+1] = 1
    
    return Architecture("new_coupling", coupling_matrix=adjacency_matrix, qubit_map=list(range(qiskit_circuits[index].num_qubits))), index

def get_circuit_from_mapper(mapper:HybridSynthesisMapper, get_exact_phases:bool=False) -> Tuple[Circuit, Architecture]:
    """Get the circuit from the mapper and return the circuit and the new architecture."""
    
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


