import math

from typing import List, Optional, Tuple

from pyzx.circuit.gates import CNOT, Gate
from pyzx.routing.architecture import Architecture



def move_control(gate:Gate, control:int, new_control:int):
    """Given a gate and a control qubit, returns a new gate with the control qubit replaced with the new control qubit"""
    gate_copy = gate.copy()
    if hasattr(gate, "control"):
        gate_copy.control = new_control
    elif hasattr(gate, "controls"):
        gate_copy.controls = [new_control if c == control else c for c in gate.controls]
    return gate_copy


def move_target(gate:Gate, new_target:int):
    """Given a gate and a target qubit, returns a new gate with the target qubit replaced with the new target qubit"""
    gate_copy = gate.copy()
    gate_copy.target = new_target
    return gate_copy


def move_control_to_next(architecture:Architecture, path:List[int], gate:Gate) -> List[Gate]:

    rerouting_result = []

    gate_copy = move_control(gate, path[0], path[-2])
    
    for i in range(len(path)-2):
        rerouting_result.append(CNOT(path[i+1], path[i]))
        rerouting_result.append(CNOT(path[i], path[i+1]))
    rerouting_result.extend(build_connection_from_architecture(architecture, gate_copy))
    for i in range(len(path)-2, 0, -1):
        rerouting_result.append(CNOT(path[i-1], path[i]))
        rerouting_result.append(CNOT(path[i], path[i-1]))

    return rerouting_result


def move_target_to_control(architecture:Architecture, path:List[int], gate:Gate) -> List[Gate]:

    rerouting_result = []

    gate_copy = move_target(gate, path[1])
    
    for i in range(len(path)-1, 1, -1):
        rerouting_result.append(CNOT(path[i-1], path[i]))
        rerouting_result.append(CNOT(path[i], path[i-1]))
    rerouting_result.extend(build_connection_from_architecture(architecture, gate_copy))
    for i in range(1, len(path)-1):
        rerouting_result.append(CNOT(path[i], path[i+1]))
        rerouting_result.append(CNOT(path[i+1], path[i]))

    return rerouting_result


def build_connection_from_architecture(architecture: Architecture, gate:Gate) -> List[CNOT]:
    """Given a gate and an architecture, returns a list of CNOTs that connect the qubits of the gate
    according to the architecture"""

    if not hasattr(gate, "target"):
        raise ValueError("Gate does not have a target qubit")
    
    target_qubit = gate.target

    if not hasattr(gate, "control"):
        if not hasattr(gate, "controls"):
            raise ValueError("Gate does not have controls")
        else:
            control_qubits = gate.controls
    else:
        control_qubits = [gate.control]

    if architecture and not architecture.is_subgraph_connected([target_qubit]+control_qubits):

        if len(control_qubits) > 1:
            path_dict = {qubit:None for qubit in [target_qubit]+control_qubits}

            for control_qubit_index in range(len(control_qubits)):
                current_control_qubit = control_qubits[control_qubit_index]

                path_to_target = architecture.shortest_path(current_control_qubit, target_qubit)
                if len(path_to_target) > 2:
                    if not path_dict[target_qubit] or len(path_dict[target_qubit]) > len(path_to_target):
                        path_dict[target_qubit] = path_to_target

                for next_control_qubit_index in range(control_qubit_index+1, len(control_qubits)):
                    next_control_qubit = control_qubits[next_control_qubit_index]
                    path_to_next_control = architecture.shortest_path(current_control_qubit, next_control_qubit)

                    if len(path_to_next_control) > 2:
                        if not path_dict[current_control_qubit] or len(path_dict[current_control_qubit]) > len(path_to_next_control):
                            path_dict[current_control_qubit] = path_to_next_control
            
            start, min_path = min(path_dict.items(), key=lambda x: len(x[1]) if x[1] else math.inf)

            if not min_path:
                return [gate]

            if start in control_qubits:
                return move_control_to_next(architecture, min_path, gate)
            elif start == target_qubit:
                return move_target_to_control(architecture, min_path, gate)

        elif len(control_qubits) == 1:
            shortest_path = architecture.shortest_path(control_qubits[0], target_qubit)

            if not shortest_path:
                raise ValueError("Architecture is not connected")
            
            if hasattr(gate, "phase"):
                return move_control_to_next(architecture, shortest_path, gate)
            
            gate_copy = move_control(gate, control_qubits[0], shortest_path[-2])

            rerouting_result = []
            
            #TODO: check for shuttling
            for i in range(len(shortest_path)-2):
                rerouting_result.append(CNOT(shortest_path[i], shortest_path[i+1]))
            rerouting_result.append(gate_copy)
            for i in range(len(shortest_path)-2, 0, -1):
                rerouting_result.append(CNOT(shortest_path[i-1], shortest_path[i]))

            for i in range(1, len(shortest_path)-2):
                rerouting_result.append(CNOT(shortest_path[i], shortest_path[i+1]))
            rerouting_result.append(gate_copy)
            for i in range(len(shortest_path)-2, 1, -1):
                rerouting_result.append(CNOT(shortest_path[i-1], shortest_path[i]))

        return rerouting_result
    else:
        return [gate]

