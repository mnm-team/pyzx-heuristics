import itertools
import math

from typing import Callable, Dict, List, Optional, Set, Tuple
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit


from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, Gate, ZPhase
from pyzx.extract import column_optimal_swap, connectivity_from_biadj, filter_duplicate_cnots, xor_rows
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.rerouting import ReroutedGate, build_connection_from_architecture
from pyzx.linalg import Z2, CNOTMaker, Mat2
from pyzx.routing.architecture import Architecture
from pyzx.routing.cnot_mapper import ElimMode, gauss
from pyzx.heuristics.tools import insert_identity
from pyzx.simplify import apply_rule, pivot, lcomp_with_boundaries
from pyzx.utils import EdgeType, FractionLike, VertexType, phase_is_true_clifford, toggle_edge



class MCP(Gate):
    name = 'MCP'
    qasm_name = 'mcp'
    print_phase = True
    def __init__(self, controls: List[int], target: int, phase: FractionLike) -> None:
        self.target = target
        self.controls = controls
        self.phase = phase

    def to_basic_gates(self):
        all_qubits = self.controls+[self.target]
        gates = []
        odd_phase = self.phase/(2**len(self.controls))
        even_phase = -odd_phase
        for degree in range(2,len(all_qubits)+1):
            combinations = list(itertools.combinations(all_qubits, degree))
            for combination in combinations:
                for idx in range(0,len(combination)-1):
                    gates.append(CNOT(combination[idx],combination[idx+1]))
                gates.append(ZPhase(combination[-1], odd_phase if degree % 2 == 1 else even_phase))
                for idx in range(len(combination)-2,-1,-1):
                    gates.append(CNOT(combination[idx],combination[idx+1]))
        for qubit in all_qubits:
            gates.append(ZPhase(qubit, odd_phase))
        return gates                

    def to_graph(self, g, q_mapper, c_mapper):
        for gate in self.to_basic_gates():
            gate.to_graph(g, q_mapper, c_mapper)
    
    def to_qasm(self) -> str:
        phase = "({}*pi)".format(float(self.phase))
        if len(self.controls) == 1:
            return "cp"+phase+" q["+str(self.controls[0])+"], q["+str(self.target)+"];"
        else:
            name = "mcp"+str(len(self.controls)+1)
            control_string = "".join(["q["+str(control)+"], " for control in self.controls])
            return name+phase+" "+control_string+" q["+str(self.target)+"];"
        

class SHUTTLE(CZ):
    name = 'Shuttle'
    qasm_name = 'shuttle'
    qc_name = 'undefined'
    quipper_name = 'undefined'

    def to_basic_gates(self):
        c1 = CNOT(self.control, self.target)
        c2 = CNOT(self.target, self.control)
        return [c1,c2,c1]

    def to_graph(self, g, q_mapper, c_mapper):
        for gate in self.to_basic_gates():
            gate.to_graph(g, q_mapper, c_mapper)



def add_gate_to_circuit(circuit: Circuit|QuantumCircuit, gate:Gate):
    if isinstance(circuit, Circuit):
        circuit.add_gate(gate)
    else:
        if hasattr(gate, "phase") and isinstance(gate.phase, Fraction):
            gate.phase = gate.phase.numerator/gate.phase.denominator
        if hasattr(gate, "theta") and isinstance(gate.theta, Fraction):
            gate.theta = gate.theta.numerator/gate.theta.denominator
        if hasattr(gate, "phi") and isinstance(gate.phi, Fraction):
            gate.phi = gate.phi.numerator/gate.phi.denominator
        if hasattr(gate, "rho") and isinstance(gate.rho, Fraction):
            gate.rho = gate.rho.numerator/gate.rho.denominator
        if hasattr(gate, "gamma") and isinstance(gate.gamma, Fraction):
            gate.gamma = gate.gamma.numerator/gate.gamma.denominator

        match gate.name:
            case "MCP":
                circuit.mcp(gate.phase, gate.controls, gate.target)
            case "CNOT":
                circuit.cx(gate.control, gate.target)
            case "CZ":
                circuit.cz(gate.control, gate.target)
            case "CY":
                circuit.cy(gate.control, gate.target)
            case "ZPhase":
                circuit.rz(gate.phase, gate.target)
            case "YPhase":
                circuit.ry(gate.phase, gate.target)
            case "XPhase":
                circuit.rx(gate.phase, gate.target)
            case "CPhase":
                circuit.cp(gate.phase, gate.phase, gate.target)
            case "SX":
                circuit.sx(gate.target)
            case "CSX":
                circuit.csx(gate.control, gate.target)
            case "CRX":
                circuit.crx(gate.phase, gate.control, gate.target)
            case "CRY":
                circuit.cry(gate.phase, gate.control, gate.target)
            case "CRZ":
                circuit.crz(gate.phase, gate.control, gate.target)
            case "CCZ":
                circuit.ccz(gate.ctrl1, gate.ctrl2, gate.target)
            case "Tof":
                circuit.ccx(gate.ctrl1, gate.ctrl2, gate.target)
            case "CSWAP":
                circuit.cswap(gate.ctrl1, gate.ctrl2, gate.target)
            case "RXX":
                circuit.rxx(gate.phase, gate.control, gate.target)
            case "RZZ":
                circuit.rzz(gate.phase, gate.control, gate.target)
            case "XCX":
                circuit.h(gate.control)
                circuit.cnot(gate.control, gate.target)
                circuit.h(gate.control)
            case "SWAP":
                circuit.swap(gate.control, gate.target)
            case "Z":
                circuit.z(gate.target)
            case "Y":
                circuit.y(gate.target)
            case "NOT":
                circuit.x(gate.target)
            case "S":
                circuit.s(gate.target)
            case "T":
                circuit.t(gate.target)
            case "HAD":
                circuit.h(gate.target)
            case "CHAD":
                circuit.ch(gate.control, gate.target)
            case "ParityPhase":
                raise NotImplementedError("ParityPhase gate not implemented for qiskit")
            case "FSim":
                raise NotImplementedError("FSim gate not implemented for qiskit")
            case "InitAncilla":
                raise NotImplementedError("InitAncilla gate not implemented for qiskit")
            case "PostSelect":
                raise NotImplementedError("PostSelect gate not implemented for qiskit")
            case "DiscardBit":
                raise NotImplementedError("DiscardBit gate not implemented for qiskit")
            case "U2":
                circuit.u(np.pi/2, gate.theta, gate.phi, gate.target)
            case "U3":
                circuit.u(gate.theta, gate.phi, gate.rho, gate.target)
            case "CU3":
                circuit.cu(gate.theta, gate.phi, gate.rho, 0, gate.control, gate.target)
            case "CU":
                circuit.cu(gate.theta, gate.phi, gate.rho, gate.gamma, gate.control, gate.target)
            case "Measure":
                circuit.measure(gate.target)



def bi_adj(g: BaseGraph[VT,ET], vs:List[VT], ws:List[VT]) -> Mat2:
    """Construct a biadjacency matrix between the supplied list of vertices
    ``vs`` and ``ws``."""
    vs_copy = vs.copy()
    for _ in range(len(ws)-len(vs)):
        vs_copy.append(-1)

    return Mat2([[0 if (w == -1 or v == -1) else int(g.connected(v,w)) for v in vs_copy] for w in ws])

def bi_adj2(g: BaseGraph[VT,ET], vs:List[VT], ws:List[VT]) -> Mat2:
    """Construct a biadjacency matrix between the supplied list of vertices
    ``vs`` and ``ws``."""
    ws_copy = ws.copy()
    for _ in range(len(vs)-len(ws)):
        ws_copy.append(-1)

    return Mat2([[0 if (w == -1 or v == -1) else int(g.connected(v,w)) for v in vs] for w in ws_copy])


def find_minimal_sums_with_architecture(m: Mat2, architecture:Architecture, result_amount_limit:int=5, reversed_search=False) -> Optional[Tuple[int, ...]]:
    """Returns a list of rows in m that can be added together to reduce one of the rows so that
    it only contains a single 1. Used in :func:`greedy_reduction`"""

    results = []

    r = m.rows()
    d = m.data
    if any(sum(r) == 1 for r in d):
        return tuple()
    combs:  Dict[Tuple[int, ...], List[Z2]] = {(i,): d[i] for i in range(r)}
    combs2: Dict[Tuple[int, ...], List[Z2]] = {}
    iterations = 0
    while True:
        combs2 = {}
        for index, l in combs.items():
            max_index: int = max(index)
            rr: range = range(max_index + 1, r) if not reversed_search else range(r - 1, max_index, -1)
            for k in rr:
                if architecture and not architecture.is_subgraph_connected([*index]+[k]): continue
                # Unrolled xor_rows(combs[index],d[k])
                row: List[Z2] = [0 if v1 == v2 else 1 for v1, v2 in zip(combs[index], d[k])]
                # row = xor_rows(combs[index],d[k])
                if sum(row) == 1:
                    results.append((*index, k))
                    # return (*index, k)
                combs2[(*index, k)] = row
                iterations += 1
            if iterations > 100000:
                return results
        if not combs2:
            return results
            # raise ValueError("Irreducible input has been given")
        if len(results) >= result_amount_limit:
            return results
        combs = combs2

def reorder_frontier(frontier: Dict[int, VT], architecture: Architecture) -> Dict[int, VT]:
    """Reorders the frontier to match the architecture"""
    new_frontier = {}
    for qubit, vertex in frontier.items():
        new_frontier[architecture.qubit_map[qubit]] = vertex
    new_frontier = dict(sorted(new_frontier.items()))
    return new_frontier


def greedy_reduction_with_architecture(m: Mat2, architecture: Architecture) -> Optional[List[List[ReroutedGate]]]:
    """Returns a list of tuples (r1,r2) that specify which row should be added to which other row
    in order to reduce one row of m to only contain a single 1. 
    Used in :func:`extract_circuit` and :func:`lookahead_extract_base`"""
    indices_list_greedy_row_add = find_minimal_sums_with_architecture(m, architecture=None)
    if indices_list_greedy_row_add == []: return None

    row_add_results: List[List[ReroutedGate]] = []
    for indices_greedy_row_add in indices_list_greedy_row_add:
        indices = list(indices_greedy_row_add)
        rows = {i:m.data[i] for i in indices}
        weights: Dict[int,int] = {i: sum(r) for i,r in rows.items()}
        result: List[ReroutedGate] = []
        while len(indices)>1:
            best = (-1,-1)
            reduction = -10000
            for i in indices:
                for j in indices:
                    if j <= i: continue
                    w = sum(xor_rows(rows[i],rows[j]))
                    cnot_cost = 1
                    if j-i > 1:
                        cnot_cost = 4*(j-i-1)
                    if weights[i] - w - cnot_cost > reduction:
                        best = (j,i) # "Add row j to i"
                        reduction = weights[i] - w - cnot_cost
                    if weights[j] - w - cnot_cost > reduction:
                        best = (i,j)
                        reduction = weights[j] - w - cnot_cost
            rerouted_gates = build_connection_from_architecture(architecture, CNOT(best[0], best[1]))
            result.append(ReroutedGate(CNOT(best[0], best[1]), rerouted_gates))
            control, target = best
            rows[target] = xor_rows(rows[control],rows[target])
            weights[target] = weights[target] - reduction
            indices.remove(control)

        row_add_results.append(result)

    return row_add_results


def greedy_reduction_with_architecture2(m: Mat2, architecture: Architecture) -> Optional[List[List[CNOT]]]:
    """Returns a list of tuples (r1,r2) that specify which row should be added to which other row
    in order to reduce one row of m to only contain a single 1. 
    Used in :func:`extract_circuit` and :func:`lookahead_extract_base`"""
    indices_list_greedy_row_add = find_minimal_sums_with_architecture(m, architecture=architecture)
    if indices_list_greedy_row_add == []: return None

    row_add_results = []
    for indices_greedy_row_add in indices_list_greedy_row_add:
        indices = list(indices_greedy_row_add)
        rows = {i:m.data[i] for i in indices}
        weights: Dict[int,int] = {i: sum(r) for i,r in rows.items()}
        result = []
        while len(indices)>1:
            best = (-1,-1)
            reduction = -10000
            for i in indices:
                for j in indices:
                    if j <= i: continue
                    if architecture and not architecture.is_subgraph_connected([i,j]): continue
                    w = sum(xor_rows(rows[i],rows[j]))
                    if weights[i] - w > reduction:
                        if not architecture or architecture.is_subgraph_connected(set(indices)-set([j])):
                            best = (j,i) # "Add row j to i"
                            reduction = weights[i] - w
                    if weights[j] - w > reduction:
                        if not architecture or architecture.is_subgraph_connected(set(indices)-set([i])):
                            best = (i,j)
                            reduction = weights[j] - w
            result.append(best)
            control, target = best
            rows[target] = xor_rows(rows[control],rows[target])
            weights[target] = weights[target] - reduction
            indices.remove(control)

        row_add_results.append(result)

    cnots_list = []
    for row_add_result in row_add_results:
        # cnots = [[CNOT(control, target) for control, target in row_add_result] for row_add_result in row_add_result_list]
        cnots = [CNOT(control, target) for control, target in row_add_result]
        cnots_list.append(cnots)
    return cnots_list



def get_best_cnot_configuration(rerouted_gate_list: List[List[ReroutedGate]], architecture: Architecture) -> List[ReroutedGate]:
    """Given a list of lists of CNOTs, returns the list with the fewest CNOTs"""
    if not rerouted_gate_list:
        return []
    
    best_result = (None, math.inf)
    
    for rerouted_gates in rerouted_gate_list:
        # if any(len(rerouted_cnots) > 1 for rerouted_cnots in cnots):
        #     shuttle_gate, new_cnots, new_cost = check_shuttling(cnots, architecture)
        #     if new_cost < best_result[2]:
        #         best_result = (new_cnots, shuttle_gate, new_cost)
        # else:
        if (sum_cnots := sum(len(rerouted_gate.gate_path) for rerouted_gate in rerouted_gates)) < best_result[1]:
            best_result = (rerouted_gates, sum_cnots)

    return best_result[0]


def check_shuttling(rerouted_gate_list: List[ReroutedGate], architecture: Architecture) -> Tuple[Optional[SHUTTLE], List[List[CNOT]], int]:
    """Given a list of rerouted CNOTs, check if a shuttling operation is better than the current configuration"""
    #TODO: This needs to be implemented
    # cnot_list_dict: list[dict] = []
    # for cnots in cnots_list:
    #     cnot_dict = {"original": None, "rerouted": None}
    #     if len(cnots) == 1:
    #         cnot_dict["original"] = cnots[0]
    #     else:
    #         cnot_dict["rerouted"] = cnots
    #         min_control = min(cnots, key=lambda x: x.control).control
    #         min_target = min(cnots, key=lambda x: x.target).target

    #         max_control = max(cnots, key=lambda x: x.control).control
    #         max_target = max(cnots, key=lambda x: x.target).target

    #         if min_control < min_target:
    #             cnot_dict["original"] = CNOT(min_control, max_target)
    #         else:
    #             cnot_dict["original"] = CNOT(max_control, min_target)
    #     cnot_list_dict.append(cnot_dict)

            

    max_rerouting_index = max(enumerate(rerouted_gate_list), key=lambda x: len(x[1].gate_path))[0]

    basic_cnots: List[CNOT] = [rerouted_gate.basic_gate for rerouted_gate in rerouted_gate_list]

    biggest_cnot = basic_cnots[max_rerouting_index]

    if abs(biggest_cnot.control - biggest_cnot.target) <= 1:
        return None, rerouted_gate_list, sum(len(rerouted_gate.gate_path) for rerouted_gate in rerouted_gate_list)

    if biggest_cnot.control < biggest_cnot.target:
        best_shuttle = SHUTTLE(biggest_cnot.control, target=biggest_cnot.target-1)
    else:
        best_shuttle = SHUTTLE(biggest_cnot.control, target=biggest_cnot.target+1)
   
    # shuttled_graph = architecture.graph.copy()
    # edges_v0 = shuttled_graph.incident_edges(best_shuttle.control)
    # edges_v1 = shuttled_graph.incident_edges(best_shuttle.target)

    # for v, w in edges_v0:
    #     if shuttled_graph.edge(v, w) in edges_v1:
    #         continue
    #     shuttled_graph.remove_edge(shuttled_graph.edge(v, w))
    #     if v == best_shuttle.control:
    #         shuttled_graph.add_edge(shuttled_graph.edge(best_shuttle.target, w))
    #     else:
    #         shuttled_graph.add_edge(shuttled_graph.edge(v, best_shuttle.target))

    # for v, w in edges_v1:
    #     if shuttled_graph.edge(v, w) in edges_v0:
    #         continue
    #     shuttled_graph.remove_edge(shuttled_graph.edge(v, w))
    #     if v == best_shuttle.target:
    #         shuttled_graph.add_edge(shuttled_graph.edge(best_shuttle.control, w))
    #     else:
    #         shuttled_graph.add_edge(shuttled_graph.edge(v, best_shuttle.control))

    # shuttled_architecture = Architecture(name="Shuttled Architecture", coupling_graph=shuttled_graph, qubit_map=list(range(shuttled_graph.num_vertices())))

    new_cnots_list = []
    for cnot in basic_cnots:
        cnot_copy = cnot.copy()

        current_target = cnot_copy.target
        current_control = cnot_copy.control

        if current_target == best_shuttle.target:
            cnot_copy.target = best_shuttle.control
        elif current_target == best_shuttle.control:
            cnot_copy.target = best_shuttle.target
        elif current_control == best_shuttle.target:
            cnot_copy.control = best_shuttle.control
        elif current_control == best_shuttle.control:
            cnot_copy.control = best_shuttle.target

        new_cnots_list.append(build_connection_from_architecture(architecture, cnot_copy))

    if sum(len(cnots) for cnots in new_cnots_list)+6 > sum(len(cnots) for cnots in rerouted_gate_list):
        return None, rerouted_gate_list, sum(len(cnots) for cnots in rerouted_gate_list)
    else:
        return best_shuttle, new_cnots_list, sum(len(cnots) for cnots in new_cnots_list)+6







def rearrange_columns_for_architecture(m: Mat2, architecture: Architecture, swap_operations: Dict[int, int] = {}, col_index: int = 0) -> Tuple[dict, Mat2] | None:
    """
    Rearranges the columns of a matrix to match the qubit mapping of an architecture.
    The function recursively tries to swap columns to match the architecture.
    If a suitable column is found, the function is called recursively for the next column.
    If no suitable column is found, the function returns the matrix as is.
    
    Args:
        m: The matrix to be rearranged
        architecture: The architecture to match the columns to
        swap_operations: A dictionary containing the swap operations that have been performed so far
        col_index: The index of the column to start the rearrangement from
    
    Returns:
        A tuple containing the swap operations that have been performed and the rearranged matrix,
        or None if no suitable column is found.
    """
    
    #TODO: This is not enough. Steiner will edit columns, which will mess up the initial column swaps. Column swaps should try to consider steiner method.
    # Create a copy of the matrix to avoid modifying the original
    matrix = m.copy()
    qubit_mapping = architecture.qubit_map

    # Iterate over the columns
    for c in range(col_index, matrix.cols()):
        # Get the actual row indices for the current column where the value is 1
        actual_indices = [qubit_mapping[r] for r in range(c, matrix.rows()) if matrix.data[r][c] == 1 or r == c]
        actual_indices.sort()

        if len(actual_indices) > 0:
            # Check if the actual indices form a contiguous series
            if actual_indices != list(range(min(actual_indices), max(actual_indices) + 1)):
                # If not, find a column to swap with
                for k in range(c + 1, matrix.cols()):
                    # Get the actual row indices for the potential swap column
                    swap_indices = [qubit_mapping[r] for r in range(c, matrix.rows()) if matrix.data[r][k] == 1 or r == c]
                    swap_indices.sort()

                    # Check if the swap indices form a contiguous series
                    if swap_indices == list(range(min(swap_indices), max(swap_indices) + 1)):
                        # If so, swap the columns
                        matrix.col_swap(c, k)
                        swap_operations[c] = k

                        # Recursively call the function for the next column
                        result = rearrange_columns_for_architecture(matrix, architecture, swap_operations.copy(), c + 1)
                        if result is not None:
                            return result

                        # If the recursive call did not find a solution, undo the swap
                        matrix.col_swap(c, k)
                        del swap_operations[c]

                # If no suitable column is found in the next columns, check the previous columns
                if c not in swap_operations:
                    for k in range(c - 1, -1, -1):
                        # Get the actual row indices for the potential swap column
                        swap_indices = [qubit_mapping[r] for r in range(c, matrix.rows()) if matrix.data[r][k] == 1 or r == c]
                        swap_indices.sort()

                        # Check if the swap indices form a contiguous series
                        if swap_indices == list(range(min(swap_indices), max(swap_indices) + 1)):
                            # If so, swap the columns
                            matrix.col_swap(c, k)

                            # Check if both columns are now correct
                            actual_indices_after_swap = [qubit_mapping[r] for r in range(c, matrix.rows()) if matrix.data[r][c] == 1 or r == c]
                            actual_indices_after_swap.sort()
                            swap_indices_after_swap = [qubit_mapping[r] for r in range(k, matrix.rows()) if matrix.data[r][k] == 1 or r == k]
                            swap_indices_after_swap.sort()

                            if actual_indices_after_swap == list(range(min(actual_indices_after_swap), max(actual_indices_after_swap) + 1)) and swap_indices_after_swap == list(range(min(swap_indices_after_swap), max(swap_indices_after_swap) + 1)):
                                # If both columns are correct, add the swap operation to the dictionary
                                swap_operations[c] = k

                                # Recursively call the function for the next column
                                result = rearrange_columns_for_architecture(matrix, architecture, swap_operations.copy(), c + 1)
                                if result is not None:
                                    return result

                                # If the recursive call did not find a solution, undo the swap
                                del swap_operations[c]
                                matrix.col_swap(c, k)
                            else:
                                matrix.col_swap(c, k)

    # If no suitable column is found, return the matrix as is
    if col_index == matrix.cols() - 1:
        return swap_operations, matrix
    else:
        return None


def neighbors_of_frontier(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        inverse: bool = False
        ) -> Set[VT]:
    """Returns the set of neighbors of the frontier. When collecting the vertices, it also checks if the vertices
    of the frontier are connected correctly to the inputs.
    If a frontier vertex is only connected to an input, it is removed from the frontier.
    If a frontier vertex is connected to an input and some other vertices, it is disconnected from the input via a new
    spider."""
    qs = g.qubits()
    rs = g.rows()
    neighbor_set = set()

    start = g.inputs() if not inverse else g.outputs()
    end = g.outputs() if not inverse else g.inputs()

    for qubit, vertex in frontier.copy().items():
        non_start_neighbors = [neighbor for neighbor in g.neighbors(vertex) if neighbor not in start]
        if any(neighbor in end for neighbor in non_start_neighbors):  # frontier vertex v is connected to an input
            if len(non_start_neighbors) == 1:  # Only connected to input, remove from frontier
                del frontier[qubit]
                continue
            # We disconnect v from the input b via a new spider
            first_end = [neighbor for neighbor in non_start_neighbors if neighbor in end][0]
            q = qs[first_end]
            r = rs[first_end]
            new_vertex = g.add_vertex(VertexType.Z, q, r + 1)
            edge = g.edge(vertex, first_end)
            edge_type = g.edge_type(edge)

            g.remove_edge(edge)
            g.add_edge(g.edge(vertex, new_vertex), EdgeType.HADAMARD)
            g.add_edge(g.edge(new_vertex, first_end), toggle_edge(edge_type))
            non_start_neighbors.remove(first_end)
            non_start_neighbors.append(new_vertex)
        neighbor_set.update(non_start_neighbors)
    return neighbor_set

def get_frontier_neighbors(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all (non-output) neighbors of the frontier as a set"""
    res = set()
    for v in frontier.values():
        res.update(set(g.neighbors(v)))
    return res.difference(set(g.outputs()))



def remove_gadget2(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        neighbor_set: Set[VT], 
        gadgets: Dict[VT, VT]
        ) -> bool:
    """Removes a gadget that is attached to a frontier vertex. Returns True if such gadget was found, False otherwise"""
    removed_gadget = False
    outputs = g.outputs()
    for neighbor in neighbor_set:
        if neighbor not in gadgets: continue
        for vertex in g.neighbors(neighbor):
            if vertex in frontier.values():
                if phase_is_true_clifford(g.phase(neighbor)):
                    apply_rule(g, lcomp_with_boundaries, [(neighbor, list(g.neighbors(neighbor)))])  # type: ignore
                else:
                    qubit_for_vertex = list(frontier.keys())[list(frontier.values()).index(vertex)]
                    apply_rule(g, pivot, [(neighbor, vertex, [], [o for o in g.neighbors(vertex) if o in outputs])])  # type: ignore
                    
                    frontier[qubit_for_vertex] = neighbor

                del gadgets[neighbor]
                removed_gadget = True
                break
    return removed_gadget


def remove_gadget(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        inverse: bool = False
        ) -> bool:
    """Removes a gadget that is attached to a frontier vertex. Returns True if such gadget was found, False otherwise"""
    gadget_set = get_frontier_gadgets(g, frontier)
    removed_gadget = False
    start = g.inputs() if not inverse else g.outputs()
    
    for root, _ in gadget_set:
        first_frontier_neighbor = [o for o in g.neighbors(root) if o in frontier.values()][0]
        if phase_is_true_clifford(g.phase(root)):
            apply_rule(g, lcomp_with_boundaries, [(root, list(g.neighbors(root)))])  # type: ignore
        else:
            qubit_for_vertex = list(frontier.keys())[list(frontier.values()).index(first_frontier_neighbor)]
            apply_rule(g, pivot, [(root, first_frontier_neighbor, [], [o for o in g.neighbors(first_frontier_neighbor) if o in start])])  # type: ignore
            
            frontier[qubit_for_vertex] = root

        removed_gadget = True
        break
    return removed_gadget


def apply_cnots(graph: BaseGraph[VT, ET], 
                circuit: Circuit | QuantumCircuit | None, 
                frontier: Dict[int, VT], 
                rerouted_cnots: List[ReroutedGate], 
                m: Mat2, 
                neighbors: List[VT],
                inverse: bool = False
                ) -> Tuple[Circuit|QuantumCircuit, int]:
    """Adds the list of CNOTs to the circuit, modifying the graph, frontier, and qubit map as needed.
    Returns the number of vertices that end up being extracted"""
    start = graph.inputs() if not inverse else graph.outputs()

    frontier_with_removed = {i: -1 for i in range(len(start))}
    frontier_with_removed.update(frontier)
    neighbors_copy = neighbors.copy()
    for _ in range(len(frontier) - len(neighbors)):
        neighbors_copy.append(-1)

    cnots = sum(rerouted_cnots, [])
    
    if len(cnots) > 0:
        cnots2 = cnots
        cnots = []
        for cnot in cnots2:
            m.row_add(cnot.control, cnot.target)
            cnots.append(CNOT(cnot.target, cnot.control))

        connectivity_from_biadj(graph, m, neighbors_copy, list(frontier_with_removed.values()))

    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            qubit: int = list(frontier_with_removed)[i]
            v = frontier_with_removed[qubit]
            w = neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[qubit] = (v, w)
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")
    hads = []

    for qubit, (v, w) in good_verts.items():  # Update frontier vertices
        hads.append(qubit)
        # c.add_gate("HAD",qubit_map[v])
        b = [o for o in graph.neighbors(v) if o in start][0]
        graph.remove_vertex(v)
        graph.add_edge(graph.edge(w, b))
        frontier[qubit] = w

    if circuit is None:
        c_extracted_gates = QuantumCircuit(len(frontier_with_removed))
        for cnot in cnots:
            c_extracted_gates.cx(cnot.control, cnot.target)
            # c_extracted_gates.add_gate("HAD",cnot.target)
            # c_extracted_gates.add_gate("CZ", cnot.control, cnot.target)
            # c_extracted_gates.add_gate("HAD",cnot.target)
        for h in hads:
            c_extracted_gates.h(h)
        return c_extracted_gates, len(good_verts)

    
    for cnot in cnots:
        add_gate_to_circuit(circuit=circuit, gate=cnot)
        # c.append(HAD(cnot.target))
        # c.append(CZ(cnot.control, cnot.target))
        # c.append(HAD(cnot.target))
    for h in hads:
        add_gate_to_circuit(circuit=circuit, gate=HAD(h))

    return circuit, len(good_verts)







def eliminate_unary_phase_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """checks whether there are unary phase gadgets left in the diagram, that is phase gadgets which have only a single neighbor"""
    while True:
        change = False
        for v in set(g.vertices()).difference(set(frontier.values())):
            
            n = list(g.neighbors(v))
            if len(n) == 1 and g.phase(n[0]) == 0 and g.type(v) == VertexType.Z and g.type(n[0]) == VertexType.Z:
                n2 = set(g.neighbors(n[0])).difference(set([v]))
                if len(n2) == 1:
                    root = n2.pop()
                    g.set_phase(root, g.phase(root)+g.phase(v))
                    g.remove_vertices([v,n[0]])
                    change = True
                    break
        if not change:
            break


def get_cnot_row_operations(
        g: BaseGraph, 
        frontier: Dict[int, VT], 
        frontier_neighbors: Set[VT],
        architecture: Architecture = None        
        ) -> List[ReroutedGate]:
    """ Compute row echelon form of adjacency matrix and save row operations as CNOTs 
     -> because of gflow the resulting matrix has a row with only a single 1"""
    
    rerouted_gate_list = get_all_cnot_operations(g, frontier, frontier_neighbors, architecture=architecture)

    if rerouted_gate_list is None:
        return None
    
    rerouted_gates = get_best_cnot_configuration(rerouted_gate_list, architecture)

    print(f"      CNOT elimination with {rerouted_gates} CNOTs")
    
    return rerouted_gates

def get_cnot_row_operations2(
        g: BaseGraph, 
        frontier: Dict[int, VT], 
        frontier_neighbors: Set[VT],
        architecture: Architecture = None        
        ) -> List[CNOT]:
    """ Compute row echelon form of adjacency matrix and save row operations as CNOTs 
     -> because of gflow the resulting matrix has a row with only a single 1"""
    frontier_with_removed = {i: -1 for i in range(len(g.outputs()))}
    frontier_with_removed.update(frontier)
    m_frontier = bi_adj(g, list(frontier_neighbors), frontier.values())
    m: Mat2 = bi_adj(g, list(frontier_neighbors), frontier_with_removed.values())
    elim_mode = ElimMode.STEINER_MODE
    
    greedy_operations = greedy_reduction_with_architecture(m, architecture)

    if not greedy_operations:
        neighbors = list(frontier_neighbors)
        perm = column_optimal_swap(m)
        perm = {v: k for k, v in perm.items()}
        neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]

        m2 = bi_adj(g, neighbors2, frontier_with_removed.values())

        if architecture:
            cnots, rank = gauss(architecture=architecture, matrix=m2, mode=elim_mode, full_reduce=True)
            cnots = filter_duplicate_cnots(cnots)
            cnots = [CNOT(cnot.target, cnot.control) for cnot in cnots]

        else:
            cnots = m2.to_cnots(optimize=True)

        if not any([sum(row) == 1 for row in m2.data]):
            return None
    else:
        cnot_list_greedy: list[list[CNOT]] = []
        for rerouted_gate_list in greedy_operations:
            rerouted_cnots: list[CNOT] = []
            for rerouted_gate in rerouted_gate_list:
                rerouted_cnots.extend(rerouted_gate.gate_path)
            cnot_list_greedy.append(rerouted_cnots)

        cnots = get_best_cnot_configuration(cnot_list_greedy, architecture)
        cnots = filter_duplicate_cnots(cnots)
        # cnots = [CNOT(cnot.target, cnot.control) for cnot in cnots]

    if not greedy_operations: print(f"      Gaussian elimination with {[CNOT(cnot.target, cnot.control) for cnot in cnots]} CNOTs")
    else: print(f"      Greedy elimination with {[CNOT(cnot.target, cnot.control) for cnot in cnots]} CNOTs")
    
    return cnots


def get_all_cnot_operations(
        g: BaseGraph, 
        frontier: Dict[int, VT], 
        frontier_neighbors: Set[VT],
        architecture: Architecture = None        
        ) -> List[List[ReroutedGate]] | None:
    """ Compute row echelon form of adjacency matrix and save row operations as CNOTs 
     -> because of gflow the resulting matrix has a row with only a single 1"""
    frontier_with_removed = {i: -1 for i in range(len(g.outputs()))}
    frontier_with_removed.update(frontier)
    m_frontier = bi_adj(g, list(frontier_neighbors), frontier.values())
    m: Mat2 = bi_adj(g, list(frontier_neighbors), frontier_with_removed.values())
    m2: Mat2 = m.copy()
    elim_mode = ElimMode.STEINER_MODE

    greedy_operations = greedy_reduction_with_architecture(m, architecture)

    if not greedy_operations:
        neighbors = list(frontier_neighbors)
        perm = column_optimal_swap(m)
        perm = {v: k for k, v in perm.items()}
        neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]

        m2 = bi_adj(g, neighbors2, frontier_with_removed.values())
        cnots = []
        m2_no_arch = m2.copy()

        if architecture:
            cnot_list_arch, rank = gauss(architecture=architecture, matrix=m2, mode=elim_mode, full_reduce=True)
            cnot_list_arch = filter_duplicate_cnots(cnot_list_arch)
            cnots_arch = [CNOT(cnot.target, cnot.control) for cnot in cnot_list_arch]

            cnots = [ReroutedGate(cnot, [cnot]) for cnot in cnot_list_arch]

        cnot_list_no_arch = m2_no_arch.to_cnots(optimize=True)
        cnot_list_no_arch = filter_duplicate_cnots(cnot_list_no_arch)

        cnots += [ReroutedGate(cnot, [cnot]) for cnot in cnot_list_no_arch]
        cnots = [cnots]

        if not any([sum(row) == 1 for row in m2.data]):
            return None
    else:
        # cnot_list_greedy_basic = []

        # # Iterate through each rerouted_gate_list in greedy_operations
        # for rerouted_gate_list in greedy_operations:
        #     # Check if any rerouted_gate in the rerouted_gate_list satisfies the condition
        #     # This replaces the any(...) part of the original code, aiming to short-circuit the evaluation
        #     if any(len(rerouted_gate.gate_path) > 1 for rerouted_gate in rerouted_gate_list):
        #         # If the condition is satisfied, extract the basic_gate from each rerouted_gate
        #         # and add the list of basic_gates to the optimized_cnot_list
        #         cnot_list_greedy_basic.append([rerouted_gate.basic_gate for rerouted_gate in rerouted_gate_list])

        # cnot_list_greedy: list[list[CNOT]] = []
        # for rerouted_gate_list in greedy_operations:
        #     rerouted_cnots: list[CNOT] = []
        #     for rerouted_gate in rerouted_gate_list:
        #         rerouted_cnots.extend(rerouted_gate.gate_path)
        #     cnot_list_greedy.append(filter_duplicate_cnots(rerouted_cnots))
        

        cnots = greedy_operations

    # for cnot_list in cnots:
    #     if not greedy_operations: print(f"      Gaussian elimination with {cnot_list} CNOTs")
    #     else: print(f"      Greedy elimination with {cnot_list} CNOTs")

    return cnots



def get_all_gadgets(g: BaseGraph[VT, ET]) -> Dict[VT, VT]:
    """Returns all phase gadgets in the graph as a dictionary where the key is the root spider and the value is the top spider"""
    gadgets = dict()
    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in g.inputs() and v not in g.outputs():
            n = list(g.neighbors(v))[0]
            gadgets[n] = v
    return gadgets


def get_frontier_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all phase gadget neighbors of the frontier as a set of tuples (root,top) 
    where root is the (phaseless) root spider, and top the 1-ary spider with phase connected to root"""
    res = set()
    for v in frontier.values():
        for n in g.neighbors(v):
            for potential_top in g.neighbors(n):
                if g.vertex_degree(potential_top) == 1 and potential_top not in g.inputs() and potential_top not in g.outputs():
                    res.add((n,potential_top))

    return res


def get_exclusive_frontier_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all phase gadget neighbors of the frontier as a set of tuples (root,top) 
    where root is the (phaseless) root spider, and top the 1-ary spider with phase connected to root"""
    res = set()
    for v in frontier.values():
        for n in g.neighbors(v):
            difference_set = set(g.neighbors(n)).union(set(frontier.values())).difference(set(frontier.values())).difference(set(g.outputs()))
            if len(difference_set) == 1: 
                top = difference_set.pop()
                if len(g.neighbors(top)) == 1:
                    #if there are no other neighbors than frontier neighbors+ gadget top
                    res.add((n,top))

    return res


def get_frontier_gadget_dict(g: BaseGraph[VT,ET], frontier: Dict[int, VT]):
    """returns a dictionary of all gadgets adjacent to only frontier vertices grouped by their degree 
    (i.e. to how many frontier vertices the gadget is connected to)"""
    frontier_gadgets = get_exclusive_frontier_gadgets(g, frontier)
    gadget_dict = dict()
    for root, top in frontier_gadgets:
        neighbors_in_frontier = set(g.neighbors(root)).difference(set([top]))
        gadget_dict.setdefault(len(neighbors_in_frontier),[]).append((root,top,neighbors_in_frontier))

    return gadget_dict



def init_frontier(g: BaseGraph[VT, ET], circuit: Circuit|QuantumCircuit, inverse:bool=False) -> Dict[int,VT]:
    """Inits the frontier of a ZX-diagram with the spiders adjacent to the inputs. Extracts Hadamard wires between inputs and frontier"""
    frontier: Dict[int,VT] = dict()
    start = list(g.inputs()) if not inverse else list(g.outputs())
    end = list(g.outputs()) if not inverse else list(g.inputs())

    for qubit, start_vertex in enumerate(start):
        v = list(g.neighbors(start_vertex))[0]
        if not v in end:
            frontier[qubit] = v
            if g.edge_type(g.edge(v,start_vertex)) == EdgeType.HADAMARD:
                add_gate_to_circuit(circuit, HAD(qubit))
                g.set_edge_type(g.edge(v,start_vertex),EdgeType.SIMPLE)
    
    return frontier


def extract_czs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit|QuantumCircuit, architecture:Architecture, optimize: bool = False):
    """Extracts connected frontier spiders as controlled Z gates and updates the diagram"""
    if optimize:
        #TODO: this might need to be changed for all types of gadgets
        optimize_czs_in_frontier(g, frontier)
    change = False
    for qubit, v in frontier.items():
        for w in set(g.neighbors(v)).intersection(set(frontier.values())):
            g.remove_edge(g.edge(v,w))
            gate = CZ(qubit, list(frontier.keys())[list(frontier.values()).index(w)])
            if architecture:
                rerouted_gates = build_connection_from_architecture(architecture, gate)
                for rerouted_gate in rerouted_gates:
                    add_gate_to_circuit(circuit, rerouted_gate)
            else:
                add_gate_to_circuit(circuit, gate)
            change = True
    return change


def extract_rzs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit|QuantumCircuit, inverse: bool = False):
    """Extracts phases of frontier spiders as ZPhase gates and updates the diagram"""
    start = list(g.inputs()) if not inverse else list(g.outputs())
    phase_change = False
    
    for qubit in list(frontier.keys()):
        while True:
            boundary = False
            v = frontier[qubit]
            phase = g.phase(v)
            if phase != 0:
                g.set_phase(v,0)
                gate = ZPhase(qubit, phase)
            else:
                neighbors = [neighbor for neighbor in g.neighbors(v) if neighbor not in start]
                if len(neighbors) != 1:
                    break

                if g.type(neighbors[0]) == VertexType.BOUNDARY:
                    boundary = True
                    if g.edge_type(g.edge(v,neighbors[0])) == EdgeType.HADAMARD:
                        gate = HAD(qubit)
                    else:
                        gate = None

                                    
                    g.add_edge(g.edge(start[qubit],neighbors[0]))
                    del frontier[qubit]
                else:
                    frontier[qubit] = neighbors[0]
                    gate = HAD(qubit)

                    first_start = [neighbor for neighbor in g.neighbors(v) if neighbor in start]
                    g.add_edge(g.edge(first_start[0], neighbors[0]))
                    
                    print("Simple vertex")
                                
                g.remove_vertex(v)

            
            if gate:
                add_gate_to_circuit(circuit, gate)
                phase_change = True

            if boundary:
                break

    return phase_change

def extract_cnots(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit|QuantumCircuit, rerouted_cnots: List[ReroutedGate]):
    """Extracts CNOT gates resulting from gaussian elimination to circuit and adds the Hadamard wires of the corresponding frontier vertices"""
    
    basic_cnots = [rerouted_gate.basic_gate for rerouted_gate in rerouted_cnots]
    full_cnots = sum(rerouted_cnots, [])

    for cnot in full_cnots:
        # CNOT = H+CZ+H
        # circuit.add_gate("HAD",target_qubit)
        # circuit.add_gate("CZ", control_qubit, target_qubit)
        # circuit.add_gate("HAD",target_qubit)
        add_gate_to_circuit(circuit, cnot)

        # add_gate_to_circuit(circuit, HAD(target_qubit))
        # add_gate_to_circuit(circuit, CZ(control_qubit, target_qubit))
        # add_gate_to_circuit(circuit, HAD(target_qubit))

    for cnot in basic_cnots:
        # Add CNOT to circuit
        control_qubit = cnot.control 
        target_qubit = cnot.target
        # Add or remove Hadamard wires in diagram according to CNOT addition

        # if control_qubit not in frontier or target_qubit not in frontier:
        #     continue

        ftarg = frontier[control_qubit]
        fcont = frontier[target_qubit] 

        neighbors_without_start = [neighbor for neighbor in g.neighbors(fcont) if neighbor not in g.inputs()]
        for v in neighbors_without_start:
            if g.type(v) == VertexType.BOUNDARY:
                # special case: neighbor of "control" spider is an input, therefore we need to insert a spider between input and control spider
                vnew = insert_identity(g, fcont, v)
                break
    
        for v in neighbors_without_start:
            # remove wire
            if g.connected(ftarg,v):
                g.remove_edge(g.edge(ftarg,v))
            # add wire
            else:
                g.add_edge(g.edge(ftarg,v), EdgeType.HADAMARD)

    return True


def optimize_czs_in_frontier(g: BaseGraph, frontier: Dict[int, VT]):
    """optimizes czs in a frontier by applying local complementations on phase gadgets so that the number of wires between frontiers decreases
    effect on overall runtime relatively small yet."""
    gadget_dict = get_frontier_gadget_dict(g, frontier)
    for degree in sorted(gadget_dict.keys()):
        for gadget_root, gadget_top, gadget_neighbors in gadget_dict[degree]:
            if g.subgraph_from_vertices(gadget_neighbors).num_edges() > len(gadget_neighbors)*(len(gadget_neighbors)-1)/4:
                #lcomp removes more edges in the frontier set than it creates
                # print("complemented away czs")
                complement_neighbors(g, list(gadget_neighbors))
                for neighbor in gadget_neighbors:
                    g.set_phase(neighbor, g.phase(neighbor)+Fraction(1,2))
                g.set_phase(gadget_top, g.phase(gadget_top)-Fraction(1,2))


def complement_neighbors(g: BaseGraph, vn: List[VT]):
    """complements connections between a set of vertices, i.e. everything connected gets disconnected and vice versa"""
    vn.sort()
    for n in vn:
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)




def convert_to_qiskit(c: Circuit):
    qc = QuantumCircuit(c.qubits)
    for gate in c.gates:
        if gate.name == "HAD":
            qc.h(gate.target)
        elif gate.name == "ZPhase":
            qc.rz(float(gate.phase)*math.pi, gate.target)
        elif gate.name == "CZ":
            qc.cz(gate.control, gate.target)
        elif gate.name == "MCP":
            qc.mcp(float(gate.phase)*math.pi, gate.controls, gate.target)
        elif gate.name == "SWAP":
            qc.swap(gate.control, gate.target)
        elif gate.name == "U3":
            qc.u3(float(gate.phases[0])*math.pi, float(gate.phases[1])*math.pi, float(gate.phases[2])*math.pi, gate.target)
        else:
            print("unknown gate",gate)
    return qc


def optimize_pyzx_circuit(c: Circuit):
    """ZPhase gate optimize for circuits consisting of CZ, MCP, HAD, ZPhase and maybe SWAPs"""
    c_opt = Circuit(c.qubits)
    current_rz_gates = [None for i in range(c.qubits)]
    to_remove = []
    for gate in c.gates:
        if gate.name not in ["CZ","HAD","ZPhase","MCP","SWAP","U3"]:
            print("This should not happen")
            print(gate.name)
            print(gate)
        assert(gate.name in ['CZ','HAD','ZPhase','MCP','SWAP','U3'])
        qubit = gate.target
        if gate.name == "ZPhase":
            if current_rz_gates[qubit] != None:
                current_rz_gates[qubit] += gate.phase
                to_remove.append(gate)
            else:
                current_rz_gates[qubit] = gate.phase
            continue
        elif gate.name == "HAD":
            if current_rz_gates[qubit] != None:
                c_opt.add_gate("ZPhase",qubit,current_rz_gates[qubit])
                current_rz_gates[qubit] = None
            c_opt.add_gate("HAD",qubit)
        elif gate.name == "SWAP":
            qubit2 = gate.control
            for q in [qubit, qubit2]:
                if current_rz_gates[q] != None:
                    c_opt.add_gate("ZPhase",q,current_rz_gates[q])
                    current_rz_gates[q] = None
            c_opt.add_gate("SWAP",qubit, qubit2)
        elif gate.name == "CZ":
            c_opt.add_gate("CZ",gate.control, gate.target)
        elif gate.name == "MCP":
            c_opt.add_gate(MCP(gate.controls, gate.target, gate.phase))
        elif gate.name == "U3":
            c_opt.add_gate("U3",gate.target, *gate.phases)
    return c_opt
