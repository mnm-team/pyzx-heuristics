import copy
import itertools
import math

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit


from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, Gate, ZPhase
from pyzx.extract import column_optimal_swap, connectivity_from_biadj, filter_duplicate_cnots, xor_rows
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.mapper import gate_mapper
from pyzx.heuristics.extraction.rerouting import ReroutedGate, build_connection_from_architecture
from pyzx.linalg import Z2, Mat2
from pyzx.routing.architecture import Architecture
from pyzx.routing.cnot_mapper import ElimMode
from pyzx.simplify import apply_rule, full_reduce, pivot, lcomp_with_boundaries
from pyzx.utils import EdgeType, FractionLike, VertexType, phase_is_true_clifford, toggle_edge

from pyzx.drawing import draw


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
        

def add_gate_to_circuit(circuit: Circuit|QuantumCircuit, gate:Gate) -> None:
    """Adds a gate to a circuit. If the circuit is a QuantumCircuit, the gate is added using the Qiskit API.
    If the circuit is a Circuit, the gate is added using the PyZX API."""

    #TODO: this should be implemented in the Gate class
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


def get_full_graph_from_partial_graph_and_circuit(partial_graph: BaseGraph, partial_circuit: Circuit, fr:bool, inverse:bool=False) -> BaseGraph:
    """Returns the full graph from a partial graph and a partial circuit. If fr is True, the full graph is fully reduced."""

    second_partial_graph = partial_circuit.to_graph()
    if inverse:
        full_graph = partial_graph + second_partial_graph
    else:
        full_graph = second_partial_graph + partial_graph

    if fr:
        full_reduce(full_graph)

    return full_graph





def bi_adj(g: BaseGraph[VT,ET], vs:List[VT], ws:List[VT]) -> Mat2:
    """Construct a biadjacency matrix between the supplied list of vertices
    ``vs`` and ``ws``.
    
    If ``vs`` has less elements than ``ws``, ``vs`` is padded with -1.
    """
    vs_copy = vs.copy()
    for _ in range(len(ws)-len(vs)):
        vs_copy.append(-1)

    return Mat2([[0 if (w == -1 or v == -1) else int(g.connected(v,w)) for v in vs_copy] for w in ws])


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


def greedy_reduction_with_architecture(m: Mat2, architecture: Architecture) -> Optional[List[List[ReroutedGate]]]:
    """Returns a list of lists of CNOTs that reduce the matrix m to a matrix with only one 1 in at least one row.
    The function uses a greedy algorithm to find the minimal sums of rows that can be added together to reduce the matrix."""
    indices_list_greedy_row_add = find_minimal_sums_with_architecture(m, architecture=None)
    # print("indices",indices_list_greedy_row_add,"for m",m)
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
            # print("result",result)
            control, target = best
            rows[target] = xor_rows(rows[control],rows[target])
            weights[target] = weights[target] - reduction
            indices.remove(control)

        row_add_results.append(result)
        # print("result",row_add_results)

    return row_add_results


def get_best_cnot_configuration(rerouted_gate_list: List[List[ReroutedGate]], architecture: Architecture) -> List[ReroutedGate]:
    """Given a list of lists of CNOTs, returns the list with the fewest CNOTs that are viable for the given architecture."""
    if not rerouted_gate_list:
        return []
    
    best_result = (None, math.inf)
    
    for rerouted_gates in rerouted_gate_list:

        if not all(rerouted_gate.is_viable_for_architecture(architecture) for rerouted_gate in rerouted_gates):
            continue

        if (sum_cnots := sum(len(rerouted_gate.gate_path) for rerouted_gate in rerouted_gates)) < best_result[1]:
            best_result = (rerouted_gates, sum_cnots)

    return best_result[0]






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


def get_neighbors_of_frontier(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        inverse: bool = False
        ) -> Set[VT]:
    """Returns the set of neighbors of the frontier."""
    neighbor_set = set()

    start = g.inputs() if not inverse else g.outputs()
    end = g.outputs() if not inverse else g.inputs()

    for _, vertex in frontier.copy().items():
        non_start_neighbors = [neighbor for neighbor in g.neighbors(vertex) if neighbor not in start+end]
        neighbor_set.update(non_start_neighbors)
    return neighbor_set


def update_graph_for_frontier_neighbor_in_end(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        inverse: bool = False
        ) -> Set[VT]:
    """Checks if the vertices of the frontier are connected correctly to the end.
    If a frontier vertex is only connected to an end vertex, it is removed from the frontier.
    If a frontier vertex is connected to an end vertex and some other vertices, it is disconnected from the input via a new
    spider."""
    qs = g.qubits()
    rs = g.rows()
    new_verticies = set()

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

            new_verticies.add(new_vertex)

    return new_verticies







def apply_gates_to_circuit(graph:BaseGraph,
                            circuit:Circuit|QuantumCircuit,
                            mapper, 
                            frontier:Dict[int, VT], 
                            rerouted_gate_list:List[List[ReroutedGate]],
                            apply_gate_function:Callable[[BaseGraph, Circuit|QuantumCircuit, Dict[int, VT], List[ReroutedGate], List[VT], bool], Circuit|QuantumCircuit],
                            inverse:bool=False,
                            ) -> Tuple[BaseGraph, Dict[int, VT], List[VT], Circuit, Architecture|None]:
    
    """Applies the Gates to the graph and circuit, updating the frontier and frontier neighbors as needed.
    Returns the updated graph, frontier, frontier neighbors, circuit, and architecture.
    If use_gate_mapping is True, the circuit is mapped to the architecture using the given mapper.
    If use_gate_mapping is False, the circuit is updated with the CNOTs directly.
    """
    gate_data = {"circuits": [], "graphs": [], "frontier": [], "neighbors": []}
    gate_list_index = 0
    while gate_list_index < len(rerouted_gate_list):
        current_gate_list = rerouted_gate_list[gate_list_index]
        if mapper:
            # If a mapper is given, each gate_list is applied to an empty circuit and stored in the gate_data dictionary.
            # The graph, frontier, and frontier_neighbors are updated accordingly and also stored in the gate_data dictionary.
            gate_data = apply_gate_operations_and_store_data(graph, frontier, copy.deepcopy(current_gate_list), gate_data, inverse=inverse, apply_gate_function=apply_gate_function)
            # print("gate data 1",gate_data)
            # if current_gate_list and any(rerouted_gate.is_gate_rerouted() for rerouted_gate in current_gate_list):
            #     basic_gates = [ReroutedGate(rerouted_gate.basic_gate.copy(), None) for rerouted_gate in current_gate_list]
            #     gate_list_index += 1
            #     rerouted_gate_list.insert(gate_list_index, basic_gates)
            #     gate_data = apply_gate_operations_and_store_data(graph, frontier, basic_gates, gate_data, inverse=inverse, apply_gate_function=apply_gate_function)
            #     print("gate data 2",gate_data)
        else:
            # If no mapper is given, the gates are applied directly to the circuit.
            # Only one gate list is supported without gate mapping
            if len(rerouted_gate_list) > 1:
                raise ValueError("Multiple Gatelists not supported without gate mapping")
            circuit = apply_gate_function(graph, circuit, frontier, current_gate_list, inverse=inverse)
            if len(rerouted_gate_list[0])>0: print(f"      Gate extraction with {rerouted_gate_list[0]}")
        gate_list_index += 1
    
    architecture_copy = None
    if mapper:
        # If a mapper is given, all stored circuits are given to the mapper to choose the optimal one to map.
        # The chosen circuit is then mapped to the architecture and the architecture is returned.
        # The according graph, frontier, and frontier_neighbors are taken from the gate_data dictionary.
        architecture_copy, circuit_index = gate_mapper(mapper, gate_data["circuits"])
        circuit += gate_data["circuits"][circuit_index] #Circuit(len(graph.inputs()))
        graph = gate_data["graphs"][circuit_index]
        frontier = gate_data["frontier"][circuit_index]
        print("applied gates:",gate_data["circuits"][circuit_index].gates)
        # frontier_neighbors = gate_data["neighbors"][circuit_index]

        if len(rerouted_gate_list[circuit_index])>0: print(f"      Gate extraction with {rerouted_gate_list[circuit_index]}")
    
    return graph, frontier, circuit, architecture_copy


def apply_gate_operations_and_store_data(graph:BaseGraph, frontier:Dict[int, VT], gate_list:List[ReroutedGate], gate_data:Dict[str, List[Any]], inverse:bool, apply_gate_function:Callable) -> Dict[str, List[Any]]:
    """Applies the given gate list to the graph and stores the resulting graph, frontier, and neighbors in the gate_data dictionary."""

    graph_copy = graph.clone()
    frontier_copy = frontier.copy()
    # neighbors_copy = frontier_neighbors.copy()
    c = apply_gate_function(graph_copy, Circuit(len(graph.inputs())), frontier_copy, gate_list, inverse)
    gate_data["circuits"].append(c)
    gate_data["graphs"].append(graph_copy)
    gate_data["frontier"].append(frontier_copy)
    # gate_data["neighbors"].append(neighbors_copy)

    return gate_data


def apply_cnots_to_circuit(g:BaseGraph, circuit:Circuit|QuantumCircuit, frontier:Dict[int, VT], rerouted_cnots: List[ReroutedGate], inverse:bool=False) -> Circuit|QuantumCircuit:
    """Applies the CNOTs to the circuit and returns the updated circuit and graph. If the circuit is a QuantumCircuit, the CNOTs are added using the Qiskit API."""
    
    #CNOT extraction
    #TODO: We need frontier_with_removed for keeping track where to add cnots, why does this work in normal pyzx?
    # frontier_with_removed = {i: -1 for i in range(len(g.outputs()))}
    # frontier_with_removed.update(frontier)
    frontier_with_removed = frontier

    basic_cnots = [rerouted_gate.basic_gate for rerouted_gate in rerouted_cnots]
    
    frontier_neighbors = list(get_neighbors_of_frontier(g, frontier))
    # Apply basic CNOTs to the matrix to save computation time
    m = bi_adj(g, frontier_neighbors, list(frontier_with_removed.values()))
    # if 33 in frontier_neighbors:
    #     import pdb
    #     pdb.set_trace()
    for cnot in basic_cnots:
        m.row_add(cnot.control, cnot.target)

    # this throws an error if we have multiple basic cnots, because then this function gets called multiple times in order to extract a vertex 
    # if all(sum(row) != 1 for row in m.data):
    #     raise Exception("CNOTs do not suffice to extract a vertex")
    
    # If we start from the inputs or outputs
    start = g.inputs() if not inverse else g.outputs()

    # Neighbors are padded with -1 since the list cant be shorter than the frontier for connectivity_from_biadj
    # neighbors_copy = frontier_neighbors.copy()
    # for _ in range(len(frontier) - len(frontier_neighbors)):
    #     neighbors_copy.append(-1)

    #Gates are reversed according to paper "there and back again"
    for rerouted_cnot in rerouted_cnots:
        rerouted_cnot.reverse_gate_qubits()

    cnots = sum(rerouted_cnots, [])

    if len(cnots) > 0:
        # if cnots[0].target == 2 and cnots[0].control == 4:
        #     import pdb
        #     pdb.set_trace()

        connectivity_from_biadj(g, m, frontier_neighbors, list(frontier_with_removed.values()))

    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            qubit: int = list(frontier_with_removed)[i]
            v = frontier_with_removed[qubit]
            w = frontier_neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[qubit] = (v, w)
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")

    if circuit is None:
        print("why?")
        import pdb
        pdb.set_trace()
        circuit = Circuit(len(frontier_with_removed))

    #TODO: Add parameter to choose between cnot and had+cz+had
    frontier_qubits = list(frontier.keys())
    for cnot in cnots:
        circuit_cnot = CNOT(frontier_qubits[cnot.control],frontier_qubits[cnot.target])
        add_gate_to_circuit(circuit=circuit, gate=circuit_cnot)

    return circuit


def apply_frontier_gates_to_circuit(g:BaseGraph, circuit:Circuit|QuantumCircuit, frontier:Dict[int, VT], rerouted_gates: List[ReroutedGate], inverse:bool=False) -> Circuit|QuantumCircuit:
    """Applies the gates to the circuit and returns the updated circuit and graph. If the circuit is a QuantumCircuit, the gates are added using the Qiskit API."""
    print("rerouted gates",rerouted_gates)
    gates = sum(rerouted_gates, [])
    frontier_with_removed = {i: -1 for i in range(len(g.inputs()))}
    frontier_with_removed.update(frontier)

    if circuit is None:
        print("why1?")
        import pdb
        pdb.set_trace()
        circuit = Circuit(len(frontier_with_removed))
    
    for gate in gates:
        add_gate_to_circuit(circuit=circuit, gate=gate)
        # c.append(HAD(cnot.target))
        # c.append(CZ(cnot.control, cnot.target))
        # c.append(HAD(cnot.target))

    return circuit






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
    
    return rerouted_gates


def get_all_cnot_operations(
        g: BaseGraph, 
        frontier_with_removed: Set[VT],
        frontier_neighbors: Set[VT],
        architecture: Architecture = None        
        ) -> List[List[ReroutedGate]] | None:
    """ Compute row echelon form of adjacency matrix and save row operations as CNOTs 
     -> because of gflow the resulting matrix has a row with only a single 1"""

    m: Mat2 = bi_adj(g, list(frontier_neighbors), frontier_with_removed)
    m2: Mat2 = m.copy()
    elim_mode = ElimMode.STEINER_MODE

    greedy_operations = greedy_reduction_with_architecture(m, architecture)

    if not greedy_operations:
        neighbors = list(frontier_neighbors)
        perm = column_optimal_swap(m)
        perm = {v: k for k, v in perm.items()}
        neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]

        m2 = bi_adj(g, neighbors2, frontier_with_removed)
        cnots = []
        m2_no_arch = m2.copy()

        # if architecture:
        #     #FIXME: This does not seem to work if a frontier is already removed
        #     #STEINER_MODE needs to be studied in more detail
        #     cnot_list_arch, rank = gauss(architecture=architecture, matrix=m2, mode=elim_mode, full_reduce=True)
        #     cnot_list_arch = filter_duplicate_cnots(cnot_list_arch)
        #     # cnots_arch = [CNOT(cnot.target, cnot.control) for cnot in cnot_list_arch]

        #     cnots.append([ReroutedGate(cnot, None) for cnot in cnot_list_arch])

        cnot_list_no_arch = m2_no_arch.to_cnots(optimize=True)
        cnot_list_no_arch = filter_duplicate_cnots(cnot_list_no_arch)
        cnots_no_arch = [CNOT(cnot.target, cnot.control) for cnot in cnot_list_no_arch]

        cnots.append([ReroutedGate(cnot, build_connection_from_architecture(architecture, cnot.copy())) for cnot in cnots_no_arch])

        if not any([sum(row) == 1 for row in m2.data]):
            return None
    else:     
        cnots = greedy_operations

    return cnots






def remove_gadget(
        g: BaseGraph[VT, ET], 
        frontier: Dict[int, VT],
        inverse: bool = False
        ) -> bool:
    """Removes a gadget that is attached to a frontier vertex. Returns True if such gadget was found, False otherwise"""
    gadget_set = get_frontier_gadgets(g, frontier)
    removed_gadget = False
    # start = g.inputs() if not inverse else g.outputs()
    
    for root, _ in gadget_set:
        first_frontier_neighbor = [o for o in g.neighbors(root) if o in frontier.values()][0]
        if phase_is_true_clifford(g.phase(root)):
            apply_rule(g, lcomp_with_boundaries, [(root, list(g.neighbors(root)))])  # type: ignore
        else:
            qubit_for_vertex = list(frontier.keys())[list(frontier.values()).index(first_frontier_neighbor)]
            apply_rule(g, pivot, [(root, first_frontier_neighbor, [], [o for o in g.neighbors(first_frontier_neighbor) if o in g.inputs()+g.outputs()])])  # type: ignore
            
            frontier[qubit_for_vertex] = root

        removed_gadget = True
        break
    return removed_gadget


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
