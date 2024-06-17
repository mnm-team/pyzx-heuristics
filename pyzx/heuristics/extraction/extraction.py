from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

from qiskit import QuantumCircuit
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, ZPhase
from pyzx.extract import graph_to_swaps, max_overlap
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.extraction_base import add_gate_to_circuit, apply_cnots, bi_adj, get_all_cnot_operations, get_all_gadgets, get_cnot_row_operations, init_frontier, neighbors_of_frontier, remove_gadget, reorder_frontier
from pyzx.heuristics.extraction.extraction_mcp import eliminate_yz_spider
from pyzx.heuristics.extraction.mapper import create_mapper, gate_mapper, get_circuit_from_mapper, init_mapper
from pyzx.heuristics.extraction.rerouting import ReroutedGate
from pyzx.linalg import Mat2
from pyzx.routing.architecture import Architecture
from pyzx.simplify import id_simp
from pyzx.utils import EdgeType


from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping



def extract_architecture_aware_circuit(
        graph: BaseGraph[VT, ET],
        architecture: Architecture,
        use_gate_mapping: bool = False,
        optimize_czs: bool = True,
        up_to_perm: bool = False,
        quiet: bool = True
        ) -> Tuple[Circuit, Architecture]:
    """Given a graph put into semi-normal form by :func:`~pyzx.simplify.full_reduce`, 
    it extracts its equivalent set of gates into an instance of :class:`~pyzx.circuit.Circuit`.
    This function implements a more optimized version of the algorithm described in
    `There and back again: A circuit extraction tale <https://arxiv.org/abs/2003.01664>`_

    Args:
        g: The ZX-diagram graph to be extracted into a Circuit.
        optimize_czs: Whether to try to optimize the CZ-subcircuits by exploiting overlap between the CZ gates
        optimize_cnots: (0,1,2,3) Level of CNOT optimization to apply.
        up_to_perm: If true, returns a circuit that is equivalent to the given graph up to a permutation of the inputs.
        randomize_frontier_tries: Number of times to try to extract the circuit with a randomized frontier. Default is 5.
        quiet: Whether to print detailed output of the extraction process.

    Warning:
        Note that this function changes the graph `g` in place. 
        In particular, if the extraction fails, the modified `g` shows 
        how far the extraction got. If you want to keep the original `g`
        then input `g.copy()` into `extract_circuit`.
    """
    outputs = list(graph.outputs())
    gadgets = get_all_gadgets(graph)
    
    if use_gate_mapping:
        circuit = QuantumCircuit(len(outputs))
        # mapper = create_mapper()
        # init_mapper(mapper, g.num_outputs())

        path = Path(__file__).parent.parent.resolve()   
        # create a neutral atom hybrid architecture
        architecture_mapper = NeutralAtomHybridArchitecture(str(path)+"/mapper_files/rubidium_sim.json")
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
        mapper = HybridSynthesisMapper(arch=architecture_mapper, params=params)

        mapper.init_mapping(graph.num_outputs(), InitialCircuitMapping.identity)
    else:   
        circuit = Circuit(len(outputs))

    frontier = init_frontier(graph, circuit, True)
    architecture_copy = Architecture(name=architecture.name, coupling_graph=architecture.graph.copy(), qubit_map=list(range(len(frontier))))
        
    while True:
        czs_saved = clean_frontier(graph, circuit, frontier, optimize_czs)

        if use_gate_mapping:
            gate_names = [f"{gate[0].name}: {gate[0].params}" for gate in circuit.data] 
            # If we have extracted some CZ gates, we need to add them to the circuit
            architecture_new, circuit_index = gate_mapper(mapper, circuit)
            if architecture_new.qubit_map != architecture_copy.qubit_map:
                architecture_copy = architecture_new
            ctest, new_arch = get_circuit_from_mapper(mapper, get_exact_phases=False)
            circuit = QuantumCircuit(len(outputs))
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = neighbors_of_frontier(graph, frontier, inverse=True)
   
        if not frontier:
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(graph, frontier, True):
            # There was a gadget in the way. Go back to the top
            continue
        
        neighbors = list(neighbor_set)

        #TODO: Check if this is needed. In theory, if this is not done it could lead to cnots which are not allowed by the architecture
        frontier_with_removed = {i: -1 for i in range(len(outputs))}
        frontier_with_removed.update(frontier)

        m = bi_adj(graph, neighbors, frontier_with_removed.values())
        if all(sum(row) != 1 for row in m.data):  # No easy vertex
            
            if use_gate_mapping:
                rerouted_cnot_list = get_all_cnot_operations(graph, frontier, neighbors, architecture_copy)
            else:
                rerouted_cnot_list = get_cnot_row_operations(graph, frontier, neighbors, architecture_copy)
                if rerouted_cnot_list:
                    rerouted_cnot_list = [rerouted_cnot_list]

            if not rerouted_cnot_list:  # No CNOTs found
                if not eliminate_yz_spider(graph, frontier, neighbors, circuit):
                    raise Exception("Extraction failed")
                rerouted_cnot_list = [[]]

            # We now have a set of CNOTs that suffice to extract at least one vertex.
        else:
            if not quiet: print("Simple vertex")
            rerouted_cnot_list = [[]]


        cnot_data = {"circuits": [], "graphs": [], "frontier": [], "matrix": [], "neighbors": []}


        for cnot_list in rerouted_cnot_list:
            if use_gate_mapping:
                apply_cnots_to_circuit(graph, frontier, neighbors, m, cnot_data, cnot_list)
                if cnot_list and all(rerouted_cnot.is_gate_rerouted() for rerouted_cnot in cnot_list):
                    basic_cnots = [ReroutedGate(rerouted_cnot.basic_gate, [rerouted_cnot.basic_gate]) for rerouted_cnot in cnot_list]
                    apply_cnots_to_circuit(graph, frontier, neighbors, m, cnot_data, basic_cnots)
            else:
                if len(rerouted_cnot_list) > 1:
                    raise ValueError("Multiple CNOTs not supported without gate mapping")
                circuit, extracted = apply_cnots(graph, circuit, frontier, cnot_list, m, neighbors, inverse=True)
                if not quiet: print("Vertices extracted:", extracted)

        if use_gate_mapping:
            architecture_new, circuit_index = gate_mapper(mapper, cnot_data["circuits"])

            gate_names = [f"{gate[0].name}: {gate[0].params}" for gate in cnot_data["circuits"][circuit_index].data] 
            ctest, new_arch = get_circuit_from_mapper(mapper, get_exact_phases=False)

            circuit = QuantumCircuit(len(outputs))
            graph = cnot_data["graphs"][circuit_index]
            frontier = cnot_data["frontier"][circuit_index]
            m = cnot_data["matrix"][circuit_index]
            neighbors = cnot_data["neighbors"][circuit_index]

            if architecture_new.qubit_map != architecture_copy.qubit_map:
                architecture_copy = architecture_new

            if not quiet and len(rerouted_cnot_list[circuit_index])>0: print(f"      Cnot elimination with {rerouted_cnot_list[circuit_index]} CNOTs")

    # for shuttle in reversed(shuttle_list):
    #     c.add_gate(shuttle)
    # Outside of loop. Finish up the permutation
    id_simp(graph, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates

    if use_gate_mapping:
        circuit, architecture_copy = get_circuit_from_mapper(mapper, get_exact_phases=False)

    circuit.gates = list(reversed(circuit.gates))
    return graph_to_swaps(graph, up_to_perm) + circuit, architecture_copy

def apply_cnots_to_circuit(graph, frontier, neighbors, m, cnot_data, cnot_list):
    graph_copy = graph.clone()
    frontier_copy = frontier.copy()
    m_copy = m.copy()
    neighbors_copy = neighbors.copy()
    c, extracted = apply_cnots(graph_copy, None, frontier_copy, cnot_list, m_copy, neighbors_copy, inverse=True)
    cnot_data["circuits"].append(c)
    cnot_data["graphs"].append(graph_copy)
    cnot_data["frontier"].append(frontier_copy)
    cnot_data["matrix"].append(m_copy)
    cnot_data["neighbors"].append(neighbors_copy)




def clean_frontier(g: BaseGraph[VT, ET], circuit: Circuit|QuantumCircuit, frontier: Dict[int, VT], optimize_czs: bool = True) -> int:
    """Remove single qubit gates from the frontier and any CZs between the vertices in the frontier
    Returns the number of CZs saved if `optimize_czs` is True; otherwise returns 0"""
    phases = g.phases()
    czs_saved = 0
    outputs = g.outputs()
    for qubit, vertex in frontier.copy().items():  # First removing single qubit gates
        b = [w for w in g.neighbors(vertex) if w in outputs][0]
        e = g.edge(vertex, b)
        if g.edge_type(e) == EdgeType.HADAMARD:
            add_gate_to_circuit(circuit, HAD(qubit))
            g.set_edge_type(e, EdgeType.SIMPLE)
        if phases[vertex]:
            add_gate_to_circuit(circuit, ZPhase(qubit, phases[vertex]))
            g.set_phase(vertex, 0)

    # And now on to CZ gates
    cz_mat = Mat2([[0 for i in range(len(outputs))] for j in range(len(outputs))])
    for vertex in frontier.values():
        for w in list(g.neighbors(vertex)):
            if w in frontier.values():
                vertex_qubit = list(frontier.keys())[list(frontier.values()).index(vertex)]
                w_qubit = list(frontier.keys())[list(frontier.values()).index(w)]
                cz_mat.data[vertex_qubit][w_qubit] = 1
                cz_mat.data[w_qubit][vertex_qubit] = 1
                g.remove_edge(g.edge(vertex, w))

    if optimize_czs:
        overlap_data = max_overlap(cz_mat)
        while len(overlap_data[1]) > 2:  # there are enough common qubits to be worth optimizing
            i, j = overlap_data[0][0], overlap_data[0][1]
            czs_saved += len(overlap_data[1]) - 2
            add_gate_to_circuit(circuit, CNOT(i,j))
            for qb in overlap_data[1]:
                add_gate_to_circuit(circuit, CZ(j, qb))
                cz_mat.data[i][qb] = 0
                cz_mat.data[j][qb] = 0
                cz_mat.data[qb][i] = 0
                cz_mat.data[qb][j] = 0
            add_gate_to_circuit(circuit, CNOT(i,j))
            overlap_data = max_overlap(cz_mat)

    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if cz_mat.data[i][j] == 1:
                add_gate_to_circuit(circuit, CZ(i,j))

    return czs_saved