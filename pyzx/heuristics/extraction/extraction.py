from pathlib import Path
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, ZPhase
from pyzx.extract import graph_to_swaps, max_overlap
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.extraction_base import apply_cnots_to_circuit, apply_frontier_gates_to_circuit, bi_adj, get_all_cnot_operations, get_cnot_row_operations, init_frontier, get_neighbors_of_frontier, remove_gadget, update_graph_for_frontier_neighbor_in_end, apply_gates_to_circuit
from pyzx.heuristics.extraction.extraction_mcp import eliminate_yz_spider
from pyzx.heuristics.extraction.mapper import get_circuit_from_mapper
from pyzx.heuristics.extraction.rerouting import ReroutedGate, build_connection_from_architecture
from pyzx.heuristics.tools import insert_identity
from pyzx.linalg import Mat2
from pyzx.routing.architecture import Architecture
from pyzx.simplify import id_simp
from pyzx.utils import EdgeType
from pyzx.drawing import draw
from pyzx.tensor import compare_tensors
from pyzx.extract import extract_circuit


from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping

def create_na_mapper(config_path: str, num_qubits: int):
    architecture_mapper = NeutralAtomHybridArchitecture(config_path)
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
    params.verbose = False

    # create mapper
    mapper = HybridSynthesisMapper(arch=architecture_mapper) #, params=params

    mapper.init_mapping(6)

    return mapper

def extract_with_mapper(g: BaseGraph[VT, ET], mapper: HybridSynthesisMapper, optimize_czs: bool = True):
    orig_circ = extract_circuit(g.copy())
    circuit, architecture = get_circuit_from_mapper(mapper, get_exact_phases=False) #TODO: exact_phase?
    frontier = init_frontier(g, circuit, False)
    while True:
        draw(g, labels=True, scale=30)
        print("check phase+cz+had frontier is",frontier)
        if not compare_tensors(orig_circ, circuit+extract_circuit(g.copy())):
            print("tensors do not match")
        else:
            print("tensors ok")
        # import pdb
        # pdb.set_trace()
        # First we extract the gates that are in the frontier (single qubit and CZs)
        extracted_gates, czs_saved = extract_frontier_gates(g, frontier, optimize_czs, architecture)
        if extracted_gates:
            g, frontier, circuit, architecture = apply_gates_to_circuit(g, circuit, mapper, frontier, [extracted_gates], apply_gate_function=apply_frontier_gates_to_circuit)
            continue
        
        if not set(g.outputs()).difference(set(frontier.values())):
            break
        print("check gadget")
        # Before CNOT extraction we check if there is a phase gadget in the way
        for v in frontier.values():
            output_neighbor = set(g.outputs()).intersection(set(g.neighbors(v)))
            if output_neighbor:
                insert_identity(g, output_neighbor.pop(), v)

        if remove_gadget(g, frontier):
            # There was a gadget in the way. Go back to the top
            continue
        #prepare frontier (i.e. add identity between output and frontier vertex if they are connected)
        frontier_neighbors = list(get_neighbors_of_frontier(g, frontier))
        # frontier_with_removed = [v for v in frontier.values() if not set(g.neighbors(v)).intersection(g.outputs())]
        # print("check cnot frontier:",frontier.values(),"frontier neighbors:",frontier_neighbors)
        m = bi_adj(g, frontier_neighbors, frontier.values())
        # print("m begin",m)
        # if all(sum(row) != 1 for row in m.data):
        print("have to find cnots")
        # if 33 in frontier_neighbors:
        """
        problem is as follows: We need complete frontier to adress correct qubits in extracted circuit
        desired architecture: one(!) frontier object where all no frontiers have value -1
        for bi_adj we remove those with value -1?
        No, why don't we just eliminate the items from the frontier, 
        the keys should not change, why do they do here?
        It is not the extracted circuit, it is the remaining graph.
        Warum funktioniert draw nicht? liegt es an der pyzx version?
        remaining graph has added 5 2 instead of 4 2
        """
        #     import pdb
        #     pdb.set_trace()
        rerouted_cnot_list = get_all_cnot_operations(g, frontier.values(), frontier_neighbors, architecture)
        if not rerouted_cnot_list:
            print("fatal: no cnots found")
            return

        g_orig = g.copy()
        circuit_orig = circuit.copy()
        g, frontier, circuit, architecture = apply_gates_to_circuit(g, circuit, mapper, frontier, rerouted_cnot_list, apply_gate_function=apply_cnots_to_circuit)
        if not compare_tensors(orig_circ, circuit+extract_circuit(g.copy())):
            print("after cnots tensors do not match")
            import pdb
            pdb.set_trace()
        else:
            print("after cnots tensors ok")
        # if 33 in frontier_neighbors:
        #     import pdb
        #     pdb.set_trace()

    id_simp(g, quiet=True)

    circuit, architecture = get_circuit_from_mapper(mapper, get_exact_phases=False)
    return circuit + graph_to_swaps(g, False), architecture #TODO: Last swaps are not architecture aware but could be arranged via qubit reordering? missing hadamards should be removed already?


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
    If an architecture is provided, the extraction will try to create a circuit that is compatible with the architecture.
    This function implements a more optimized version of the algorithm described in
    `There and back again: A circuit extraction tale <https://arxiv.org/abs/2003.01664>`_

    Args:
        g: The ZX-diagram graph to be extracted into a Circuit.
        architecture: The architecture to be used for the extraction.
        use_gate_mapping: Whether to use the mapper for the extraction.
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
    init_graph = graph.clone()
    
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
        params.verbose = False

        # create mapper
        mapper = HybridSynthesisMapper(arch=architecture_mapper, params=params)

        mapper.init_mapping(graph.num_inputs(), InitialCircuitMapping.identity)
    else:   
        circuit = Circuit(len(outputs))
        mapper = None

    frontier = init_frontier(graph, circuit, True)
    architecture_copy = Architecture(name=architecture.name, coupling_graph=architecture.graph.copy(), qubit_map=list(range(len(frontier))))
    
    it_step = 0
    frontier_neighbors = []

    if mapper:
        # Get initial architecture from mapper
        _, architecture_copy = get_circuit_from_mapper(mapper, get_exact_phases=False)
    
    while True:
        # First we extract the gates that are in the frontier (single qubit and CZs)
        frontier_gates, czs_saved = extract_frontier_gates(graph, frontier, optimize_czs, None)

        # Apply the gates to the circuit
        # If the mapper changed the architecture, we need to update it. If not mapper is used, the architecture is not changed
        graph, frontier, frontier_neighbors, circuit, architecture_new = apply_gates_to_circuit(graph, circuit, mapper, frontier, frontier_neighbors, [frontier_gates], inverse=True, apply_gate_function=apply_frontier_gates_to_circuit)
        if architecture_new:
            architecture_copy = architecture_new
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = get_neighbors_of_frontier(graph, frontier, inverse=True)
        new_vertices = update_graph_for_frontier_neighbor_in_end(graph, frontier, True)
        if new_vertices:
            neighbor_set.update(new_vertices)
   
        if not frontier:
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(graph, frontier, True):
            # There was a gadget in the way. Go back to the top
            continue
        
        frontier_neighbors = list(neighbor_set)

        #TODO: Check if this is needed. In theory, if this is not done it could lead to cnots which are not allowed by the architecture
        frontier_with_removed = {i: -1 for i in range(len(outputs))}
        frontier_with_removed.update(frontier)

        m = bi_adj(graph, frontier_neighbors, frontier_with_removed.values())
        if all(sum(row) != 1 for row in m.data):  # No easy vertex
            
            if use_gate_mapping:
                rerouted_cnot_list = get_all_cnot_operations(graph, frontier, frontier_neighbors, architecture_copy)
            else:
                rerouted_cnot_list = get_cnot_row_operations(graph, frontier, frontier_neighbors, architecture_copy)
                if rerouted_cnot_list:
                    rerouted_cnot_list = [rerouted_cnot_list]

            if not rerouted_cnot_list:  # No CNOTs found
                #TODO: This should not be needed since remove_gadget can remove xz and yz gadgets
                if not eliminate_yz_spider(graph, frontier, frontier_neighbors, circuit):
                    raise Exception("Extraction failed")
                rerouted_cnot_list = [[]]

            # We now have a set of CNOTs that suffice to extract at least one vertex.
        else:
            if not quiet: print("Simple vertex")
            rerouted_cnot_list = [[]]

        graph, frontier, frontier_neighbors, circuit, architecture_new = apply_gates_to_circuit(graph, circuit, mapper, frontier, frontier_neighbors, rerouted_cnot_list, inverse=True, apply_gate_function=apply_cnots_to_circuit)
        if architecture_new:
            architecture_copy = architecture_new


        # if use_gate_mapping:
        #     circuit_after_step,_ = get_circuit_from_mapper(mapper)
        #     fig_name = f"full_graph_mapper_{it_step}.png"
        # else:
        #     circuit_after_step = circuit.copy()
        #     fig_name = f"full_graph_{it_step}.png"

        # graph_after_step = get_full_graph_from_partial_graph_and_circuit(graph.clone(), circuit_after_step, False)
        # draw_matplotlib(graph_after_step, figsize=(16,4)).savefig(fig_name)
        it_step += 1

        # assert compare_tensors(graph_after_step, init_graph)
            
    # Outside of loop. Finish up the permutation
    id_simp(graph, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates

    if use_gate_mapping:
        circuit, architecture_copy = get_circuit_from_mapper(mapper, get_exact_phases=False)

    circuit.gates = list(reversed(circuit.gates))
    return graph_to_swaps(graph, up_to_perm) + circuit, architecture_copy





def extract_frontier_gates(g: BaseGraph[VT, ET], frontier: Dict[int, VT], optimize_czs: bool = True, architecture:Architecture|None = None) -> Tuple[List[ReroutedGate], int]:
    """Remove single qubit gates from the frontier and any CZs between the vertices in the frontier
    Returns the extracted gates and the number of CZs saved if `optimize_czs` is True; otherwise returns 0"""
    single_qubit_gates = get_single_qubit_gates(g, frontier)
    two_qubit_gates, czs_saved = get_two_qubit_gates(g, frontier, optimize_czs, architecture)

    return single_qubit_gates+two_qubit_gates, czs_saved

def get_two_qubit_gates(graph:BaseGraph, frontier:Dict[int,VT], optimize_czs:bool, architecture:Architecture|None = None) -> Tuple[List[ReroutedGate], int]:
    """Extracts two qubit gates from the frontier and removes them from the graph.
    Returns a list of the extracted two qubit gates."""

    outputs = graph.inputs()

    czs_saved = 0
    two_qubit_gates = []
    
    cz_mat = Mat2([[0 for i in range(len(outputs))] for j in range(len(outputs))])
    for vertex in frontier.values():
        # if vertex in graph.outputs():
        #     continue
        for w in list(graph.neighbors(vertex)):
            if w in frontier.values():
                vertex_qubit = list(frontier.keys())[list(frontier.values()).index(vertex)]
                w_qubit = list(frontier.keys())[list(frontier.values()).index(w)]
                cz_mat.data[vertex_qubit][w_qubit] = 1
                cz_mat.data[w_qubit][vertex_qubit] = 1
                graph.remove_edge(graph.edge(vertex, w))

    if optimize_czs:
        overlap_data = max_overlap(cz_mat)
        while len(overlap_data[1]) > 2:  # there are enough common qubits to be worth optimizing
            i, j = overlap_data[0][0], overlap_data[0][1]
            czs_saved += len(overlap_data[1]) - 2
            if architecture:
                two_qubit_gates.append(ReroutedGate(CNOT(i,j), build_connection_from_architecture(architecture, CNOT(i,j))))
            else:
                two_qubit_gates.append(ReroutedGate(CNOT(i,j), None))

            for qb in overlap_data[1]:
                if architecture:
                    two_qubit_gates.append(ReroutedGate(CZ(j, qb), build_connection_from_architecture(architecture, CZ(j, qb))))
                else:
                    two_qubit_gates.append(ReroutedGate(CZ(j, qb), None))

                cz_mat.data[i][qb] = 0
                cz_mat.data[j][qb] = 0
                cz_mat.data[qb][i] = 0
                cz_mat.data[qb][j] = 0

            if architecture:
                two_qubit_gates.append(ReroutedGate(CNOT(i,j), build_connection_from_architecture(architecture, CNOT(i,j))))
            else:
                two_qubit_gates.append(ReroutedGate(CNOT(i,j), None))

            overlap_data = max_overlap(cz_mat)
    
    print("architecture is",architecture)
    draw(architecture.graph, labels=True)
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if cz_mat.data[i][j] == 1:
                if architecture:
                    two_qubit_gates.append(ReroutedGate(CZ(i,j), build_connection_from_architecture(architecture, CZ(i,j))))
                else:
                    two_qubit_gates.append(ReroutedGate(CZ(i,j), None))

    return two_qubit_gates, czs_saved

def get_single_qubit_gates(graph:BaseGraph, frontier:Dict[int, VT], reverse = False) -> List[ReroutedGate]:
    """Extracts single qubit gates from the frontier and removes them from the graph.
    Returns a list of the extracted single qubit gates."""
    
    phases = graph.phases()
    if reverse:
        outputs = graph.outputs()
    else:
        outputs = graph.inputs()
    single_qubit_gates = []
    for qubit, vertex in frontier.items():  # First removing single qubit gates
        if vertex in graph.outputs():
            continue
        b = [w for w in graph.neighbors(vertex) if w in outputs][0]
        e = graph.edge(vertex, b)
        if graph.edge_type(e) == EdgeType.HADAMARD:
            single_qubit_gates.append(ReroutedGate(HAD(qubit), None))
            graph.set_edge_type(e, EdgeType.SIMPLE)
        if phases[vertex]:
            single_qubit_gates.append(ReroutedGate(ZPhase(qubit, phases[vertex]), None))
            graph.set_phase(vertex, 0)
        neighbors = list(graph.neighbors(vertex))
        if len(neighbors) == 2:
            hcount = sum([graph.edge_type(graph.edge(vertex, neighbor)) == EdgeType.HADAMARD for neighbor in neighbors])
            graph.remove_vertex(vertex)
            if hcount % 2 == 1:
                single_qubit_gates.append(ReroutedGate(HAD(qubit), None))
            graph.add_edge(graph.edge(neighbors[0],neighbors[1]), EdgeType.SIMPLE)
            # print("rem vert",vertex,"add edge",neighbors[0],neighbors[1])
            # if set(graph.outputs()).intersection(set(neighbors)):
            #     print("remove ",qubit,"from frontier")
            #     frontier[qubit] = -1
            # else:
            frontier[qubit] = neighbors[0] if neighbors[1] in graph.inputs() else neighbors[1]
        

    return single_qubit_gates