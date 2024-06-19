import itertools

from pathlib import Path
from typing import Dict, List, Set, Tuple
from fractions import Fraction

from qiskit import QuantumCircuit


from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, ZPhase
from pyzx.extract import connectivity_from_biadj, graph_to_swaps
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.extraction_base import MCP, add_gate_to_circuit, bi_adj, complement_neighbors, eliminate_unary_phase_gadgets, extract_cnots, extract_czs, extract_rzs, get_all_cnot_operations, get_cnot_row_operations, get_frontier_gadget_dict, init_frontier, get_neighbors_of_frontier, update_graph_for_frontier_neighbor_in_end
from pyzx.heuristics.extraction.mapper import gate_mapper, get_circuit_from_mapper
from pyzx.heuristics.extraction.rerouting import ReroutedGate, build_connection_from_architecture
from pyzx.heuristics.tools import insert_identity
from pyzx.routing.architecture import Architecture
from pyzx.utils import EdgeType, VertexType

from mqt.qmap import HybridSynthesisMapper, NeutralAtomHybridArchitecture, HybridMapperParameters, InitialCoordinateMapping, InitialCircuitMapping


def mcp_aware_extract(
        graph: BaseGraph[VT, ET], 
        allow_insertions: bool = False, 
        cz_optimize: bool = False, 
        architecture: Architecture = None,
        use_gate_mapping: bool = False
        ) -> Tuple[Circuit, Architecture] | None:
    """
    Extracts a ZX-diagram to circuit with gateset H,CZ,RZ and CnP (=MCP). 
    phase gadgets are extracted as MCP gates whenever possible. 
    allow_insertions specifies whether every phase gadget should be extracted as (M)CP gate 
    (by inserting missing phase gadgets to complete mcp graph structure) 
    or only those phase gadgets already present in the diagram should be considered 
    (therefore we may need to resolve some phase gadgets with degree > 2 via pivoting resulting in higher CZ,H counts)
    """
    assert(graph.num_inputs()==graph.num_outputs())
    circuit = Circuit(qubit_amount=graph.num_inputs())
    init_graph = graph.clone()

    if use_gate_mapping:
        circuit = QuantumCircuit(graph.num_inputs())
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

        mapper.init_mapping(graph.num_inputs(), InitialCircuitMapping.identity)
    else:   
        circuit = Circuit(graph.num_inputs())
   
    frontier = init_frontier(graph, circuit)

    if architecture:
        architecture_copy = Architecture(name=architecture.name, coupling_graph=architecture.graph.copy(), qubit_map=list(range(len(frontier))))
    else:
        architecture_copy = None
    
    it_step = 0
    #iterative process
    while True:

        #CZ extraction + MCP extraction
        cz_gates = extract_czs(graph, frontier, circuit, None, cz_optimize)
        mcp_gates = extract_mcp(graph, frontier, circuit, None, allow_insertions)
        #Phase + Hadamard extraction
        rz_gates = extract_rzs(graph, frontier, circuit)

        if use_gate_mapping and len(circuit.data) > 0:
            gate_names = [f"{gate[0].name}: {gate[0].params}" for gate in circuit.data] 
            # If we have extracted some CZ gates, we need to add them to the circuit
            architecture_copy, circuit_index = gate_mapper(mapper, circuit)

            circuit = QuantumCircuit(len(graph.inputs()))


        if frontier and not (rz_gates or mcp_gates or cz_gates):
            frontier_neighbors = list(get_neighbors_of_frontier(graph, frontier))
            new_vertices = update_graph_for_frontier_neighbor_in_end(graph, frontier, False)
            if new_vertices:
                frontier_neighbors.extend(new_vertices)

            if use_gate_mapping:
                rerouted_cnot_list = get_all_cnot_operations(graph, frontier, frontier_neighbors, architecture=architecture_copy)
            else:
                rerouted_cnot_list = get_cnot_row_operations(graph, frontier, frontier_neighbors, architecture=architecture_copy)
                if rerouted_cnot_list:
                    rerouted_cnot_list = [rerouted_cnot_list]

            if not rerouted_cnot_list:  # No CNOTs found
                if not eliminate_yz_spider(graph, frontier, frontier_neighbors, circuit):
                    raise Exception("Extraction failed")
                    # import pdb
                    # pdb.set_trace()
            else:
                cnot_data = {"circuits": [], "graphs": [], "frontier": [], "neighbors": []}
                for cnot_list in rerouted_cnot_list:
                    if use_gate_mapping:
                        graph_copy = graph.clone()
                        frontier_copy = frontier.copy()
                        neighbors_copy = frontier_neighbors.copy()
                        c = apply_cnots(graph_copy, None, frontier_copy, cnot_list, neighbors_copy)
                        cnot_data["circuits"].append(c)
                        cnot_data["graphs"].append(graph_copy)
                        cnot_data["frontier"].append(frontier_copy)
                        cnot_data["neighbors"].append(neighbors_copy)
                    else:
                        if len(rerouted_cnot_list) > 1:
                            raise ValueError("Multiple CNOTs not supported without gate mapping")
                        circuit = apply_cnots(graph, circuit, frontier, cnot_list, frontier_neighbors)

                if use_gate_mapping:
                    architecture_copy, circuit_index = gate_mapper(mapper, cnot_data["circuits"])
                    circuit = QuantumCircuit(len(graph.inputs()))
                    graph = cnot_data["graphs"][circuit_index]
                    frontier = cnot_data["frontier"][circuit_index]
                    frontier_neighbors = cnot_data["neighbors"][circuit_index]

                    if len(rerouted_cnot_list[circuit_index])>0: print(f"      Cnot elimination with {rerouted_cnot_list[circuit_index]} CNOTs")

        # if use_gate_mapping:
        #     circuit_after_step,_ = get_circuit_from_mapper(mapper)
        # else:
        #     circuit_after_step = circuit.copy()

        # graph_after_step = get_full_graph_from_partial_graph_and_circuit(graph.clone(), circuit_after_step, False)

        # draw_matplotlib(graph_after_step).savefig(f"full_graph_{it_step}.png")

        # assert compare_tensors(graph_after_step, init_graph)

        it_step += 1
                
        if graph.num_vertices() == graph.num_inputs() + graph.num_outputs():
            try:
                if use_gate_mapping:
                    circuit, architecture_copy = get_circuit_from_mapper(mapper)
                
                return circuit + graph_to_swaps(graph), architecture_copy
            except:
                print("extraction failed")
                # import pdb
                # pdb.set_trace()
                return None
            

def apply_cnots(g:BaseGraph, circuit:Circuit|QuantumCircuit, frontier:Dict[int, VT], rerouted_cnots: List[ReroutedGate], frontier_neighbors:List[VT]):
    #CNOT extraction
    frontier_with_removed = {i: -1 for i in range(len(g.inputs()))}
    frontier_with_removed.update(frontier)

    basic_cnots = [rerouted_gate.basic_gate for rerouted_gate in rerouted_cnots]
    
    m = bi_adj(g, frontier_neighbors, list(frontier_with_removed.values()))
    m_copy = m.copy()
    for cnot in basic_cnots:
        m.row_add(cnot.control, cnot.target)

    if all(sum(row) != 1 for row in m.data):
        raise Exception("CNOTs do not suffice to extract a vertex")
    
    start = g.inputs()

    neighbors_copy = frontier_neighbors.copy()
    for _ in range(len(frontier) - len(frontier_neighbors)):
        neighbors_copy.append(-1)

    for rerouted_cnot in rerouted_cnots:
        rerouted_cnot.reverse_gate_qubits()

    cnots = sum(rerouted_cnots, [])

    if circuit is None:
        circuit = QuantumCircuit(len(frontier_with_removed))

    if len(cnots) > 0:
        connectivity_from_biadj(g, m, neighbors_copy, list(frontier_with_removed.values()))

    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            qubit: int = list(frontier_with_removed)[i]
            v = frontier_with_removed[qubit]
            w = frontier_neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[qubit] = (v, w)
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")
    hads = []

    for qubit, (v, w) in good_verts.items():  # Update frontier vertices
        hads.append(qubit)
        # c.add_gate("HAD",qubit_map[v])
        try:
            b = [o for o in g.neighbors(v) if o in start][0]
        except:
            pass
        g.remove_vertex(v)
        g.add_edge(g.edge(w, b))
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

    return circuit


def apply_cnots2(g:BaseGraph, circuit:Circuit|QuantumCircuit, frontier:Dict[int, VT], rerouted_cnots: List[ReroutedGate], frontier_neighbors:List[VT]):
    #CNOT extraction
    frontier_with_removed = {i: -1 for i in range(len(g.inputs()))}
    frontier_with_removed.update(frontier)

    cnots = [rerouted_gate.basic_gate for rerouted_gate in rerouted_cnots]
    
    m = bi_adj(g, frontier_neighbors, list(frontier_with_removed.values()))
    m_copy = m.copy()
    for cnot in cnots:
        m.row_add(cnot.control, cnot.target)

    if all(sum(row) != 1 for row in m.data):
        raise Exception("CNOTs do not suffice to extract a vertex")
    
    if circuit is None:
        circuit = QuantumCircuit(len(frontier_with_removed))

    for rerouted_cnot in rerouted_cnots:
        rerouted_cnot.reverse_gate_qubits()
    
    # apply_cnots(g, c, frontier, cnots, m, frontier_neighbors)
    extract_cnots(g, frontier, circuit, rerouted_cnots)
    #eliminate possible unary phase gadgets (this happens since we do not immediately remove every YZ spider connected to the frontier)
    eliminate_unary_phase_gadgets(g, frontier)
    #Repeat rz extraction, in case unary phase gadget elimination created a new phase on the frontier
    rz_gates = extract_rzs(g, frontier, circuit)

    return circuit


def get_maximal_mcp(g: BaseGraph, frontier: Dict[int,VT]):
    """Given a graph and a frontier set, this method finds the largest set of (frontier neighbor) phase gadgets 
    which we can extract as a single multi controlled phase gate.
    returns: dict of gadgets grouped by vertex degree"""    
    gadget_dict = get_frontier_gadget_dict(g, frontier) 
    mcp = dict()
    if not gadget_dict or len(gadget_dict.keys()) != max(gadget_dict.keys())-1:
        return mcp
    #check for mcp starting from the highest degree gadgets
    for degree in range(max(gadget_dict.keys()), 1, -1):
        for max_gadget_root, max_top, max_gadget_neighbors in gadget_dict[degree]:
            max_root_gadget = dict()
            valid_root_gadget = True
            # choose a gadget and search for corresponding sub gadgets
            for lower_degree in range(degree-1, 1, -1):
                current_gadgets = []
                permutations = list(itertools.combinations(max_gadget_neighbors, lower_degree))

                #generate permutations, we need gadgets whose connectivity is as specified by the permutations
                for permutation in permutations:
                    for root, top, neighbors_in_frontier in gadget_dict[lower_degree]:
                        #search for phase gadgets which has the same neighbors as given by the permutation
                        if neighbors_in_frontier == set(permutation):
                            current_gadgets.append((root, top, neighbors_in_frontier))
                            break
                                  
                if len(current_gadgets) == len(list(permutations)):
                    # on this level there exist all phase gadgets required
                    max_root_gadget[lower_degree] = current_gadgets
                else:
                    # start with another max gadget root
                    valid_root_gadget = False
                    break
            
            if valid_root_gadget:
                mcp = max_root_gadget
                mcp[degree] = (max_gadget_root, max_top, max_gadget_neighbors)
                break
        
        if mcp:
            # if there is a mcp for this degree, we extract it, otherwise we go to a lower level
            break
    
    return mcp


def gadget_unfusion(g: BaseGraph[VT, ET], root: VT, top: VT, neighbors_in_frontier: List[VT], desired_phase: Fraction):
    """unfuses a gadget by creating a second gadget with the same neighbors. The phases are split up between the two gadgets so that the first has the desired phase
    corresponds to a reverse application of the GF rule in the T-count reduction paper"""
    original_phase = g.phase(top)
    root2 = g.add_vertex(VertexType.Z, g.qubit(root), g.row(root)+0.5, 0)
    top2 = g.add_vertex(VertexType.Z, g.qubit(top), g.row(top)+0.5, original_phase-desired_phase)
    g.add_edge(g.edge(root2,top2),EdgeType.HADAMARD)
    for frontier_vertex in neighbors_in_frontier:
        g.add_edge(g.edge(root2,frontier_vertex),EdgeType.HADAMARD)
    g.set_phase(top,desired_phase)


def frontier_unfusion(g: BaseGraph[VT, ET], frontier_vertex: VT, frontier: Dict[int,VT], circuit: Circuit|QuantumCircuit, desired_phase: Fraction):
    """unfuses the phase of a frontier vertex as ZPhase gate on the circuit, such that in the frontier vertex the desired phase remains"""
    qubit = [qubit for qubit, vertex in frontier.items() if vertex == frontier_vertex][0]
    gate = ZPhase(qubit, g.phase(frontier_vertex)-desired_phase)
    # gate = U3(qubit, 0, 0, g.phase(frontier_vertex)-desired_phase)
    add_gate_to_circuit(circuit, gate)
    g.set_phase(frontier_vertex, desired_phase)
    return True


def insert_z_gadget(g: BaseGraph[VT, ET], neighbors: List[VT]):
    """inserts empty z gadget into graph; this allows for adding the missing graph structures to extract higher order mcp gates"""
    row = g.row(neighbors[0])
    root = g.add_vertex(VertexType.Z, -1, row)
    top = g.add_vertex(VertexType.Z, -2, row)
    g.add_edge(g.edge(root,top), EdgeType.HADAMARD)
    for neighbor in neighbors:
        g.add_edge(g.edge(neighbor, root), EdgeType.HADAMARD)
    return (root, top, neighbors)


def complete_mcp_structure(g: BaseGraph[VT, ET], gadget: Tuple[VT,VT,List[VT]], gadget_dict: Dict[int,Tuple[VT,VT,List[VT]]]):
    """given a gadget searches all sub gadgets required to extract mcp or inserts the missing sub gadgets"""
    _gadget_root, _gadget_top, gadget_neighbors = gadget
    mcp_dict = {len(gadget_neighbors): gadget}
    for degree in range(len(gadget_neighbors)-1, 1, -1):
        permutations = list(itertools.combinations(gadget_neighbors, degree))

        #generate permutations, we need gadgets whose connectivity is as specified by the permutations
        for permutation in permutations:
            current_gadget = None
            if degree in gadget_dict.keys():
                for root, top, neighbors_in_frontier in gadget_dict[degree]:
                    #search for phase gadgets which has the same neighbors as given by the permutation
                    if neighbors_in_frontier == set(permutation):
                        current_gadget = (root, top, neighbors_in_frontier)
                        break
            
            if not current_gadget:
                current_gadget = insert_z_gadget(g, permutation)
            
            mcp_dict.setdefault(degree,[]).append(current_gadget)
        
    return mcp_dict


def construct_maximal_mcp(g: BaseGraph[VT, ET], frontier: Dict[int,VT]):
    """returns mcp for gadget with maximum degree; inserts missing sub gadgets"""
    gadget_dict = get_frontier_gadget_dict(g, frontier)
    if not gadget_dict:
        return dict()
    max_degree = max(gadget_dict.keys())
    return complete_mcp_structure(g, gadget_dict[max_degree][0], gadget_dict)


def extract_mcp(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit|QuantumCircuit, architecture:Architecture, allow_insertions: bool = False): 
    """extracts a single (multi or normal) controlled phase gate from the diagram"""
    if allow_insertions:
        mcp = construct_maximal_mcp(g, frontier)
    else:
        mcp = get_maximal_mcp(g, frontier)
    
    if mcp:
        # for now we take the phase of the gadget with highest degree as the desired phase of the mcp, since unfusion here is costly
        # may not be optimal though
        max_degree = max(mcp.keys())
        odd_phase = g.phase(mcp[max_degree][1]) if max_degree % 2 == 1 else -g.phase(mcp[max_degree][1])
        even_phase = -odd_phase
        g.remove_vertices([mcp[max_degree][0],mcp[max_degree][1]])

        for degree in range(max_degree-1,1,-1):
            for root, top, neighbors_in_frontier in mcp[degree]:
                desired_phase = odd_phase if degree % 2 == 1 else even_phase
                if g.phase(top) != desired_phase % Fraction(2,1):
                    gadget_unfusion(g, root,top,neighbors_in_frontier,desired_phase)

                g.remove_vertices([root,top])

        for frontier_vertex in mcp[max_degree][2]:
            if g.phase(frontier_vertex) != odd_phase:
                frontier_unfusion(g, frontier_vertex, frontier, circuit, odd_phase)

            g.set_phase(frontier_vertex,0)
        
        mcp_phase = odd_phase*(2**(max_degree-1)) % Fraction(2,1)
        # sometimes spider phases are too large and sum up to a mcp phase rotation which is a multiple of 2pi. 
        # We can simply remove those vertices and not extract a gate since they implement the identity
        if mcp_phase != 0:
            qubits = [qubit for qubit, vertex in frontier.items() if vertex in mcp[max_degree][2]]
            gate = MCP(qubits[:-1],qubits[-1],mcp_phase) #target qubit is the last one, but doesn't matter since mcps are symmetric
            if architecture:
                rerouted_gates = build_connection_from_architecture(architecture, gate)
                for rerouted_gate in rerouted_gates:
                    add_gate_to_circuit(circuit, rerouted_gate)
            else:
                add_gate_to_circuit(circuit, gate)
        return True
    else:
        return False



def eliminate_yz_spider(g: BaseGraph[VT,ET], frontier: Dict[int,VT], frontier_neighbors: Set, circuit: Circuit|QuantumCircuit, inverse: bool = False) -> bool:
    """Finds a YZ measured spider which is connected to a spider in frontier and applies a pivot on them. 
    By that, the YZ spider transforms to a XY spider"""

    start = g.inputs() if not inverse else g.outputs()
    
    for n in frontier_neighbors:
        for candidate in g.neighbors(n):
            candidate_neighbors = [neighbor for neighbor in g.neighbors(candidate) if neighbor not in start]
            if len(candidate_neighbors) == 1 and g.type(candidate) == VertexType.Z:
                frontier_vertex = list(set(g.neighbors(n)).intersection(set(frontier.values())))[0]
                frontier_vertex_qubit = list(frontier.keys())[list(frontier.values()).index(frontier_vertex)]
                frontier_vertex_neighbors = [neighbor for neighbor in g.neighbors(frontier_vertex) if neighbor not in start]
                for neighbor in frontier_vertex_neighbors:
                    # special case if frontier vertex is connected to an output
                    if g.type(neighbor) == VertexType.BOUNDARY:
                        new_vertex = insert_identity(g, frontier_vertex, neighbor)
                        frontier[frontier_vertex_qubit] = new_vertex
                        break

                pivot_mcp(g, n, frontier_vertex)
                frontier_vertex_neighbors = [neighbor for neighbor in g.neighbors(frontier_vertex) if neighbor not in start]
                for shared_neighbor in set(g.neighbors(n)).intersection(set(frontier_vertex_neighbors)):
                    g.set_phase(shared_neighbor,g.phase(shared_neighbor)+Fraction(1,1))
                g.set_phase(n, g.phase(n)+g.phase(candidate))
                g.remove_vertex(candidate)

                add_gate_to_circuit(circuit, HAD(frontier_vertex_qubit))      

                return True
    return False


def pivot_mcp(g: BaseGraph[VT,ET], u: VT, v: VT) -> bool:
    """Graph theoretic pivot on graph-like diagram
    g: A graph instance
    u: First vertex
    v: Second vertex"""
    if not lcomp_mcp(g, u):
        return False
    success = lcomp_mcp(g, v)
    lcomp_mcp(g, u)
    return success


def lcomp_mcp(g: BaseGraph[VT,ET], u: VT):
    """graph theoretic local complementation on graph-like diagram"""
    vn = [neighbor for neighbor in g.neighbors(u) if len(g.neighbors(neighbor)) > 1]
    complement_neighbors(g, vn)
    return True