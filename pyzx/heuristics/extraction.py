from fractions import Fraction
import itertools
import logging
import math
import random
from typing import Dict, List, Optional, Set, Tuple
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, Gate, ZPhase
from pyzx.extract import bi_adj, clean_frontier, column_optimal_swap, connectivity_from_biadj, filter_duplicate_cnots, graph_to_swaps, greedy_reduction, neighbors_of_frontier, remove_gadget
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.linalg import CNOTMaker, Mat2
from pyzx.routing.architecture import Architecture
from pyzx.routing.cnot_mapper import ElimMode, gauss
from pyzx.simplify import id_simp
from pyzx.utils import EdgeType, FractionLike, VertexType

def build_architecture_from_frontier(g: BaseGraph[VT, ET], architecture: Architecture, frontier: Dict[int,VT] | List[int], qubit_map: List[int] | None = None) -> Architecture:
    
    if isinstance(frontier, list):
        if qubit_map is None:
            raise ValueError("qubit_map must be provided when frontier is a list")
        frontier_dict = {qubit_map[frontier[i]]: frontier[i] for i in range(len(frontier))}
    else:
        frontier_dict = frontier

    architecture_qubit_map = list(frontier_dict.keys())
    # if len(frontier_dict) > 0:
    #     min_qubit = min(architecture_qubit_map)
    #     architecture_qubit_map = [q - min_qubit for q in architecture_qubit_map]

    new_graph = architecture.graph.copy()
    new_edges = []

    for edge in architecture.graph.edges():
        new_graph.remove_edge(edge)
        if edge[0] in architecture_qubit_map and edge[1] in architecture_qubit_map:
            # new_edges.append((architecture_qubit_map.index(edge[0]), architecture_qubit_map.index(edge[1])))
            # new_edges.append((architecture_qubit_map[edge[0]], architecture_qubit_map[edge[1]]))
            new_edges.append((edge[0], edge[1]))

    new_graph.add_edges(new_edges)

    v_list = list(new_graph.vertices())
    for v in v_list:
        if new_graph.vertex_degree(v) == 0:
            new_graph.remove_vertex(v)

    # if len(frontier_dict) < architecture.n_qubits and len(frontier_dict) > 0:
    #     new_graph = architecture.graph.copy()
    #     for i in range(architecture.n_qubits - len(frontier_dict)):
    #         new_graph.remove_vertex(architecture.vertices[-(i+1)])
    #     return Architecture(name=architecture.name, coupling_graph=new_graph, qubit_map=architecture_qubit_map)

    # return Architecture(name=architecture.name, coupling_graph=new_graph, qubit_map=list(range(len(frontier_dict))))
    # return Architecture(name=architecture.name, coupling_graph=new_graph)
    return Architecture(name=architecture.name, coupling_graph=new_graph, qubit_map=architecture_qubit_map)

def extract_architecture_aware_circuit(
        g: BaseGraph[VT, ET],
        architecture: Architecture,
        optimize_czs: bool = True,
        up_to_perm: bool = False,
        output_ordering: List[int] = None,
        quiet: bool = True
        ) -> Circuit:
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
    gadgets = {}
    inputs = g.inputs()
    outputs = g.outputs()
    
    original_graph = g.clone()

    output_list = list(outputs)
    output_list_copy = list(outputs)

    if output_ordering is not None:
        for index, output_mapping in enumerate(output_ordering):
            output_list[index] = output_list_copy[output_mapping]

    c = Circuit(len(outputs))

    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in inputs and v not in outputs:
            n = list(g.neighbors(v))[0]
            gadgets[n] = v
    qubit_map: Dict[VT,int] = dict()
    frontier = []

    if not quiet: print("Outputs:", output_list)

    for i, o in enumerate(output_list):
        v = list(g.neighbors(vertex=o))[0]
        # v = random.choice(vs)
        if v in inputs:
            continue
        frontier.append(v)
        qubit_map[v] = i
    czs_saved = 0

    print(frontier)
    
    while True:
        # preprocessing
        # FIXME: CZ's are not placed according to the architecture
        czs_saved += clean_frontier(g, c, frontier, qubit_map, optimize_czs)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = neighbors_of_frontier(g, frontier)
   
        if not frontier:
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(g, frontier, qubit_map, neighbor_set, gadgets):
            # There was a gadget in the way. Go back to the top
            continue
            
        neighbors = list(neighbor_set)
        m = bi_adj(g, neighbors, frontier)
        if all(sum(row) != 1 for row in m.data):  # No easy vertex

            perm = column_optimal_swap(m)
            perm = {v: k for k, v in perm.items()}
            neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]

           
            m2 = bi_adj(g, neighbors2, frontier)
            m3 = m2.copy()
            m_input = m2.copy()

            # frontier_mapping = [qubit_map[frontier[i]] for i in range(len(frontier))]
            # for index, mapping in enumerate(frontier_mapping):
            #     m3.data[index] = m2.data[mapping]

            # architecture_copy = build_architecture_from_frontier(g, architecture, {i: frontier[i] for i in range(len(frontier))})
            architecture_copy = build_architecture_from_frontier(g, architecture, frontier, qubit_map)

            elim_mode = ElimMode.STEINER_MODE
            # elim_mode = ElimMode.GAUSS_MODE
            # elim_mode = ElimMode.GENETIC_STEINER_MODE
            
            swaps = []

            num_tries = 0
            exception = None
            while True:
                num_tries += 1
                if num_tries > 1:
                    raise exception
                try:
                    cnots, rank = gauss(architecture=architecture_copy, matrix=m3, mode=elim_mode, full_reduce=True)
                    break  # If gauss function works, break the loop
                except Exception as e:
                    # If gauss function fails, randomly swap two columns and try again
                    col1, col2 = random.sample(range(m_input.cols()), 2)
                    swaps.append((col1, col2))
                    m_input.col_swap(col1, col2)
                    m3 = m_input.copy()
                    exception = e
                    # for index, mapping in enumerate(frontier_mapping):
                    #     m3.data[index] = m_input.data[mapping]
                    # if not quiet: print(f"WARNING: Gauss elimination failed. Trying again with swapped columns. Try {num_tries}")

            # try:
            #     cnots, rank = gauss(architecture=architecture_copy, matrix=m3, mode=elim_mode, full_reduce=True)
            # except Exception as e:
            #     if randomize_frontier_tries <= 0:
            #         raise e
            #     if not quiet: print(f"WARNING: Gauss elimination failed. Trying again with randomized frontier. Remaining tries: {randomize_frontier_tries-1}")
            #     return extract_architecture_aware_circuit(g=original_graph, architecture=architecture, optimize_czs=optimize_czs, up_to_perm=up_to_perm, randomize_frontier_tries=randomize_frontier_tries-1, quiet=quiet)

            cnots = filter_duplicate_cnots(cnots)

            # for cnot in cnots:
            #     cnot.control = frontier_mapping[cnot.control]
            #     cnot.target = frontier_mapping[cnot.target]

            neighbors_input = neighbors2[:]
            for swap in swaps:
                neighbors_input[swap[0]], neighbors_input[swap[1]] = neighbors_input[swap[1]], neighbors_input[swap[0]]
            
            m = m_input
            neighbors = neighbors_input

            mapped_cnots = [CNOT(qubit_map[frontier[cnot.target]], qubit_map[frontier[cnot.control]]) for cnot in cnots]
            if not quiet: print(f"Gaussian elimination with {mapped_cnots} CNOTs")
            # We now have a set of CNOTs that suffice to extract at least one vertex.
        else:
            if not quiet: print("Simple vertex")
            cnots = []

        extracted = apply_cnots_with_architecture(g, c, frontier, qubit_map, cnots, m, neighbors)
        if not quiet: print("Vertices extracted:", extracted)
            
    if optimize_czs:
        if not quiet: print("CZ gates saved:", czs_saved)
    # Outside of loop. Finish up the permutation
    id_simp(g, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))
    return graph_to_swaps(g, up_to_perm) + c


def apply_cnots_with_architecture(g: BaseGraph[VT, ET], 
                                  c: Circuit, 
                                  frontier: List[VT], 
                                  qubit_map: Dict[VT, int], 
                                  cnots: List[CNOT], 
                                  m: Mat2, 
                                  neighbors: List[VT]
                                  ) -> int:
    """Adds the list of CNOTs to the circuit, modifying the graph, frontier, and qubit map as needed.
    Returns the number of vertices that end up being extracted"""
    if len(cnots) > 0:
        cnots2 = cnots
        cnots = []
        frontier_map = {qubit_map[frontier[i]]: i for i in range(len(frontier))}
        for cnot in cnots2:
            # Why is this reversed?
            # m.row_add(cnot.target, cnot.control)
            m.row_add(cnot.control, cnot.target)
            cnots.append(CNOT(qubit_map[frontier[cnot.target]], qubit_map[frontier[cnot.control]]))
            # cnots.append(CNOT(qubit_map[frontier[cnot.control]], qubit_map[frontier[cnot.target]]))
        connectivity_from_biadj(g, m, neighbors, frontier)

    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            v = frontier[i]
            w = neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[v] = w
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")
    hads = []
    outputs = g.outputs()
    for v, w in good_verts.items():  # Update frontier vertices
        hads.append(qubit_map[v])
        # c.add_gate("HAD",qubit_map[v])
        qubit_map[w] = qubit_map[v]
        b = [o for o in g.neighbors(v) if o in outputs][0]
        g.remove_vertex(v)
        g.add_edge(g.edge(w, b))
        f_index = frontier.index(v)
        frontier.remove(v)
        frontier.insert(f_index, w)

    for cnot in cnots:
        c.add_gate(cnot)
    for h in hads:
        c.add_gate("HAD", h)

    return len(good_verts)


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


def mcp_aware_extract(
        g: BaseGraph[VT, ET], 
        allow_insertions: bool = False, 
        cz_optimize: bool = False, 
        architecture: Architecture = None,
        randomize_frontier_tries: int = 0,
        ) -> Circuit | None:
    """
    Extracts a ZX-diagram to circuit with gateset H,CZ,RZ and CnP (=MCP). 
    phase gadgets are extracted as MCP gates whenever possible. 
    allow_insertions specifies whether every phase gadget should be extracted as (M)CP gate 
    (by inserting missing phase gadgets to complete mcp graph structure) 
    or only those phase gadgets already present in the diagram should be considered 
    (therefore we may need to resolve some phase gadgets with degree > 2 via pivoting resulting in higher CZ,H counts)
    """
    assert(g.num_inputs()==g.num_outputs())
    c = Circuit(qubit_amount=g.num_inputs())
   
    frontier = init_frontier(g, c, randomize_frontier_tries)
    
    #we proceed iteratively by removing unnecessary edges until wires directly from inputs to outputs remain
    for input in g.inputs():
        neighbor = list(g.neighbors(input))[0]
        if not neighbor in g.outputs():
            g.remove_edge(g.edge(input,neighbor))

    #iterative process
    while True: 
        #CZ extraction + MCP extraction
        cz_gates = extract_czs(g, frontier, c, cz_optimize)
        mcp_gates = extract_mcp(g, frontier, c, allow_insertions)
        #Phase + Hadamard extraction
        rz_gates = extract_rzs(g, frontier, c)
        # If we cannot proceed with H,RZ,CZ,MCP gate extractions remove Hadamard Wires via CNOT row additions using gaussian elimination
        if frontier and not (rz_gates or mcp_gates or cz_gates):
            frontier_neighbors = get_frontier_neighbors(g, frontier)

            if architecture:
                architecture_copy = build_architecture_from_frontier(g, architecture, frontier)
            else:
                architecture_copy = None

            try:
                cnots = get_cnot_row_operations(g, frontier, frontier_neighbors, architecture_copy)
            except Exception as e:
                if randomize_frontier_tries <= 0:
                    raise e
                print(f"WARNING: Gauss elimination failed. Trying again with randomized frontier. Remaining tries: {randomize_frontier_tries-1}")
                return mcp_aware_extract(g, allow_insertions, cz_optimize, architecture, randomize_frontier_tries-1)

            # If there is no row with a single 1 there has to be a YZ measured spider in frontier neighbors which we can eliminate
            # Note that this will only be called if allow_insertions=False, because otherwise each phase gadget gets extracted as (M)CP
            if not cnots:
                if not eliminate_yz_spider(g, frontier, frontier_neighbors, c):
                    print("This should not happen")
                    # import pdb
                    # pdb.set_trace()
            else:
                #CNOT extraction
                extract_cnots(g, frontier, c, cnots)
                #eliminate possible unary phase gadgets (this happens since we do not immediately remove every YZ spider connected to the frontier)
                eliminate_unary_phase_gadgets(g, frontier)
                #Repeat rz extraction, in case unary phase gadget elimination created a new phase on the frontier
                rz_gates = extract_rzs(g, frontier, c)

                
        if g.num_vertices() == g.num_inputs() + g.num_outputs():
            try:
                return c + graph_to_swaps(g)
            except:
                print("extraction failed")
                # import pdb
                # pdb.set_trace()
                return None


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
        ) -> list | List[CNOT]:
    """ Compute row echelon form of adjacency matrix and save row operations as CNOTs 
     -> because of gflow the resulting matrix has a row with only a single 1"""
    m: Mat2 = bi_adj(g, list(frontier_neighbors), frontier.values())
    m2: Mat2 = m.copy()
    elim_mode = ElimMode.STEINER_MODE

    if architecture:
        init_cnots, rank = gauss(architecture=architecture, matrix=m2, mode=elim_mode, full_reduce=True)
        init_cnots = filter_duplicate_cnots(init_cnots)
        for cnot in init_cnots:
            temp = cnot.control
            cnot.control = cnot.target
            cnot.target = temp
    else:
        cnot_maker = CNOTMaker()
        m2.gauss(x=cnot_maker, full_reduce=True)
        init_cnots = cnot_maker.cnots

    if not any([sum(row) == 1 for row in m2.data]):
        return []
    
    greedy_operations = greedy_reduction(m)

    if not greedy_operations:
        neighbors = list(frontier_neighbors)
        perm = column_optimal_swap(m)
        perm = {v: k for k, v in perm.items()}
        neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]

        m2 = bi_adj(g, neighbors2, frontier.values())

        if architecture:
            architecture_copy = build_architecture_from_frontier(g, architecture, frontier)
            
            cnots, rank = gauss(architecture=architecture_copy, matrix=m2, mode=elim_mode, full_reduce=True)
            cnots = filter_duplicate_cnots(cnots)
            for cnot in cnots:
                temp = cnot.control
                cnot.control = cnot.target
                cnot.target = temp

        else:
            cnots = m2.to_cnots(optimize=True)
    else:
        cnots = [CNOT(target, control) for control, target in greedy_operations]
    
    if not cnots:
        #to_cnots may return an empty list because the blocksize only goes to len(rows)
        cnots = init_cnots
    
    return cnots

def get_frontier_neighbors(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all (non-output) neighbors of the frontier as a set"""
    res = set()
    for v in frontier.values():
        res.update(set(g.neighbors(v)))
    return res.difference(set(g.outputs()))


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

def frontier_unfusion(g: BaseGraph[VT, ET], frontier_vertex: VT, frontier: Dict[int,VT], circuit: Circuit, desired_phase: Fraction):
    """unfuses the phase of a frontier vertex as ZPhase gate on the circuit, such that in the frontier vertex the desired phase remains"""
    qubit = [qubit for qubit, vertex in frontier.items() if vertex == frontier_vertex][0]
    gate = ZPhase(qubit, g.phase(frontier_vertex)-desired_phase)
    # gate = U3(qubit, 0, 0, g.phase(frontier_vertex)-desired_phase)
    circuit.add_gate(gate)
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


def extract_mcp(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit, allow_insertions: bool = False): 
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
            circuit.add_gate(gate)
        return True
    else:
        return False

def init_frontier(g: BaseGraph[VT, ET], circuit: Circuit, randomize_frontier_tries: int) -> Dict[int,VT]:
    """Inits the frontier of a ZX-diagram with the spiders adjacent to the inputs. Extracts Hadamard wires between inputs and frontier"""
    frontier: Dict[int,VT] = dict()

    input_list = list(g.inputs())
    if randomize_frontier_tries > 0:
        random.shuffle(input_list)

    for qubit, input in enumerate(input_list):
        v = list(g.neighbors(input))[0]
        if not v in g.outputs():
            frontier[qubit] = v
            if g.edge_type(g.edge(v,input)) == EdgeType.HADAMARD:
                gate = HAD(qubit)
                # gate = U3(qubit, Fraction(1,2), 0, Fraction(1,1))
                circuit.add_gate(gate)
                g.set_edge_type(g.edge(v,input),EdgeType.SIMPLE)
    
    return frontier

def extract_czs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit, optimize: bool = False):
    """Extracts connected frontier spiders as controlled Z gates and updates the diagram"""
    if optimize:
        optimize_czs_in_frontier(g, frontier)
    change = False
    for qubit, v in frontier.items():
        for w in set(g.neighbors(v)).intersection(set(frontier.values())):
            g.remove_edge(g.edge(v,w))
            gate = CZ(qubit, list(frontier.keys())[list(frontier.values()).index(w)])
            circuit.add_gate(gate)
            change = True
    return change

def extract_rzs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    """Extracts phases of frontier spiders as ZPhase gates and updates the diagram"""
    change = False
    for qubit in list(frontier.keys()):
        while True:
            boundary = False
            v = frontier[qubit]
            phase = g.phase(v)
            if phase != 0:
                g.set_phase(v,0)
                gate = ZPhase(qubit, phase)
            else:
                neighbors = list(g.neighbors(v))
                if len(neighbors) != 1:
                    break
                
                if g.type(neighbors[0]) == VertexType.BOUNDARY:
                    boundary = True
                    if g.edge_type(g.edge(v,neighbors[0])) == EdgeType.HADAMARD:
                        gate = HAD(qubit)
                    else:
                        gate = None

                    g.add_edge(g.edge(list(g.inputs())[qubit],neighbors[0]))
                    del frontier[qubit]
                else:
                    frontier[qubit] = neighbors[0]
                    gate = HAD(qubit)
                
                g.remove_vertex(v)
            
            if gate:
                circuit.add_gate(gate)
                change = True

            if boundary:
                break

    return change

def extract_cnots(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit, cnots: List[CNOT]):
    """Extracts CNOT gates resulting from gaussian elimination to circuit and adds the Hadamard wires of the corresponding frontier vertices"""
    for cnot in cnots:
        # Add CNOT to circuit
        control_qubit = list(frontier)[cnot.control]
        target_qubit = list(frontier)[cnot.target]
        # CNOT = H+CZ+H
        circuit.add_gate("HAD",target_qubit)
        circuit.add_gate("CZ", control_qubit, target_qubit)
        circuit.add_gate("HAD",target_qubit)

        # Add or remove Hadamard wires in diagram according to CNOT addition
        ftarg = frontier[control_qubit]
        fcont = frontier[target_qubit]
        for v in g.neighbors(fcont):
            if g.type(v) == VertexType.BOUNDARY:
                # special case: neighbor of "control" spider is an input, therefore we need to insert a spider between input and control spider
                vnew = insert_identity(g, fcont, v)
                break
    
        for v in g.neighbors(fcont):
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


def eliminate_yz_spider(g: BaseGraph[VT,ET], frontier: Dict[int,VT], frontier_neighbors: Set, circuit: Circuit):
    """Finds a YZ measured spider which is connected to a spider in frontier and applies a pivot on them. 
    By that, the YZ spider transforms to a XY spider"""
    
    for n in frontier_neighbors:
        for candidate in g.neighbors(n):
            if len(g.neighbors(candidate)) == 1 and g.type(candidate) == VertexType.Z:
                frontier_vertex = list(set(g.neighbors(n)).intersection(set(frontier.values())))[0]
                for neighbor in g.neighbors(frontier_vertex):
                    # special case if frontier vertex is connected to an output
                    if g.type(neighbor) == VertexType.BOUNDARY:
                        insert_identity(g, frontier_vertex, neighbor)
                        break

                pivot(g, n, frontier_vertex)
                for shared_neighbor in set(g.neighbors(n)).intersection(set(g.neighbors(frontier_vertex))):
                    g.set_phase(shared_neighbor,g.phase(shared_neighbor)+Fraction(1,1))
                g.set_phase(n, g.phase(n)+g.phase(candidate))
                g.remove_vertex(candidate)

                qubit = list(frontier)[list(frontier.values()).index(frontier_vertex)]
                gate = HAD(qubit)
                circuit.add_gate(gate)                

                return True
    return False

def pivot(g: BaseGraph[VT,ET], u: VT, v: VT) -> bool:
    """Graph theoretic pivot on graph-like diagram
    g: A graph instance
    u: First vertex
    v: Second vertex"""
    if not lcomp(g, u):
        return False
    success = lcomp(g, v)
    lcomp(g, u)
    return success

def lcomp(g: BaseGraph[VT,ET], u: VT):
    """graph theoretic local complementation on graph-like diagram"""
    vn = [neighbor for neighbor in g.neighbors(u) if len(g.neighbors(neighbor)) > 1]
    complement_neighbors(g, vn)
    return True

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

def insert_identity(g: BaseGraph[VT,ET], v1: VT, v2: VT) -> VT:
    """Inserts an empty Z-spider surrounded by two hadamard wires between two spiders a graph-like diagram."""
    orig_type = g.edge_type(g.edge(v1, v2))
    if g.connected(v1, v2):
        g.remove_edge(g.edge(v1, v2))
    vmid = g.add_vertex(VertexType.Z,g.qubits()[v1],g.rows()[v1] -1)
    g.add_edge((v1,vmid), EdgeType.HADAMARD)
    if orig_type == EdgeType.HADAMARD:
        g.add_edge((vmid,v2), EdgeType.SIMPLE)
    else:
        g.add_edge((vmid,v2), EdgeType.HADAMARD)
    return vmid

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
