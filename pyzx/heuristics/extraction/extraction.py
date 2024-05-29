


from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT
from pyzx.extract import column_optimal_swap, filter_duplicate_cnots, graph_to_swaps
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.heuristics.extraction.extraction_base import apply_cnots, bi_adj, extract_cnots, extract_czs, extract_rzs, get_best_cnot_configuration, get_cnot_row_operations, greedy_reduction_with_architecture, init_frontier, neighbors_of_frontier, remove_gadget
from pyzx.routing.architecture import Architecture
from pyzx.routing.cnot_mapper import ElimMode, gauss
from pyzx.simplify import id_simp



def extract_architecture_aware_circuit(
        g: BaseGraph[VT, ET],
        architecture: Architecture,
        optimize_czs: bool = True,
        up_to_perm: bool = False,
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

    c = Circuit(len(outputs))

    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in inputs and v not in outputs:
            n = list(g.neighbors(v))[0]
            gadgets[n] = v

    frontier = init_frontier(g, c, True)

    czs_saved = 0

    architecture_copy = Architecture(name=architecture.name, coupling_graph=architecture.graph.copy(), qubit_map=list(range(len(frontier))))
    shuttle_list = []
    
    while True:
        # preprocessing
        cz_gates = extract_czs(g, frontier, c, architecture_copy, optimize=False)
        rz_gates = extract_rzs(g, frontier, c, True)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = neighbors_of_frontier(g, frontier)
   
        if not frontier:
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(g, frontier, neighbor_set, gadgets):
            # There was a gadget in the way. Go back to the top
            continue
            
        neighbors = list(neighbor_set)

        #TODO: Check if this is needed. In theory, if this is not done it could lead to cnots which are not allowed by the architecture
        frontier_with_removed = {i: -1 for i in range(len(outputs))}
        frontier_with_removed.update(frontier)

        m = bi_adj(g, neighbors, frontier_with_removed.values())
        if all(sum(row) != 1 for row in m.data):  # No easy vertex

            cnots, shuttle = get_cnot_row_operations(g, frontier, neighbors, architecture_copy)
            cnots = [CNOT(control=cnot.target, target=cnot.control) for cnot in cnots]

            if shuttle:
                shuttle_list.append(shuttle)
                c.add_gate(shuttle)

                temp = frontier[shuttle.control]
                frontier[shuttle.control] = frontier[shuttle.target]
                frontier[shuttle.target] = temp

                m.row_add(shuttle.control, shuttle.target)
                m.row_add(shuttle.target, shuttle.control)
                m.row_add(shuttle.control, shuttle.target)

            # We now have a set of CNOTs that suffice to extract at least one vertex.
        else:
            if not quiet: print("Simple vertex")
            cnots = []

        extracted = apply_cnots(g, c, frontier, cnots, m, neighbors)
        if not quiet: print("Vertices extracted:", extracted)
            
    if optimize_czs:
        if not quiet: print("CZ gates saved:", czs_saved)

    for shuttle in reversed(shuttle_list):
        c.add_gate(shuttle)
    # Outside of loop. Finish up the permutation
    id_simp(g, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))
    return graph_to_swaps(g, up_to_perm) + c