from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.circuit.gates import Gate, CNOT
from pyzx.routing.architecture import Architecture
from pyzx.linalg import Mat2, Z2
from pyzx.extract import xor_rows, greedy_reduction, column_optimal_swap, bi_adj, filter_duplicate_cnots
from typing import List, Dict, Optional, Tuple
from .extractionutils import get_neighbors_of_frontier


def calculate_addition_options(g: BaseGraph[VT,ET], frontier: Dict[int,VT], architecture: Architecture) -> List[List[CNOT]]:
    options = []
    frontier_vertices_without_outputs = [v for k,v in frontier.items() if not any([n for n in g.neighbors(v) if n in g.outputs()]) and not v in g.outputs()]
    frontier_neighbors = list(get_neighbors_of_frontier(g,frontier_vertices_without_outputs))

    m: Mat2 = bi_adj(g, frontier_neighbors, frontier_vertices_without_outputs)
    # print("calculateaddition biadj",m,frontier_neighbors, frontier_vertices_without_outputs)
    if all(sum(row) != 1 for row in m.data):

        path_options = greedy_reduction_with_architecture(m, architecture)
        if path_options:
            for path in path_options:
                options.append(path)

        greedy_addition = greedy_reduction(m)
        if greedy_addition:
            path = []
            for cnot in greedy_addition:
                path.append(CNOT(cnot[1],cnot[0]))
            if path:
                options.append(path)

        perm = column_optimal_swap(m)
        perm = {v: k for k, v in perm.items()}
        neighbors2 = [frontier_neighbors[perm[i]] for i in range(len(frontier_neighbors))]
        m2 = bi_adj(g, neighbors2, frontier)
        standard_cnots = m2.to_cnots(optimize=True)
        standard_cnots = filter_duplicate_cnots(standard_cnots)
        if standard_cnots:
            options.append(standard_cnots)

        #fit options to graph (because removed frontiers change the indexing)
        adjusted_options = []
        # print(options)
        frontier_qubits = list(frontier.values())
        for option in options:
            new_option = []
            for cnot in option:
                #hacky, could be improved with better frontier structure
                ctrl = frontier_qubits.index(frontier_vertices_without_outputs[cnot.control])
                targ = frontier_qubits.index(frontier_vertices_without_outputs[cnot.target])
                new_option.append(CNOT(ctrl,targ))
            adjusted_options.append(new_option)

        # print("adjusted",adjusted_options)
        
        options = adjusted_options

    return options


def greedy_reduction_with_architecture(m: Mat2, architecture: Architecture) -> Optional[List[List[Gate]]]:
    """Returns a list of lists of CNOTs that reduce the matrix m to a matrix with only one 1 in at least one row.
    The function uses a greedy algorithm to find the minimal sums of rows that can be added together to reduce the matrix."""
    indices_list_greedy_row_add = find_minimal_sums_with_architecture(m, architecture)
    # print("indices",indices_list_greedy_row_add,"for m",m)
    if indices_list_greedy_row_add == []: return []

    row_add_results: List[List[Gate]] = []
    for indices_greedy_row_add in indices_list_greedy_row_add:
        indices = list(indices_greedy_row_add)
        rows = {i:m.data[i] for i in indices}
        weights: Dict[int,int] = {i: sum(r) for i,r in rows.items()}
        result: List[Gate] = []
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
            result.append(CNOT(best[0], best[1]))
            # print("result",result)
            control, target = best
            rows[target] = xor_rows(rows[control],rows[target])
            weights[target] = weights[target] - reduction
            indices.remove(control)

        row_add_results.append(result)
        # print("result",row_add_results)

    return row_add_results

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