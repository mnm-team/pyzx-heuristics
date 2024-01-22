
from fractions import Fraction
from pyzx.rules import apply_rule, lcomp, pivot
from .heuristics import PhaseType, get_phase_type, lcomp_heuristic, lcomp_heuristic_neighbor_unfusion, pivot_heuristic, pivot_heuristic_neighbor_unfusion
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Tuple, List
from pyzx.utils import VertexType, EdgeType
from .tools import split_phases, insert_identity
from pyzx.gflow import gflow
from .gflow_calculation import update_gflow_from_double_insertion, update_gflow_from_lcomp, update_gflow_from_pivot
from .simplify import get_random_match
from pyzx.rules import match_ids_parallel, remove_ids, match_spider_parallel, spider
import math
import random


MatchLcompHeuristicNeighbourType = Tuple[float,Tuple[VT,List[VT],VT]]

def lcomp_matcher(graph: BaseGraph[VT,ET], graph_flow=None) -> List[MatchLcompHeuristicNeighbourType]:
    """
    Find all vertices in the graph that can be matched for local complementation.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    graph_flow (dict): The dictionary representing the flow of the graph.

    Returns:
    list: A list of tuples, each containing the heuristic value, the vertex and its neighbors, and the neighbor for unfusion.
    """
    candidate_vertices = graph.vertex_set()
    vertex_types = graph.types()

    matches = []
    while len(candidate_vertices) > 0:

        current_vertex = candidate_vertices.pop()
        current_vertex_type = vertex_types[current_vertex]
        current_vertex_phase = graph.phase(current_vertex)
        
        if current_vertex_type != VertexType.Z: continue
        current_vertex_neighbors = list(graph.neighbors(current_vertex))
        boundary = False

        for neighbor in current_vertex_neighbors:
            if vertex_types[neighbor] != VertexType.Z:
                boundary = True
        
        if boundary: continue # for the moment

        if get_phase_type(graph.phase(current_vertex)) == PhaseType.TRUE_CLIFFORD:
            # Append a tuple with the heuristic value, the current vertex and its neighbors, and None for the neighbor for unfusion
            matches.append((lcomp_heuristic(graph,current_vertex),(current_vertex,current_vertex_neighbors),None))
        else:
            for neighbor in get_possible_unfusion_neighbours(graph, current_vertex, None, graph_flow):
                matches.append((lcomp_heuristic_neighbor_unfusion(graph,current_vertex,neighbor),(current_vertex,current_vertex_neighbors),neighbor))

    return matches

MatchPivotHeuristicNeighbourType = Tuple[float,Tuple[VT,VT],VT,VT]

def pivot_matcher(graph: BaseGraph[VT,ET], graph_flow=None) -> List[MatchPivotHeuristicNeighbourType]:
    """
    Find all edges in the graph that can be matched for a pivot operation.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    graph_flow (dict): The dictionary representing the flow of the graph.

    Returns:
    list: A list of tuples, each containing the heuristic value, the vertices of the edge, and the neighbors for unfusion.
    """
    candidate_edges = graph.edge_set()
    vertex_types = graph.types()

    matches = []
    while len(candidate_edges) > 0:
        current_edge = candidate_edges.pop()
        if graph.edge_type(current_edge) != EdgeType.HADAMARD: continue
        
        vertex_0, vertex_1 = graph.edge_st(current_edge)
        if not (vertex_types[vertex_0] == VertexType.Z and vertex_types[vertex_1] == VertexType.Z): continue
        
        boundary = False

        for neighbor in graph.neighbors(vertex_0):
            if vertex_types[neighbor] != VertexType.Z: # no boundaries
                boundary = True

        for neighbor in graph.neighbors(vertex_1):
            if vertex_types[neighbor] != VertexType.Z: # no boundaries
                boundary = True

        # If the vertices are on the boundary, skip it
        if boundary: continue

        if get_phase_type(graph.phase(vertex_0)) == PhaseType.CLIFFORD:
            if get_phase_type(graph.phase(vertex_1)) == PhaseType.CLIFFORD:
                # Append a tuple with the heuristic value, the vertices of the edge, and None for the neighbors for unfusion
                matches.append((pivot_heuristic(graph,current_edge),(vertex_0,vertex_1),None,None))
            else:
                for neighbor in get_possible_unfusion_neighbours(graph, vertex_1, vertex_0, graph_flow):
                    matches.append((pivot_heuristic_neighbor_unfusion(graph,current_edge,None,neighbor),(vertex_0,vertex_1),None,neighbor))
        else:
            if get_phase_type(graph.phase(vertex_1)) == PhaseType.CLIFFORD:
                for neighbor in get_possible_unfusion_neighbours(graph, vertex_0, vertex_1, graph_flow):
                    matches.append((pivot_heuristic_neighbor_unfusion(graph,current_edge,neighbor,None),(vertex_0,vertex_1),neighbor,None))
            else:
                for neighbor_v0 in get_possible_unfusion_neighbours(graph, vertex_0, vertex_1, graph_flow):
                    for neighbor_v1 in get_possible_unfusion_neighbours(graph, vertex_1, vertex_0, graph_flow):
                        matches.append((pivot_heuristic_neighbor_unfusion(graph,current_edge,neighbor_v0,neighbor_v1),(vertex_0,vertex_1),neighbor_v0,neighbor_v1))

    return matches

def get_possible_unfusion_neighbours(graph: BaseGraph[VT,ET], current_vertex, exclude_vertex=None, flow=None):
    """
    Get the possible neighbors for unfusion of a given vertex in a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    current_vertex (VT): The vertex to find possible unfusion neighbors for.
    exclude_vertex (VT, optional): A vertex to exclude from the possible unfusion neighbors.
    graph_flow (dict, optional): The dictionary representing the flow of the graph.

    Returns:
    list: A list of vertices that are possible neighbors for unfusion.
    """
    possible_unfusion_neighbours = []

    if len(flow[current_vertex]) == 1:
        possible_unfusion_neighbours.append(next(iter(flow[current_vertex]))) #get first element of set

    for neighbor in graph.neighbors(current_vertex):
        if current_vertex in flow[neighbor] and len(flow[neighbor]) == 1:
            possible_unfusion_neighbours.append(neighbor)

    if exclude_vertex and exclude_vertex in possible_unfusion_neighbours:
        possible_unfusion_neighbours.remove(exclude_vertex)

    return possible_unfusion_neighbours


def unfuse_to_neighbor(graph, current_vertex, neighbor_vertex, desired_phase):
    """
    Unfuse a vertex to its neighbor in a graph.

    Parameters:
    graph (BaseGraph): The graph to perform the operation on.
    current_vertex (VT): The vertex to unfuse.
    neighbor_vertex (VT): The neighbor vertex to unfuse to.
    desired_phase (float): The desired phase for the unfused vertex.

    Returns:
    tuple: A tuple containing the phaseless spider and the phase spider.
    """
    new_phase = split_phases(graph.phases()[current_vertex], desired_phase)
    phase_spider = graph.add_vertex(VertexType.Z, -2, graph.rows()[current_vertex], new_phase)

    graph.set_phase(current_vertex, desired_phase)
    graph.add_edge((phase_spider, neighbor_vertex), EdgeType.HADAMARD)
    graph.add_edge((current_vertex, phase_spider), EdgeType.SIMPLE)

    phaseless_spider = insert_identity(graph, current_vertex, phase_spider)

    graph.remove_edge(graph.edge(current_vertex, neighbor_vertex))

    return (phaseless_spider, phase_spider)

def apply_lcomp(graph: BaseGraph[VT,ET], match, flow):
    """
    Apply a local complementation operation to a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    match (tuple): The match to apply the operation to.
    flow (dict): The dictionary representing the flow of the graph.
    """
    vertex, neighbors = match[1]
    unfusion_neighbor = match[2]
    neighbors_copy = neighbors[:]

    if unfusion_neighbor:
        phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbor, Fraction(1,2))
        update_gflow_from_double_insertion(flow, vertex, unfusion_neighbor, phaseless_spider, phase_spider)
        neighbors_copy = [phaseless_spider if neighbor == unfusion_neighbor else neighbor for neighbor in neighbors_copy]

    update_gflow_from_lcomp(graph, vertex, flow)
    apply_rule(graph, lcomp, [(vertex, neighbors_copy)])


def apply_pivot(graph: BaseGraph[VT,ET], match, flow):
    """
    Apply a pivot operation to a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    match (tuple): The match to apply the operation to.
    flow (dict): The dictionary representing the flow of the graph.
    """
    vertex_1, vertex_2 = match[1]

    unfusion_neighbors = {}
    unfusion_neighbors[vertex_1] = match[2]
    unfusion_neighbors[vertex_2] = match[3]

    for vertex in [vertex_1, vertex_2]:
        if unfusion_neighbors[vertex]:
            phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbors[vertex], Fraction(0,1))
            update_gflow_from_double_insertion(flow, vertex, unfusion_neighbors[vertex], phaseless_spider, phase_spider)
            
    update_gflow_from_pivot(graph, vertex_1, vertex_2, flow)
    apply_rule(graph, pivot, [(vertex_1, vertex_2, [], [])])

def generate_matches(graph, flow, max_vertex=None, threshold=1):
    """
    Generate matches for local complementation and pivot operations in a graph.

    Parameters:
    graph (BaseGraph): The graph to generate matches for.
    flow (dict): The dictionary representing the flow of the graph.
    max_vertex (int, optional): The maximum vertex to consider for matches.
    threshold (int, optional): The minimum wire reduction for a match to be considered.

    Returns:
    tuple: A tuple containing the filtered matches for local complementation and pivot operations.
    """
    local_complement_matches = lcomp_matcher(graph, flow)
    pivot_matches = pivot_matcher(graph, flow)

    filtered_local_complement_matches = []
    filtered_pivot_matches = []

    for match in local_complement_matches:
        wire_reduction, vertices, neighbor = match
        if wire_reduction < threshold:
            continue
        if max_vertex and wire_reduction <= 0 and vertices[0] > max_vertex:
            continue
        filtered_local_complement_matches.append((wire_reduction, vertices, neighbor))

    for match in pivot_matches:
        wire_reduction, vertices, neighbor1, neighbor2 = match
        if wire_reduction < threshold:
            continue
        if max_vertex and wire_reduction <= 0 and vertices[0] > max_vertex and vertices[1] > max_vertex:
            continue
        filtered_pivot_matches.append((wire_reduction, vertices, neighbor1, neighbor2))

    return (filtered_local_complement_matches, filtered_pivot_matches)

def greedy_wire_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex=None, threshold=1, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using a greedy approach with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex (int, optional): The maximum vertex to consider for matches.
    threshold (int, optional): The minimum wire reduction for a match to be considered.
    quiet (bool, optional): Whether to suppress output.
    stats (dict, optional): A dictionary to store statistics.

    Returns:
    int: The number of iterations performed.
    """
    changes_made = True
    iteration_count = 0
    flow = gflow(graph)[1]

    while changes_made:
        changes_made = False
        local_complement_matches, pivot_matches = generate_matches(graph, graph_flow=flow, max_vertex=max_vertex, threshold=threshold)
        if apply_best_match(graph, local_complement_matches, pivot_matches, flow):
            iteration_count += 1
            changes_made = True
            flow = gflow(graph)[1]

    return iteration_count

def random_wire_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex=None, threshold=1, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using a random approach with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex (int, optional): The maximum vertex to consider for matches.
    threshold (int, optional): The minimum wire reduction for a match to be considered.
    quiet (bool, optional): Whether to suppress output.
    stats (dict, optional): A dictionary to store statistics.

    Returns:
    int: The number of iterations performed.
    """
    changes_made = True
    iteration_count = 0
    flow = gflow(graph)[1]

    while changes_made:
        changes_made = False
        local_complement_matches, pivot_matches = generate_matches(graph, graph_flow=flow, max_vertex=max_vertex, threshold=threshold)
        if apply_random_match(graph, local_complement_matches, pivot_matches, flow):
            iteration_count += 1
            changes_made = True
            flow = gflow(graph)[1]

    return iteration_count

def sim_annealing_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex=None, iterations=100, cooling_rate=0.95, threshold=-100000, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using simulated annealing with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex (int, optional): The maximum vertex to consider for matches.
    iterations (int, optional): The number of iterations to perform.
    cooling_rate (float, optional): The rate at which the temperature decreases.
    threshold (int, optional): The minimum wire reduction for a match to be considered.
    quiet (bool, optional): Whether to suppress output.
    stats (dict, optional): A dictionary to store statistics.

    Returns:
    BaseGraph[VT,ET]: The graph with the reduced number of wires.
    """
    temperature = iterations
    min_temperature = 1
    iteration_count = 0
    flow = gflow(graph)[1]

    best_graph = graph.copy()
    best_evaluation = graph.num_edges()
    current_evaluation = best_evaluation

    backtrack = False

    while temperature > min_temperature:
        iteration_count += 1
        local_complement_matches, pivot_matches = generate_matches(graph, graph_flow=flow, max_vertex=max_vertex, threshold=threshold)
        operation, match = get_best_match(local_complement_matches, pivot_matches)
        if match[0] <= 0:
            if backtrack:
                graph = best_graph.copy()
                current_evaluation = best_evaluation
                backtrack = False
                flow = gflow(graph)[1]
                continue
            else:
                operation, match = get_random_match(local_complement_matches, pivot_matches)
                backtrack = True

        if operation == "none":
            temperature = 0
            break

        acceptance_probability = math.exp(match[0]/temperature)
        # If the wire reduction of the match is positive or the acceptance probability is greater than a random number
        if match[0] > 0 or acceptance_probability > random.random():
            current_evaluation -= match[0]

            if operation == "pivot":
                apply_pivot(graph, match, flow)
            else:
                apply_lcomp(graph, match, flow)

            if current_evaluation < best_evaluation:
                best_graph = graph.copy()
                best_evaluation = current_evaluation

            flow = gflow(graph)[1]

        temperature *= cooling_rate

    print("final num edges: ", best_graph.num_edges())
    return best_graph

def apply_random_match(graph, local_complement_matches, pivot_matches, flow):
    """
    Apply a random match from the list of local complementation and pivot matches to a graph.

    Parameters:
    graph (BaseGraph): The graph to apply the match to.
    local_complement_matches (list): The list of local complementation matches.
    pivot_matches (list): The list of pivot matches.
    graph_flow (dict): The dictionary representing the flow of the graph.

    Returns:
    bool: True if a match was applied, False otherwise.
    """
    operation_type, match = get_random_match(local_complement_matches, pivot_matches)

    if operation_type == "pivot":
        apply_pivot(graph, match, flow)
    elif operation_type == "lcomp":
        apply_lcomp(graph, match, flow)
    else:
        return False

    return True

# def apply_random_match(g, lcomp_matches, pivot_matches, gfl):
#     # lcomp_matches.sort(key= lambda m: m[0],reverse=True)
#     # pivot_matches.sort(key= lambda m: m[0],reverse=True)
#     method = "pivot"

#     if len(lcomp_matches) > 0:
#         if len(pivot_matches) > 0:
#             method = "lcomp" if random.random() < 0.5 else "pivot"    
#         else:
#             method = "lcomp"
#     else:
#         if len(pivot_matches) == 0:
#             return False

#     if method == "pivot":
#         apply_pivot(g,pivot_matches[0], gfl)
#     else:
#         apply_lcomp(g,lcomp_matches[0], gfl)
#     return True

def apply_best_match(graph, local_complement_matches, pivot_matches, flow):
    """
    Apply the best match from the list of local complementation and pivot matches to a graph.

    Parameters:
    graph (BaseGraph): The graph to apply the match to.
    local_complement_matches (list): The list of local complementation matches.
    pivot_matches (list): The list of pivot matches.
    flow (dict): The dictionary representing the flow of the graph.

    Returns:
    bool: True if a match was applied, False otherwise.
    """
    # Sort the local complementation and pivot matches in descending order of their scores
    local_complement_matches.sort(key=lambda match: match[0], reverse=True)
    pivot_matches.sort(key=lambda match: match[0], reverse=True)

    operation_type = "pivot"

    if len(local_complement_matches) > 0:
        if len(pivot_matches) > 0 and local_complement_matches[0][0] > pivot_matches[0][0]:
            operation_type = "lcomp"
        elif len(pivot_matches) == 0:
            operation_type = "lcomp"

    elif len(pivot_matches) == 0:
        return False

    if operation_type == "pivot":
        apply_pivot(graph, pivot_matches[0], flow)
    else:
        apply_lcomp(graph, local_complement_matches[0], flow)

    return True

def get_best_match(local_complement_matches, pivot_matches):
    """
    Get the best match from the list of local complementation and pivot matches.

    Parameters:
    local_complement_matches (list): The list of local complementation matches.
    pivot_matches (list): The list of pivot matches.

    Returns:
    tuple: A tuple containing the operation type and the best match.
    """
    # Sort the local complementation and pivot matches in descending order of their scores
    local_complement_matches.sort(key=lambda match: match[0], reverse=True)
    pivot_matches.sort(key=lambda match: match[0], reverse=True)

    operation_type = "pivot"

    if len(local_complement_matches) > 0:
        if len(pivot_matches) > 0 and local_complement_matches[0][0] > pivot_matches[0][0]:
            operation_type = "lcomp"
        elif len(pivot_matches) == 0:
            operation_type = "lcomp"

    elif len(pivot_matches) == 0:
        return ("none", None)

    if operation_type == "pivot":
        return ("pivot", pivot_matches[0])
    else:
        return ("lcomp", local_complement_matches[0])
