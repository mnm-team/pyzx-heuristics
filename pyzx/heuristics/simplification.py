from itertools import combinations
import logging
import math
import random

from fractions import Fraction
import time
from typing import Set, Tuple, List, Optional, Callable
from functools import partial

from .heuristics import PhaseType, get_phase_type, lcomp_heuristic, pivot_heuristic
from .tools import insert_identity, insert_phase_gadget

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rules import apply_rule, lcomp, pivot
from pyzx.utils import VertexType, EdgeType


MatchLcompHeuristicType = Tuple[float,Tuple[VT,List[VT]],int]

MatchPivotHeuristicType = Tuple[float,Tuple[VT,VT]]



def check_lcomp_match(graph, vertex, include_boundaries, include_gadgets, calculate_heuristic=True):
    vertex_types = graph.types()

    current_vertex_type = vertex_types[vertex]
    current_vertex_phase = graph.phase(vertex)
    
    if current_vertex_type != VertexType.Z: return None
    # Check if the vertex needs to be transformed into an XZ spider
    needs_gadget = get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD
    # Skip if gadgets are not allowed and the vertex needs to be gadgetized
    if include_gadgets == False and needs_gadget: return None
    # Skip if the vertex has only one neighbor (i.e., it's a leaf node)
    if len(graph.neighbors(vertex)) == 1: return None
            
    current_vertex_neighbors = list(graph.neighbors(vertex))

    if vertex in current_vertex_neighbors: return None

    is_already_gadget = False
    boundary_count = 0
    for neighbor in current_vertex_neighbors:
        # Check if the neighbor is a leaf node and the vertex needs to be gadgetized
        if len(graph.neighbors(neighbor)) == 1 and get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD:
            is_already_gadget = True
        if vertex_types[neighbor] != VertexType.Z:
            boundary_count += 1

    if is_already_gadget and needs_gadget: return None
    if not include_boundaries and boundary_count > 0: return None

    if not calculate_heuristic:
        return (0,(vertex,current_vertex_neighbors),0)
    
    spider_count = -1 + boundary_count + (2 if needs_gadget else 0)

    if boundary_count > 0:
        return (lcomp_heuristic(graph,vertex)-boundary_count,(vertex,current_vertex_neighbors),spider_count)
    
    return (lcomp_heuristic(graph,vertex),(vertex,current_vertex_neighbors),spider_count)

def check_pivot_match(graph, edge, include_boundaries, include_gadgets, calculate_heuristic=True):
    
    vertex_types = graph.types()

    if graph.edge_type(edge) != EdgeType.HADAMARD: return None

    # Get the vertices at the ends of this edge
    vertex0, vertex1 = graph.edge_st(edge)

    if vertex0 == vertex1: return None

    # Skip this edge if both vertices are not Z vertices
    if not (vertex_types[vertex0] == VertexType.Z and vertex_types[vertex1] == VertexType.Z): return None

    is_vertex0_not_clifford = get_phase_type(graph.phase(vertex0)) != PhaseType.CLIFFORD
    is_vertex1_not_clifford = get_phase_type(graph.phase(vertex1)) != PhaseType.CLIFFORD

    if include_gadgets == False and (is_vertex0_not_clifford or is_vertex1_not_clifford): return None
    # Skip if both vertices are either true clifford or not clifford phase types
    if is_vertex0_not_clifford and is_vertex1_not_clifford: return None 
    # Skip if the vertices have only one neighbor (i.e., they are leaf nodes)
    if len(graph.neighbors(vertex0)) == 1 or len(graph.neighbors(vertex1)) == 1: return None 

    vertex0_already_gadget = False
    vertex1_already_gadget = False
    boundary_count = 0

    for neighbor in graph.neighbors(vertex0):
        if vertex_types[neighbor] != VertexType.Z: 
            boundary_count += 1

        # Set the flag if the neighbor is a root of a second phase gadget
        if len(graph.neighbors(neighbor)) == 1 and get_phase_type(graph.phases()[vertex0]) != PhaseType.CLIFFORD: 
            vertex0_already_gadget = True

    for neighbor in graph.neighbors(vertex1):
        if vertex_types[neighbor] != VertexType.Z: 
            boundary_count += 1

        # Set the flag if the neighbor is a root of a second phase gadget
        if len(graph.neighbors(neighbor)) == 1 and get_phase_type(graph.phases()[vertex1]) != PhaseType.CLIFFORD: 
            vertex1_already_gadget = True

    if (vertex0_already_gadget and is_vertex0_not_clifford) or (vertex1_already_gadget and is_vertex1_not_clifford): return None
    if not include_boundaries and boundary_count > 0: return None

    if not calculate_heuristic:
        return (0,(vertex0,vertex1),0)

    spider_count = -2 + boundary_count + (2 if is_vertex0_not_clifford else 0) + (2 if is_vertex1_not_clifford else 0)

    if include_boundaries:
        return (pivot_heuristic(graph,edge)-boundary_count,(vertex0,vertex1), spider_count)
    else:
        return (pivot_heuristic(graph,edge),(vertex0,vertex1), spider_count)

def lcomp_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, calculate_heuristic=True) -> List[MatchLcompHeuristicType]:
    """
    Generates all matches for local complementation in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ spiders by the rule application)

    Returns:
    List[MatchLcompHeuristicType]: A list of match tuples (heuristic,vertices,spider_count), where heuristic is the LCH, vertices the tuple needed for rule application and spider_count the amount of saved/added spiders
    """
    vertex_candidates = graph.vertex_set()

    matches = []

    while len(vertex_candidates) > 0:
        current_vertex = vertex_candidates.pop()
        match = check_lcomp_match(graph, current_vertex, include_boundaries, include_gadgets, calculate_heuristic)

        if match is not None:
            matches.append(match)
    
    return matches

def pivot_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, calculate_heuristic=True) -> List[MatchPivotHeuristicType]:
    """
    Generates all matches for pivoting in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into YZ spiders by the rule application)

    Returns:
    List[MatchPivotHeuristicType]: A list of match tuples (x,y,z), where x is the PH, y the tuple needed for rule application and z the amount of saved/added spiders
    """
    edge_candidates = graph.edge_set()
    matches = []

    while len(edge_candidates) > 0:
        edge = edge_candidates.pop()
        match = check_pivot_match(graph, edge, include_boundaries, include_gadgets, calculate_heuristic)

        if match is not None:
            matches.append(match)

    return matches


def update_lcomp_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: List[VT], lcomp_matches: List[MatchLcompHeuristicType], neighbors_of_neighbors: Set[VT], include_boundaries=False, include_gadgets=False) -> List[MatchLcompHeuristicType]:
    # Iterate over the current local complement matches
    lcomp_matches_copy = lcomp_matches[:]

    indices_to_remove = set()
    for i, (heuristic, (vertex_match, vertex_match_neighbors), spider_count) in enumerate(lcomp_matches_copy):

        if vertex_match in removed_vertices:
            indices_to_remove.add(i)
            continue

        if any(element in vertex_match_neighbors for element in removed_vertices):
            match = check_lcomp_match(graph, vertex_match, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
            if match is None:
                indices_to_remove.add(i)
            else:
                lcomp_matches_copy[i] = match
            continue

        # If the vertex is in the set of neighbors of neighbors, recalculate the heuristic
        if vertex_match in neighbors_of_neighbors:
            new_heuristic = lcomp_heuristic(graph, vertex_match)
            lcomp_matches_copy[i] = (new_heuristic, (vertex_match, vertex_match_neighbors), spider_count)
    
    # Remove the matches that need to be removed
    for index in sorted(indices_to_remove, reverse=True):
        del lcomp_matches_copy[index]

    # Check for new local complement matches in the vertex neighbors
    for neighbor in vertex_neighbors:
        match = check_lcomp_match(graph, neighbor, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
        if match is not None:
            lcomp_matches_copy.append(match)

    return lcomp_matches_copy

def update_pivot_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: List[VT], pivot_matches: List[MatchPivotHeuristicType], neighbors_of_neighbors: Set[VT], include_boundaries=False, include_gadgets=False) -> List[MatchPivotHeuristicType]:
    
    pivot_matches_copy = pivot_matches[:]

    indices_to_remove = set()
    # Iterate over the current pivot matches
    for i, (heuristic, (vertex0, vertex1), spider_count) in enumerate(pivot_matches_copy):
        if vertex0 in removed_vertices or vertex1 in removed_vertices:
            indices_to_remove.add(i)
            continue
        
        if not graph.connected(vertex0, vertex1):
            indices_to_remove.add(i)
            continue

        # If the vertices are in the set of neighbors of neighbors, recalculate the heuristic
        if vertex0 in neighbors_of_neighbors or vertex1 in neighbors_of_neighbors:
            edge = graph.edge(vertex0, vertex1)
            match = check_pivot_match(graph, edge, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
            if match is None:
                indices_to_remove.add(i)
            else:
                pivot_matches_copy[i] = match

    # Remove the matches that need to be removed
    for index in sorted(indices_to_remove, reverse=True):
        del pivot_matches_copy[index]

    # Check for new pivot matches in the vertex neighbors
    for vertex_neighbor in vertex_neighbors:
        for neighbor_of_neighbor in graph.neighbors(vertex_neighbor):
            if graph.connected(vertex_neighbor, neighbor_of_neighbor):
                edge = graph.edge(vertex_neighbor, neighbor_of_neighbor)
                match = check_pivot_match(graph, edge, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
                
                if match is not None:

                    already_added_to_list = False
                    for i, (_, edge_canidate, _) in enumerate(pivot_matches_copy):
                        if edge_canidate == edge and not already_added_to_list:
                            pivot_matches_copy[i] = match
                            already_added_to_list = True
                    if not already_added_to_list:
                        pivot_matches_copy.append(match)

    return pivot_matches_copy

def deep_tuple(lst):
    return tuple(deep_tuple(i) if isinstance(i, list) or isinstance(i, tuple) else i for i in lst)

def update_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: List[VT], lcomp_matches: List[MatchLcompHeuristicType], pivot_matches: List[MatchPivotHeuristicType], include_boundaries=False, include_gadgets=False) -> Tuple[List[MatchLcompHeuristicType], List[MatchPivotHeuristicType]]:
    """
    Updates the list of local complement and pivot matches after a local complementation or pivot has been applied.

    Parameters:
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    vertex_neighbors (List[VT]): The neighbors of the vertex where the local complementation or pivot was applied
    removed_vertices (List[VT]): The vertices that were removed by the local complementation or pivot
    lcomp_matches (List[MatchLcompHeuristicType]): The current list of local complement matches
    pivot_matches (List[MatchPivotHeuristicType]): The current list of pivot matches

    Returns:
    Tuple[List[MatchLcompHeuristicType], List[MatchPivotHeuristicType]]: The updated lists of local complement and pivot matches
    """
    graph_test = graph.clone()

    # Initialize a set of neighbors of neighbors
    neighbors_of_neighbors = set()

    # Iterate over the neighbors of the vertex
    for neighbor in vertex_neighbors:
        # Iterate over the neighbors of the current neighbor
        for neighbor_of_neighbor in graph.neighbors(neighbor):
            # If the neighbor of the neighbor is not in the vertex_neighbors list, add it to the set
            if neighbor_of_neighbor not in vertex_neighbors:
                neighbors_of_neighbors.add(neighbor_of_neighbor)

    lcomp_matches = update_lcomp_matches(graph=graph, 
                                         vertex_neighbors=vertex_neighbors, 
                                         removed_vertices=removed_vertices, 
                                         lcomp_matches=lcomp_matches, 
                                         neighbors_of_neighbors=neighbors_of_neighbors, 
                                         include_boundaries=include_boundaries, 
                                         include_gadgets=include_gadgets)
    
    pivot_matches = update_pivot_matches(graph=graph, 
                                         vertex_neighbors=vertex_neighbors, 
                                         removed_vertices=removed_vertices, 
                                         pivot_matches=pivot_matches, 
                                         neighbors_of_neighbors=neighbors_of_neighbors, 
                                         include_boundaries=include_boundaries, 
                                         include_gadgets=include_gadgets)
    
    lcomp_test = lcomp_matcher(graph_test, include_boundaries=include_boundaries, include_gadgets=include_gadgets, calculate_heuristic=True)
    pivot_test = pivot_matcher(graph_test, include_boundaries=include_boundaries, include_gadgets=include_gadgets, calculate_heuristic=True)
    
    lcomp_matches_diff = set(deep_tuple(lcomp_matches)) - set(deep_tuple(lcomp_test))
    lcomp_test_diff = set(deep_tuple(lcomp_test)) - set(deep_tuple(lcomp_matches))

    pivot_matches_diff = set(deep_tuple(pivot_matches)) - set(deep_tuple(pivot_test))
    pivot_test_diff = set(deep_tuple(pivot_test)) - set(deep_tuple(pivot_matches))

    if lcomp_matches_diff or lcomp_test_diff or pivot_matches_diff or pivot_test_diff:
        raise Exception(f"lcomp_matches_diff: {lcomp_matches_diff}\n"
                f"lcomp_test_diff: {lcomp_test_diff}\n"
                f"pivot_matches_diff: {pivot_matches_diff}\n"
                f"pivot_test_diff: {pivot_test_diff}")

    return lcomp_matches, pivot_matches 



def generate_filtered_matches(graph, include_boundaries=False, include_gadgets=False, max_vertex_index=None, threshold=1, check_sub_branches=False, calculate_heuristic=True):
    """
    Collects and filters all matches for local complementation and pivoting

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of heuristic_threshold<0).
    threshold (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

    Returns: 
    Tuple (List[MatchLcompHeuristicType], List[MatchPivotHeuristicType]): A tuple with all filtered matches for local complementation and pivoting
    """
    if not check_sub_branches:
        time_get_matches = time.perf_counter()

    local_complement_matches = lcomp_matcher(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, calculate_heuristic=calculate_heuristic)
    pivot_matches = pivot_matcher(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, calculate_heuristic=calculate_heuristic)

    if not check_sub_branches:
        time_get_matches = time.perf_counter()-time_get_matches

    filtered_local_complement_matches = []
    filtered_pivot_matches = []
    num_sub_branch_local_complement_list = []
    num_sub_branch_pivot_list = []

    time_sub_branches_all = 0
    time_apply_rule = 0
    if check_sub_branches:
        time_filter_matches = 0
        time_get_matches = 0

    if not check_sub_branches:
        time_filter_matches = time.perf_counter()

    for match in local_complement_matches:
        wire_reduction, vertices, spider_count = match
        # Skip matches that do not meet the heuristic threshold
        # if wire_reduction < threshold:
        #     continue
        # Skip matches that could cause non-termination
        if max_vertex_index and wire_reduction <= 0 and vertices[0] > max_vertex_index:
            continue
        
        num_sub_branches = 0
        if check_sub_branches:
            time_sub_branches = time.perf_counter()
            sub_branch_graph = graph.clone()
            apply_lcomp(sub_branch_graph, vertices)
            time_apply_rule += time.perf_counter()-time_sub_branches
            (t0, t1), (sub_branch_local_complement_matches, sub_branch_pivot_matches) = generate_filtered_matches(sub_branch_graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_sub_branches=False)
            num_sub_branches = len(sub_branch_local_complement_matches)+len(sub_branch_pivot_matches)
            num_sub_branch_local_complement_list.append(num_sub_branches)
            time_sub_branches_all += time.perf_counter()-time_sub_branches
            time_filter_matches += t0
            time_get_matches += t1

        filtered_local_complement_matches.append((wire_reduction, vertices, spider_count))
    
    for match in pivot_matches:
        wire_reduction, vertices, spider_count = match
        # Skip matches that do not meet the heuristic threshold
        #TODO fix this
        # if wire_reduction < threshold:
        #     continue
        # Skip matches that could cause non-termination
        if max_vertex_index and wire_reduction <= 0 and vertices[0] > max_vertex_index and vertices[1] > max_vertex_index:
            continue

        num_sub_branches = 0
        if check_sub_branches:
            time_sub_branches = time.perf_counter()
            sub_branch_graph = graph.clone()
            apply_pivot(sub_branch_graph, vertices)
            time_apply_rule += time.perf_counter()-time_sub_branches
            (t0, t1), (sub_branch_local_complement_matches, sub_branch_pivot_matches) = generate_filtered_matches(sub_branch_graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_sub_branches=False)
            num_sub_branches = len(sub_branch_local_complement_matches)+len(sub_branch_pivot_matches)
            num_sub_branch_pivot_list.append(num_sub_branches)
            time_sub_branches_all += time.perf_counter()-time_sub_branches
            time_filter_matches += t0
            time_get_matches += t1

        filtered_pivot_matches.append((wire_reduction, vertices, spider_count))

    if not check_sub_branches:
        time_filter_matches = time.perf_counter()-time_filter_matches

    if check_sub_branches:
        logging.debug(f"Time to calculate sub-branches: {time_sub_branches_all}")
        logging.debug(f"Time to filter matches: {time_filter_matches}")
        logging.debug(f"Time to get matches: {time_get_matches}")
        logging.debug(f"Time to apply rule: {time_apply_rule}")
        num_matches = len(filtered_local_complement_matches)+len(filtered_pivot_matches)

        filtered_local_complement_matches_new_heu = [(heu + (sub_branches-num_matches), match, spider_count) for (heu, match, spider_count), sub_branches in zip(filtered_local_complement_matches, num_sub_branch_local_complement_list)]
        filtered_pivot_matches_new_heu = [(heu + (sub_branches-num_matches), match, spider_count) for (heu, match, spider_count), sub_branches in zip(filtered_pivot_matches, num_sub_branch_local_complement_list)]
        return (filtered_local_complement_matches_new_heu, filtered_pivot_matches_new_heu)
    
    return (filtered_local_complement_matches, filtered_pivot_matches)
    # return (time_filter_matches, time_get_matches),(filtered_local_complement_matches, filtered_pivot_matches)


def get_random_match(local_complement_matches, pivot_matches):
    """
    Randomly selects a rule application out of the given matches

    Parameters: 
    local_complement_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
    pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

    Returns:
    Tuple (string, MatchLcompHeuristicType | MatchPivotHeuristicType): Tuple of rule name and match
    """
    rule_to_apply = "pivot"

    # If there are local complement matches and a 50%/50% percent chance is true
    if len(local_complement_matches) > 0 and random.randint(0, 1) == 1:
        rule_to_apply = "lcomp"

    if len(local_complement_matches) > 0:
        # If there are pivot matches and a 50%/50% percent chance is true
        if len(pivot_matches) > 0 and random.randint(0, 1) == 1:
            rule_to_apply = "lcomp"
        else:
            rule_to_apply = "lcomp"
    else:
        if len(pivot_matches) == 0:
            return ("none", None)

    if rule_to_apply == "pivot":
        return ("pivot", pivot_matches[random.randint(0, len(pivot_matches) - 1)])
    else:
        return ("lcomp", local_complement_matches[random.randint(0, len(local_complement_matches) - 1)])

def get_matches_from_beginning_middle_end(n, local_complement_matches, pivot_matches):
    """
    Get n best matches incrementally from the beginning, middle, and end of local complement and pivot matches.
    Starting with the first element of the lists.

    This function sorts the local complement and pivot matches in descending order based on the heuristic result. It then selects n best matches incrementally from the beginning, middle, and end of the sorted matches, choosing the better match between local complement and pivot matches.

    Parameters:
    n (int): The number of matches to return.
    local_complement_matches (list): A list of local complement matches.
    pivot_matches (list): A list of pivot matches.

    Returns:
    list: A list of dictionaries, where each dictionary contains a match and its type ("lcomp" or "pivot").
    """
    # Sort the matches in descending order based on the heuristic result
    local_complement_matches.sort(key=lambda match: match[0], reverse=True)
    pivot_matches.sort(key=lambda match: match[0], reverse=True)

    matches = []

    # Initialize separate indices for local complement and pivot matches
    lcomp_indices = [0, len(local_complement_matches) // 3, len(local_complement_matches)-1]
    pivot_indices = [0, len(pivot_matches) // 3, len(pivot_matches)-1]

    # Store the initial indices to check for repetition later
    initial_lcomp_indices = [0, len(local_complement_matches) // 3, 2* len(local_complement_matches) // 3]
    initial_pivot_indices = [0, len(pivot_matches) // 3, 2* len(pivot_matches) // 3]

    for i in range(n):
        # Select the index based on the current iteration
        lcomp_index = lcomp_indices[i % 3]
        pivot_index = pivot_indices[i % 3]

        # Check if the index is within the length of the list and has not reached the initial index of the next index set
        if lcomp_index < len(local_complement_matches) and lcomp_index != initial_lcomp_indices[(i+1) % 3]:
            lcomp_match = local_complement_matches[lcomp_index]
        else:
            lcomp_match = None

        if pivot_index < len(pivot_matches) and pivot_index != initial_pivot_indices[(i+1) % 3]:
            pivot_match = pivot_matches[pivot_index]
        else:
            pivot_match = None

        if lcomp_match and pivot_match:
            if lcomp_match[0] > pivot_match[0]:
                matches.append({"match": lcomp_match, "match type": "lcomp"})
                if i%3 == 2:
                    lcomp_indices[i % 3] -= 1
                else:
                    lcomp_indices[i % 3] += 1
            else:
                matches.append({"match": pivot_match, "match type": "pivot"})
                if i%3 == 2:
                    pivot_indices[i % 3] -= 1
                else:
                    pivot_indices[i % 3] += 1
        elif lcomp_match:
            matches.append({"match": lcomp_match, "match type": "lcomp"})
            if i%3 == 2:
                lcomp_indices[i % 3] -= 1
            else:
                lcomp_indices[i % 3] += 1
        elif pivot_match:
            matches.append({"match": pivot_match, "match type": "pivot"})
            if i%3 == 2:
                pivot_indices[i % 3] -= 1
            else:
                pivot_indices[i % 3] += 1

    return matches



def apply_best_match(graph, local_complement_matches, pivot_matches):
    """
    Applies the rule with the best heuristic result, i.e., the rule which eliminates the most Hadamard wires

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e., ZX-diagram
    local_complement_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
    pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

    Returns:
    match (Dict[str, MatchLcompHeuristicType | MatchPivotHeuristicType]): The match that has been applied. None if no match has been applied.
    """
    # Sort the matches in descending order based on the heuristic result
    local_complement_matches.sort(key=lambda match: match[0], reverse=True)
    pivot_matches.sort(key=lambda match: match[0], reverse=True)

    method_to_apply = "pivot"

    # If there are local complement matches
    if len(local_complement_matches) > 0:
        # If there are also pivot matches and the best local complement match is better than the best pivot match
        if len(pivot_matches) > 0:
            if local_complement_matches[0][0] > pivot_matches[0][0]:
                method_to_apply = "lcomp"      
        else:
            method_to_apply = "lcomp"
    else:
        if len(pivot_matches) == 0:
            return None

    if method_to_apply == "pivot":
        apply_pivot(graph, pivot_matches[0][1])
        return {"match": pivot_matches[0], "match type": "pivot"}
    else:
        apply_lcomp(graph, local_complement_matches[0][1])
        return {"match": local_complement_matches[0], "match type": "lcomp"}

def apply_random_match(graph, local_complement_matches, pivot_matches):
    """
    Applies a randomly selected rule on the given graph.

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e., ZX-diagram
    local_complement_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
    pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

    Returns:
    match: The match that has been applied. None if no match has been applied.
    """
    rule_type, selected_match = get_random_match(local_complement_matches, pivot_matches)

    if rule_type == "pivot":
        apply_pivot(graph, selected_match[1])
        return {"match": selected_match, "match type": "pivot"}
    elif rule_type == "lcomp":
        apply_lcomp(graph, selected_match[1])
        return {"match": selected_match, "match type": "lcomp"}
    else:
        return None



def apply_lcomp(graph: BaseGraph[VT,ET], match):
    """
    Applies local complementation on a vertex depending on its phase and whether it is a boundary vertex or not.

    - If the vertex is a boundary, an additional spider and (Hadamard) wire is inserted to make the vertex interior.
    - If the vertex v0 has a non-Clifford phase p, two additional vertices v1 and v2 are inserted on top:
      v0 has phase π/2, v1 has phase 0 and is connected to v0, v2 has phase p - π/2 and is connected to v1.
      Local complementation removes v0, so v1 becomes a XZ spider with v2 corresponding to the measurement effect.

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram.
    match (V,List[V]): Tuple of a vertex and its neighbors.

    Returns: 
    Nothing
    """
    vertex, neighbors = match
    neighbors_copy = neighbors[:]

    identity_inserted = True
    while identity_inserted:
        identity_inserted = False
        for neighbor in graph.neighbors(vertex):
            if graph.types()[neighbor] == VertexType.BOUNDARY:
                new_vertex = insert_identity(graph, vertex, neighbor)
                neighbors_copy = [new_vertex if i == neighbor else i for i in neighbors_copy]
                identity_inserted = True
                break    

    phase_type = get_phase_type(graph.phases()[vertex])

    # If the phase type is not a true Clifford, insert a phase gadget
    if phase_type != PhaseType.TRUE_CLIFFORD:
        mid_vertex, gadget_top_vertex = insert_phase_gadget(graph, vertex, Fraction(1,2))
        neighbors_copy.append(mid_vertex)
        apply_rule(graph, lcomp, [(vertex, neighbors_copy)])
    else:
        apply_rule(graph, lcomp, [(vertex, neighbors_copy)])

def apply_pivot(graph: BaseGraph[VT,ET], matched_vertices):
    """
    Applies pivoting on edge dependent on phase of its adjacent vertices and whether they are boundary or not

    - For each adjacent vertex v0 which is boundary an additional vertex and (Hadamard) wire is inserted to make v0 interior
    - For each adjacent vertex v0 with non-Clifford phase p, two additional vertices v1 and v2 are inserted on top:
    v0 has phase 0, v1 has phase 0 and is connected to v0, v2 has phase p and is connected to v1
    Pivoting removes v0, so v1 becomes a YZ spider with v2 corresponding to the measurement effect.

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    matched_vertices (V,V): adjacent vertices of edge

    Returns: 
    Nothing
    """
    vertex1, vertex2 = matched_vertices

    for vertex in [vertex1, vertex2]:
        identity_inserted = True
        while identity_inserted:
            identity_inserted = False

            for neighbor in graph.neighbors(vertex):
                if graph.types()[neighbor] == VertexType.BOUNDARY:
                    insert_identity(graph, vertex, neighbor)
                    identity_inserted = True
                    break

        phase_type = get_phase_type(graph.phases()[vertex])

        if phase_type != PhaseType.CLIFFORD:
            # Insert a phase gadget at the vertex with phase 0
            _mid_vertex, _gadget_top_vertex = insert_phase_gadget(graph, vertex, Fraction(0,1))

    # Apply the pivot rule to the matched vertices
    apply_rule(graph, pivot, [(vertex1, vertex2, [], [])])

def apply_match(graph: BaseGraph[VT,ET], match, local_complement_matches, pivot_matches, include_boundaries=False, include_gadgets=False):
    """
    Applies the given match to the graph and updates the lists of local complement and pivot matches.

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e., ZX-diagram
    match (Dict[str, str | MatchLcompHeuristicType | MatchPivotHeuristicType]): The match to apply.
    local_complement_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
    pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting
    include_boundaries (bool): whether to include boundary spiders

    Returns:
    Tuple[List[MatchLcompHeuristicType], List[MatchPivotHeuristicType]]: The updated lists of local complement and pivot matches
    """
    if match["match type"] == "lcomp":
        apply_lcomp(graph=graph, match=match["match"][1])
        local_complement_matches, pivot_matches = update_matches(graph=graph, vertex_neighbors=match["match"][1][1], removed_vertices=[match["match"][1][0]], lcomp_matches=local_complement_matches, pivot_matches=pivot_matches, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
    else:
        vertex_neighbors = set()
        for vertex in match["match"][1]:
            for vertex_neighbor in graph.neighbors(vertex):
                if vertex_neighbor not in match["match"][1]:
                    vertex_neighbors.add(vertex_neighbor)

        apply_pivot(graph=graph, matched_vertices=match["match"][1])
        local_complement_matches, pivot_matches = update_matches(graph=graph, vertex_neighbors=list(vertex_neighbors), removed_vertices=match["match"][1], lcomp_matches=local_complement_matches, pivot_matches=pivot_matches, include_boundaries=include_boundaries, include_gadgets=include_gadgets)

    
    
    
    return local_complement_matches, pivot_matches




def full_search_match_with_best_result_at_depth(
    graph: BaseGraph[VT, ET], 
    local_complement_matches: List[Tuple], 
    pivot_matches: List[Tuple], 
    lookahead: int, 
    include_boundaries: bool, 
    include_gadgets: bool
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
    """
    Perform a depth-first search on the graph to find the best result at a specific depth.

    This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
    it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
    The search explores every single path to the maximum lookahead depth.

    Parameters:
    graph (Graph): The graph to search.
    local_complement_matches (list): List of local complement matches.
    pivot_matches (list): List of pivot matches.
    lookahead (int): The depth at which to find the best result.
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)

    Returns:
    The best result found at the lookahead depth and the match that led to it.
    """
    apply_match_partial = partial(
        apply_match, 
        include_boundaries=include_boundaries, 
        include_gadgets=include_gadgets
    )
    return _depth_search_for_best_result(
        graph=graph, 
        local_complement_matches=local_complement_matches, 
        pivot_matches=pivot_matches, 
        lookahead=lookahead, 
        apply_match=apply_match_partial,
        full_subgraphs=True
    )

def search_match_with_best_result_at_depth(
    graph: BaseGraph[VT, ET], 
    local_complement_matches: List[Tuple], 
    pivot_matches: List[Tuple], 
    lookahead: int, 
    include_boundaries: bool, 
    include_gadgets: bool
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
    """
    Perform a depth-first search on the graph to find the best result at a specific depth.

    This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
    it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
    The search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.

    Parameters:
    graph (Graph): The graph to search.
    local_complement_matches (list): List of local complement matches.
    pivot_matches (list): List of pivot matches.
    lookahead (int): The depth at which to find the best result.
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)

    Returns:
    The best result found at the lookahead depth and the match that led to it.
    """
    apply_partial = partial(
        apply_match, 
        include_boundaries=include_boundaries, 
        include_gadgets=include_gadgets
    )
    return _depth_search_for_best_result(
        graph=graph, 
        local_complement_matches=local_complement_matches, 
        pivot_matches=pivot_matches, 
        lookahead=lookahead, 
        apply_match=apply_partial
    )

def _update_best_result(
    current_result: Optional[List[dict]], 
    best_result: Optional[List[dict]]
    )-> Tuple[Optional[List[dict]], Optional[int]]:
    
    if best_result is None or sum([match["match"][0] for match in current_result]) > sum([match["match"][0] for match in best_result]):
        best_result = current_result
    return best_result


def _depth_search_for_best_result(
    graph: BaseGraph[VT, ET], 
    local_complement_matches: List[Tuple], 
    pivot_matches: List[Tuple],
    apply_match: Callable,
    lookahead: int,
    depth: int = 0, 
    best_result: Optional[List[dict]] = None, 
    current_match_list: List[dict] = [],
    full_subgraphs: bool = False
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
    """
    Perform a depth-first search on the graph to find the best result at a specific depth.
    
    This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
    it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
    If 'full_subgraphs' is False, the search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.
    
    Parameters:
    graph (Graph): The graph to search.
    local_complement_matches (list): List of local complement matches.
    pivot_matches (list): List of pivot matches.
    lookahead (int): The depth at which to find the best result.
    apply_match (Callable): The function to apply a match to the graph.
    depth (int): The current depth of the search.
    best_result (list): The best result found so far.
    current_match_list (list): The current list of matches.
    full_subgraphs (bool): Whether to consider full subgraphs or not.
    """

    if depth == lookahead:
        current_results = get_matches_from_beginning_middle_end(n=1, local_complement_matches=local_complement_matches, pivot_matches=pivot_matches)
        if not current_results or (sum([match["match"][0] for match in current_match_list]) <= 0 and len(current_match_list) > 0):
            return best_result
        current_result = current_results[0]
        best_result = _update_best_result(current_result=current_match_list + [current_result], best_result=best_result)
        return best_result if sum([match["match"][0] for match in best_result]) > 0 else None

    if not local_complement_matches and not pivot_matches:
        return _update_best_result(current_result=current_match_list, best_result=best_result) if current_match_list else best_result

    num_matches = len(local_complement_matches) + len(pivot_matches)
    num_sub_branches = max(int(num_matches  ** (1/(depth+2))), 1) if not full_subgraphs else max(num_matches, 1)

    matches = get_matches_from_beginning_middle_end(n=num_sub_branches, local_complement_matches=local_complement_matches, pivot_matches=pivot_matches)

    for match in matches:
        lookahead_graph = graph.clone()
        lookahead_local_complement_matches, lookahead_pivot_matches = apply_match(graph=lookahead_graph, match=match, local_complement_matches=local_complement_matches, pivot_matches=pivot_matches)
        
        current_result = _depth_search_for_best_result(
            graph=lookahead_graph, 
            local_complement_matches=lookahead_local_complement_matches, 
            pivot_matches=lookahead_pivot_matches, 
            lookahead=lookahead, 
            apply_match=apply_match, 
            depth=depth + 1, 
            best_result=best_result, 
            current_match_list=current_match_list + [match]
        )
        
        best_result = _update_best_result(current_result=current_result, best_result=best_result)

    return best_result

def greedy_wire_reduce(
    graph: BaseGraph[VT,ET], 
    include_boundaries=False, 
    include_gadgets=False, 
    max_vertex_index=None, 
    threshold=1, 
    lookahead=0, 
    quiet=True, 
    stats=None
    ):

    """
    Perform a greedy Hadamard wire reduction on the given graph.

    This function iteratively applies the best local complement or pivot match to the graph until no further improvements can be made. The "best" match is determined by a heuristic that considers the number of Hadamard wires added by the match.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to simplify.
    include_boundaries (bool): Whether to include boundary spiders in the search for matches. Defaults to False.
    include_gadgets (bool): Whether to include non-Clifford spiders in the search for matches. Defaults to False.
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine. This is needed to prevent non-termination in the case of heuristic_threshold<0.
    threshold (int): Lower bound for heuristic result. Any rule application which adds more than this number of Hadamard wires is filtered out. Defaults to 1.
    lookahead (int): The number of steps to look ahead when searching for the best match. Defaults to 0.

    Returns:
    int: The number of rule applications, i.e., the number of iterations the function went through to simplify the graph.
    """
    has_changes_occurred = True
    rule_application_count = 0

    applied_matches = []

    local_complement_matches, pivot_matches = generate_filtered_matches(
        graph=graph, 
        include_boundaries=include_boundaries, 
        include_gadgets=include_gadgets, 
        max_vertex_index=max_vertex_index, 
        threshold=threshold
    )     

    while has_changes_occurred:
        
        has_changes_occurred = False
           
        best_match_list = search_match_with_best_result_at_depth(
            graph=graph, 
            local_complement_matches=local_complement_matches, 
            pivot_matches=pivot_matches, 
            lookahead=lookahead, 
            include_boundaries=include_boundaries, 
            include_gadgets=include_gadgets
        )
        
        if best_match_list is not None:
            local_complement_matches, pivot_matches = apply_match(
                graph=graph, 
                match=best_match_list[0], 
                local_complement_matches=local_complement_matches, 
                pivot_matches=pivot_matches
            )

            rule_application_count += 1
            has_changes_occurred = True

            applied_matches.append(best_match_list[0])

    return rule_application_count, applied_matches

def random_wire_reduce(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, max_vertex_index=None, threshold=1, quiet=True, stats=None):
    """
    Random Hadamard wire reduction

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of heuristic_threshold<0).
    threshold (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

    Returns:
    int: The number of iterations, i.e. rule applications
    """
    has_changes_occurred = True
    rule_application_count = 0

    while has_changes_occurred:
        has_changes_occurred = False
        local_complement_matches, pivot_matches = generate_filtered_matches(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, max_vertex_index=max_vertex_index, threshold=threshold)
        if apply_random_match(graph, local_complement_matches, pivot_matches):
            rule_application_count += 1
            has_changes_occurred = True

    return rule_application_count

def simulated_annealing_reduce(graph: BaseGraph[VT,ET], initial_temperature=100, cooling_factor=0.95, threshold=-100000, quiet=True, stats=None):
    """
    Hadamard wire reduction with simulated annealing (does not work very well yet)

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    initial_temperature (int): Initial temperature for the simulated annealing process
    cooling_factor (float): Factor by which the temperature is reduced in each iteration
    threshold (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

    Returns:
    int: 0
    """
    temperature = initial_temperature
    minimum_temperature = 0.01
    iteration_count = 0

    while temperature > minimum_temperature:
        iteration_count += 1
        local_complement_matches, pivot_matches = generate_filtered_matches(graph, include_boundaries=True, include_gadgets=True, threshold=threshold)

        rule_type, selected_match = get_random_match(local_complement_matches, pivot_matches)
        if rule_type == "none":
            temperature = 0
            break

        if selected_match[0] < 0:
            # If the probability of accepting the match is less than a random probability
            if math.exp(selected_match[0]/temperature) < random.random():
                temperature *= cooling_factor
                continue

        if rule_type == "pivot":
            apply_pivot(graph, selected_match[1])
        else:
            apply_lcomp(graph, selected_match[1])

        temperature *= cooling_factor

    return 0