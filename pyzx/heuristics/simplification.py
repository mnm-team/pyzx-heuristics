import math
import random

from fractions import Fraction
from typing import Tuple, List
from functools import partial

from .heuristics import PhaseType, get_phase_type, lcomp_heuristic, pivot_heuristic
from .tools import insert_identity, insert_phase_gadget

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rules import apply_rule, lcomp, pivot
from pyzx.utils import VertexType, EdgeType


MatchLcompHeuristicType = Tuple[float,Tuple[VT,List[VT]],int]

def lcomp_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False) -> List[MatchLcompHeuristicType]:
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
    vertex_types = graph.types()

    matches = []
    while len(vertex_candidates) > 0:
        current_vertex = vertex_candidates.pop()
        current_vertex_type = vertex_types[current_vertex]
        current_vertex_phase = graph.phase(current_vertex)
        
        if current_vertex_type != VertexType.Z: continue
        # Check if the vertex needs to be transformed into an XZ spider
        needs_gadget = get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD
        # Skip if gadgets are not allowed and the vertex needs to be gadgetized
        if include_gadgets == False and needs_gadget: continue
        # Skip if the vertex has only one neighbor (i.e., it's a leaf node)
        if len(graph.neighbors(current_vertex)) == 1: continue
                
        current_vertex_neighbors = list(graph.neighbors(current_vertex))
        is_already_gadget = False
        boundary_count = 0
        for neighbor in current_vertex_neighbors:
            # Check if the neighbor is a leaf node and the vertex needs to be gadgetized
            if len(graph.neighbors(neighbor)) == 1 and get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD:
                is_already_gadget = True
            if vertex_types[neighbor] != VertexType.Z:
                boundary_count += 1

        if is_already_gadget and needs_gadget: continue
        if not include_boundaries and boundary_count > 0: continue

        spider_count = -1 + boundary_count + (2 if needs_gadget else 0)

        # Calculate the heuristic and add the match to the list
        if boundary_count > 0:
            matches.append((lcomp_heuristic(graph,current_vertex)-boundary_count,(current_vertex,current_vertex_neighbors),spider_count))
        else:
            matches.append((lcomp_heuristic(graph,current_vertex),(current_vertex,current_vertex_neighbors),spider_count))
        
    return matches

MatchPivotHeuristicType = Tuple[float,Tuple[VT,VT]]

def pivot_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False) -> List[MatchPivotHeuristicType]:
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
    vertex_types = graph.types()
    matches = []

    while len(edge_candidates) > 0:
        edge = edge_candidates.pop()
        if graph.edge_type(edge) != EdgeType.HADAMARD: continue

        # Get the vertices at the ends of this edge
        vertex0, vertex1 = graph.edge_st(edge)

        # Skip this edge if both vertices are not Z vertices
        if not (vertex_types[vertex0] == VertexType.Z and vertex_types[vertex1] == VertexType.Z): continue

        is_vertex0_not_clifford = get_phase_type(graph.phase(vertex0)) != PhaseType.CLIFFORD
        is_vertex1_not_clifford = get_phase_type(graph.phase(vertex1)) != PhaseType.CLIFFORD

        if include_gadgets == False and (is_vertex0_not_clifford or is_vertex1_not_clifford): continue
        # Skip if both vertices are either true clifford or not clifford phase types
        if is_vertex0_not_clifford and is_vertex1_not_clifford: continue 
        # Skip if the vertices have only one neighbor (i.e., they are leaf nodes)
        if len(graph.neighbors(vertex0)) == 1 or len(graph.neighbors(vertex1)) == 1: continue 

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

        if (vertex0_already_gadget and is_vertex0_not_clifford) or (vertex1_already_gadget and is_vertex1_not_clifford): continue
        if not include_boundaries and boundary_count > 0: continue

        spider_count = -2 + boundary_count + (2 if is_vertex0_not_clifford else 0) + (2 if is_vertex1_not_clifford else 0)

        if include_boundaries:
            matches.append((pivot_heuristic(graph,edge)-boundary_count,(vertex0,vertex1), spider_count))
        else:
            matches.append((pivot_heuristic(graph,edge),(vertex0,vertex1), spider_count))

    return matches

def generate_filtered_matches(graph, include_boundaries=False, include_gadgets=False, max_vertex_index=None, threshold=1):
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
    local_complement_matches = lcomp_matcher(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets)
    pivot_matches = pivot_matcher(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets)

    filtered_local_complement_matches = []
    filtered_pivot_matches = []

    for match in local_complement_matches:
        wire_reduction, vertices, spider_count = match
        # Skip matches that do not meet the heuristic threshold
        if wire_reduction < threshold:
            continue
        # Skip matches that could cause non-termination
        if max_vertex_index and wire_reduction <= 0 and vertices[0] > max_vertex_index:
            continue
        filtered_local_complement_matches.append((wire_reduction, vertices, spider_count))
    
    for match in pivot_matches:
        wire_reduction, vertices, spider_count = match
        # Skip matches that do not meet the heuristic threshold
        if wire_reduction < threshold:
            continue
        # Skip matches that could cause non-termination
        if max_vertex_index and wire_reduction <= 0 and vertices[0] > max_vertex_index and vertices[1] > max_vertex_index:
            continue
        filtered_pivot_matches.append((wire_reduction, vertices, spider_count))

    return (filtered_local_complement_matches, filtered_pivot_matches)

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

def get_best_match(local_complement_matches, pivot_matches):
    """
    Returns the rule and matchin vertex with the best heuristic result, i.e., the rule which eliminates the most Hadamard wires

    Parameters:
    local_complement_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
    pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

    Returns:
    match (Dict[str, MatchLcompHeuristicType | MatchPivotHeuristicType]): The best match. None if no match has been applied.
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
        return {"match": pivot_matches[0], "match type": "pivot"}
    else:
        return {"match": local_complement_matches[0], "match type": "lcomp"}



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



def search_match_with_best_result_at_depth(graph: BaseGraph[VT,ET], local_complement_matches, pivot_matches, lookahead, include_boundaries, include_gadgets, threshold, max_vertex_index):
    """
    Perform a depth-first search on the graph to find the best result at a specific depth.

    This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
    it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
    The search continues until all paths have been explored to the lookahead depth.

    Parameters:
    graph (Graph): The graph to search.
    local_complement_matches (list): List of local complement matches.
    pivot_matches (list): List of pivot matches.
    lookahead (int): The depth at which to find the best result.
    include_boundaries (bool): whether to include boundary spiders
    include_gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of heuristic_threshold<0).
    threshold (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

    Returns:
    The best result found at the lookahead depth and the match that led to it.
    """
    generate_filtered_matches_partial = partial(generate_filtered_matches, include_boundaries=include_boundaries, include_gadgets=include_gadgets, max_vertex_index=max_vertex_index, threshold=threshold)
    return _depth_search_for_best_result(graph, local_complement_matches, pivot_matches, lookahead, generate_filtered_matches_partial)

def _depth_search_for_best_result(graph: BaseGraph[VT,ET], local_complement_matches, pivot_matches, lookahead, generate_filtered_matches, depth=0, best_result=None, best_match=None):
    """
    Parameters:
    depth (int, optional): Current depth of the search. Defaults to 0.
    best_result: The best result found so far. Defaults to None.
    best_match: The match that led to the best result. Defaults to None.

    Returns:
    The best result found at the look-ahead depth and the match that led to it.
    """
    if depth == lookahead:
        current_result = get_best_match(local_complement_matches, pivot_matches)
        if best_result is None or current_result["match"][0] > best_result["match"][0]:
            best_result = current_result

        return best_result
    
    elif depth == 0:
        for local_complement_match in local_complement_matches:
            lookahead_graph = graph.clone()
            apply_lcomp(lookahead_graph, local_complement_match[1])
            lookahead_local_complement_matches, lookahead_pivot_matches = generate_filtered_matches(lookahead_graph)
            current_result = _depth_search_for_best_result(lookahead_graph, lookahead_local_complement_matches, lookahead_pivot_matches, lookahead, generate_filtered_matches, depth + 1, best_result)
            if best_result is None or current_result["match"][0] > best_result["match"][0]:
                best_result = current_result
                best_match = {"match": local_complement_match, "match type": "lcomp"}

        for pivot_match in pivot_matches:
            lookahead_graph = graph.clone()
            apply_pivot(lookahead_graph, pivot_match[1])
            lookahead_local_complement_matches, lookahead_pivot_matches = generate_filtered_matches(lookahead_graph)
            current_result = _depth_search_for_best_result(lookahead_graph, lookahead_local_complement_matches, lookahead_pivot_matches, lookahead, generate_filtered_matches, depth + 1, best_result)
            if best_result is None or current_result["match"][0] > best_result["match"][0]:
                best_result = current_result
                best_match = {"match": pivot_match, "match type": "pivot"}

        return best_match

    else:
        for local_complement_match in local_complement_matches:
            lookahead_graph = graph.clone()
            apply_lcomp(lookahead_graph, local_complement_match[1])
            lookahead_local_complement_matches, lookahead_pivot_matches = generate_filtered_matches(lookahead_graph)
            best_result = _depth_search_for_best_result(lookahead_graph, lookahead_local_complement_matches, lookahead_pivot_matches, lookahead, generate_filtered_matches, depth + 1, best_result)

        for pivot_match in pivot_matches:
            lookahead_graph = graph.clone()
            apply_pivot(lookahead_graph, pivot_match[1])
            lookahead_local_complement_matches, lookahead_pivot_matches = generate_filtered_matches(lookahead_graph)
            best_result = _depth_search_for_best_result(lookahead_graph, lookahead_local_complement_matches, lookahead_pivot_matches, lookahead, generate_filtered_matches, depth + 1, best_result)

        return best_result



def greedy_wire_reduce(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, max_vertex_index=None, threshold=1, lookahead=0, quiet=True, stats=None):
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

    while has_changes_occurred:

        has_changes_occurred = False
        local_complement_matches, pivot_matches = generate_filtered_matches(graph, include_boundaries=include_boundaries, include_gadgets=include_gadgets, max_vertex_index=max_vertex_index, threshold=threshold)        
        best_match = search_match_with_best_result_at_depth(graph, local_complement_matches, pivot_matches, lookahead, include_boundaries, include_gadgets, threshold, max_vertex_index)
        
        if best_match is not None:
            if best_match["match type"] == "lcomp":
                apply_lcomp(graph, best_match["match"][1])
            else:
                apply_pivot(graph, best_match["match"][1])

            rule_application_count += 1
            has_changes_occurred = True

    return rule_application_count

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