
from enum import Enum
import logging
import math
import random

from fractions import Fraction
import time
from typing import Callable, Dict, Set, Tuple, List
import warnings

import numpy as np

from .heuristics import PhaseType, get_phase_type, lcomp_heuristic, lcomp_heuristic_neighbor_unfusion, pivot_heuristic, pivot_heuristic_neighbor_unfusion
from .tools import split_phases, insert_identity
from .simplification import get_random_match
from .flow_calculation import cflow, update_gflow_from_double_insertion, update_gflow_from_lcomp, update_gflow_from_pivot

from pyzx.rules import apply_rule, lcomp, pivot
from pyzx.utils import VertexType, EdgeType
from pyzx.gflow import gflow
from pyzx.graph.base import BaseGraph, VT, ET


MatchLcompHeuristicType = Tuple[float, List[VT], VT]

MatchPivotHeuristicType = Tuple[float, VT, VT]

def check_lcomp_match(graph, vertex, include_boundaries=False, include_gadgets=False, check_for_unfusions = True, calculate_heuristic=True) -> List[Tuple[Tuple[VT], MatchLcompHeuristicType]] | None:
    vertex_types = graph.types()

    current_vertex_type = vertex_types[vertex]
    current_vertex_phase = graph.phase(vertex)
    
    if current_vertex_type != VertexType.Z: return None
    current_vertex_neighbors = list(graph.neighbors(vertex))

    # TODO: implement include_gadgets
    # Check if the vertex needs to be transformed into an XZ spider
    # needs_gadget = get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD
    # Skip if gadgets are not allowed and the vertex needs to be gadgetized
    # if include_gadgets == False and needs_gadget: return None
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

    # if is_already_gadget and needs_gadget: return None
    if not include_boundaries and boundary_count > 0: return None

    if not calculate_heuristic:
        return ((vertex,), (0,current_vertex_neighbors,0))
    
    # spider_count = -1 + boundary_count + (2 if needs_gadget else 0)

    if not calculate_heuristic:
        return ((vertex,), (0,current_vertex_neighbors,None))
    
    if not include_boundaries:
        boundary_count = 0

    if get_phase_type(current_vertex_phase) == PhaseType.TRUE_CLIFFORD:
        return [((vertex,), (lcomp_heuristic(graph,vertex)-boundary_count,current_vertex_neighbors,None))]
    elif check_for_unfusions:
        matches = []
        for neighbor in get_all_possible_unfusion_neighbours(graph, vertex, None):
            matches.append(((vertex,),(lcomp_heuristic_neighbor_unfusion(graph,vertex,neighbor)-boundary_count,current_vertex_neighbors,neighbor)))
        if len(matches) > 0:
            return matches
        return None

def check_pivot_match(graph, edge, include_boundaries=False, include_gadgets=False, check_for_unfusions=True, calculate_heuristic=True) -> List[Tuple[Tuple[VT, VT], MatchPivotHeuristicType]] | None:
    
    vertex_types = graph.types()

    if graph.edge_type(edge) != EdgeType.HADAMARD: return None
        
    vertex0, vertex1 = graph.edge_st(edge)

    if vertex0 == vertex1: return None

    if not (vertex_types[vertex0] == VertexType.Z and vertex_types[vertex1] == VertexType.Z): return None
    
    is_vertex0_not_clifford = get_phase_type(graph.phase(vertex0)) != PhaseType.CLIFFORD
    is_vertex1_not_clifford = get_phase_type(graph.phase(vertex1)) != PhaseType.CLIFFORD

    # TODO: implement include_gadgets
    # if include_gadgets == False and (is_vertex0_not_clifford or is_vertex1_not_clifford): return None
    # Skip if both vertices are either true clifford or not clifford phase types
    # if is_vertex0_not_clifford and is_vertex1_not_clifford: return None 
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

    # if (vertex0_already_gadget and is_vertex0_not_clifford) or (vertex1_already_gadget and is_vertex1_not_clifford): return None
    if not include_boundaries and boundary_count > 0: return None

    if not calculate_heuristic:
        return ((vertex0, vertex1), (0, None, None))
    
    if not include_boundaries:
        boundary_count = 0

    matches = []
    if get_phase_type(graph.phase(vertex0)) == PhaseType.CLIFFORD:
        if get_phase_type(graph.phase(vertex1)) == PhaseType.CLIFFORD:
            return [((vertex0, vertex1), (pivot_heuristic(graph,edge)-boundary_count,None,None))]
        elif check_for_unfusions:
            for neighbor in get_all_possible_unfusion_neighbours(graph, vertex1, vertex0):
                matches.append(((vertex0, vertex1),(pivot_heuristic_neighbor_unfusion(graph,edge,None,neighbor)-boundary_count,None,neighbor)))
    elif check_for_unfusions:
        if get_phase_type(graph.phase(vertex1)) == PhaseType.CLIFFORD:
            for neighbor in get_all_possible_unfusion_neighbours(graph, vertex0, vertex1):
                matches.append(((vertex0, vertex1),(pivot_heuristic_neighbor_unfusion(graph,edge,neighbor,None)-boundary_count,neighbor,None)))
        else:
            for neighbor_v0 in get_all_possible_unfusion_neighbours(graph, vertex0, vertex1):
                for neighbor_v1 in get_all_possible_unfusion_neighbours(graph, vertex1, vertex0):
                    matches.append(((vertex0, vertex1), (pivot_heuristic_neighbor_unfusion(graph,edge,neighbor_v0,neighbor_v1)-boundary_count,neighbor_v0,neighbor_v1)))
    if len(matches) > 0:
        return matches
    return None




def lcomp_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, check_for_unfusions=True, calculate_heuristic=True) -> Dict[Tuple[VT], MatchLcompHeuristicType]:
    """
    Generates all matches for local complementation in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders.
    include_gadgets (bool): whether to include non-Clifford spiders.
    check_for_unfusions (bool): whether to check for unfusions.
    calculate_heuristic (bool): whether to calculate the heuristic value for each match

    Returns:
    Dict[Tuple[VT], MatchLcompHeuristicType]: A dictionary of match tuples match_key:(heuristic,vertices,spider_count), where heuristic is the LCH, vertices are the neighbor vertices and spider_count the amount of saved/added spiders
    """
    vertex_candidates = graph.vertex_set()

    matches = {}

    while len(vertex_candidates) > 0:
        current_vertex = vertex_candidates.pop()
        match_list = check_lcomp_match(graph, current_vertex, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions, calculate_heuristic=calculate_heuristic)

        if match_list is not None:
            for match in match_list:
                match_key, match_value = match
                matches[match_key] = match_value
    
    return matches

def pivot_matcher(graph: BaseGraph[VT,ET], include_boundaries=False, include_gadgets=False, check_for_unfusions=True, calculate_heuristic=True) -> Dict[Tuple[VT,VT], MatchPivotHeuristicType]:
    """
    Generates all matches for pivoting in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    include_boundaries (bool): whether to include boundary spiders.
    include_gadgets (bool): whether to include non-Clifford spiders.
    check_for_unfusions (bool): whether to check for unfusions.
    calculate_heuristic (bool): whether to calculate the heuristic value for each match

    Returns:
    Dict[Tuple[VT,VT], MatchPivotHeuristicType]: A dictionary of match tuples match_key:(heuristic,spider_count), where heuristic is the LCH and spider_count the amount of saved/added spiders
    """
    edge_candidates = graph.edge_set()
    matches = {}

    while len(edge_candidates) > 0:
        edge = edge_candidates.pop()
        match_list = check_pivot_match(graph, edge, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions, calculate_heuristic=calculate_heuristic)

        if match_list is not None:
            for match in match_list:
                match_key, match_value = match
                matches[match_key] = match_value

    return matches



def update_lcomp_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: Tuple[VT], lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], neighbors_of_neighbors: Set[VT], include_boundaries=False, include_gadgets=False, check_for_unfusions=True) -> Dict[VT, MatchLcompHeuristicType]:
    # Iterate over the current local complement matches
    lcomp_matches_copy = lcomp_matches.copy()
    keys_to_remove = set()

    for vertex_match, (heuristic, vertex_match_neighbors, unfusion_vertex) in lcomp_matches_copy.items():

        if vertex_match[0] in removed_vertices:
            keys_to_remove.add(vertex_match)
            continue

        if any(element in vertex_match_neighbors for element in removed_vertices):
            new_matches = check_lcomp_match(graph, vertex_match[0], include_boundaries=include_boundaries, include_gadgets=include_gadgets)
            if new_matches is None:
                keys_to_remove.add(vertex_match)
            else:
                for match in new_matches:
                    match_key, match_value = match
                    lcomp_matches_copy[match_key] = match_value
            continue

        # If the vertex is in the set of neighbors of neighbors, recalculate the heuristic
        if vertex_match in neighbors_of_neighbors:
            new_matches = check_lcomp_match(graph, neighbor, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions)
            if new_matches is not None:
                for match in new_matches:
                    match_key, match_value = match
                    lcomp_matches_copy[match_key] = match_value

    for key in keys_to_remove:
        del lcomp_matches_copy[key]

    # Check for new local complement matches in the vertex neighbors
    for neighbor in vertex_neighbors:
        new_matches = check_lcomp_match(graph, neighbor, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions)
        if new_matches is not None:
            for match in new_matches:
                match_key, match_value = match
                lcomp_matches_copy[match_key] = match_value

    return lcomp_matches_copy

def update_pivot_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: Tuple[VT], pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType], neighbors_of_neighbors: Set[VT], include_boundaries=False, include_gadgets=False, check_for_unfusions=True) -> Dict[Tuple[VT,VT], MatchPivotHeuristicType]:
    
    pivot_matches_copy = pivot_matches.copy()
    keys_to_remove = set()

    for edge, (heuristic, neighbor1, neighbor2) in pivot_matches_copy.items():
        vertex0, vertex1 = edge

        if vertex0 in removed_vertices or vertex1 in removed_vertices:
            keys_to_remove.add(edge)
            continue

        if not graph.connected(vertex0, vertex1):
            keys_to_remove.add(edge)
            continue

        # TODO maybe remove as well wenn neighbor1 or 2 are in removed_vertices
        if neighbor1 in removed_vertices or neighbor2 in removed_vertices:
            keys_to_remove.add(edge)
            continue

        # If the vertices are in the set of neighbors of neighbors, recalculate the heuristic
        if vertex0 in neighbors_of_neighbors or vertex1 in neighbors_of_neighbors:
            new_matches = check_pivot_match(graph, edge, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions)
            if new_matches is None:
                keys_to_remove.add(edge)
            else:
                for match in new_matches:
                    match_key, match_value = match
                    pivot_matches_copy[match_key] = match_value
    
    for key in keys_to_remove:
        del pivot_matches_copy[key]

    # Check for new pivot matches in the vertex neighbors
    for vertex_neighbor in vertex_neighbors:
        for neighbor_of_neighbor in graph.neighbors(vertex_neighbor):
            if graph.connected(vertex_neighbor, neighbor_of_neighbor):
                edge = graph.edge(vertex_neighbor, neighbor_of_neighbor)
                new_matches = check_pivot_match(graph, edge, include_boundaries=include_boundaries, include_gadgets=include_gadgets, check_for_unfusions=check_for_unfusions)
                if new_matches is not None:
                    for match in new_matches:
                        match_key, match_value = match
                        pivot_matches_copy[match_key] = match_value

    return pivot_matches_copy

def update_matches(graph: BaseGraph[VT,ET], vertex_neighbors: List[VT], removed_vertices: Tuple[VT], lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType], include_boundaries=False, include_gadgets=False, check_for_unfusions=True, max_vertex_index=None) -> Tuple[Dict[VT, MatchLcompHeuristicType], Dict[Tuple[VT,VT], MatchPivotHeuristicType]]:
    """
    Updates the dict of local complement and pivot matches after a local complementation or pivot has been applied.

    Parameters:
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    vertex_neighbors (List[VT]): The neighbors of the vertex where the local complementation or pivot was applied
    removed_vertices (Tuple[VT]): The vertices that were removed by the local complementation or pivot
    lcomp_matches (Dict[Tuple[VT], MatchLcompHeuristicType]): The current dict of local complement matches
    pivot_matches (Dict[Tuple[VT,VT], MatchPivotHeuristicType]): The current dict of pivot matches
    include_boundaries (bool): whether to include boundary spiders.
    include_gadgets (bool): whether to include non-Clifford spiders.
    check_for_unfusions (bool): whether to check for unfusions.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.

    Returns:
    Tuple[Dict[VT, MatchLcompHeuristicType], Dict[Tuple[VT,VT], MatchPivotHeuristicType]]: The updated dictonaries of local complement and pivot matches.
    """

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
                                         check_for_unfusions=check_for_unfusions,
                                         include_boundaries=include_boundaries, 
                                         include_gadgets=include_gadgets)
    
    pivot_matches = update_pivot_matches(graph=graph, 
                                         vertex_neighbors=vertex_neighbors, 
                                         removed_vertices=removed_vertices, 
                                         pivot_matches=pivot_matches, 
                                         neighbors_of_neighbors=neighbors_of_neighbors, 
                                         check_for_unfusions=check_for_unfusions,
                                         include_boundaries=include_boundaries, 
                                         include_gadgets=include_gadgets)

    for match_key in list(lcomp_matches.keys()):
        if max_vertex_index and match_key[0] > max_vertex_index:
            del lcomp_matches[match_key]

    for match_key in list(pivot_matches.keys()):
        if max_vertex_index and (match_key[0] > max_vertex_index or match_key[1] > max_vertex_index):
            del pivot_matches[match_key]

    return lcomp_matches, pivot_matches




def get_possible_unfusion_neighbours(graph: BaseGraph[VT,ET], current_vertex, exclude_vertex=None):
    """
    Get the possible neighbors for unfusion of a given vertex in a graph.
    Only neighbors with 2 or less neighbors are considered.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    current_vertex (VT): The vertex to find possible unfusion neighbors for.
    exclude_vertex (VT, optional): A vertex to exclude from the possible unfusion neighbors.

    Returns:
    list: A list of vertices that are possible neighbors for unfusion.
    """
    possible_unfusion_neighbours = set()

    # TODO: check if the following is correct
    for neighbor in graph.neighbors(current_vertex):
        neighbors_of_neighbor = graph.neighbors(neighbor)
        if len(neighbors_of_neighbor) <= 2:
            possible_unfusion_neighbours.add(neighbor)

    if exclude_vertex and exclude_vertex in possible_unfusion_neighbours:
        possible_unfusion_neighbours.remove(exclude_vertex)

    return possible_unfusion_neighbours


def get_all_possible_unfusion_neighbours(graph: BaseGraph[VT,ET], current_vertex, exclude_vertex=None):
    """
    Get all the possible neighbors for unfusion of a given vertex in a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    current_vertex (VT): The vertex to find possible unfusion neighbors for.
    exclude_vertex (VT, optional): A vertex to exclude from the possible unfusion neighbors.

    Returns:
    list: A list of vertices that are possible neighbors for unfusion.
    """
    possible_unfusion_neighbours = set(graph.neighbors(current_vertex))

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


def apply_lcomp(graph: BaseGraph[VT,ET], match) -> Tuple[VT, VT] | None:
    """
    Apply a local complementation operation to a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    match (tuple): The match to apply the operation to.

    Returns:
    tuple: A tuple containing the added unfusion vertices.
    """
    match_key, match_value = match
    vertex = match_key[0]
    neighbors = match_value[1]
    unfusion_neighbor = match_value[2]

    neighbors_copy = neighbors[:]

    new_vertices = None

    if unfusion_neighbor:
        phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbor, Fraction(1,2))
        new_vertices = (phaseless_spider, phase_spider)
        # update_gflow_from_double_insertion(flow, vertex, unfusion_neighbor, phaseless_spider, phase_spider)
        neighbors_copy = [phaseless_spider if neighbor == unfusion_neighbor else neighbor for neighbor in neighbors_copy]

    #TODO: check if update_gflow_from_lcomp is calculating the correct flow after the lcomp
    # update_gflow_from_lcomp(graph, vertex, flow)
    apply_rule(graph, lcomp, [(vertex, neighbors_copy)])

    return new_vertices

def apply_pivot(graph: BaseGraph[VT,ET], match) -> Tuple[VT, ...] | None:
    """
    Apply a pivot operation to a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    match (tuple): The match to apply the operation to.

    Returns:
    tuple: A tuple containing the added unfusion vertices.
    """

    match_key, match_value = match
    vertex_1, vertex_2 = match_key

    unfusion_neighbors = {}
    unfusion_neighbors[vertex_1] = match_value[1]
    unfusion_neighbors[vertex_2] = match_value[2]

    new_vertices = []

    for vertex in [vertex_1, vertex_2]:
        if unfusion_neighbors[vertex]:
            phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbors[vertex], Fraction(0,1))
            new_vertices.append(phaseless_spider)
            new_vertices.append(phase_spider)
            # update_gflow_from_double_insertion(flow, vertex, unfusion_neighbors[vertex], phaseless_spider, phase_spider)
            
    # FIXME: update gflow is not correctly calculating the flow after the pivot
    # update_gflow_from_pivot(graph, vertex_1, vertex_2, flow)

    apply_rule(graph, pivot, [(vertex_1, vertex_2, [], [])])

    return tuple(new_vertices) if len(new_vertices) > 0 else None


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
        new_verticies = apply_pivot(graph, pivot_matches[0], flow)
    else:
        new_verticies = apply_lcomp(graph, local_complement_matches[0], flow)

    return True

def apply_random_match(graph, local_complement_matches, pivot_matches, flow):
    """
    Apply a random match from the list of local complementation and pivot matches to a graph.

    Parameters:
    graph (BaseGraph): The graph to apply the match to.
    local_complement_matches (list): The list of local complementation matches.
    pivot_matches (list): The list of pivot matches.
    flow (dict): The dictionary representing the flow of the graph.

    Returns:
    bool: True if a match was applied, False otherwise.
    """
    operation_type, match = get_random_match(local_complement_matches, pivot_matches)

    if operation_type == "pivot":
        new_verticies = apply_pivot(graph, match, flow)
    elif operation_type == "lcomp":
        new_verticies = apply_lcomp(graph, match, flow)
    else:
        return False

    return True


class FilterFlowFunc(Enum):
    """
    An enumeration of the filter flow functions.
    """
    NONE = lambda _: {}
    G_FLOW_PRESERVING = gflow
    C_FLOW_PRESERVING = cflow

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

class WireReducer:
    def __init__(
        self,
        graph: BaseGraph[VT, ET],
        include_boundaries=False,
        include_gadgets=False,
        use_neighbor_unfusion=False,
        max_vertex_index=None,
        threshold=0,
        lookahead=0,
        flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
        quiet=True,
        stats=None,
    ):
        """
        Initializes a WireReducer object.

        Args:
            graph (BaseGraph[VT, ET]): The graph to be reduced.
            include_boundaries (bool): Whether to include boundary spiders.
            include_gadgets (bool): Whether to include non-Clifford spiders.
            use_neighbor_unfusion (bool): Whether to use neighbor unfusion.
            max_vertex_index: The maximum vertex index.
            threshold (int): Lower bound for heuristic result.
            lookahead (int): The depth at which to find the best result.
            flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches.
            quiet (bool): Whether to suppress output.
            stats: Statistics object to track the reduction process.
        """
        self.graph = graph
        self.include_boundaries = include_boundaries
        self.include_gadgets = include_gadgets
        self.use_neighbor_unfusion = use_neighbor_unfusion
        self.max_vertex_index = max_vertex_index
        self.threshold = threshold
        self.lookahead = lookahead
        self.flow_function = flow_function
        self.quiet = quiet
        self.stats = stats

        self._lookup_flow_for_unfusion: Dict[Tuple[VT, VT], bool] = {}
        self._use_lookup_flow_for_unfusion = False

        self._possibly_non_flow_preserving_matches = []

        # For logging purposes
        self._rule_application_count = 0
        self._reduction_per_match = []
        self._applied_matches = []
        self._remaining_matches = []
        self._skipped_matches_until_reset = [0 for _ in range(lookahead+1)]
        self._skipped_filter_func_evals = 0
        self._neighbor_unfusions = 0
        self._total_evals = 0
        self._rehabilitated_non_flow_preserving_matches = 0
        

        if self.use_neighbor_unfusion:
            self._use_lookup_flow_for_unfusion = True

        if self.use_neighbor_unfusion and self.flow_function == FilterFlowFunc.NONE:
            self.flow_function = FilterFlowFunc.G_FLOW_PRESERVING
            warnings.warn("Neighbor unfusion requires a flow function. Using G-flow preserving function.")

        if not self.use_neighbor_unfusion and self.flow_function == FilterFlowFunc.G_FLOW_PRESERVING:
            self.flow_function = FilterFlowFunc.NONE
            warnings.warn("G-flow preserving function is not needed without neighbor unfusion. Using no flow function.")
            

    def greedy_wire_reduce(self):
        self.has_changes_occurred = True

        local_complement_matches = lcomp_matcher(self.graph, include_boundaries=self.include_boundaries, include_gadgets=self.include_gadgets, check_for_unfusions=self.use_neighbor_unfusion, calculate_heuristic=True)
        pivot_matches = pivot_matcher(self.graph, include_boundaries=self.include_boundaries, include_gadgets=self.include_gadgets, check_for_unfusions=self.use_neighbor_unfusion, calculate_heuristic=True)

        while self.has_changes_occurred:
            self.has_changes_occurred = False
            local_complement_matches, pivot_matches = self._apply_and_find_new_matches(local_complement_matches=local_complement_matches, pivot_matches=pivot_matches)

            # For testing purposes
            true_flow = self._calculate_flow(self.graph)
            if true_flow is None:
                raise Exception("Flow is not preserved after applying the match")
            
        logging.info(f"Total rule applications: {self._rule_application_count}, Total reduction: {sum(self._reduction_per_match)}, Std reduction: {np.std(self._reduction_per_match)}")
        logging.info(f"Total skipped filter function evaluations: {self._skipped_filter_func_evals}, Total neighbor unfusions: {self._neighbor_unfusions}, Total skipped matches: {self._skipped_matches_until_reset}")
        return sum(self._reduction_per_match), self._applied_matches



    def _reset_lookup_flow(self):
        self._lookup_flow_for_unfusion = {}

    def _lookup_flow_preserving(self, graph: BaseGraph[VT, ET], match) -> bool:
        """
        Looks up whether the flow is preserved if neighbor unfusion is applied to a given edge.

        Args:
            graph (BaseGraph[VT, ET]): The graph to check.
            edge (Tuple[VT, VT]): The edge to check.

        Returns:
            bool: True if the flow is preserved, False otherwise.
        """
        if not self._use_lookup_flow_for_unfusion:
            return self._calculate_flow(graph) is not None

        edges = []
        match_key, match_value = match

        if len(match_key) == 1:
            vertex = match_key[0]
            unfusion_neighbor = match_value[2]
            if unfuse_to_neighbor and unfusion_neighbor:
                edges.append(graph.edge(vertex, unfusion_neighbor))

        elif len(match_key) == 2:
            unfusion_neighbors = {vertex: match_value[i+1] for i, vertex in enumerate(match_key)}
            edges.extend(graph.edge(vertex, unfusion_neighbors[vertex]) for vertex in match_key if unfusion_neighbors[vertex])

        if not all(edge in self._lookup_flow_for_unfusion for edge in edges):
            flow = self._calculate_flow(graph) is not None
            self._neighbor_unfusions += 1
            for edge in edges:
                self._lookup_flow_for_unfusion[edge] = flow
            return flow

        if len(edges) == 0:
            if self.flow_function != FilterFlowFunc.G_FLOW_PRESERVING:
                return self._calculate_flow(graph) is not None
            else:
                # self._skipped_filter_func_evals += 1
                return True

        #FIXME: check if this is correct
        # flows = all(self._lookup_flow_for_unfusion[edge] for edge in edges)
        # self._neighbor_unfusions += 1
        # self._skipped_filter_func_evals += 1
        # return flows
        flow = self._calculate_flow(graph) is not None
        self._neighbor_unfusions += 1
        for edge in edges:
            self._lookup_flow_for_unfusion[edge] = flow
        return flow

    def _calculate_flow(self, graph: BaseGraph[VT, ET]) -> Dict[VT, Set[VT]] | None:
        if self.flow_function == FilterFlowFunc.G_FLOW_PRESERVING:
            return self.flow_function(graph)[1] if self.flow_function(graph) else None
        elif self.flow_function == FilterFlowFunc.C_FLOW_PRESERVING:
            return self.flow_function(graph) if self.flow_function(graph) else None
        else:
            return self.flow_function(graph)

    def _sort_matches(self, lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType]) -> Dict[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]:
        """
        Sorts the matches based on their heuristic result.
        Puts the possibly non-flow-preserving matches at the end of the dict.

        Args:
            lcomp_matches (Dict[Tuple[VT], MatchLcompHeuristicType]): A dict of matches for local complementation.
            pivot_matches (Dict[Tuple[VT,VT], MatchPivotHeuristicType]): A dict of matches for pivoting.

        Returns:
            Dict[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]: A sorted dict of matches.    
        """
        match_dict = dict(sorted({**lcomp_matches, **pivot_matches}.items(), key=lambda item: item[1][0], reverse=True))

        removed_matches = set()

        for (possible_non_flow_preserving_match_key, possible_non_flow_preserving_match_value) in self._possibly_non_flow_preserving_matches:
            if possible_non_flow_preserving_match_key in match_dict:
                del match_dict[possible_non_flow_preserving_match_key]
                match_dict[possible_non_flow_preserving_match_key] = possible_non_flow_preserving_match_value
                pass
            else:
                removed_matches.add((possible_non_flow_preserving_match_key, possible_non_flow_preserving_match_value))
        
        for removed_match in removed_matches:
            self._possibly_non_flow_preserving_matches.remove(removed_match)

        return match_dict
    
    def _get_best_and_random_matches(
        self,
        n:int, 
        lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], 
        pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType]
        ) -> Dict[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]:
        """
        Get the best and a random match out of the given matches

        Parameters: 
        n (int): The number of matches to return.
        lcomp_matches (Dict[Tuple[VT], MatchLcompHeuristicType]): A dict of matches for local complementation
        pivot_matches (Dict[Tuple[VT,VT], MatchPivotHeuristicType]): A dict of matches for pivoting

        Returns:
        Dict[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]: A dict of the best and random matches
        """
        if len(lcomp_matches) == 0 and len(pivot_matches) == 0:
            return None
        
        if n <= 0:
            raise ValueError("n must be greater than 0")
        
        # Sort the matches in descending order based on the heuristic result
        match_dict = self._sort_matches(lcomp_matches, pivot_matches)

        best_matches = list(match_dict.items())[:max(n//3, 1)]
        random_matches = random.sample(list(match_dict.items())[max(n//3, 1):], n-max(n//3, 1))

        return {**dict(best_matches), **dict(random_matches)}

    def _apply_best_match(
        self,
        graph: BaseGraph[VT,ET],
        lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], 
        pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType],
        ) -> Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
        """
        Finds the best match in the given graph based on lcomp_matches and pivot_matches dictionaries and applies it to the graph.
        
        Args:
            graph (BaseGraph): The graph to search for matches in.
            lcomp_matches (Dict[Tuple[VT], MatchLcompHeuristicType]): Dictionary of matches based on lcomp heuristic.
            pivot_matches (Dict[Tuple[VT,VT], MatchPivotHeuristicType]): Dictionary of matches based on pivot heuristic.
            
        Returns:
            Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None: The best match found in the graph, 
            represented as a tuple of match key and match value. Returns None if no matches are found.
        """
        if len(lcomp_matches) == 0 and len(pivot_matches) == 0:
            return None
        
        match_dict = self._sort_matches(lcomp_matches, pivot_matches)

        for match_key, match_value in match_dict.items():
            filter_graph = graph.clone()
            match_result = self._apply_match(filter_graph, (match_key, match_value))
            if match_result is not None:
                self._total_evals += 1
                return match_key, match_value
            self._skipped_matches_until_reset[self.lookahead] += 1
            
        return None

    def _apply_match(
        self,
        graph: BaseGraph[VT, ET],
        match: Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType], 
        skip_flow_calculation: bool = False
        ) -> Tuple[List[VT], Tuple[VT, ...]] | None:
        """
        Applies the given match to the graph and updates the dicts of local complement and pivot matches.

        Parameters:
        graph (BaseGraph[VT, ET]): The graph to apply the match to.
        match (Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]): The match to apply.
        skip_flow_calculation (bool, optional): Whether to skip the flow calculation. Defaults to False.

        Returns:
        Tuple[List[VT], Tuple[VT, ...]] | None: The neighbors of the matched vertices and the removed vertices. None if the match is not flow-preserving.
        """
        match_key, match_value = match
        vertex_neighbors = set()

        if len(match_key) == 2:
            for vertex in match_key:
                for vertex_neighbor in graph.neighbors(vertex):
                    if vertex_neighbor not in match_key:
                        vertex_neighbors.add(vertex_neighbor)
            new_verticies = apply_pivot(graph=graph, match=match)

        elif len(match_key) == 1:
            _, vertex_neighbors, _ = match_value
            new_verticies = apply_lcomp(graph, match=match)

        else:
            raise ValueError("Match key must be a tuple of length 1 or 2")

        if not skip_flow_calculation:
            if not self._lookup_flow_preserving(graph, match=match):
                if match not in self._possibly_non_flow_preserving_matches:
                    self._possibly_non_flow_preserving_matches.append(match)
                logging.debug(f"Match {match} is not flow-preserving")
                return None
        
        if match in self._possibly_non_flow_preserving_matches:
            self._possibly_non_flow_preserving_matches.remove(match)
            self._rehabilitated_non_flow_preserving_matches += 1

        if new_verticies:
            vertex_neighbors = set(vertex_neighbors).union(set(new_verticies))

        return list(vertex_neighbors), match_key


    def _update_best_result(
        self,
        current_result: Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType], 
        best_result: Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None,
    ) -> Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]:
        """
        Updates the best result based on the current result.

        Args:
            current_result (Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]): The current result to compare.
            best_result (Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None): The current best result.

        Returns:
            Dict[Tuple, Tuple[MatchLcompHeuristicType | MatchPivotHeuristicType, Dict[VT, Set[VT]]]]: The updated best result.
        """

        if current_result is None:
            raise ValueError("current_result cannot be None")
        
        if best_result is None or sum([match[0] for match in current_result.values()]) > sum([match[0] for match in best_result.values()]):
            best_result = current_result
        return best_result

    def _depth_search_for_best_result(
        self,
        graph: BaseGraph[VT, ET],
        lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType],
        pivot_matches: Dict[Tuple[VT, VT], MatchPivotHeuristicType],
        depth: int = 0, 
        best_result: Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None = None,
        current_match_dict: Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] = {},
        full_subgraphs: bool = False
        ) -> Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.
        
        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        If 'full_subgraphs' is False, the search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.
        
        Parameters:
        graph (BaseGraph[VT, ET]): The graph to search for matches in.
        lcomp_matches (Dict[Tuple[VT], MatchLcompHeuristicType]): Dict of local complement matches.
        pivot_matches (Dict[Tuple[VT,VT], MatchPivotHeuristicType]): Dict of pivot matches.
        depth (int): The current depth of the search.
        threshold (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out
        best_result (Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None): The best result found so far.
        current_match_list (Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]): The current list of matches.
        full_subgraphs (bool): Whether to consider full subgraphs or not.
        """

        if depth == self.lookahead:
            current_results = self._apply_best_match(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches)
            if not current_results or (sum([match[0] for match in current_match_dict.values()]) <= self.threshold and len(current_match_dict) > 0):
                return best_result
            
            current_key, current_result = current_results
            lookahead_current_match_dict = current_match_dict.copy()
            lookahead_current_match_dict[current_key] = current_result
            best_result = self._update_best_result(current_result=lookahead_current_match_dict, best_result=best_result)
            
            match_heuristic = sum([match[0] for match in best_result.values()])
            if match_heuristic < self.threshold:
                logging.debug(f"Best match could not be applied at depth {depth} due to heuristic result {match_heuristic} <= {self.threshold}.")
            # return best_result
            
            self.log_data(depth)
            return best_result if match_heuristic >= self.threshold else None

        if not lcomp_matches and not pivot_matches:
            return self._update_best_result(current_result=current_match_dict, best_result=best_result) if current_match_dict else best_result

        num_matches = len(lcomp_matches) + len(pivot_matches)
        num_sub_branches = max(int(num_matches  ** (1/(depth+1.5))), 1) if not full_subgraphs else max(num_matches, 1)

        matches = self._get_best_and_random_matches(n=num_sub_branches, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches)

        for match in matches.items():
            lookahead_graph = graph.clone()
            match_result = self._apply_match(lookahead_graph, match, skip_flow_calculation=False)

            if match_result is not None:
                vertex_neighbors, removed_vertices = match_result
                
                # For testing purposes
                # true_flow = self._calculate_flow(lookahead_graph)
                # if true_flow is None:
                #     raise Exception("Flow is not preserved after applying the match")

                lookahead_lcomp_matches, lookahead_pivot_matches = update_matches(graph=lookahead_graph, vertex_neighbors=vertex_neighbors, removed_vertices=removed_vertices, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, check_for_unfusions=self.use_neighbor_unfusion, include_boundaries=self.include_boundaries, include_gadgets=self.include_gadgets)
                lookahead_current_match_dict = current_match_dict.copy()

                self._reset_lookup_flow()

                lookahead_current_match_dict[match[0]] = match[1]
                
                current_result = self._depth_search_for_best_result(
                    graph=lookahead_graph, 
                    lcomp_matches=lookahead_lcomp_matches, 
                    pivot_matches=lookahead_pivot_matches, 
                    depth=depth + 1, 
                    best_result=best_result, 
                    current_match_dict=lookahead_current_match_dict
                )

                if current_result is not None:
                    best_result = self._update_best_result(current_result=current_result, best_result=best_result)
            else:
                self._skipped_matches_until_reset[depth] += 1

        self.log_data(depth)
        return best_result

    def log_data(self, depth):
        logging.debug(f"Skipped {self._skipped_matches_until_reset} matches at depth {depth}")
        logging.debug(f"Skipped {self._skipped_filter_func_evals} filter function evaluations at depth {depth}")
        logging.debug(f"Applied {self._neighbor_unfusions} neighbor unfusions at depth {depth}")
        logging.debug(f"Total evaluations at depth {depth}: {self._total_evals}")
        logging.debug(f"Rehabilitated {self._rehabilitated_non_flow_preserving_matches} non flow-preserving matches at depth {depth}")

    def _full_search_match_with_best_result_at_depth(self, lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], pivot_matches: Dict[Tuple[VT, VT], MatchPivotHeuristicType]) -> Dict[Tuple, Tuple[MatchLcompHeuristicType | MatchPivotHeuristicType, Dict[VT, Set[VT]]]] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search explores every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search_for_best_result(graph=self.graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, full_subgraphs=True)

    def _search_match_with_best_result_at_depth(self, lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], pivot_matches: Dict[Tuple[VT, VT], MatchPivotHeuristicType]) -> Dict[Tuple, Tuple[MatchLcompHeuristicType | MatchPivotHeuristicType, Dict[VT, Set[VT]]]] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search_for_best_result(graph=self.graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, full_subgraphs=False)

    def _apply_and_find_new_matches(self, local_complement_matches: Dict[Tuple[VT], MatchLcompHeuristicType], pivot_matches: Dict[Tuple[VT, VT], MatchPivotHeuristicType]) -> Tuple[Dict[Tuple[VT], MatchLcompHeuristicType]]:
        """
        Apply the best match found by searching for the best result at a specific depth.

        This function searches for the best result at a specific depth using depth-first search.
        It applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        """

        stop_search = False

        # If the standard deviation over the last couple of local and pivot matches is too low, then we should stop
        # if len(self._remaining_matches) >= 5:
        #     last_matches = self._remaining_matches[-5:]
        #     local_complement_results = [match[0] for match in last_matches]
        #     pivot_results = [match[1] for match in last_matches]
        #     if np.std(local_complement_results) < 1 and np.std(pivot_results) < 1:
        #         self.has_changes_occurred = False
        #         warnings.warn(f"Std of lcomp: {np.std(local_complement_results)} and pivot: {np.std(pivot_results)} is too low. Stopping.")
        #         stop_search = True
        #     logging.debug(f"Std of lcomp: {np.std(local_complement_results)} and pivot: {np.std(pivot_results)} of the last 5 matches")

        if len(self._reduction_per_match) >= 20:
            last_matches = self._reduction_per_match[-20:]
            if np.std(last_matches) < 0.1:
                self.has_changes_occurred = False
                warnings.warn(message=f"Std of reduction: {np.std(last_matches)} is too low. Stopping.")
                stop_search = True
            logging.debug(f"Std of reduction: {np.std(last_matches)} of the last 20 matches")

        if not stop_search:
            
            best_match_dict = self._search_match_with_best_result_at_depth(lcomp_matches=local_complement_matches, pivot_matches=pivot_matches)

            if best_match_dict is not None:
                best_key, best_result = next(iter(best_match_dict.items()))

                match_result = self._apply_match(self.graph, (best_key, best_result), skip_flow_calculation=True)

                if match_result is not None:
                    vertex_neighbors, removed_vertices = match_result
                    local_complement_matches, pivot_matches = update_matches(graph=self.graph, vertex_neighbors=vertex_neighbors, removed_vertices=removed_vertices, lcomp_matches=local_complement_matches, pivot_matches=pivot_matches, check_for_unfusions=self.use_neighbor_unfusion, include_boundaries=self.include_boundaries, include_gadgets=self.include_gadgets)
                else:
                    raise Exception(f"Best match: {best_key} was found but could not be applied.")

                logging.debug(f"Applied match #{self._rule_application_count}: {best_key} with heuristic result: {best_result[0]}")
                logging.debug(f"Found {len(local_complement_matches)} local complement matches and {len(pivot_matches)} pivot matches after applying match")

                self.has_changes_occurred = True
                self._rule_application_count += 1
                self._reduction_per_match.append(best_result[0])

                self._applied_matches.append((best_key, best_result))
                self._remaining_matches.append((len(local_complement_matches), len(pivot_matches)))
            else:
                logging.info("No more matches found")
        
        # self._skipped_matches_until_reset = [0]
        # self._neighbor_unfusions = 0
        self._total_evals = 0

        return local_complement_matches, pivot_matches




def greedy_wire_reduce(
    graph: BaseGraph[VT, ET],
    include_boundaries=False,
    include_gadgets=False,
    max_vertex_index=None,
    threshold=0,
    lookahead=0,
    use_neighbor_unfusion=True,
    flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
    quiet=True,
    stats=None,
):
    """
    Perform a greedy Hadamard wire reduction on the given graph.

    This function iteratively applies the best local complement or pivot match to the graph until no further improvements can be made. The "best" match is determined by a heuristic that considers the number of Hadamard wires added by the match.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to simplify.
    include_boundaries (bool): Whether to include boundary spiders in the search for matches. Defaults to False.
    include_gadgets (bool): Whether to include non-Clifford spiders in the search for matches. Defaults to False.
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine. This is needed to prevent non-termination in the case of heuristic_threshold<0.
    threshold (int): Lower bound for heuristic result. Any rule application which adds more than this number of Hadamard wires is filtered out. Defaults to 0.
    lookahead (int): The number of steps to look ahead when searching for the best match. Defaults to 0.
    use_neighbor_unfusion (bool): Whether to use neighbor unfusion. Defaults to False.
    flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches. Defaults to lambda x: True.

    Returns:
    int: The number of rule applications, i.e., the number of iterations the function went through to simplify the graph.
    """
    reducer = WireReducer(
        graph=graph,
        include_boundaries=include_boundaries,
        include_gadgets=include_gadgets,
        use_neighbor_unfusion=use_neighbor_unfusion,
        max_vertex_index=max_vertex_index,
        threshold=threshold,
        lookahead=lookahead,
        flow_function=flow_function,
        quiet=quiet,
        stats=stats,
    )
    return reducer.greedy_wire_reduce()












def greedy_wire_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex_index=None, threshold=1, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using a greedy approach with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.
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
        local_complement_matches, pivot_matches = generate_matches(graph, flow=flow, max_vertex_index=max_vertex_index, threshold=threshold)
        if apply_best_match(graph, local_complement_matches, pivot_matches, flow):
            iteration_count += 1
            changes_made = True
            flow = gflow(graph)[1]

    return iteration_count

def random_wire_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex_index=None, threshold=1, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using a random approach with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.
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
        local_complement_matches, pivot_matches = generate_matches(graph, flow=flow, max_vertex_index=max_vertex_index, threshold=threshold)
        if apply_random_match(graph, local_complement_matches, pivot_matches, flow):
            iteration_count += 1
            changes_made = True
            flow = gflow(graph)[1]

    return iteration_count

def sim_annealing_reduce_neighbor(graph: BaseGraph[VT,ET], max_vertex_index=None, initial_temperature=100, cooling_rate=0.95, threshold=-100000, quiet:bool=False, stats=None):
    """
    Reduce the number of wires in a graph using simulated annealing with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.
    initial_temperature (int, optional): initial_temperature (int): Initial temperature for the simulated annealing process.
    cooling_rate (float, optional): The rate at which the temperature decreases.
    threshold (int, optional): The minimum wire reduction for a match to be considered.
    quiet (bool, optional): Whether to suppress output.
    stats (dict, optional): A dictionary to store statistics.

    Returns:
    BaseGraph[VT,ET]: The graph with the reduced number of wires.
    """
    temperature = initial_temperature
    min_temperature = 1
    iteration_count = 0
    flow = gflow(graph)[1]

    best_graph = graph.copy()
    best_evaluation = graph.num_edges()
    current_evaluation = best_evaluation

    backtrack = False

    while temperature > min_temperature:
        iteration_count += 1
        local_complement_matches, pivot_matches = generate_matches(graph, flow=flow, max_vertex_index=max_vertex_index, threshold=threshold)
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
                new_verticies = apply_pivot(graph, match, flow)
            else:
                new_verticies = apply_lcomp(graph, match, flow)

            if current_evaluation < best_evaluation:
                best_graph = graph.copy()
                best_evaluation = current_evaluation

            flow = gflow(graph)[1]

        temperature *= cooling_rate

    print("final num edges: ", best_graph.num_edges())
    return best_graph