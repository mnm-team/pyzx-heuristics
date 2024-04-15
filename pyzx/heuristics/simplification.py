
from enum import Enum, member
import logging
import math
import os
import random
import heapq
import time
from tqdm import tqdm

from fractions import Fraction
from typing import Callable, Dict, Optional, Set, Tuple, List
import warnings

import numpy as np

from .heuristics import PhaseType, get_phase_type, lcomp_heuristic, lcomp_heuristic_neighbor_unfusion, pivot_heuristic, pivot_heuristic_neighbor_unfusion
from .tools import disentangle_outputs, split_phases, insert_identity
from .flow_calculation import Flow, identify_cflow, identify_gflow, identify_gflow_with_gadgets

from pyzx.rules import apply_rule, lcomp, lcomp_with_boundaries, pivot
from pyzx.utils import VertexType, EdgeType
from pyzx.graph.base import BaseGraph, VT, ET


import psutil

class MatchType(Enum):
    LCOMP = "lcomp"
    PIVOT = "pivot"


MatchLcompHeuristicType = Tuple[float, List[VT], VT]

MatchPivotHeuristicType = Tuple[float, VT, VT]

def is_vertex_next_to_boundary(graph, vertex):
    """
    Check if a vertex is next to a boundary in a graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    vertex (VT): The vertex to check.

    Returns:
    bool: True if the vertex is next to a boundary, False otherwise.
    """
    for neighbor in graph.neighbors(vertex):
        if graph.type(neighbor) != VertexType.Z:
            return True
    return False

def check_lcomp_match(graph, vertex, check_for_unfusions=True, check_for_xz_phase_gadgets=False) -> Tuple[Tuple[VT], List[MatchLcompHeuristicType]] | None:
    vertex_types = graph.types()

    current_vertex_type = vertex_types[vertex]
    current_vertex_phase = graph.phase(vertex)
    
    if current_vertex_type != VertexType.Z: return None
    current_vertex_neighbors = list(graph.neighbors(vertex))

    # Check if the vertex needs to be transformed into an XZ spider
    needs_gadget = get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD
    # Skip if the vertex has only one neighbor (i.e., it's a leaf node)
    if len(graph.neighbors(vertex)) == 1: return None
    if vertex in current_vertex_neighbors: return None

    is_already_gadget = False
    boundary_neighbours = set()
    for neighbor in current_vertex_neighbors:
        # Check if the neighbor is a leaf node and the vertex needs to be gadgetized
        if len(graph.neighbors(neighbor)) == 1 and get_phase_type(current_vertex_phase) != PhaseType.TRUE_CLIFFORD and graph.type(neighbor) != VertexType.BOUNDARY:
            is_already_gadget = True
        if vertex_types[neighbor] != VertexType.Z:
            boundary_neighbours.add(neighbor)

    if is_already_gadget and needs_gadget: return None
    boundary_count = len(boundary_neighbours)
    if boundary_count > 1: return None

    if check_for_xz_phase_gadgets:
        if boundary_count == 1 and needs_gadget:
            return (vertex,), [(lcomp_heuristic(graph,vertex)-boundary_count,current_vertex_neighbors,-1)]
    
    # spider_count = -1 + boundary_count + (2 if needs_gadget else 0)

    matches = []
    if not needs_gadget:
        return (vertex,), [(lcomp_heuristic(graph,vertex)-boundary_count,current_vertex_neighbors,None)]
    if check_for_unfusions:
        for neighbor in get_all_possible_unfusion_neighbours(graph, vertex, None):
            boundary_count_for_unfusion = boundary_count
            if neighbor in boundary_neighbours:
                boundary_count_for_unfusion -= 1
            matches.append((lcomp_heuristic_neighbor_unfusion(graph,vertex,neighbor)-boundary_count,current_vertex_neighbors,neighbor))
    if check_for_xz_phase_gadgets:
        matches.append((lcomp_heuristic(graph,vertex)-boundary_count,current_vertex_neighbors,-1))

    if len(matches) > 0:
        return (vertex,), matches
    return None

def check_pivot_match(graph, edge, check_for_unfusions=True, check_for_phase_gadgets=True) -> Tuple[Tuple[VT, VT], List[MatchPivotHeuristicType]] | None:

    vertex_types = graph.types()
    if graph.edge_type(edge) != EdgeType.HADAMARD: return None

    vertex0, vertex1 = graph.edge_st(edge)
    if vertex0 == vertex1: return None

    vertex0_type = vertex_types[vertex0]
    vertex1_type = vertex_types[vertex1]
    if not (vertex0_type == VertexType.Z and vertex1_type == VertexType.Z): return None

    vertex0_phase = graph.phase(vertex0)
    vertex1_phase = graph.phase(vertex1)
    vertex0_needs_unfusion = get_phase_type(vertex0_phase) != PhaseType.CLIFFORD
    vertex1_needs_unfusion = get_phase_type(vertex1_phase) != PhaseType.CLIFFORD

    vertex0_neighbors = graph.neighbors(vertex0)
    vertex1_neighbors = graph.neighbors(vertex1)
    if len(vertex0_neighbors) == 1 or len(vertex1_neighbors) == 1: return None 

    vertex0_is_gadget = False
    vertex1_is_gadget = False
    vertex0_boundary = set()
    vertex1_boundary = set()

    for neighbor in vertex0_neighbors:
        if vertex_types[neighbor] != VertexType.Z:
            vertex0_boundary.add(neighbor)
        if len(graph.neighbors(neighbor)) == 1 and vertex0_needs_unfusion and graph.type(neighbor) != VertexType.BOUNDARY: 
            vertex0_is_gadget = True

    for neighbor in vertex1_neighbors:
        if vertex_types[neighbor] != VertexType.Z:
            vertex1_boundary.add(neighbor)
        if len(graph.neighbors(neighbor)) == 1 and vertex1_needs_unfusion and graph.type(neighbor) != VertexType.BOUNDARY: 
            vertex1_is_gadget = True

    # Check if vertex needs unfusion but is already a gadget
    if (vertex0_is_gadget and vertex0_needs_unfusion) or (vertex1_is_gadget and vertex1_needs_unfusion): return None
    # Check if unfusion is not allowd but needed
    if (vertex0_needs_unfusion or vertex1_needs_unfusion):
        if not check_for_unfusions and not check_for_phase_gadgets: return None

    boundary_count = len(vertex0_boundary) + len(vertex1_boundary)
    if boundary_count > 1: return None
    
    # TODO: check heuristic function for gadgets and boundaries

    if check_for_phase_gadgets:
        #TODO: check if correct. test_phase_gadget_matches
        if vertex0_boundary and vertex0_needs_unfusion and vertex1_boundary and vertex1_needs_unfusion: 
            return None
        elif vertex0_boundary and vertex0_needs_unfusion:
            return (vertex0, vertex1), [(pivot_heuristic(graph,edge)-boundary_count-1,-1,None)]
        elif vertex1_boundary and vertex1_needs_unfusion:
            return (vertex0, vertex1), [(pivot_heuristic(graph,edge)-boundary_count-1,None,-1)]
    
    # No need for unfusion or phase gadget
    if not vertex0_needs_unfusion and not vertex1_needs_unfusion:
        return (vertex0, vertex1), [(pivot_heuristic(graph,edge)-boundary_count,None,None)]

    matches = []
    
    if vertex0_needs_unfusion and vertex1_needs_unfusion:
        # Unfuse both vertices to all possible neighbors
        if check_for_unfusions:
            for neighbor_v0 in get_all_possible_unfusion_neighbours(graph, vertex0, vertex1):
                boundary_count_for_unfusion = boundary_count
                if neighbor_v0 in vertex0_boundary:
                    boundary_count_for_unfusion -= 1
                for neighbor_v1 in get_all_possible_unfusion_neighbours(graph, vertex1, vertex0):
                    if neighbor_v1 in vertex1_boundary:
                        boundary_count_for_unfusion -= 1
                    matches.append((pivot_heuristic_neighbor_unfusion(graph,edge,neighbor_v0,neighbor_v1)-boundary_count_for_unfusion,neighbor_v0,neighbor_v1))
    elif vertex0_needs_unfusion:
        # Unfuse vertex0 to all possible neighbors
        if check_for_unfusions:
            for neighbor in get_all_possible_unfusion_neighbours(graph, vertex0, vertex1):
                boundary_count_for_unfusion = boundary_count
                if neighbor in vertex0_boundary:
                    boundary_count_for_unfusion -= 1
                matches.append((pivot_heuristic_neighbor_unfusion(graph,edge,neighbor, None)-boundary_count_for_unfusion,neighbor,None))
        # FIXME: pivot heuristic does not seem to be correct for phase gadgets. It will calculate positive heuristic even if there is no reduction in wires.
        # -1 to the heuristic value seems to work for now
        # Phase gadget unfusion for vertex0
        if check_for_phase_gadgets:
            matches.append((pivot_heuristic(graph,edge)-boundary_count-1,-1,None))
    elif vertex1_needs_unfusion:
        # Unfuse vertex1 to all possible neighbors
        if check_for_unfusions:
            for neighbor in get_all_possible_unfusion_neighbours(graph, vertex1, vertex0):
                boundary_count_for_unfusion = boundary_count
                if neighbor in vertex1_boundary:
                    boundary_count_for_unfusion -= 1
                matches.append((pivot_heuristic_neighbor_unfusion(graph,edge,None, neighbor)-boundary_count_for_unfusion,None,neighbor))
        # Phase gadget unfusion for vertex1
        if check_for_phase_gadgets:
            matches.append((pivot_heuristic(graph,edge)-boundary_count-1,None,-1))
        
    if matches:
        return (vertex0, vertex1), matches
    return None




def lcomp_matcher(graph: BaseGraph[VT,ET], check_for_unfusions=True, check_for_xz_phase_gadgets=True) -> Dict[Tuple[VT], List[MatchLcompHeuristicType]]:
    """
    Generates all matches for local complementation in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    check_for_unfusions (bool): whether to check for unfusions.
    check_for_xz_phase_gadgets (bool): whether to check for phase gadgets.

    Returns:
    Dict[Tuple[VT], List[MatchLcompHeuristicType]]: A dictionary of match tuples match_key:(heuristic,vertices,spider_count), where heuristic is the LCH, vertices are the neighbor vertices and spider_count the amount of saved/added spiders
    """
    vertex_candidates = graph.vertex_set()

    matches = {}

    while len(vertex_candidates) > 0:
        current_vertex = vertex_candidates.pop()
        match = check_lcomp_match(graph, current_vertex, check_for_unfusions=check_for_unfusions, check_for_xz_phase_gadgets=check_for_xz_phase_gadgets)

        if match is not None:
            match_key, match_values = match
            matches[match_key] = match_values
    
    return matches

def pivot_matcher(graph: BaseGraph[VT,ET], check_for_unfusions=True, check_for_phase_gadgets=True) -> Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]:
    """
    Generates all matches for pivoting in a graph-like ZX-diagram

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram


    check_for_unfusions (bool): whether to check for unfusions.
    check_for_phase_gadgets (bool): whether to check for phase gadgets.

    Returns:
    Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]: A dictionary of match tuples match_key:(heuristic,spider_count), where heuristic is the LCH and spider_count the amount of saved/added spiders
    """
    edge_candidates = graph.edge_set()
    matches = {}

    while len(edge_candidates) > 0:
        edge = edge_candidates.pop()

        match = check_pivot_match(graph, edge, check_for_unfusions=check_for_unfusions, check_for_phase_gadgets=check_for_phase_gadgets)

        if match is not None:
            match_key, match_values = match
            matches[match_key] = match_values

    return matches



def update_lcomp_matches(
        graph: BaseGraph[VT,ET], 
        vertex_neighbors: List[VT], 
        removed_vertices: Tuple[VT], 
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
        neighbors_of_neighbors: Set[VT],
        check_for_unfusions=True,
        check_for_xz_phase_gadgets=True
        ) -> Dict[Tuple[VT], List[MatchLcompHeuristicType]]:
    
    # Iterate over the current local complement matches
    lcomp_matches_copy = lcomp_matches.copy()
    keys_to_remove = set()

    for vertex_match, match_values in lcomp_matches_copy.items():

        if vertex_match[0] in removed_vertices:
            keys_to_remove.add((vertex_match, None))
            continue

        if any(element in match_values[0][1] for element in removed_vertices):
            
            new_match = check_lcomp_match(graph, vertex_match[0], check_for_unfusions=check_for_unfusions, check_for_xz_phase_gadgets=check_for_xz_phase_gadgets)
            if new_match is None:
                keys_to_remove.add((vertex_match, None))
            else:
                match_key, match_values = new_match
                lcomp_matches_copy[match_key] = match_values
            continue

        # If the vertex is in the set of neighbors of neighbors, recalculate the heuristic
        if vertex_match[0] in neighbors_of_neighbors:
            new_match = check_lcomp_match(graph, vertex_match[0], check_for_unfusions=check_for_unfusions, check_for_xz_phase_gadgets=check_for_xz_phase_gadgets)
            if new_match is not None:
                match_key, match_values = new_match
                lcomp_matches_copy[match_key] = match_values
            else:
                keys_to_remove.add((vertex_match, None))

    for key, index in keys_to_remove:
        if index is None:
            del lcomp_matches_copy[key]
        else:
            del lcomp_matches_copy[key][index]

    # Check for new local complement matches in the vertex neighbors
    for neighbor in vertex_neighbors:
        new_match = check_lcomp_match(graph, neighbor, check_for_unfusions=check_for_unfusions, check_for_xz_phase_gadgets=check_for_xz_phase_gadgets)
        if new_match is not None:
            match_key, match_values = new_match
            lcomp_matches_copy[match_key] = match_values

    return lcomp_matches_copy

def update_pivot_matches(
        graph: BaseGraph[VT,ET], 
        vertex_neighbors: List[VT], 
        removed_vertices: Tuple[VT], 
        pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]], 
        neighbors_of_neighbors: Set[VT],
        check_for_unfusions=True,
        check_for_phase_gadgets=True
        ) -> Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]:
    
    pivot_matches_copy = pivot_matches.copy()
    keys_to_remove = set()

    for edge, match_values in pivot_matches_copy.items():

        vertex0, vertex1 = edge

        if vertex0 in removed_vertices or vertex1 in removed_vertices:
            keys_to_remove.add((edge, None))
            continue

        if not graph.connected(vertex0, vertex1):
            keys_to_remove.add((edge, None))
            continue

        # If the vertices are in the set of neighbors of neighbors, recalculate the heuristic
        if vertex0 in neighbors_of_neighbors or vertex1 in neighbors_of_neighbors:
            new_match = check_pivot_match(graph, edge, check_for_unfusions=check_for_unfusions, check_for_phase_gadgets=check_for_phase_gadgets)
            if new_match is None:
                keys_to_remove.add((edge, None))
            else:
                match_key, match_values = new_match
                pivot_matches_copy[match_key] = match_values
    
    for key, index in keys_to_remove:
        if index is None:
            del pivot_matches_copy[key]
        else:
            del pivot_matches_copy[key][index]

    # Check for new pivot matches in the vertex neighbors
    for vertex_neighbor in vertex_neighbors:
        for neighbor_of_neighbor in graph.neighbors(vertex_neighbor):
            if graph.connected(vertex_neighbor, neighbor_of_neighbor):
                edge = graph.edge(vertex_neighbor, neighbor_of_neighbor)
                new_match = check_pivot_match(graph, edge, check_for_unfusions=check_for_unfusions, check_for_phase_gadgets=check_for_phase_gadgets)
                if new_match is not None:
                    match_key, match_values = new_match
                    pivot_matches_copy[match_key] = match_values

    return pivot_matches_copy

def update_matches(
        graph: BaseGraph[VT,ET], 
        vertex_neighbors: List[VT], 
        removed_vertices: Tuple[VT], 
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
        pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]],
        check_for_unfusions=True,
        check_for_yz_phase_gadgets=True,
        check_for_xz_phase_gadgets=True,
        max_vertex_index=None
        ) -> Tuple[Dict[Tuple[VT], List[MatchLcompHeuristicType]], Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]]:
    """
    Updates the dict of local complement and pivot matches after a local complementation or pivot has been applied.

    Parameters:
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    vertex_neighbors (List[VT]): The neighbors of the vertex where the local complementation or pivot was applied
    removed_vertices (Tuple[VT]): The vertices that were removed by the local complementation or pivot
    lcomp_matches (Dict[Tuple[VT], List[MatchLcompHeuristicType]]): The current dict of local complement matches
    pivot_matches (Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]): The current dict of pivot matches
    check_for_unfusions (bool): whether to check for unfusions.
    check_for_yz_phase_gadgets (bool): whether to check for phase gadgets in pivoting.
    check_for_xz_phase_gadgets (bool): whether to check for phase gadgets in local complementation.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.

    Returns:
    Tuple[Dict[Tuple[VT], List[MatchLcompHeuristicType]], Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]]: The updated dictonaries of local complement and pivot matches.
    """

    # Initialize a set of neighbors of neighbors
    neighbors_of_neighbors = set()

    # Iterate over the neighbors of the vertex
    for neighbor in vertex_neighbors:
        # Iterate over the neighbors of the current neighbor
        try:
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                # If the neighbor of the neighbor is not in the vertex_neighbors list, add it to the set
                if neighbor_of_neighbor not in vertex_neighbors:
                    neighbors_of_neighbors.add(neighbor_of_neighbor)
        except:
            pass

    lcomp_matches = update_lcomp_matches(graph=graph, 
                                         vertex_neighbors=vertex_neighbors, 
                                         removed_vertices=removed_vertices, 
                                         lcomp_matches=lcomp_matches, 
                                         neighbors_of_neighbors=neighbors_of_neighbors, 
                                         check_for_unfusions=check_for_unfusions,
                                         check_for_xz_phase_gadgets=check_for_xz_phase_gadgets
                                         )
    
    pivot_matches = update_pivot_matches(graph=graph, 
                                         vertex_neighbors=vertex_neighbors, 
                                         removed_vertices=removed_vertices, 
                                         pivot_matches=pivot_matches, 
                                         neighbors_of_neighbors=neighbors_of_neighbors, 
                                         check_for_unfusions=check_for_unfusions,
                                         check_for_phase_gadgets=check_for_yz_phase_gadgets
                                         )

    for match_key in list(lcomp_matches.keys()):
        if max_vertex_index and match_key[0] > max_vertex_index:
            del lcomp_matches[match_key]

    for match_key in list(pivot_matches.keys()):
        if max_vertex_index and (match_key[0] > max_vertex_index or match_key[1] > max_vertex_index):
            del pivot_matches[match_key]

    return lcomp_matches, pivot_matches



def get_possible_gflow_preserving_unfusion_neighbours(graph: BaseGraph[VT,ET], current_vertex, exclude_vertex=None):
    """
    Get the possible neighbors for unfusion of a given vertex in a graph.
    Only neighbors with 2 or less neighbors are considered to preserve the G-flow.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to perform the operation on.
    current_vertex (VT): The vertex to find possible unfusion neighbors for.
    exclude_vertex (VT, optional): A vertex to exclude from the possible unfusion neighbors.

    Returns:
    list: A list of vertices that are possible neighbors for unfusion.
    """
    
    possible_unfusion_neighbours = set()

    for neighbor in graph.neighbors(current_vertex):
        neighbors_of_neighbor = graph.neighbors(neighbor)
        if len(neighbors_of_neighbor) <= 2 and graph.type(neighbor) == VertexType.Z:
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
    #TODO: dont allow boundaries as long as fixme in rules.py is not fixed
    # possible_unfusion_neighbours = set(neighbor for neighbor in graph.neighbors(current_vertex) if graph.type(neighbor) == VertexType.Z)
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
    new_phase = split_phases(graph.phase(current_vertex), desired_phase)
    phase_spider = graph.add_vertex(VertexType.Z,-2,graph.rows()[current_vertex],new_phase)
    graph.set_phase(current_vertex, desired_phase)

    if neighbor_vertex == -1:
        # TODO: Track phase index
        # graph.update_phase_index(current_vertex,phase_spider)
        graph.add_edge((phase_spider,current_vertex), EdgeType.SIMPLE)

        # TODO: check if this is correct. In rules.py no extra phaseless spieder is added
        phaseless_spider = insert_identity(graph, current_vertex, phase_spider)
        # phaseless_spider = None

        return (phaseless_spider, phase_spider)

    graph.add_edge((phase_spider, neighbor_vertex), EdgeType.HADAMARD)
    graph.add_edge((current_vertex, phase_spider), EdgeType.SIMPLE)

    phaseless_spider = insert_identity(graph, current_vertex, phase_spider)

    graph.remove_edge(graph.edge(current_vertex, neighbor_vertex))

    return (phaseless_spider, phase_spider)



def apply_pivot(graph: BaseGraph[VT,ET], match: Tuple[Tuple[VT, VT], MatchPivotHeuristicType], flow_function: Callable[[BaseGraph, Tuple[VT, VT]], bool] = None) -> Tuple[Tuple[VT, ...], Dict[Tuple[VT, VT], bool] | None] | None:
    """
        Apply a pivot operation to a graph.
        If a flow function is provided, the flow of the graph is calculated for each edge unfused.

        Parameters:
        graph (BaseGraph[VT,ET]): The graph to perform the operation on.
        match (tuple): The match to apply the operation to.
        flow_function (optinal[Callable]): A function to calculate the flow of the graph for each edge unfused.

        Returns:
        tuple[tuple[tuple, dict], dict]: A tuple containing the added unfusion vertices, a dictionary storing the flow preserving attribute of each edge, and a dictionary containing the time for each processing step.
    """
    match_key, match_value = match
    vertex_0, vertex_1 = match_key

    if graph.num_vertices() == 46 and graph.num_edges() == 58:
        print("Before pivot")
        print(graph)
    
    unfusion_neighbors = {vertex_0: match_value[1], vertex_1: match_value[2]}
    
    vertex0_neighbors = set(graph.neighbors(vertex_0))
    vertex1_neighbors = set(graph.neighbors(vertex_1))
    
    vertex0_boundary = [vertex for vertex in vertex0_neighbors if graph.type(vertex) != VertexType.Z]
    vertex1_boundary = [vertex for vertex in vertex1_neighbors if graph.type(vertex) != VertexType.Z]
    
    if len(vertex0_boundary) + len(vertex1_boundary) > 1:
        return None, None
    if unfusion_neighbors[vertex_0] in vertex0_boundary or unfusion_neighbors[vertex_1] in vertex1_boundary:
        return None, None
    
    if vertex0_boundary and (unfusion_neighbors[vertex_0] is not None and unfusion_neighbors[vertex_0] != -1):
        return None, None
    if vertex1_boundary and (unfusion_neighbors[vertex_1] is not None and unfusion_neighbors[vertex_1] != -1):
        return None, None
    
    new_vertices = []
    flow = {}
    was_neighbor_unfused = False
    
    # Initialize a dictionary to store all time-related variables
    time_dict = {"time_to_unfuse": 0, "time_to_claculate_flow": 0, "time_to_apply_rule": 0}
    
    for vertex, unfusion_neighbor in unfusion_neighbors.items():
        if unfusion_neighbor is not None:

            time_to_unfuse_start = time.perf_counter()
            phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbor, Fraction(0,1))
            if phaseless_spider: new_vertices.append(phaseless_spider) 
            if phase_spider: new_vertices.append(phase_spider)
            time_dict["time_to_unfuse"] += time.perf_counter() - time_to_unfuse_start
    
            time_to_claculate_flow_start = time.perf_counter()
            if flow_function:
                unfused_edge = graph.edge(vertex, unfusion_neighbor) if unfusion_neighbor != -1 else graph.edge(vertex, phase_spider)
                flow[unfused_edge] = flow_function(graph, unfused_edge)
            else:
                flow = None
            time_dict["time_to_claculate_flow"] += time.perf_counter() - time_to_claculate_flow_start
            was_neighbor_unfused = True
    
    time_to_apply_rule_start = time.perf_counter()
    apply_rule(graph, pivot, [(vertex_0, vertex_1, vertex0_boundary, vertex1_boundary)])
    # apply_rule(graph, pivot, [(vertex_0, vertex_1, [], [])])
    time_dict["time_to_apply_rule"] = time.perf_counter() - time_to_apply_rule_start
    
    if was_neighbor_unfused:
        return (tuple(new_vertices), flow), time_dict
    
    return None, time_dict

def apply_lcomp(graph: BaseGraph[VT,ET], match: Tuple[Tuple[VT,], MatchLcompHeuristicType], flow_function: Callable[[BaseGraph, Tuple[VT, VT]], bool] = None) -> Tuple[Tuple[VT, VT], Dict[Tuple[VT, VT], bool] | None] | None:
    """
        Apply a local complementation operation to a graph.
        If a flow function is provided, the flow of the graph is calculated for each edge unfused.

        Parameters:
        graph (BaseGraph[VT,ET]): The graph to perform the operation on.
        match (tuple): The match to apply the operation to.
        flow_function (optinal[Callable]): A function to calculate the flow of the graph for each edge unfused.

        Returns:
        tuple: A tuple containing the added unfusion vertices and a dictionary storing the flow preserving attribute of each edge.
    """

    match_key, match_value = match
    vertex = match_key[0]
    neighbors = match_value[1]
    unfusion_neighbor = match_value[2]

    if match_key == (128,) and match_value == (-4.0, [0, 136, 127], 0):
        pass

    vertex_boundary = [vertex for vertex in neighbors if graph.type(vertex) != VertexType.Z]
    if len(vertex_boundary) > 1:
        return None, None

    was_neighbor_unfused = False
    time_dict = {"time_to_unfuse": 0, "time_to_claculate_flow": 0, "time_to_apply_rule": 0}

    if unfusion_neighbor is not None:
        time_to_unfuse_start = time.perf_counter()
        phaseless_spider, phase_spider = unfuse_to_neighbor(graph, vertex, unfusion_neighbor, Fraction(1,2))
        new_vertices = [v for v in [phaseless_spider, phase_spider] if v]
        time_dict["time_to_unfuse"] = time.perf_counter() - time_to_unfuse_start

        time_to_claculate_flow_start = time.perf_counter()
        # If unfusion to a phase gadget add the gadget to the neighbors
        if unfusion_neighbor == -1:
            neighbors_copy = neighbors + [phaseless_spider]
        # If unfusion to a neighbor, replace the neighbor with the phase spider
        else:
            neighbors_copy = [phaseless_spider if neighbor == unfusion_neighbor else neighbor for neighbor in neighbors]
        unfused_edge = graph.edge(vertex, unfusion_neighbor) if unfusion_neighbor != -1 else graph.edge(vertex, phase_spider)
        flow = {unfused_edge: flow_function(graph, unfused_edge)} if flow_function else None
        time_dict["time_to_claculate_flow"] = time.perf_counter() - time_to_claculate_flow_start

        was_neighbor_unfused = True
    else:
        neighbors_copy = neighbors.copy()
        flow = None

    time_to_apply_rule_start = time.perf_counter()
    apply_rule(graph, lcomp_with_boundaries, [(vertex, neighbors_copy)])
    time_dict["time_to_apply_rule"] = time.perf_counter() - time_to_apply_rule_start

    if was_neighbor_unfused:
        return (tuple(new_vertices), flow), time_dict

    return None, time_dict


def get_match_type(match: Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]) -> MatchType:
    """
    Get the type of a match.

    Parameters:
    match (tuple): The match to get the type of.

    Returns:
    str: The type of the match.
    """
    match_key, _ = match

    if len(match_key) == 1:
        return MatchType.LCOMP
    elif len(match_key) == 2:
        return MatchType.PIVOT
    raise ValueError("Match key must be a tuple of length 1 or 2")

def is_match_unfusing(match: Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]) -> bool | None:
    """
    Check if a match is unfusing.

    Parameters:
    match (tuple): The match to check.

    Returns:
    bool: True if the match is unfusing, False otherwise.
    """
    _, match_value = match

    if get_match_type(match) == MatchType.LCOMP:
        return match_value[-1] is not None
    elif get_match_type(match) == MatchType.PIVOT:
        return match_value[-1] is not None or match_value[-2] is not None
    
def is_match_boundary(graph:BaseGraph, match: Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]) -> bool:
    """
    Check if a match is boundary.

    Parameters:
    match (tuple): The match to check.
    
    Returns:
    bool: True if the match is boundary, False otherwise.
    """
    match_key, _ = match

    for vertex in match_key:
        if is_vertex_next_to_boundary(graph, vertex):
            return True
    return False



class FilterFlowFunc(Enum):
    """
    An enumeration of the filter flow functions.
    """
    NONE = member(lambda _: ({}, {}))
    G_FLOW_PRESERVING =  member(identify_gflow)
    C_FLOW_PRESERVING =  member(identify_cflow)
    G_FLOW_PRESERVING_GADGET =  member(identify_gflow_with_gadgets)

    def __call__(self, *args, **kwargs) -> Optional[Flow]:
        return self.value(*args, **kwargs)

class WireReducer:

    def __init__(
        self,
        graph: BaseGraph[VT, ET],
        use_neighbor_unfusion:bool=False,
        use_yz_phase_gadgets:bool=False,
        use_xz_phase_gadgets:bool=False,
        max_vertex_index:int|None=None,
        threshold:int=0,
        lookahead:int=0,
        flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
        quiet:bool=True,
        stats=None,
    ):
        """
        Initializes a WireReducer object.

        Args:
            graph (BaseGraph[VT, ET]): The graph to be reduced.
        
        
            use_neighbor_unfusion (bool): Whether to use neighbor unfusion.
            use_yz_phase_gadgets (bool): Whether to use phase gadgets in pivot matches.
            use_xz_phase_gadgets (bool): Whether to use phase gadgets in lcomp matches.
            max_vertex_index (optional[int]): The maximum vertex index.
            threshold (int): Lower bound for heuristic result.
            lookahead (int): The depth at which to find the best result.
            flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches.
            quiet (bool): Whether to suppress output.
            stats: Statistics object to track the reduction process.
        """
        self.graph = graph
        self.use_neighbor_unfusion = use_neighbor_unfusion
        self.use_yz_phase_gadgets = use_yz_phase_gadgets
        self.use_xz_phase_gadgets = use_xz_phase_gadgets
        self.max_vertex_index = max_vertex_index
        self.threshold = threshold
        self.lookahead = lookahead
        self.flow_function: FilterFlowFunc = flow_function
        self.quiet = quiet
        self.stats = stats

        self._lookup_flow_for_unfusion: Dict[Tuple[VT, VT], bool] = {}
        self._use_lookup_flow_for_unfusion = False

        self._possibly_non_flow_preserving_matches = []

        self._apply_all_lookahead_matches = False

        # For logging purposes
        self._rule_application_count = 0
        self._reduction_per_match = []
        self._applied_matches = []
        self._remaining_matches = []
        self._matches = {"total matches": [0], "total unfusion matches": [0], "total boundary matches":[0], "evaluated true": [0], "evaluated false": [0], "evaluated unfusion true": [0], "evaluated unfusion false": [0]}
        self._neighbor_unfusions = {"total": [0], "gadgets": [0], "skipped true": [0], "skipped false": [0], "evaluated true": [0], "evaluated false": [0]}
        self._rehabilitated_non_flow_preserving_matches = 0

        self._time_to_apply_match = [0]
        self._time_to_unfuse = [0]
        self._time_to_calculate_flow = [0]
        self._time_dict = {"time_to_unfuse": [0], "time_to_claculate_flow": [0], "time_to_apply_rule": [0]}

        self.graph.track_phases = False
        
        # if self.max_vertex_index is None:
        #     self.max_vertex_index = max(self.graph.vertex_set())
        #     warnings.warn("No maximum vertex index provided. Using the maximum vertex index in the graph.")

        if self.use_neighbor_unfusion:
            self._use_lookup_flow_for_unfusion = True

        if self.flow_function == FilterFlowFunc.NONE:
            if self.use_neighbor_unfusion:
                if self.use_yz_phase_gadgets or self.use_xz_phase_gadgets:
                    self.flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
                    warnings.warn(f"Neighbor unfusion with phase gadgets requires a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")
                    logging.warning(f"Neighbor unfusion with phase gadgets requires a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")
                else:
                    self.flow_function = FilterFlowFunc.G_FLOW_PRESERVING
                    warnings.warn(f"Neighbor unfusion requires a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")
                    logging.warning(f"Neighbor unfusion requires a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")
            elif self.use_yz_phase_gadgets or self.use_xz_phase_gadgets:
                self.flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
                warnings.warn(f"Phase gadgets require a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")
                logging.warning(f"Phase gadgets require a flow function. Using {FilterFlowFunc(self.flow_function).name} function.")

        if (not self.use_neighbor_unfusion and self.flow_function == FilterFlowFunc.G_FLOW_PRESERVING) or (not (self.use_yz_phase_gadgets or self.use_xz_phase_gadgets) and self.flow_function == FilterFlowFunc.G_FLOW_PRESERVING_GADGET):
            warnings.warn("G-flow preserving function is not needed without neighbor unfusion or phase gadgets. This will cause unnecessary overhead.")
            logging.warning("G-flow preserving function is not needed without neighbor unfusion or phase gadgets. This will cause unnecessary overhead.")

        if self.flow_function == FilterFlowFunc.C_FLOW_PRESERVING:
            if not self._is_graph_flow_preserving(self.graph):
                disentangle_outputs(self.graph)
                if not self._is_graph_flow_preserving(self.graph):
                    raise Exception("Graph is not C-flow preserving")

    def greedy_wire_reduce(self):
        self.has_changes_occurred = True

        local_complement_matches = lcomp_matcher(self.graph, check_for_unfusions=self.use_neighbor_unfusion, check_for_xz_phase_gadgets=self.use_xz_phase_gadgets)
        pivot_matches = pivot_matcher(self.graph, check_for_unfusions=self.use_neighbor_unfusion, check_for_phase_gadgets=self.use_yz_phase_gadgets)

        while self.has_changes_occurred:
            self.has_changes_occurred = False
            local_complement_matches, pivot_matches = self._apply_and_find_new_matches(lcomp_matches=local_complement_matches, pivot_matches=pivot_matches, find_matches_method=self._search_match_with_best_result_at_depth)

            # For testing purposes
            if not self._is_graph_flow_preserving(self.graph):
                raise Exception("Flow is not preserved after applying the match")
        
        total_matches_info = {key: sum(value) for key, value in self._matches.items()}
        total_neighbor_unfusions_info = {key: sum(value) for key, value in self._neighbor_unfusions.items()}
        logging.info(f"Total rule applications: {self._rule_application_count}, Total reduction: {sum(self._reduction_per_match)}, Average reduction: {sum(self._reduction_per_match)/len(self._reduction_per_match) if len(self._reduction_per_match) > 0 else 0}")
        logging.info(f"Neighbor unfusions: {total_neighbor_unfusions_info}")
        logging.info(f"Matches: {total_matches_info}")
        logging.info(f"Number of vertices: {len(self.graph.vertex_set())}, Number of edges: {len(self.graph.edge_set())}")
        return sum(self._reduction_per_match), self._applied_matches
    
    def random_wire_reduce(self):
        self.has_changes_occurred = True

        local_complement_matches = lcomp_matcher(self.graph, check_for_unfusions=self.use_neighbor_unfusion, check_for_xz_phase_gadgets=self.use_xz_phase_gadgets)
        pivot_matches = pivot_matcher(self.graph, check_for_unfusions=self.use_neighbor_unfusion, check_for_phase_gadgets=self.use_yz_phase_gadgets)

        while self.has_changes_occurred:
            self.has_changes_occurred = False
            local_complement_matches, pivot_matches = self._apply_and_find_new_matches(lcomp_matches=local_complement_matches, pivot_matches=pivot_matches, find_matches_method=self._search_match_with_random_result_at_depth)

            # For testing purposes
            if not self._is_graph_flow_preserving(self.graph):
                raise Exception("Flow is not preserved after applying the match")
        
        total_matches_info = {key: sum(value) for key, value in self._matches.items()}
        total_neighbor_unfusions_info = {key: sum(value) for key, value in self._neighbor_unfusions.items()}
        logging.info(f"Total rule applications: {self._rule_application_count}, Total reduction: {sum(self._reduction_per_match)}, Average reduction: {sum(self._reduction_per_match)/len(self._reduction_per_match) if len(self._reduction_per_match) > 0 else 0}")
        logging.info(f"Neighbor unfusions: {total_neighbor_unfusions_info}")
        logging.info(f"Matches: {total_matches_info}")
        return sum(self._reduction_per_match), self._applied_matches


    def _reset_lookup_flow(self):
        self._lookup_flow_for_unfusion = {}

    def _lookup_flow_preserving_for_edge(self, graph: BaseGraph[VT, ET], edge) -> bool:
        """
        Looks up whether the flow is preserved if neighbor unfusion is applied to a given edge.

        Args:
            graph (BaseGraph[VT, ET]): The graph to check.
            edge (Tuple[VT, VT]): The edge to check.

        Returns:
            bool: True if the flow is preserved, False otherwise.
        """
        #TODO: all unfusions with 0 or 1 neighbors are gflow preserving
        self._neighbor_unfusions["total"][-1] += 1
        
        if edge not in self._lookup_flow_for_unfusion:
            flow = self._is_graph_flow_preserving(graph)
            self._lookup_flow_for_unfusion[edge] = flow
            if flow is True:
                self._neighbor_unfusions["evaluated true"][-1] += 1
            else:
                self._neighbor_unfusions["evaluated false"][-1] += 1
            return flow

        flow = self._lookup_flow_for_unfusion[edge]
        if flow is True:
            self._neighbor_unfusions["skipped true"][-1] += 1
        else:
            self._neighbor_unfusions["skipped false"][-1] += 1
        return flow

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
            return self._is_graph_flow_preserving(graph)

        edges = []
        match_key, match_value = match

        if get_match_type(match) == MatchType.LCOMP:
            vertex = match_key[0]
            unfusion_neighbor = match_value[2]
            if unfuse_to_neighbor and unfusion_neighbor:
                edges.append(graph.edge(vertex, unfusion_neighbor))

        elif get_match_type(match) == MatchType.PIVOT:
            unfusion_neighbors = {vertex: match_value[i+1] for i, vertex in enumerate(match_key)}
            edges.extend(graph.edge(vertex, unfusion_neighbors[vertex]) for vertex in match_key if unfusion_neighbors[vertex])

        if not all(edge in self._lookup_flow_for_unfusion for edge in edges):
            flow = self._is_graph_flow_preserving(graph)
            # self._neighbor_unfusions["total"][-1] += 1
            # if flow is True:
            #     self._neighbor_unfusions["evaluated true"][-1] += 1
            # else:
            #     self._neighbor_unfusions["evaluated false"][-1] += 1
            for edge in edges:
                self._lookup_flow_for_unfusion[edge] = flow
            return flow

        # if no unfusion is applied, the g flow is allready preserved
        if len(edges) == 0:
            if self.flow_function != FilterFlowFunc.G_FLOW_PRESERVING:
                return self._is_graph_flow_preserving(graph)
            else:
                return True

        #FIXME: check if this is correct
        flows = all(self._lookup_flow_for_unfusion[edge] for edge in edges)
        # self._neighbor_unfusions["total"][-1] += 1
        # if flow is True:
        #     self._neighbor_unfusions["skipped true"][-1] += 1
        # else:
        #     self._neighbor_unfusions["skipped false"][-1] += 1
        return flows

    def _calculate_flow(self, graph: BaseGraph[VT, ET]) -> Dict[VT, Set[VT]] | None:
        return self.flow_function(graph)
    
    def _is_graph_flow_preserving(self, graph: BaseGraph[VT, ET]) -> bool:
        return self._calculate_flow(graph) is not None

    def _sort_matches(
            self, 
            lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
            pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]
            ) -> Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]]:
        """
        Sorts the matches based on their heuristic result.
        Puts the possibly non-flow-preserving matches at the end of the dict.

        Args:
            lcomp_matches (Dict[Tuple[VT], List[MatchLcompHeuristicType]]): A dict of matches for local complementation.
            pivot_matches (Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]): A dict of matches for pivoting.

        Returns:
            Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]]: A sorted dict of matches.    
        """

        # Merge the dictionaries
        merged_dict = {**lcomp_matches, **pivot_matches}

        # Sort the lists in each dictionary
        merged_dict = {k: sorted(v, reverse=True, key=lambda item: item[0]) for k, v in merged_dict.items()}

        # Sort the dictionary based on the first (i.e., highest) value in each list
        merged_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1][0][0], reverse=True))

        removed_matches = []

        for (possible_non_flow_preserving_match_key, possible_non_flow_preserving_match_value) in self._possibly_non_flow_preserving_matches:
            if possible_non_flow_preserving_match_key in merged_dict.keys() and possible_non_flow_preserving_match_value in merged_dict[possible_non_flow_preserving_match_key]:
                # merged_dict[possible_non_flow_preserving_match_key].remove(possible_non_flow_preserving_match_value)
                # if len(merged_dict[possible_non_flow_preserving_match_key]) == 0:
                #     del merged_dict[possible_non_flow_preserving_match_key]
                #     merged_dict[possible_non_flow_preserving_match_key] = possible_non_flow_preserving_match_value
                pass
            else:
                removed_matches.append((possible_non_flow_preserving_match_key, possible_non_flow_preserving_match_value))
        
        for removed_match in removed_matches:
            self._possibly_non_flow_preserving_matches.remove(removed_match)

        return merged_dict

    def _get_n_best_matches(
        self,
        n: int,
        match_dict: Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]]
    ) -> List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]:
        """
        Get the best n matches out of the given matches.
        Excludes the possibly non-flow-preserving matches.
        
        Parameters:
        n (int): The number of matches to return.
        match_dict (Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]]): A dict of matches
        
        Returns:
        List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]: A list of the best n matches
        """
        
        if len(match_dict) == 0:
            return None

        if n <= 0:
            raise ValueError("n must be greater than 0")

        # Create a set of matches to remove for faster lookup
        remove_matches = list((match[0], match[1]) for match in self._possibly_non_flow_preserving_matches)

        # Create a new list of matches, excluding those that need to be removed
        matches = [(match[0], key, match) for key, match_list in match_dict.items() for match in match_list if (key, match) not in remove_matches]

        # Use a heap to keep the best n matches
        best_matches_heap = heapq.nlargest(n, matches)

        best_matches_list = []
        for value, key, match in best_matches_heap:
            best_matches_list.append((key, match))

        return best_matches_list
    
    def _get_n_random_matches(
        self,
        n: int,
        match_dict: Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]],
        matches_to_exclude: List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]] = []
    ) -> List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]:
        """
        Get n random matches out of the given matches.
        
        Parameters:
        n (int): The number of matches to return.
        match_dict (Dict[Tuple[VT, ...], List[MatchLcompHeuristicType | MatchPivotHeuristicType]]): A dict of matches
        matches_to_exclude (List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]], optional): A dict of matches to exclude from the random matches
        
        Returns:
        List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]: A list of n random matches
        """
            
        if len(match_dict) == 0:
            return None

        if n <= 0:
            raise ValueError("n must be greater than 0")

        # Create a new list of matches, excluding those that need to be excluded
        matches = [(match[0], key, match) for key, match_list in match_dict.items() for match in match_list if (key, match) not in matches_to_exclude]

        # Use a heap to keep the best n matches
        random_matches = random.sample(matches, n)

        # Convert the heap back to the desired list format
        random_matches_list = []
        for value, key, match in random_matches:
            random_matches_list.append((key, match))

        return random_matches_list        
    
    def _get_best_and_random_matches(
        self,
        n:int, 
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
        pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]
        ) -> List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]:
        """
        Get a mix of the best and random matches out of the given matches.
        1/3 of the matches are the best matches and 2/3 are random matches.
        Possibly non-flow-preserving matches are excluded from the best matches, but can still be inculed in the random matches.

        Parameters: 
        n (int): The number of matches to return.
        lcomp_matches (Dict[Tuple[VT], List[MatchLcompHeuristicType]]): A dict of matches for local complementation
        pivot_matches (Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]): A dict of matches for pivoting

        Returns:
        List[Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType]]: A list of the best and random matches
        """
        if len(lcomp_matches) == 0 and len(pivot_matches) == 0:
            return None
        
        if n <= 0:
            raise ValueError("n must be greater than 0")
        
        # Sort the matches in descending order based on the heuristic result
        # match_dict = self._sort_matches(lcomp_matches, pivot_matches)
        match_dict = {**lcomp_matches, **pivot_matches}

        best_matches = self._get_n_best_matches(max(n//3, 1), match_dict)
        random_matches = self._get_n_random_matches(n-max(n//3, 1), match_dict, best_matches) if max(n//3, 1) < n else []

        return best_matches + random_matches

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
        
        match_list = self._get_n_best_matches(len(lcomp_matches)+len(pivot_matches), {**lcomp_matches, **pivot_matches})

        for match_key, match_value in match_list:
            filter_graph = graph.clone()
            match_result = self._apply_match(filter_graph, (match_key, match_value))
            if match_result is not None:
                return match_key, match_value

        return None

    def _apply_random_match(
        self,
        graph: BaseGraph[VT,ET],
        lcomp_matches: Dict[Tuple[VT], MatchLcompHeuristicType], 
        pivot_matches: Dict[Tuple[VT,VT], MatchPivotHeuristicType],
        ) -> Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
        """
        Finds a random match in the given graph based on lcomp_matches and pivot_matches dictionaries and applies it to the graph.
        
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
        
        match_list = self._get_n_random_matches(len(lcomp_matches)+len(pivot_matches), {**lcomp_matches, **pivot_matches})

        for match_key, match_value in match_list:
            filter_graph = graph.clone()
            match_result = self._apply_match(filter_graph, (match_key, match_value))
            if match_result is not None:
                return match_key, match_value
            
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

        if match_value == (1, None, 410) and match_key == (313, 315):
            pass

        graph_copy = graph.clone()

        flow_function = self._lookup_flow_preserving_for_edge if not skip_flow_calculation and self.use_neighbor_unfusion and not (self.use_yz_phase_gadgets or self.use_xz_phase_gadgets) else None

        if get_match_type(match) == MatchType.PIVOT:
            vertex_neighbors = {vertex_neighbor for vertex in match_key for vertex_neighbor in graph.neighbors(vertex) if vertex_neighbor not in match_key}
            match_result_with_time = apply_pivot(graph=graph, match=match, flow_function=flow_function)
        elif get_match_type(match) == MatchType.LCOMP:
            _, vertex_neighbors, _ = match_value
            match_result_with_time = apply_lcomp(graph, match=match, flow_function=flow_function)
        
        match_result, time_info = match_result_with_time
        if time_info is not None:
            self._time_dict["time_to_unfuse"][-1] += time_info["time_to_unfuse"]
            self._time_dict["time_to_claculate_flow"][-1] += time_info["time_to_claculate_flow"]
            self._time_dict["time_to_apply_rule"][-1] += time_info["time_to_apply_rule"]
        
        # If an unfusion to a neighbor or a phase gadget was applied, check if the flow is preserved
        if match_result is not None and not skip_flow_calculation:
            new_vertices, edge_flow = match_result

            # If at least one edge is not flow-preserving, return None
            if edge_flow is not None and not all(edge_flow.values()):
                if match not in self._possibly_non_flow_preserving_matches:
                    self._possibly_non_flow_preserving_matches.append(match)
                not_flow_preserving_edges = [edge for edge in edge_flow if not edge_flow[edge]]
                # logging.debug(f"Edges {not_flow_preserving_edges} are not flow-preserving")
                self._matches["evaluated unfusion false"][-1] += 1
                self._matches["total unfusion matches"][-1] += 1
                return None
            # If no edge flow was calculated (vertex was unfused to a phase gadget), check if the flow is preserved for the match
            elif edge_flow is None:
                self._neighbor_unfusions["gadgets"][-1] += 1
                if not self._is_graph_flow_preserving(graph):
                    if match not in self._possibly_non_flow_preserving_matches:
                        self._possibly_non_flow_preserving_matches.append(match)
                    # logging.debug(f"Match {match} is not flow-preserving")
                    self._matches["evaluated unfusion false"][-1] += 1
                    self._matches["total unfusion matches"][-1] += 1
                    return None
                    
            vertex_neighbors = set(vertex_neighbors).union(set(new_vertices))
        
        else:
            # If no unfusion was applied and the FilterFlowFunc may not be met by the match, check if the match is flow-preserving
            # All pivot and lcomp matches are inherently gflow-preserving when no phase gadgets are used
            if not skip_flow_calculation and self.flow_function != FilterFlowFunc.G_FLOW_PRESERVING:
                if not self._is_graph_flow_preserving(graph):
                    if match not in self._possibly_non_flow_preserving_matches:
                        self._possibly_non_flow_preserving_matches.append(match)
                    # logging.debug(f"Match {match} is not flow-preserving")
                    self._matches["evaluated false"][-1] += 1
                    self._matches["total matches"][-1] += 1
                    return None
            
        if match in self._possibly_non_flow_preserving_matches:
            self._possibly_non_flow_preserving_matches.remove(match)
            self._rehabilitated_non_flow_preserving_matches += 1

        if not skip_flow_calculation:
            if is_match_unfusing(match):
                self._matches["evaluated unfusion true"][-1] += 1
                self._matches["total unfusion matches"][-1] += 1
            elif is_match_boundary(graph_copy, match):
                self._matches["total boundary matches"][-1] += 1
                self._matches["evaluated true"][-1] += 1
            else:
                self._matches["evaluated true"][-1] += 1
                self._matches["total matches"][-1] += 1

        removed_vertices = [key for key in match_key if key not in graph.vertex_set()]

        return list(vertex_neighbors), removed_vertices


    def _update_best_result(
        self,
        current_result: List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]], 
        best_result: List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]] | None,
    ) -> List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]:
        """
        Updates the best result based on the current result.

        Args:
            current_result (List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]): The current result to compare.
            best_result (List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]] | None): The current best result.

        Returns:
            List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]: The updated best result.
        """

        if current_result is None:
            raise ValueError("current_result cannot be None")
        
        if best_result is None:
            best_result = current_result
            return best_result
        
        current_heuristic = sum([match_value[0] for _, match_value in current_result])
        best_heuristic = sum([match_value[0] for _, match_value in best_result])
        if current_heuristic == best_heuristic:
            if max([match_value[0] for _, match_value in current_result]) > max([match_value[0] for _, match_value in best_result]):
                best_result = current_result
            elif random.choice([True, False]):
                best_result = current_result
        elif current_heuristic > best_heuristic:
            best_result = current_result

        return best_result
    
    def _depth_search(
        self,
        graph: BaseGraph[VT, ET],
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]],
        pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]],
        get_matches_method: Callable,
        apply_match_method: Callable,
        depth: int = 0, 
        best_result: List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]] | None = None,
        current_match_list: List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]] = [],
        full_subgraphs: bool = False
    ) -> Dict[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.
        
        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        If 'full_subgraphs' is False, the search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.
        
        Parameters:
        graph (BaseGraph[VT, ET]): The graph to search for matches in.
        lcomp_matches (Dict[Tuple[VT], List[MatchLcompHeuristicType]]): Dictionary of matches based on lcomp heuristic.
        pivot_matches (Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]]): Dictionary of matches based on pivot heuristic.
        get_matches_method (Callable): The method to get the matches.
        apply_match_method (Callable): The method to apply the matches.
        depth (int): The current depth of the search.
        best_result (List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]] | None): The best result found so far.
        current_match_dict (List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]): The current matches found so far.
        full_subgraphs (bool): Whether to consider full subgraphs or not.
        """

        if depth == self.lookahead:
            current_results = apply_match_method(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches)
            if not current_results:
                return best_result
            
            current_key, current_result = current_results
            lookahead_current_match_list = current_match_list.copy()
            lookahead_current_match_list.append((current_key, current_result))

            if (sum([match_value[0] for _, match_value in lookahead_current_match_list]) < self.threshold and len(lookahead_current_match_list) > 0):
                return best_result

            best_result = self._update_best_result(current_result=lookahead_current_match_list, best_result=best_result)
            
            match_heuristic = sum([match_value[0] for _, match_value in best_result])
            if match_heuristic < self.threshold:
                logging.debug(f"Best match could not be applied at depth {depth} due to heuristic result {match_heuristic} <= {self.threshold}.")
            # return best_result
            
            # self.log_data(depth)
            return best_result if match_heuristic >= self.threshold else None

        if not lcomp_matches and not pivot_matches:
            return self._update_best_result(current_result=current_match_list, best_result=best_result) if current_match_list else best_result

        num_matches = sum([len(match_list) for match_list in lcomp_matches.values()]) + sum([len(match_list) for match_list in pivot_matches.values()])
        
        num_sub_branches = max(int(num_matches  ** (1/(depth+2))), 1) if not full_subgraphs else max(num_matches, 1)

        matches = get_matches_method(n=num_sub_branches, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches)

        if depth == 0:
            iterator = tqdm(matches)
        else:
            iterator = matches

        for match in iterator:
            lookahead_graph = graph.clone()
            
            match_result = self._apply_match(lookahead_graph, match, skip_flow_calculation=False)

            if match_result is not None:
                vertex_neighbors, removed_vertices = match_result
                
                # For testing purposes
                # if not self._is_graph_flow_preserving(lookahead_graph):
                #     raise Exception("Flow is not preserved after applying the match")
                
                lookahead_lcomp_matches, lookahead_pivot_matches = update_matches(graph=lookahead_graph, vertex_neighbors=vertex_neighbors, removed_vertices=removed_vertices, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, check_for_unfusions=self.use_neighbor_unfusion, check_for_yz_phase_gadgets=self.use_yz_phase_gadgets, check_for_xz_phase_gadgets=self.use_xz_phase_gadgets)
                lookahead_current_match_list = current_match_list.copy()

                self._reset_lookup_flow()

                lookahead_current_match_list.append(match)
                
                current_result = self._depth_search(
                    graph=lookahead_graph, 
                    lcomp_matches=lookahead_lcomp_matches, 
                    pivot_matches=lookahead_pivot_matches, 
                    get_matches_method=get_matches_method,
                    apply_match_method=apply_match_method,
                    depth=depth + 1, 
                    best_result=best_result, 
                    current_match_list=lookahead_current_match_list
                )

                if current_result is not None:
                    best_result = self._update_best_result(current_result=current_result, best_result=best_result)

        self._log_data()
        return best_result

    def _log_data(self):
        current_match_info = {key: values[-1] for key, values in self._matches.items()}
        current_neighbor_unfusions = {key: values[-1] for key, values in self._neighbor_unfusions.items()}
        current_times = {key: times[-1] for key, times in self._time_dict.items()}
        logging.debug(f"Current Neighbor unfusions {current_neighbor_unfusions}")
        logging.debug(f"Current Matches {current_match_info}")
        logging.debug(f"Current Time {current_times}")
        logging.debug(f"Rehabilitated {self._rehabilitated_non_flow_preserving_matches} non flow-preserving matches")
        logging.debug(f"Number of vertices: {len(self.graph.vertex_set())}, Number of edges: {len(self.graph.edge_set())}")

    def _full_search_match_with_best_result_at_depth(self, graph, lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]]) -> Dict[Tuple, List[MatchLcompHeuristicType | MatchPivotHeuristicType]] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search explores every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, get_matches_method=self._get_best_and_random_matches, apply_match_method=self._apply_best_match, full_subgraphs=True)

    def _search_match_with_best_result_at_depth(self, graph, lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]]) -> Dict[Tuple, List[MatchLcompHeuristicType | MatchPivotHeuristicType]] | None:
        """
        Perform a depth-first search on the graph to find the best result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, get_matches_method=self._get_best_and_random_matches, apply_match_method=self._apply_best_match, full_subgraphs=False)

    def _full_search_match_with_random_result_at_depth(self, graph, lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]]) -> Dict[Tuple, List[MatchLcompHeuristicType | MatchPivotHeuristicType]] | None:
        """
        Perform a depth-first search on the graph to find a random result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies a random match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search explores every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, get_matches_method=self._get_random_matches, apply_match_method=self._apply_random_match, full_subgraphs=True)

    def _search_match_with_random_result_at_depth(self, graph, lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]]) -> Dict[Tuple, List[MatchLcompHeuristicType | MatchPivotHeuristicType]] | None:
        """
        Perform a depth-first search on the graph to find a random result at a specific depth.

        This function recursively explores the graph using depth-first search. When the specified depth (lookahead) is reached,
        it applies a random match (either a local complement or a pivot) to the graph and updates the best result found so far.
        The search progressively narrows down the percentage of top paths it considers as it delves deeper, meaning it doesn't explore every single path to the maximum lookahead depth.

        Returns:
        The best result found at the lookahead depth and the match that led to it.
        """
        return self._depth_search(graph=graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, get_matches_method=self._get_random_matches, apply_match_method=self._apply_random_match, full_subgraphs=False)


    def _apply_and_find_new_matches(
            self, 
            lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
            pivot_matches: Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]],
            find_matches_method: Callable,
            ) -> Tuple[Dict[Tuple[VT], List[MatchLcompHeuristicType]], Dict[Tuple[VT, VT], List[MatchPivotHeuristicType]]]:
        """
        Apply the best match found by searching for the best result at a specific depth.

        This function searches for the best result at a specific depth using depth-first search.
        It applies the best match (either a local complement or a pivot) to the graph and updates the best result found so far.
        """

        stop_search = False

        num_matches = 10
        if len(self._reduction_per_match) >= num_matches:
            last_matches = self._reduction_per_match[-num_matches:]
            if sum(last_matches) <= self.threshold:
                self.has_changes_occurred = False
                warnings.warn(message=f"Reduction of last {num_matches} matches: {sum(last_matches)} is too low. Stopping.")
                logging.warning(f"Reduction of the last {num_matches} matches: {sum(last_matches)}")
                stop_search = True
            else:
                logging.debug(f"Reduction of the last {num_matches} matches: {sum(last_matches)}")

        if not stop_search:
            
            best_match_list = find_matches_method(graph=self.graph, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches)

            if best_match_list is not None:
                
                if not self._apply_all_lookahead_matches:
                    matches_to_apply = [next(iter(best_match_list))]
                else:
                    matches_to_apply = best_match_list

                for best_key, best_result in matches_to_apply:

                    number_of_edges_before = self.graph.num_edges()

                    match_result = self._apply_match(self.graph, (best_key, best_result), skip_flow_calculation=True)


                    if 90 not in self.graph.vertex_set():
                        pass
                    # For testing purposes
                    # if not self._is_graph_flow_preserving(self.graph):
                    #     raise Exception("Flow is not preserved after applying the match")

                    if match_result is not None:
                        vertex_neighbors, removed_vertices = match_result
                        lcomp_matches, pivot_matches = update_matches(graph=self.graph, vertex_neighbors=vertex_neighbors, removed_vertices=removed_vertices, lcomp_matches=lcomp_matches, pivot_matches=pivot_matches, check_for_unfusions=self.use_neighbor_unfusion, check_for_yz_phase_gadgets=self.use_yz_phase_gadgets, check_for_xz_phase_gadgets=self.use_xz_phase_gadgets)
                    else:
                        raise Exception(f"Best match: {best_key} was found but could not be applied.")

                    self.has_changes_occurred = True
                    self._rule_application_count += 1
                    self._reduction_per_match.append(number_of_edges_before - self.graph.num_edges())


                    self._applied_matches.append((best_key, best_result))
                    self._remaining_matches.append((len(lcomp_matches), len(pivot_matches)))

                    logging.info(f"Applied match #{self._rule_application_count}: {best_key}, {best_result}")
                    logging.debug(f"Found {len(lcomp_matches)} local complement matches and {len(pivot_matches)} pivot matches after applying match")
                
                for key in self._matches.keys():
                    self._matches[key].append(0)
                for key in self._neighbor_unfusions.keys():
                    self._neighbor_unfusions[key].append(0)
                
                self._time_dict["time_to_apply_rule"].append(0)
                self._time_dict["time_to_unfuse"].append(0)
                self._time_dict["time_to_claculate_flow"].append(0)

                pid = os.getpid()
                python_process = psutil.Process(pid)
                memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
                cpuUse = python_process.cpu_percent()
                logging.debug(f"Memory usage: {memoryUse} GB, CPU usage: {cpuUse}%")

            else:
                logging.info("No more matches found")

        return lcomp_matches, pivot_matches




def greedy_wire_reduce(
    graph: BaseGraph[VT, ET],
    max_vertex_index=None,
    threshold=0,
    lookahead=0,
    use_neighbor_unfusion=True,
    use_yz_phase_gadgets=True,
    use_xz_phase_gadgets=True,
    flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
    quiet=True,
    stats=None,
) -> Tuple[int, List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]]:
    """
    Perform a greedy Hadamard wire reduction on the given graph.

    This function iteratively applies the best local complement or pivot match to the graph until no further improvements can be made. The "best" match is determined by a heuristic that considers the number of Hadamard wires added by the match.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to simplify.
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine. This is needed to prevent non-termination in the case of heuristic_threshold<0.
    threshold (int): Lower bound for heuristic result. Any rule application which adds more than this number of Hadamard wires is filtered out. Defaults to 0.
    lookahead (int): The number of steps to look ahead when searching for the best match. Defaults to 0.
    use_neighbor_unfusion (bool): Whether to use neighbor unfusion. Defaults to True.
    use_yz_phase_gadgets (bool): Whether to use phase gadgets in pivot matches. Defaults to True.
    use_xz_phase_gadgets (bool): Whether to use phase gadgets in local complement matches. Defaults to True.
    flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches. Defaults to lambda x: True.

    Returns:
    tuple: A tuple containing the amount of reduced wires and the list of all applied matches
    """
    reducer = WireReducer(
        graph=graph,
        use_neighbor_unfusion=use_neighbor_unfusion,
        use_yz_phase_gadgets=use_yz_phase_gadgets,
        use_xz_phase_gadgets=use_xz_phase_gadgets,
        max_vertex_index=max_vertex_index,
        threshold=threshold,
        lookahead=lookahead,
        flow_function=flow_function,
        quiet=quiet,
        stats=stats,
    )
    reducer._apply_all_lookahead_matches = True
    return reducer.greedy_wire_reduce()

def random_wire_reduce(
    graph: BaseGraph[VT, ET],
    max_vertex_index=None,
    threshold=0,
    lookahead=0,
    use_neighbor_unfusion=True,
    use_yz_phase_gadgets=True,
    use_xz_phase_gadgets=True,
    flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
    quiet=True,
    stats=None,
) -> Tuple[int, List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]]:
    """
    Perform a random Hadamard wire reduction on the given graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to simplify.
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine. This is needed to prevent non-termination in the case of heuristic_threshold<0.
    threshold (int): Lower bound for heuristic result. Any rule application which adds more than this number of Hadamard wires is filtered out. Defaults to 0.
    lookahead (int): The number of steps to look ahead when searching for the best match. Defaults to 0.
    use_neighbor_unfusion (bool): Whether to use neighbor unfusion. Defaults to True.
    use_yz_phase_gadgets (bool): Whether to use phase gadgets in pivot matches. Defaults to True.
    use_xz_phase_gadgets (bool): Whether to use phase gadgets in local complement matches. Defaults to True.
    flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches. Defaults to lambda x: True.

    Returns:
    tuple: A tuple containing the amount of reduced wires and the list of all applied matches
    """

    reducer = WireReducer(
        graph=graph,
        use_neighbor_unfusion=use_neighbor_unfusion,
        use_yz_phase_gadgets=use_yz_phase_gadgets,
        use_xz_phase_gadgets=use_xz_phase_gadgets,
        max_vertex_index=max_vertex_index,
        threshold=threshold,
        lookahead=lookahead,
        flow_function=flow_function,
        quiet=quiet,
        stats=stats,
    )
    return reducer.random_wire_reduce()

def sim_annealing_wire_reduce(
    graph: BaseGraph[VT,ET], 
    max_vertex_index=None,
    use_neighbor_unfusion=True,
    initial_temperature=100, 
    cooling_rate=0.95, 
    threshold=-100000, 
    flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
    quiet:bool=False, 
    stats=None
) -> Tuple[int, List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]]:
    """
    Perform a random Hadamard wire reduction on the given graph.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to simplify.
    max_vertex_index (int): The highest index of any vertex present at the beginning of the heuristic simplification routine. This is needed to prevent non-termination in the case of heuristic_threshold<0.
    use_neighbor_unfusion (bool): Whether to use neighbor unfusion. Defaults to False.
    initial_temperature (int): The initial temperature for the simulated annealing algorithm. Defaults to 100.
    cooling_rate (float): The cooling rate for the simulated annealing algorithm. Defaults to 0.95.
    threshold (int): Lower bound for heuristic result. Any rule application which adds more than this number of Hadamard wires is filtered out. Defaults to -100000.
    flow_function (FilterFlowFunc): A function to filter out non-flow-preserving matches. Defaults to lambda x: True.

    Returns:
    tuple: A tuple containing the amount of reduced wires and the list of all applied matches
    """
    if use_neighbor_unfusion and flow_function == FilterFlowFunc.NONE:
        flow_function = FilterFlowFunc.G_FLOW_PRESERVING
        warnings.warn("Neighbor unfusion requires a flow function. Using G-flow preserving function.")
        logging.warning("Neighbor unfusion requires a flow function. Using G-flow preserving function.")

    if not use_neighbor_unfusion and flow_function == FilterFlowFunc.G_FLOW_PRESERVING:
        flow_function = FilterFlowFunc.NONE
        warnings.warn("G-flow preserving function is not needed without neighbor unfusion. Using no flow function.")
        logging.warning("G-flow preserving function is not needed without neighbor unfusion. Using no flow function.")

    return _sim_annealing_reduce(
        graph=graph,
        max_vertex_index=max_vertex_index,
        use_neighbor_unfusion=use_neighbor_unfusion,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        threshold=threshold,
        flow_function=flow_function,
        quiet=quiet,
        stats=stats,
    )




def _get_best_match(
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
        pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]], 
        max_vertex_index=None, 
        threshold=-100000
        ) -> Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
    """
    Get the best match from the list of local complementation and pivot matches.

    Parameters:
    lcomp_matches (dict): The dict of local complementation matches.
    pivot_matches (dict): The dict of pivot matches.
    max_vertex_index (int, optional): The maximum vertex to consider for matches. Defaults to None.
    threshold (int, optional): The threshold for the heuristic result. Defaults to -100000.

    Returns:
    Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType] | None: The best match from the list of local complementation and pivot matches.
    """

    match_dict = {**lcomp_matches, **pivot_matches}

    # Create a new list of matches, excluding those that need to be removed
    matches = [(match[0], key, match) for key, match_list in match_dict.items() for match in match_list if not max_vertex_index is None and max(match[0]) <= max_vertex_index and match[1] >= threshold]

    # Use a heap to keep the best n matches
    best_matches_heap = heapq.nlargest(1, matches)

    return best_matches_heap[0] if best_matches_heap else None

def _get_random_match(
        lcomp_matches: Dict[Tuple[VT], List[MatchLcompHeuristicType]], 
        pivot_matches: Dict[Tuple[VT,VT], List[MatchPivotHeuristicType]], 
        max_vertex_index=None, 
        threshold=-100000
        ) -> Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType] | None:
    """
    Get a random match from the list of local complementation and pivot matches.

    Parameters:
    local_complement_matches (list): The list of local complementation matches.
    pivot_matches (list): The list of pivot matches.

    Returns:
    Tuple[Tuple[VT, ...], MatchLcompHeuristicType | MatchPivotHeuristicType] | None: A random match from the list of local complementation and pivot matches.
    """
    if len(lcomp_matches) == 0 and len(pivot_matches) == 0:
        return ("none", None)

    match_dict = {**lcomp_matches, **pivot_matches}

    # Create a new list of matches, excluding those that need to be removed
    matches = [(match[0], key, match) for key, match_list in match_dict.items() for match in match_list if not max_vertex_index is None and max(match[0]) <= max_vertex_index and match[1] >= threshold]

    random_match = random.choice(matches)

    return random_match if random_match else None

def _apply_match(
        graph: BaseGraph[VT, ET], 
        match: Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType], 
        flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
        ) -> Tuple[List[VT], Tuple[VT, ...]] | None:
    """
    Applies the given match to the graph and updates the dicts of local complement and pivot matches.
    
    Args:
        graph (BaseGraph): The graph to search for matches in.
        match (Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]): The match to apply.
        flow_function (FilterFlowFunc, optional): A function to filter out non-flow-preserving matches. Defaults to FilterFlowFunc.NONE.
        
    Returns:
        Tuple[List[VT], Tuple[VT, ...]] | None: The neighbors of the matched vertices and the removed vertices. None if the match is not flow-preserving.
    """

    match_key, match_value = match
    vertex_neighbors = set()

    if get_match_type(match) == MatchType.PIVOT:
        vertex_neighbors = {vertex_neighbor for vertex in match_key for vertex_neighbor in graph.neighbors(vertex) if vertex_neighbor not in match_key}
        match_result_with_time = apply_pivot(graph=graph, match=match)
    elif get_match_type(match) == MatchType.LCOMP:
        _, vertex_neighbors, _ = match_value
        match_result_with_time = apply_lcomp(graph, match=match)
    
    match_result, time_info = match_result_with_time

    if match_result is not None:
        new_vertices, _ = match_result
        
        # Check if the match is flow-preserving
        if flow_function(graph) is None:
            logging.debug(f"Match {match} is not flow-preserving")
            return None
                
        vertex_neighbors = set(vertex_neighbors).union(set(new_vertices))
    else:
        raise Exception(f"Match {match} was found but could not be applied.")
    
    return list(vertex_neighbors), match_key

def _sim_annealing_reduce(
        graph: BaseGraph[VT,ET], 
        max_vertex_index=None,
        use_neighbor_unfusion=True,
        use_yz_phase_gadgets=True,
        use_xz_phase_gadgets=True,
        initial_temperature=100, 
        cooling_rate=0.95, 
        threshold=-100000, 
        flow_function: FilterFlowFunc = FilterFlowFunc.NONE,
        quiet:bool=False, 
        stats=None
    ) -> Tuple[int, List[Tuple[Tuple, MatchLcompHeuristicType | MatchPivotHeuristicType]]]:
    """
    Reduce the number of wires in a graph using simulated annealing with neighbor unfusion.

    Parameters:
    graph (BaseGraph[VT,ET]): The graph to reduce the number of wires in.
    max_vertex_index (int, optional): The maximum vertex to consider for matches.
    use_neighbor_unfusion (bool): Whether to use neighbor unfusion. Defaults to True.
    use_yz_phase_gadgets (bool): Whether to use phase gadgets in pivoting. Defaults to True.
    use_xz_phase_gadgets (bool): Whether to use phase gadgets in local complementations. Defaults to True.
    initial_temperature (int, optional): initial_temperature (int): Initial temperature for the simulated annealing process.
    cooling_rate (float, optional): The rate at which the temperature decreases.
    threshold (int, optional): The minimum wire reduction for a match to be considered.
    flow_function (FilterFlowFunc, optional): A function to filter out non-flow-preserving matches. Defaults to FilterFlowFunc.NONE.
    quiet (bool, optional): Whether to suppress output.
    stats (dict, optional): A dictionary to store statistics.

    Returns:
    tuple: A tuple containing the total wire reduction and the list of all applied matches.
    """
    temperature = initial_temperature
    min_temperature = 1
    iteration_count = 0

    best_graph = graph.copy()
    best_evaluation = graph.num_edges()
    current_evaluation = best_evaluation

    backtrack = False

    applied_matches = []
    reduction_per_match = []

    local_complement_matches = lcomp_matcher(best_graph, check_for_unfusions=use_neighbor_unfusion, check_for_xz_phase_gadgets=use_xz_phase_gadgets)
    pivot_matches = pivot_matcher(best_graph, check_for_unfusions=use_neighbor_unfusion, check_for_phase_gadgets=use_yz_phase_gadgets)

    while temperature > min_temperature:
        iteration_count += 1

        match = _get_best_match(local_complement_matches, pivot_matches, max_vertex_index, threshold)
        if match is None:
            temperature = 0
            break

        if match[1][0] <= 0:
            if backtrack:
                current_evaluation = best_evaluation
                backtrack = False
                continue
            else:
                match = _get_random_match(local_complement_matches, pivot_matches, max_vertex_index, threshold)
                backtrack = True

        if match is None:
            temperature = 0
            break

        match_key, match_value = match

        acceptance_probability = math.exp(match_value[0]/temperature)
        # If the wire reduction of the match is positive or the acceptance probability is greater than a random number
        if match_value[0] > 0 or acceptance_probability > random.random():
            
            graph_copy = graph.copy()
            match_result = _apply_match(graph_copy, match, flow_function=flow_function)
            if match_result is None:
                continue
            
            graph = graph_copy
            vertex_neighbors, removed_vertices = match_result

            current_evaluation -= match_value[0]

            if current_evaluation < best_evaluation:
                best_graph = graph.copy()
                best_evaluation = current_evaluation

            applied_matches.append(match)
            reduction_per_match.append(match_value[0])

            local_complement_matches, pivot_matches = update_matches(graph, vertex_neighbors, removed_vertices, local_complement_matches, pivot_matches, check_for_unfusions=use_neighbor_unfusion, check_for_yz_phase_gadgets=use_yz_phase_gadgets, check_for_xz_phase_gadgets=use_xz_phase_gadgets, max_vertex_index=max_vertex_index)

        temperature *= cooling_rate

    logging.info(f"Total rule applications: {len(applied_matches)}, Total reduction: {sum(reduction_per_match)}, Average reduction: {sum(reduction_per_match)/len(reduction_per_match) if len(reduction_per_match) > 0 else 0}")
    return (sum(reduction_per_match), applied_matches)