
from fractions import Fraction
from pyzx.rules import apply_rule, lcomp, pivot
from .heuristics import get_phase_type, lcomp_heuristic, lcomp_heuristic_neighbor_unfusion, pivot_heuristic, pivot_heuristic_neighbor_unfusion
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Tuple, List
from pyzx.utils import VertexType, EdgeType
from .tools import split_phases, insert_identity
from pyzx.gflow import gflow
from .gflow_calculation import update_gflow_from_double_insertion, update_gflow_from_lcomp, update_gflow_from_pivot


MatchLcompHeuristicNeighbourType = Tuple[float,Tuple[VT,List[VT],VT]]

def lcomp_matcher(g: BaseGraph[VT,ET], gfl=None) -> List[MatchLcompHeuristicNeighbourType]:
    candidates = g.vertex_set()
    types = g.types()

    m = []
    while len(candidates) > 0:
        v = candidates.pop()
        vt = types[v]
        va = g.phase(v)
        
        if vt != VertexType.Z: continue
                
        vn = list(g.neighbors(v))
        boundary = False
        for neighbor in vn:
            if types[neighbor] != VertexType.Z: # no boundaries
                boundary = True
        
        if boundary: continue # for the moment
        if get_phase_type(g.phase(v)) == 1:
            m.append((lcomp_heuristic(g,v),(v,vn),None))
        else:
            for neighbor in get_possible_unfusion_neighbours(g, v, None, gfl):
                m.append((lcomp_heuristic_neighbor_unfusion(g,v,neighbor),(v,vn),neighbor))
    return m

MatchPivotHeuristicNeighbourType = Tuple[float,Tuple[VT,VT],VT,VT]

def pivot_matcher(g: BaseGraph[VT,ET], gfl=None) -> List[MatchPivotHeuristicNeighbourType]:
    candidates = g.edge_set()
    types = g.types()
    m = []
    while len(candidates) > 0:
        e = candidates.pop()
        if g.edge_type(e) != EdgeType.HADAMARD: continue
        v0, v1 = g.edge_st(e)
        if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z): continue
        
        boundary = False
        for neighbor in g.neighbors(v0):
            if types[neighbor] != VertexType.Z: #no boundaries
                boundary = True

        for neighbor in g.neighbors(v1):
            if types[neighbor] != VertexType.Z: #no boundaries
                boundary = True

        if boundary: continue

        if get_phase_type(g.phase(v0)) == 2:
            if get_phase_type(g.phase(v1)) == 2:
                m.append((pivot_heuristic(g,e),(v0,v1),None,None))
            else:
                for neighbor in get_possible_unfusion_neighbours(g, v1, v0, gfl):
                    m.append((pivot_heuristic_neighbor_unfusion(g,e,None,neighbor),(v0,v1),None,neighbor))
        else:
            if get_phase_type(g.phase(v1)) == 2:
                for neighbor in get_possible_unfusion_neighbours(g, v0, v1, gfl):
                    m.append((pivot_heuristic_neighbor_unfusion(g,e,neighbor,None),(v0,v1),neighbor,None))
            else:
                for neighbor_v0 in get_possible_unfusion_neighbours(g, v0, v1, gfl):
                    for neighbor_v1 in get_possible_unfusion_neighbours(g, v1, v0, gfl):
                        m.append((pivot_heuristic_neighbor_unfusion(g,e,neighbor_v0,neighbor_v1),(v0,v1),neighbor_v0,neighbor_v1))

    return m

def get_possible_unfusion_neighbours(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, gfl=None):
    possible_unfusion_neighbours = []
    if len(gfl[vertex]) == 1:
        possible_unfusion_neighbours.append(next(iter(gfl[vertex]))) #get first element of set
    for neighbor in g.neighbors(vertex):
        if vertex in gfl[neighbor] and len(gfl[neighbor]) == 1:
            possible_unfusion_neighbours.append(neighbor)

    if exclude_vertex and exclude_vertex in possible_unfusion_neighbours:
        # print("removed an exclude vertex ",exclude_vertex)
        possible_unfusion_neighbours.remove(exclude_vertex)
    return possible_unfusion_neighbours


def unfuse_to_neighbor(g,vertex,neighbor,desired_phase):
    new_phase = split_phases(g.phases()[vertex], desired_phase)
    phase_spider = g.add_vertex(VertexType.Z,-2,g.rows()[vertex],new_phase)
    g.set_phase(vertex, desired_phase)
    g.add_edge((phase_spider,neighbor), EdgeType.HADAMARD)
    g.add_edge((vertex,phase_spider), EdgeType.SIMPLE)
    phaseless_spider = insert_identity(g,vertex,phase_spider)

    g.remove_edge(g.edge(vertex,neighbor))
    # print("unfuse to neighbor ",vertex, neighbor, phaseless_spider, phase_spider)
    return (phaseless_spider, phase_spider)

def apply_lcomp(g: BaseGraph[VT,ET], match, gfl):
    # print("apply lcomp match ",match)
    v,neighbors = match[1]
    unfusion_neighbor = match[2]
    neighbor_copy = neighbors[:]
    if unfusion_neighbor:
        phaseless_s, phase_s = unfuse_to_neighbor(g, v, unfusion_neighbor, Fraction(1,2))
        update_gflow_from_double_insertion(gfl, v, unfusion_neighbor, phaseless_s, phase_s)
        neighbor_copy = [phaseless_s if i==unfusion_neighbor else i for i in neighbor_copy]
    update_gflow_from_lcomp(g, v, gfl)
    apply_rule(g, lcomp, [(v, neighbor_copy)])


def apply_pivot(g: BaseGraph[VT,ET], match, gfl):
    # print("apply pivot match ",match)
    v1,v2 = match[1]
    unfusion_neighbors = {}
    unfusion_neighbors[v1] = match[2]
    unfusion_neighbors[v2] = match[3]
    for vertex in [v1,v2]:
        if unfusion_neighbors[vertex]:
            phaseless_s, phase_s = unfuse_to_neighbor(g, vertex, unfusion_neighbors[vertex], Fraction(0,1))
            update_gflow_from_double_insertion(gfl, vertex, unfusion_neighbors[vertex], phaseless_s, phase_s)
            
    update_gflow_from_pivot(g, v1, v2, gfl)
    apply_rule(g, pivot, [(v1,v2,[],[])])

def generate_matches(g, gfl,max_v=None, cap=1):
    lcomp_matches = lcomp_matcher(g, gfl)
    pivot_matches = pivot_matcher(g, gfl)
    # spider count > 0, spider count == 0, spider count < 0
    # wire_count > 0, wire_count == 0, wire count < 0
    # wire_count >= cap, wire_count < cap
    # match <= max_v, match > max_v
    filtered_lcomp_matches = []
    filtered_pivot_matches = []
    for match in lcomp_matches:
        wire_reduce, vertices, neighbor = match
        if wire_reduce < cap:
            continue
        if max_v and wire_reduce <= 0 and vertices[0] > max_v: # and spider_count >= 0 prevent non-termination
            continue
        filtered_lcomp_matches.append((wire_reduce, vertices, neighbor))
    
    for match in pivot_matches:
        wire_reduce, vertices, neighbor1, neighbor2 = match
        if wire_reduce < cap:
            continue
        if max_v and wire_reduce <= 0 and vertices[0] > max_v and vertices[1] > max_v: # and spider_count >= 0 prevent non-termination
            continue
        filtered_pivot_matches.append((wire_reduce, vertices, neighbor1, neighbor2))

    return (filtered_lcomp_matches, filtered_pivot_matches)

def greedy_wire_reduce_neighbor(g: BaseGraph[VT,ET], max_v=None, cap=1, quiet:bool=False, stats=None):
    changes = True
    iterations = 0
    gfl = gflow(g)[1]
    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, gfl=gfl, max_v=max_v, cap=cap)
        if apply_best_match(g, lcomp_matches, pivot_matches, gfl):
            iterations += 1
            changes = True
            gfl = gflow(g)[1]
        else:
            continue

    return iterations

def apply_best_match(g, lcomp_matches, pivot_matches, gfl):
    lcomp_matches.sort(key= lambda m: m[0],reverse=True)
    pivot_matches.sort(key= lambda m: m[0],reverse=True)
    method = "pivot"

    if len(lcomp_matches) > 0:
        if len(pivot_matches) > 0:
            if lcomp_matches[0][0] > pivot_matches[0][0]:
                method = "lcomp"      
        else:
            method = "lcomp"
    else:
        if len(pivot_matches) == 0:
            return False

    if method == "pivot":
        apply_pivot(g,pivot_matches[0], gfl)
    else:
        apply_lcomp(g,lcomp_matches[0], gfl)
    return True
