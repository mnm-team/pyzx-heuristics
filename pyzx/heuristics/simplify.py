from fractions import Fraction
from pyzx.rules import apply_rule, lcomp, pivot
from .heuristics import get_phase_type, lcomp_heuristic, pivot_heuristic, lcomp_heuristic_boundary, pivot_heuristic_boundary
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Tuple, List
from pyzx.utils import VertexType, EdgeType
from .tools import split_phases, insert_identity
import math
import random

MatchLcompHeuristicType = Tuple[float,Tuple[VT,List[VT]],int]

def lcomp_matcher(g: BaseGraph[VT,ET], boundaries=False, gadgets=False) -> List[MatchLcompHeuristicType]:
    candidates = g.vertex_set()
    types = g.types()

    m = []
    while len(candidates) > 0:
        v = candidates.pop()
        vt = types[v]
        va = g.phase(v)
        
        if vt != VertexType.Z: continue
        gadgetize = get_phase_type(va) != 1
        if gadgets == False and gadgetize: continue # no gadgets if not specified
        if len(g.neighbors(v)) == 1: continue # no phase gadget top
                
        vn = list(g.neighbors(v))
        is_already_gadget = False
        boundary_count = 0
        for neighbor in vn:
            if len(g.neighbors(neighbor)) == 1 and get_phase_type(va) != 1: # no phase gadget root
                is_already_gadget = True
            if types[neighbor] != VertexType.Z: # count boundary neigbors
                boundary_count += 1

        if is_already_gadget and gadgetize: continue
        if not boundaries and boundary_count > 0: continue

        spider_count = -1 + boundary_count + (2 if gadgetize else 0)

        if boundary_count > 0:
            m.append((lcomp_heuristic(g,v)-boundary_count,(v,vn),spider_count))
        else:
            m.append((lcomp_heuristic(g,v),(v,vn),spider_count))
        
    return m

MatchPivotHeuristicType = Tuple[float,Tuple[VT,VT]]

def pivot_matcher(g: BaseGraph[VT,ET], boundaries=False, gadgets=False) -> List[MatchPivotHeuristicType]:
    candidates = g.edge_set()
    types = g.types()
    m = []
    while len(candidates) > 0:
        e = candidates.pop()
        if g.edge_type(e) != EdgeType.HADAMARD: continue
        v0, v1 = g.edge_st(e)
        if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z): continue
        gadgetize_v0 = get_phase_type(g.phase(v0)) != 2
        gadgetize_v1 = get_phase_type(g.phase(v1)) != 2
        if gadgets == False and (gadgetize_v0 or gadgetize_v1): continue # no phase gadget generation if not wanted
        if get_phase_type(g.phase(v0)) != 2 and get_phase_type(g.phase(v1)) != 2: continue # not both phase gadgets
        if len(g.neighbors(v0)) == 1 or len(g.neighbors(v1)) == 1: continue # no phase gadget top
        
        v0_is_already_gadget = False
        v1_is_already_gadget = False
        boundary_count = 0
        for neighbor in g.neighbors(v0):
            if types[neighbor] != VertexType.Z: #no boundaries
                boundary_count += 1

            if len(g.neighbors(neighbor)) == 1 and get_phase_type(g.phases()[v0]) != 2: #no second phase gadget on root
                v0_is_already_gadget = True

        for neighbor in g.neighbors(v1):
            if types[neighbor] != VertexType.Z: #no boundaries
                boundary_count += 1

            if len(g.neighbors(neighbor)) == 1 and get_phase_type(g.phases()[v1]) != 2: #no second phase gadget on root
                v1_is_already_gadget = True

        if (v0_is_already_gadget and gadgetize_v0) or (v1_is_already_gadget and gadgetize_v1): continue
        if not boundaries and boundary_count > 0: continue

        spider_count = -2 + boundary_count + (2 if gadgetize_v0 else 0) + (2 if gadgetize_v1 else 0)

        if boundaries:
            m.append((pivot_heuristic(g,e)-boundary_count,(v0,v1), spider_count))
        else:
            m.append((pivot_heuristic(g,e),(v0,v1), spider_count))

    return m

def apply_best_match(g, lcomp_matches, pivot_matches):
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
        apply_pivot(g,pivot_matches[0])
    else:
        apply_lcomp(g,lcomp_matches[0])
    return True

def generate_matches(g, boundaries=False, gadgets=False, max_v=None, cap=1):
    lcomp_matches = lcomp_matcher(g, boundaries=boundaries, gadgets=gadgets)
    pivot_matches = pivot_matcher(g, boundaries=boundaries, gadgets=gadgets)
    # spider count > 0, spider count == 0, spider count < 0
    # wire_count > 0, wire_count == 0, wire count < 0
    # wire_count >= cap, wire_count < cap
    # match <= max_v, match > max_v
    filtered_lcomp_matches = []
    filtered_pivot_matches = []
    for match in lcomp_matches:
        wire_reduce, vertices, spider_count = match
        if wire_reduce < cap:
            continue
        if max_v and wire_reduce <= 0 and vertices[0] > max_v: # prevent non-termination
            continue
        filtered_lcomp_matches.append((wire_reduce, vertices, spider_count))
    
    for match in pivot_matches:
        wire_reduce, vertices, spider_count = match
        if wire_reduce < cap:
            continue
        if max_v and wire_reduce <= 0 and vertices[0] > max_v and vertices[1] > max_v: #prevent non-termination
            continue
        filtered_pivot_matches.append((wire_reduce, vertices, spider_count))

    return (filtered_lcomp_matches, filtered_pivot_matches)

def get_random_match(lcomp_matches, pivot_matches):
    method = "pivot"
    if len(lcomp_matches) > 0 and random.randint(0, 1) == 1:
        method = "lcomp"
    

    if len(lcomp_matches) > 0:
        if len(pivot_matches) > 0:
            if random.randint(0, 1) == 1:
                method = "lcomp"      
        else:
            method = "lcomp"
    else:
        if len(pivot_matches) == 0:
            return ("none",None)
    if method == "pivot":
        return ("pivot", pivot_matches[random.randint(0, len(pivot_matches)-1)])
    else:
        return ("lcomp", lcomp_matches[random.randint(0, len(lcomp_matches)-1)])


def apply_random_match(g, lcomp_matches, pivot_matches):
    method, match = get_random_match(lcomp_matches, pivot_matches)

    if method == "pivot":
        apply_pivot(g,match)
    elif method == "lcomp":
        apply_lcomp(g,match)
    else:
        return False
    return True

def random_wire_reduce(g: BaseGraph[VT,ET], boundaries=False, gadgets=False, max_v=None, cap=1, quiet:bool=False, stats=None):
    changes = True
    iterations = 0

    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, boundaries=boundaries, gadgets=gadgets, max_v=max_v, cap=cap)
        if apply_random_match(g, lcomp_matches, pivot_matches):
            iterations += 1
            changes = True

    return iterations

def greedy_wire_reduce(g: BaseGraph[VT,ET], boundaries=False, gadgets=False, max_v=None, cap=1, quiet:bool=False, stats=None):
    changes = True
    iterations = 0

    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, boundaries=boundaries, gadgets=gadgets, max_v=max_v, cap=cap)
        if apply_best_match(g, lcomp_matches, pivot_matches):
            iterations += 1
            changes = True

    return iterations

def simulated_annealing_reduce(g: BaseGraph[VT,ET], iterations=100, alpha=0.95, cap=-100000, quiet:bool=False, stats=None):
    temperature = iterations
    epsilon = 0.01
    it = 0
    while temperature > epsilon:
        it += 1
        lcomp_matches, pivot_matches = generate_matches(g, boundaries=True, gadgets=True, cap=cap)
        method, match = get_random_match(lcomp_matches, pivot_matches)
        if method == "none":
            temperature = 0
            break
        if match[0] < 0:
            if math.exp(match[0]/temperature) < random.random():
                temperature *=alpha
                continue
        if method == "pivot":
            apply_pivot(g,match)
        else:
            apply_lcomp(g,match)

        temperature *=alpha

    return 0

def apply_lcomp(g: BaseGraph[VT,ET], match):
    v,neighbors = match[1]
    neighbor_copy = neighbors[:]
    identity_insert = True
    while identity_insert:
        identity_insert = False
        for neighbor in g.neighbors(v):
            if g.types()[neighbor] == VertexType.BOUNDARY:
                new_v = insert_identity(g,v,neighbor)
                neighbor_copy = [new_v if i==neighbor else i for i in neighbor_copy]
                identity_insert = True
                break    

    phase_type = get_phase_type(g.phases()[v])
    if phase_type != 1:
        v_mid, gadget_top = insert_phase_gadget(g, v, Fraction(1,2))
        neighbor_copy.append(v_mid)
        apply_rule(g, lcomp, [(v, neighbor_copy)])
    else:
        apply_rule(g, lcomp, [(v, neighbor_copy)])


def apply_pivot(g: BaseGraph[VT,ET], match):
    v1,v2 = match[1]
    for vertex in [v1,v2]:
        identity_insert = True
        while identity_insert:
            identity_insert = False
            for neighbor in g.neighbors(vertex):
                if g.types()[neighbor] == VertexType.BOUNDARY:
                    insert_identity(g,vertex,neighbor)
                    identity_insert = True
                    break
        phase_type = get_phase_type(g.phases()[vertex])
        if phase_type != 2:
            _v_mid, _gadget_top = insert_phase_gadget(g, vertex, Fraction(0,1))
            
    apply_rule(g, pivot, [(v1,v2,[],[])])


def insert_phase_gadget(g,vertex,desired_phase):
    new_phase = split_phases(g.phases()[vertex], desired_phase)
    gadget_top = g.add_vertex(VertexType.Z,-2,g.rows()[vertex],new_phase)
    g.set_phase(vertex, desired_phase)
    g.add_edge((vertex,gadget_top), EdgeType.SIMPLE)
    v_mid = insert_identity(g,vertex,gadget_top)  
    return (v_mid, gadget_top)
