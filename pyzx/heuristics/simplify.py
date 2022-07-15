from fractions import Fraction
from pyzx.rules import apply_rule, lcomp, pivot
from .heuristics import get_phase_type, lcomp_heuristic, pivot_heuristic, lcomp_heuristic_boundary, pivot_heuristic_boundary
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Tuple, List
from pyzx.utils import VertexType, EdgeType
from .tools import split_phases, insert_identity, insert_phase_gadget
import math
import random

MatchLcompHeuristicType = Tuple[float,Tuple[VT,List[VT]],int]

"""
Generates all matches for local complementation in a graph-like ZX-diagram

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
boundaries (bool): whether to include boundary spiders
gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ spiders by the rule application)

Returns:
List[MatchLcompHeuristicType]: A list of match tuples (x,y,z), where x is the LCH, y the tuple needed for rule application and z the amount of saved/added spiders
"""
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

"""
Generates all matches for pivoting in a graph-like ZX-diagram

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
boundaries (bool): whether to include boundary spiders
gadgets (bool): whether to include non-Clifford spiders (which are transformed into YZ spiders by the rule application)

Returns:
List[MatchPivotHeuristicType]: A list of match tuples (x,y,z), where x is the PH, y the tuple needed for rule application and z the amount of saved/added spiders
"""
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

"""
Applies rule with the best heuristic result, i.e. the rule which eliminates the most Hadamard wires

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
lcomp_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

Returns:
bool: True if some rule has been applied, False if both match lists are empty
"""
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
        apply_pivot(g,pivot_matches[0][1])
    else:
        apply_lcomp(g,lcomp_matches[0][1])
    return True

"""
Collects and filters all matches for local complementation and pivoting

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
boundaries (bool): whether to include boundary spiders
gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
max_v (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of cap<0).
cap (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

Returns: 
Tuple (List[MatchLcompHeuristicType], List[MatchPivotHeuristicType]): A tuple with all filtered matches for local complementation and pivoting
"""
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

"""
Randomly selects a rule application out of the given matches

Parameters: 
lcomp_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

Returns:
Tuple (string, MatchLcompHeuristicType | MatchPivotHeuristicType): Tuple of rule name and match
"""
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

"""
Applies random rule application on diagram

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
lcomp_matches (List[MatchLcompHeuristicType]): A list of matches for local complementation
pivot_matches (List[MatchPivotHeuristicType]): A list of matches for pivoting

Returns:
bool: True if some rule has been applied, False if both match lists are empty
"""
def apply_random_match(g, lcomp_matches, pivot_matches):
    method, match = get_random_match(lcomp_matches, pivot_matches)

    if method == "pivot":
        apply_pivot(g,match[1])
    elif method == "lcomp":
        apply_lcomp(g,match[1])
    else:
        return False
    return True

"""
Random Hadamard wire reduction

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
boundaries (bool): whether to include boundary spiders
gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
max_v (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of cap<0).
cap (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

Returns:
int: The number of iterations, i.e. rule applications
"""
def random_wire_reduce(g: BaseGraph[VT,ET], boundaries=False, gadgets=False, max_v=None, cap=1):
    changes = True
    iterations = 0

    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, boundaries=boundaries, gadgets=gadgets, max_v=max_v, cap=cap)
        if apply_random_match(g, lcomp_matches, pivot_matches):
            iterations += 1
            changes = True

    return iterations

"""
Greedy Hadamard wire reduction

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
boundaries (bool): whether to include boundary spiders
gadgets (bool): whether to include non-Clifford spiders (which are transformed into XZ or YZ spiders by the rule application)
max_v (int): The highest index of any vertex present at the beginning of the heuristic simplification routine (needed to prevent non-termination in the case of cap<0).
cap (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

Returns:
int: The number of iterations, i.e. rule applications
"""
def greedy_wire_reduce(g: BaseGraph[VT,ET], boundaries=False, gadgets=False, max_v=None, cap=1):
    changes = True
    iterations = 0

    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, boundaries=boundaries, gadgets=gadgets, max_v=max_v, cap=cap)
        if apply_best_match(g, lcomp_matches, pivot_matches):
            iterations += 1
            changes = True

    return iterations

"""
Hadamard wire reduction with simulated annealing (does not work very well yet)

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
iterations (int): Initial temperature
alpha (float): Cooling factor
cap (int): Lower bound for heuristic result. I.e. -5 means any rule application which adds more than 5 Hadamard wires is filtered out

Returns:
int: 0
"""
def simulated_annealing_reduce(g: BaseGraph[VT,ET], iterations=100, alpha=0.95, cap=-100000):
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
            apply_pivot(g,match[1])
        else:
            apply_lcomp(g,match[1])

        temperature *=alpha

    return 0

"""
Applies local complementation on vertex dependent on phase and whether vertex is boundary or not

- If vertex is boundary an additional spider and (Hadamard) wire is inserted to make the vertex interior
- If vertex v0 has non-Clifford phase p, two additional vertices v1 and v2 are inserted on top:
  v0 has phase π/2, v1 has phase 0 and is connected to v0, v2 has phase p - π/2 and is connected to v1
  Local complementation removes v0, so v1 becomes a XZ spider with v2 corresponding to the measurement effect.

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
match (V,List[V]): Tuple of vertex and its neighbors

Returns: 
Nothing
"""
def apply_lcomp(g: BaseGraph[VT,ET], match):
    v,neighbors = match
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

"""
Applies pivoting on edge dependent on phase of its adjacent vertices and whether they are boundary or not

- For each adjacent vertex v0 which is boundary an additional vertex and (Hadamard) wire is inserted to make v0 interior
- For each adjacent vertex v0 with non-Clifford phase p, two additional vertices v1 and v2 are inserted on top:
  v0 has phase 0, v1 has phase 0 and is connected to v0, v2 has phase p and is connected to v1
  Pivoting removes v0, so v1 becomes a YZ spider with v2 corresponding to the measurement effect.

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
match (V,V): adjacent vertices of edge

Returns: 
Nothing
"""
def apply_pivot(g: BaseGraph[VT,ET], match):
    v1,v2 = match
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

