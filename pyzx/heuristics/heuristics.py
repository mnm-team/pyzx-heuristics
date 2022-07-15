from pyzx.utils import VertexType
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
"""
Calculates local complementation heuristic (LCH)

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
vertex (int): spider where local complementation is to be applied
debug (bool): print details of calculation

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying local complementation on the given vertex 
"""
def lcomp_heuristic(g: BaseGraph[VT,ET], vertex, debug=False):
    vn_set = set(g.neighbors(vertex))
    akk = 0 #connected neighbours
    gauss_sum = len(vn_set)*(len(vn_set)-1)/2 #maximal number of connections
    for neighbor in vn_set:
        akk += len(vn_set & g.neighbors(neighbor))
    # akk /= 2 #each edge is counted twice so divide by two
    res = akk - gauss_sum
    if debug:
        print("akk ",akk,"max_sum ",res,"choice ",get_phase_type(g.phases()[vertex]))
    return {1: res + len(vn_set), 2: res, 3: res-1}.get(get_phase_type(g.phases()[vertex]),res-1)

"""
Calculates pivoting heuristic (PH)

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
edge (int): edge where pivoting is to be applied
debug (bool): print details of calculation

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying pivoting on the given edge 
"""
def pivot_heuristic(g: BaseGraph[VT,ET], edge, debug=False):
    v1, v2 = g.edge_st(edge)
    vn_set1 = set(g.neighbors(v1))
    vn_set1.remove(v2)
    vn_set2 = set(g.neighbors(v2))
    vn_set2.remove(v1)
    shared_n = set(vn_set1 & vn_set2)
    vn_set1.difference_update(shared_n)
    vn_set2.difference_update(shared_n)
    akk = 0
    max_sum = len(vn_set1) * len(vn_set2) + len(vn_set1) * len(shared_n) + len(vn_set2) * len(shared_n) #maximal number of connections
    for neighbour in vn_set1:
        for neighbour2 in g.neighbors(neighbour):
            if neighbour2 in vn_set2:
                akk += 1
            elif neighbour2 in shared_n:
                akk += 1
    for neighbour in vn_set2:
        for neighbour2 in g.neighbors(neighbour):
            if neighbour2 in shared_n:
                akk += 1
    
    res = 2*akk - max_sum
    pt1 = get_phase_type(g.phases()[v1]) != 2
    pt2 = get_phase_type(g.phases()[v2]) != 2
    if debug:
        print("akk ",akk,"max_sum ",max_sum,"choice ",pt1+pt2*2)
    return {0: res + len(g.neighbors(v1)) + len(g.neighbors(v2)) - 1,
     1: res + len(g.neighbors(v2)) - 1,
      2: res + len(g.neighbors(v1)) - 1,
       3: res - 2
       }.get(pt1*2+pt2,res - 2)

"""
Helper function for distinguishing phases between True Clifford, Clifford and non-Clifford

Parameters: 
phase (Fraction): phase between 0 and 2π represented as Fraction of π

Returns:
int: 1 for True Clifford, 2 for Clifford and 3 for non-Clifford phase
"""
def get_phase_type(phase):
    if phase == Fraction(1,2) or phase == Fraction(3,2):
        return 1
    elif phase == Fraction(1,1) or phase == 0:
        return 2
    else:
        return 3

"""
Calculates pivoting heuristic (PH) if one or both spiders adjacent to the edge are boundary spiders

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
edge (int): edge where pivoting is to be applied

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying pivoting on the given edge 
"""
def pivot_heuristic_boundary(g: BaseGraph[VT,ET], edge):
    v1, v2 = g.edge_st(edge)
    n_count = 0
    for neighbor in g.neighbors(v1):
        if g.type(neighbor) == VertexType.BOUNDARY:
            n_count += 1
    for neighbor in g.neighbors(v2):
        if g.type(neighbor) == VertexType.BOUNDARY:
            n_count += 1 
    return pivot_heuristic(g,edge)-n_count

"""
Calculates local complementation heuristic (LCH) if spider is a boundary spider

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
vertex (int): spider where local complementation is to be applied

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying local complementation on the given vertex 
"""
def lcomp_heuristic_boundary(g: BaseGraph[VT,ET], vertex):
    n_count = 0
    for neighbor in g.neighbors(vertex):
        if g.type(neighbor) == VertexType.BOUNDARY:
            n_count += 1
    return lcomp_heuristic(g,vertex)-n_count

"""
Calculates heuristic for neighbor unfusion + local complementation

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
vertex (int): spider where local complementation is to be applied
neighbor (int): neighbor of vertex to which the non-Clifford phase is unfused
debug (bool): print details of calculation

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying neighbor unfusion with local complementation on the given vertex 
"""
def lcomp_heuristic_neighbor_unfusion(g: BaseGraph[VT,ET], vertex, neighbor, debug=False):
    vn_set = set(g.neighbors(vertex))
    akk = 0 #connected neighbours
    gauss_sum = len(vn_set)*(len(vn_set)-1)/2 #maximal number of connections
    vn_set.remove(neighbor) #remove neighbor from calculation
    for neighbor in vn_set:
        akk += len(vn_set & g.neighbors(neighbor))
    res = akk - gauss_sum
    if debug:
        print("akk ",akk,"max_sum ",res)
    return res + len(vn_set) - 2

"""
Calculates heuristic for neighbor unfusion + pivoting

Parameters: 
g (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
edge (int): spider where pivoting is to be applied
neighbor_u (int): neighbor of adjacent vertex u to which the non-Clifford phase of u is unfused
neighbor_v (int): neighbor of adjacent vertex v to which the non-Clifford phase of v is unfused
debug (bool): print details of calculation

Returns:
int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying neighbor unfusion with pivoting on the given edge 
"""
def pivot_heuristic_neighbor_unfusion(g: BaseGraph[VT,ET], edge, neighbor_u, neighbor_v, debug=False):
    v1, v2 = g.edge_st(edge)
    vn_set1 = set(g.neighbors(v1))
    vn_set1.remove(v2)
    vn_set2 = set(g.neighbors(v2))
    vn_set2.remove(v1)
    shared_n = set(vn_set1 & vn_set2)
    vn_set1.difference_update(shared_n)
    vn_set2.difference_update(shared_n)
    akk = 0
    if neighbor_u and neighbor_u in shared_n:
        shared_n.remove(neighbor_u)
        vn_set1.add(neighbor_u)
        vn_set2.add(neighbor_u)
    if neighbor_v and neighbor_v in shared_n:
        shared_n.remove(neighbor_v)
        vn_set1.add(neighbor_v)
        vn_set2.add(neighbor_v)

    max_sum = len(vn_set1) * len(vn_set2) + len(vn_set1) * len(shared_n) + len(vn_set2) * len(shared_n) #maximal number of connections
    gain_decrease = 0
    if neighbor_u:
        vn_set1.discard(neighbor_u)
        gain_decrease += 2
    if neighbor_v:
        vn_set2.discard(neighbor_v)
        gain_decrease += 2
    for neighbour in vn_set1:
        for neighbour2 in g.neighbors(neighbour):
            if neighbour2 in vn_set2:
                akk += 1
            elif neighbour2 in shared_n:
                akk += 1
    for neighbour in vn_set2:
        for neighbour2 in g.neighbors(neighbour):
            if neighbour2 in shared_n:
                akk += 1
    
    res = 2*akk - max_sum
    if debug:
        print("akk ",akk,"max_sum ",max_sum, "vn1 ",vn_set1, "vn2 ",vn_set2,"sh ",shared_n)
    return res + len(g.neighbors(v1)) + len(g.neighbors(v2)) - 1 - gain_decrease
