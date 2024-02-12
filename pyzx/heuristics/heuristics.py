from enum import Enum
from pyzx.utils import VertexType
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction

class PhaseType(Enum):
    TRUE_CLIFFORD = 1
    CLIFFORD = 2
    NON_CLIFFORD = 3

def get_phase_type(phase):
    """
    Helper function for distinguishing phases between True Clifford, Clifford and non-Clifford

    Parameters: 
    phase (Fraction): phase between 0 and 2π represented as Fraction of π

    Returns:
    PhaseType: PhaseType.TRUE_CLIFFORD for True Clifford, PhaseType.CLIFFORD for Clifford and PhaseType.NON_CLIFFORD for non-Clifford phase
    """
    if phase == Fraction(1,2) or phase == Fraction(3,2):
        return PhaseType.TRUE_CLIFFORD
    elif phase == Fraction(1,1) or phase == 0:
        return PhaseType.CLIFFORD
    else:
        return PhaseType.NON_CLIFFORD



def lcomp_heuristic(graph: BaseGraph[VT,ET], target_vertex, debug=False):
    """
    Calculates local complementation heuristic (LCH)

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    target_vertex (int): spider where local complementation is to be applied
    debug (bool): print details of calculation

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying local complementation on the given vertex 
    """
    target_vertex_neighbors = set(graph.neighbors(target_vertex))
    connected_neighbors = 0 #connected neighbours
    max_connections = len(target_vertex_neighbors)*(len(target_vertex_neighbors)-1)/2 #maximal number of connections
    for neighbor in target_vertex_neighbors:
        connected_neighbors += len(target_vertex_neighbors & graph.neighbors(neighbor))
    # connected_neighbors /= 2 #each edge is counted twice so divide by two
    heuristic_result = connected_neighbors - max_connections

    # Get the phase type of the target vertex
    phase_type = get_phase_type(graph.phases()[target_vertex])

    if debug:
        print("connected_neighbors ",connected_neighbors,"max_connections ",heuristic_result,"choice ",phase_type)

    # Check the phase type and return the corresponding calculation
    if phase_type == PhaseType.TRUE_CLIFFORD:
        return heuristic_result + len(target_vertex_neighbors)
    elif phase_type == PhaseType.CLIFFORD:
        return heuristic_result
    elif phase_type == PhaseType.NON_CLIFFORD:
        return heuristic_result - 1
    else:
        return heuristic_result - 1

def lcomp_heuristic_for_boundary(graph: BaseGraph[VT,ET], target_vertex):
    """
    Calculates local complementation heuristic (LCH) if spider is a boundary spider

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    target_vertex (int): spider where local complementation is to be applied

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying local complementation on the given vertex 
    """
    boundary_neighbors_count = 0
    for neighbor in graph.neighbors(target_vertex):
        if graph.type(neighbor) == VertexType.BOUNDARY:
            boundary_neighbors_count += 1
    return lcomp_heuristic(graph, target_vertex) - boundary_neighbors_count

def lcomp_heuristic_neighbor_unfusion(graph: BaseGraph[VT,ET], target_vertex, unfused_neighbor, debug=False):
    """
    Calculates heuristic for neighbor unfusion + local complementation

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    target_vertex (int): spider where local complementation is to be applied
    unfused_neighbor (int): neighbor of target_vertex to which the non-Clifford phase is unfused
    debug (bool): print details of calculation

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying neighbor unfusion with local complementation on the given vertex 
    """
    target_vertex_neighbors = set(graph.neighbors(target_vertex))
    connected_neighbors_count = 0 #connected neighbours
    max_connections = len(target_vertex_neighbors)*(len(target_vertex_neighbors)-1)/2 #maximal number of connections
    target_vertex_neighbors.remove(unfused_neighbor) #remove unfused_neighbor from calculation
    for neighbor in target_vertex_neighbors:
        connected_neighbors_count += len(target_vertex_neighbors & graph.neighbors(neighbor))
    heuristic_result = connected_neighbors_count - max_connections
    if debug:
        print("connected_neighbors ",connected_neighbors_count,"max_connections ",heuristic_result)
    return heuristic_result + len(target_vertex_neighbors) - 2



def pivot_heuristic(graph: BaseGraph[VT,ET], edge, debug=False):
    """
    Calculates pivoting heuristic (PH)

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    edge (int): edge where pivoting is to be applied
    debug (bool): print details of calculation

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying pivoting on the given edge 
    """
    vertex1, vertex2 = graph.edge_st(edge)
    vertex1_neighbors = set(graph.neighbors(vertex1))
    vertex1_neighbors.remove(vertex2)
    vertex2_neighbors = set(graph.neighbors(vertex2))
    vertex2_neighbors.remove(vertex1)
    shared_neighbors = set(vertex1_neighbors & vertex2_neighbors)
    vertex1_neighbors.difference_update(shared_neighbors)
    vertex2_neighbors.difference_update(shared_neighbors)
    connected_neighbors_count = 0
    max_connections = len(vertex1_neighbors) * len(vertex2_neighbors) + len(vertex1_neighbors) * len(shared_neighbors) + len(vertex2_neighbors) * len(shared_neighbors) #maximal number of connections
    for neighbor in vertex1_neighbors:
        for neighbor2 in graph.neighbors(neighbor):
            if neighbor2 in vertex2_neighbors or neighbor2 in shared_neighbors:
                connected_neighbors_count += 1
    for neighbor in vertex2_neighbors:
        for neighbor2 in graph.neighbors(neighbor):
            if neighbor2 in shared_neighbors:
                connected_neighbors_count += 1
    
    heuristic_result = 2*connected_neighbors_count - max_connections
    phase_type1 = get_phase_type(graph.phases()[vertex1])
    phase_type2 = get_phase_type(graph.phases()[vertex2])

    if debug:
        print("connected_neighbors ",connected_neighbors_count,"max_connections ",max_connections,"choice ",phase_type1.value+phase_type2.value*2)

    if phase_type1 == PhaseType.CLIFFORD and phase_type2 == PhaseType.CLIFFORD:
        return heuristic_result + len(graph.neighbors(vertex1)) + len(graph.neighbors(vertex2)) - 1
    elif phase_type1 != PhaseType.CLIFFORD and phase_type2 == PhaseType.CLIFFORD:
        return heuristic_result + len(graph.neighbors(vertex2)) - 1
    elif phase_type1 == PhaseType.CLIFFORD and phase_type2 != PhaseType.CLIFFORD:
        return heuristic_result + len(graph.neighbors(vertex1)) - 1
    elif phase_type1 != PhaseType.CLIFFORD and phase_type2 != PhaseType.CLIFFORD:
        return heuristic_result - 2
    else:
        return heuristic_result - 2

def pivot_heuristic_for_boundary(graph: BaseGraph[VT,ET], edge):
    """
    Calculates pivoting heuristic (PH) if one or both spiders adjacent to the edge are boundary spiders

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    edge (int): edge where pivoting is to be applied

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying pivoting on the given edge 
    """
    vertex1, vertex2 = graph.edge_st(edge)
    boundary_neighbors_count = 0
    for neighbor in graph.neighbors(vertex1):
        if graph.type(neighbor) == VertexType.BOUNDARY:
            boundary_neighbors_count += 1
    for neighbor in graph.neighbors(vertex2):
        if graph.type(neighbor) == VertexType.BOUNDARY:
            boundary_neighbors_count += 1 
    return pivot_heuristic(graph,edge) - boundary_neighbors_count

def pivot_heuristic_neighbor_unfusion(graph: BaseGraph[VT,ET], edge, unfused_neighbor_u, unfused_neighbor_v, debug=False):
    """
    Calculates heuristic for neighbor unfusion + pivoting

    Parameters: 
    graph (BaseGraph[VT,ET]): An instance of a Graph, i.e. ZX-diagram
    edge (int): spider where pivoting is to be applied
    unfused_neighbor_u (int): neighbor of adjacent vertex u to which the non-Clifford phase of u is unfused
    unfused_neighbor_v (int): neighbor of adjacent vertex v to which the non-Clifford phase of v is unfused
    debug (bool): print details of calculation

    Returns:
    int: Amount of saved (positive number) or added (negative number) Hadamard wires when applying neighbor unfusion with pivoting on the given edge 
    """
    vertex_u, vertex_v = graph.edge_st(edge)
    vertex_u_neighbors = set(graph.neighbors(vertex_u))
    vertex_u_neighbors.remove(vertex_v)
    vertex_v_neighbors = set(graph.neighbors(vertex_v))
    vertex_v_neighbors.remove(vertex_u)
    shared_neighbors = set(vertex_u_neighbors & vertex_v_neighbors)
    vertex_u_neighbors.difference_update(shared_neighbors)
    vertex_v_neighbors.difference_update(shared_neighbors)
    connected_neighbors_count = 0
    if unfused_neighbor_u and unfused_neighbor_u in shared_neighbors:
        shared_neighbors.remove(unfused_neighbor_u)
        vertex_u_neighbors.add(unfused_neighbor_u)
        vertex_v_neighbors.add(unfused_neighbor_u)
    if unfused_neighbor_v and unfused_neighbor_v in shared_neighbors:
        shared_neighbors.remove(unfused_neighbor_v)
        vertex_u_neighbors.add(unfused_neighbor_v)
        vertex_v_neighbors.add(unfused_neighbor_v)

    max_connections = len(vertex_u_neighbors) * len(vertex_v_neighbors) + len(vertex_u_neighbors) * len(shared_neighbors) + len(vertex_v_neighbors) * len(shared_neighbors) #maximal number of connections
    gain_decrease = 0
    if unfused_neighbor_u:
        vertex_u_neighbors.discard(unfused_neighbor_u)
        gain_decrease += 2
    if unfused_neighbor_v:
        vertex_v_neighbors.discard(unfused_neighbor_v)
        gain_decrease += 2
    for neighbor in vertex_u_neighbors:
        for neighbor2 in graph.neighbors(neighbor):
            if neighbor2 in vertex_v_neighbors or neighbor2 in shared_neighbors:
                connected_neighbors_count += 1
    for neighbor in vertex_v_neighbors:
        for neighbor2 in graph.neighbors(neighbor):
            if neighbor2 in shared_neighbors:
                connected_neighbors_count += 1
    
    heuristic_result = 2*connected_neighbors_count - max_connections
    if debug:
        print("connected_neighbors ",connected_neighbors_count,"max_connections ",max_connections, "vertex_u_neighbors ",vertex_u_neighbors, "vertex_v_neighbors ",vertex_v_neighbors,"shared_neighbors ",shared_neighbors)
    return heuristic_result + len(graph.neighbors(vertex_u)) + len(graph.neighbors(vertex_v)) - 1 - gain_decrease