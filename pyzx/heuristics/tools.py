from fractions import Fraction
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType, EdgeType


'''
When unfusing a spider into two spiders this returns phase for second spider if the first spider has to get a certain desired phase like pi/2
E.g. split_phases(7pi/4,pi/2) returns 5pi/4
'''

def split_phases(orig_phase: Fraction, desired_phase: Fraction):
    extend_denom = max(orig_phase.denominator,desired_phase.denominator)
    orig_phase_n = int(orig_phase.numerator*(extend_denom/orig_phase.denominator))
    desired_phase_n = int(desired_phase.numerator*(extend_denom/desired_phase.denominator))
    return Fraction( int((orig_phase_n- desired_phase_n) % (extend_denom*2)), extend_denom)
    
def insert_identity(g, v1, v2) -> int:
    '''
    inserts hadamard wire + empty Z + hadamard wire between two vertices.
    This does not change the standard interpretation, as two hadamards are equal to the identity
    and the empty z spider as well
    CAUTION: may break gflow property of graph if applied to the wrong vertices (see heuristics/get_possible_unfusion_neighbours)
    '''
    orig_type = g.edge_type(g.edge(v1, v2))
    if g.connected(v1, v2):
        g.remove_edge(g.edge(v1, v2))
    vmid = g.add_vertex(VertexType.Z, g.qubits()[v1], g.rows()[v1] -1)
    g.add_edge((v1,vmid), EdgeType.HADAMARD)
    if orig_type == EdgeType.HADAMARD:
        g.add_edge((vmid,v2), EdgeType.SIMPLE)
    else:
        g.add_edge((vmid,v2), EdgeType.HADAMARD)
    return vmid

def disentangle_outputs(g: BaseGraph):
    """helper function to put outputs of graph-like diagram in a form where they have an empty phase and are not interconnected
    This may be needed to have flow"""
    output_neighbors = dict()
    for o in g.outputs():
        n = list(g.neighbors(o))[0]
        output_neighbors[o] = n
    for o, n in output_neighbors.items():
        if g.phase(n) != 0 or set(g.neighbors(n)).intersection(set(output_neighbors.values())):
            insert_identity(g, n, o) 


def insert_phase_gadget(g,vertex,desired_phase):
    new_phase = split_phases(g.phases()[vertex], desired_phase)
    gadget_top = g.add_vertex(VertexType.Z,-2,g.rows()[vertex],new_phase)
    g.set_phase(vertex, desired_phase)
    g.add_edge((vertex,gadget_top), EdgeType.SIMPLE)
    v_mid = insert_identity(g,vertex,gadget_top)  
    return (v_mid, gadget_top)