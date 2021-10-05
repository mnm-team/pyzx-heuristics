from fractions import Fraction
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


'''
inserts hadamard wire + empty Z + hadamard wire between two vertices.
This does not change the standard interpretation, as two hadamards are equal to the identity
and the empty z spider as well
CAUTION: may break gflow property of graph if applied to the wrong vertices (see heuristics/get_possible_unfusion_neighbours)
'''
    
def insert_identity(g, v1, v2) -> int:
    orig_type = g.edge_type(g.edge(v1, v2))
    if g.connected(v1, v2):
        g.remove_edge(g.edge(v1, v2))
    vmid = g.add_vertex(VertexType.Z,-1,g.rows()[v1])
    g.add_edge((v1,vmid), EdgeType.HADAMARD)
    if orig_type == EdgeType.HADAMARD:
        g.add_edge((vmid,v2), EdgeType.SIMPLE)
    else:
        g.add_edge((vmid,v2), EdgeType.HADAMARD)
    return vmid