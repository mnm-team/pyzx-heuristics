from fractions import Fraction
from unittest import skip
from .heuristics.tools import insert_phase_gadget
from .simplify import simp, pivot_boundary_simp, gadget_simp, id_simp, to_gh, spider_simp
from .rules import lcomp, pivot
from pyzx.graph.base import BaseGraph, VT, ET
from typing import List, Callable, Optional, Tuple
from pyzx.utils import VertexType, EdgeType

MatchLcompType = Tuple[VT,List[VT]]
MatchPivotType = Tuple[VT,VT,List[VT],List[VT]]

def to_phase_gadget_form(g):
    for v in g.vertex_set():
        if g.phases()[v].denominator >= 2:
            insert_phase_gadget(g,v,Fraction(0,1))

# def skip_gadget_roots_lcomp(g,v):
#     return len(g.neighbors(v)) > 2

# def skip_gadget_roots_pivot(g,e):
#     v0, v1 = g.edge_st(e)
#     return len(g.neighbors(v0)) > 2 and len(g.neighbors(v1)) > 2

def get_gadget_roots(g):
    roots = []
    for v in g.vertex_set():
        if len(g.neighbors(v)) == 2 and g.phases()[v].numerator == 0:
            roots.append(v)
    return roots

def eliminate_cliffords(g):
    
    roots = get_gadget_roots(g)
    # print("roots are ", roots)

    # skip_gadget_roots_lcomp = lambda g,v: not v in roots
    skip_gadget_roots_pivot = lambda g,e: not g.edge_st(e)[0] in roots and not g.edge_st(e)[1] in roots
    iterations = 0
    while True:
        # print("lcomp matches ",_match_lcomp_parallel(g, vertexf=skip_gadget_roots_lcomp))
        # i1 = simp(g, 'lcomp_simp', _match_lcomp_parallel, lcomp, matchf=skip_gadget_roots_lcomp)
        i1 = simp(g, 'pivot_simp', _match_pivot_parallel, pivot, matchf=skip_gadget_roots_pivot)
        if i1 == 0:
            break
        iterations += 1
        print("iteration: ",iterations)

    # lcomp_matches = _match_lcomp_parallel(g, skip_gadget_roots_lcomp)
    # pivot_matches = _match_pivot_parallel(g, skip_gadget_roots_pivot)

    # etab, rem_verts, rem_edges, _check_isolated_vertices = lcomp(g, lcomp_matches)
    # g.add_edge_table(etab)
    # g.remove_edges(rem_edges)
    # g.remove_vertices(rem_verts)
    # g.remove_isolated_vertices()
    
    # etab, rem_verts, rem_edges, _check_isolated_vertices = pivot(g, pivot_matches)
    # g.add_edge_table(etab)
    # g.remove_edges(rem_edges)
    # g.remove_vertices(rem_verts)
    # g.remove_isolated_vertices()

def prepare_gadget_form(g, quiet=True, stats=None):
    spider_simp(g, quiet=quiet, stats=stats)
    to_gh(g)
    id_simp(g, quiet=quiet, stats=stats)
    spider_simp(g, quiet=quiet, stats=stats)
    to_phase_gadget_form(g)

# def topt(g, quiet=True, stats=None):
    
#     eliminate_cliffords(g)
#     gadget_simp(g)

def _match_lcomp_parallel(
        g: BaseGraph[VT,ET], 
        vertexf:Optional[Callable[[VT],bool]]=None, 
        num:int=-1, 
        check_edge_types:bool=True
        ) -> List[MatchLcompType[VT]]:
    """Finds noninteracting matchings of the local complementation rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param vertexf: An optional filtering function for candidate vertices, should
       return True if a vertex should be considered as a match. Passing None will
       consider all vertices.
    :rtype: List of 2-tuples ``(vertex, neighbors)``.
    """
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(g,v)])
    else: candidates = g.vertex_set()
    types = g.types()
    phases = g.phases()
    
    i = 0
    m = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        vt = types[v]
        va = g.phase(v)
        
        if vt != VertexType.Z: continue
        if not (va == Fraction(1,2) or va == Fraction(3,2)): continue

        if check_edge_types and not (
            all(g.edge_type(e) == EdgeType.HADAMARD for e in g.incident_edges(v))
            ): continue
                
        vn = list(g.neighbors(v))

        if not all(types[n] == VertexType.Z for n in vn): continue

        for n in vn: candidates.discard(n)
        m.append((v,vn))
    return m

def _match_pivot_parallel(
        g: BaseGraph[VT,ET], 
        matchf:Optional[Callable[[ET],bool]]=None, 
        num:int=-1, 
        check_edge_types:bool=True
        ) -> List[MatchPivotType[VT]]:
    """Finds non-interacting matchings of the pivot rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
       are of the correct type (Hadamard edges).
    :param matchf: An optional filtering function for candidate edge, should
       return True if a edge should considered as a match. Passing None will
       consider all edges.
    :rtype: List of 4-tuples. See :func:`pivot` for the details.
    """
    if matchf is not None: candidates = set([e for e in g.edges() if matchf(g,e)])
    else: candidates = g.edge_set()
    types = g.types()
    phases = g.phases()
    
    i = 0
    m = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if check_edge_types and g.edge_type(e) != EdgeType.HADAMARD: continue
        v0, v1 = g.edge_st(e)

        if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z): continue

        v0a = phases[v0]
        v1a = phases[v1]
        if not ((v0a in (0,1)) and (v1a in (0,1))): continue

        invalid_edge = False

        v0n = list(g.neighbors(v0))
        v0b = []
        for n in v0n:
            et = g.edge_type(g.edge(v0,n))
            if types[n] == VertexType.Z and et == EdgeType.HADAMARD: pass
            elif types[n] == VertexType.BOUNDARY: v0b.append(n)
            else:
                invalid_edge = True
                break

        if invalid_edge: continue

        v1n = list(g.neighbors(v1))
        v1b = []
        for n in v1n:
            et = g.edge_type(g.edge(v1,n))
            if types[n] == VertexType.Z and et == EdgeType.HADAMARD: pass
            elif types[n] == VertexType.BOUNDARY: v1b.append(n)
            else:
                invalid_edge = True
                break

        if invalid_edge: continue
        if len(v0b) + len(v1b) > 1: continue

        i += 1
        for v in v0n:
            for c in g.incident_edges(v): candidates.discard(c)
        for v in v1n:
            for c in g.incident_edges(v): candidates.discard(c)
        b0 = list(v0b)
        b1 = list(v1b)
        m.append((v0,v1,b0,b1))
    return m

