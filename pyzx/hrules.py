# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Tuple, Callable, Optional, Set, FrozenSet
from .utils import EdgeType, VertexType, toggle_edge, toggle_vertex, FractionLike, FloatInt, vertex_is_zx
from .simplify import *
from .graph.base import BaseGraph, ET, VT
from . import rules


def match_hadamards(g: BaseGraph[VT,ET],
        vertexf: Optional[Callable[[VT],bool]] = None
        ) -> List[VT]:
    """Matches all the H-boxes with arity 2 and phase 1, i.e. all the Hadamard gates."""
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    m : Set[VT] = set()
    ty = g.types()
    taken: Set[VT] = set()
    for v in candidates:
        if ty[v] == VertexType.H_BOX and g.vertex_degree(v) == 2 and g.phase(v) == 1:
            n1,n2 = g.neighbors(v)
            if n1 in taken or n2 in taken: continue
            if n1 not in m and n2 not in m: 
                m.add(v)
                taken.add(n1)
                taken.add(n2)

    return list(m)

def hadamard_to_h_edge(g: BaseGraph[VT,ET], matches: List[VT]) -> rules.RewriteOutputType[ET,VT]:
    """Converts a matching of H-boxes with arity 2 and phase 1, i.e. Hadamard gates, to Hadamard edges."""
    rem_verts = []
    etab = {}
    for v in matches:
        rem_verts.append(v)
        w1,w2 = list(g.neighbors(v))
        et1 = g.edge_type(g.edge(w1,v))
        et2 = g.edge_type(g.edge(w2,v))
        if et1 == et2:
            etab[g.edge(w1,w2)] = [0,1]
        else:
            etab[g.edge(w1,w2)] = [1,0]
    g.scalar.add_power(len(matches)) # Correct for the sqrt(2) difference in H-boxes and H-edges
    return (etab, rem_verts, [], True)

def match_connected_hboxes(g: BaseGraph[VT,ET],
        edgef: Optional[Callable[[ET],bool]] = None
        ) -> List[ET]:
    """Matches Hadamard-edges that are connected to H-boxes, as these can be fused,
    see the rule (HS1) of https://arxiv.org/pdf/1805.02175.pdf."""
    if edgef is not None: candidates = set([e for e in g.edges() if edgef(e)])
    else: candidates = g.edge_set()
    m : Set[ET] = set()
    ty = g.types()
    while candidates:
        e = candidates.pop()
        if g.edge_type(e) != EdgeType.HADAMARD: continue
        v1,v2 = g.edge_st(e)
        if ty[v1] != VertexType.H_BOX or ty[v2] != VertexType.H_BOX: continue
        if g.phase(v1) != 1 and g.phase(v2) != 1: continue
        m.add(e)
        candidates.difference_update(g.incident_edges(v1))
        candidates.difference_update(g.incident_edges(v2))
    return list(m)

def fuse_hboxes(g: BaseGraph[VT,ET], matches: List[ET]) -> rules.RewriteOutputType[ET,VT]:
    """Fuses two neighboring H-boxes together. 
    See rule (HS1) of https://arxiv.org/pdf/1805.02175.pdf."""
    rem_verts = []
    etab = {}
    for e in matches:
        v1, v2 = g.edge_st(e)
        if g.phase(v2) != 1: # at most one of v1 and v2 has a phase different from 1
            v1, v2 = v2, v1
        rem_verts.append(v2)
        g.scalar.add_power(1)
        for n in g.neighbors(v2):
            if n == v1: continue
            e2 = g.edge(v2,n)
            if g.edge_type(e2) == EdgeType.SIMPLE:
                etab[g.edge(v1,n)] = [1,0]
            else:
                etab[g.edge(v1,n)] = [0,1]
    
    return (etab, rem_verts, [], True)



MatchCopyType = Tuple[VT,VT,EdgeType.Type,FractionLike,FractionLike,List[VT]]

def match_copy(
        g: BaseGraph[VT,ET], 
        vertexf:Optional[Callable[[VT],bool]]=None
        ) -> List[MatchCopyType[VT]]:
    """Finds arity-1 spiders (with a 0 or pi phase) that can be copied through their neighbor.""" 
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    phases = g.phases()
    types = g.types()
    m = []
    taken: Set[VT] = set()

    while len(candidates) > 0:
        v = candidates.pop()
        if phases[v] not in (0,1) or not vertex_is_zx(types[v]) or g.vertex_degree(v) != 1:
                    continue
        w = list(g.neighbors(v))[0]
        if w in taken: continue
        tv = types[v]
        tw = types[w]
        if tw == VertexType.BOUNDARY: continue
        e = g.edge(v,w)
        et = g.edge_type(e)
        if vertex_is_zx(types[w]) and ((tw != tv and et==EdgeType.HADAMARD) or
                                       (tw == tv and et==EdgeType.SIMPLE)):
            continue
        if tw == VertexType.H_BOX and ((et==EdgeType.SIMPLE and tv != VertexType.X) or
                                       (et==EdgeType.HADAMARD and tv != VertexType.Z)):
            continue
        neigh = [n for n in g.neighbors(w) if n != v]
        m.append((v,w,et,phases[v],phases[w],neigh))
        candidates.discard(w)
        candidates.difference_update(neigh)
        taken.add(w)
        taken.update(neigh)

    return m

def apply_copy(
        g: BaseGraph[VT,ET], 
        matches: List[MatchCopyType[VT]]
        ) -> rules.RewriteOutputType[ET,VT]:
    """Copy arity-1 spider through their neighbor."""
    rem = []
    types = g.types()
    for v,w,t,a,alpha,neigh in matches:
        rem.append(v)
        if (types[w] == VertexType.H_BOX and a == 1): 
            continue # Don't have to do anything more for this case
        rem.append(w)
        vt: VertexType.Type = VertexType.Z
        if vertex_is_zx(types[w]):
            vt = types[v] if t == EdgeType.SIMPLE else toggle_vertex(types[v])
            if a: g.scalar.add_phase(alpha)
            g.scalar.add_power(-(len(neigh)-1))
        else: #types[w] == H_BOX
            g.scalar.add_power(1)
        for n in neigh: 
            r = 0.7*g.row(w) + 0.3*g.row(n)
            q = 0.7*g.qubit(w) + 0.3*g.qubit(n)
            
            u = g.add_vertex(vt, q, r, a)
            e = g.edge(n,w)
            et = g.edge_type(e)
            g.add_edge(g.edge(n,u), et)

    return ({}, rem, [], True)


def match_hbox_parallel_not(
        g: BaseGraph[VT,ET], 
        vertexf:Optional[Callable[[VT],bool]]=None
        ) -> List[Tuple[VT,VT,VT]]:
    """Finds H-boxes that are connected to a Z-spider both directly and via a NOT.""" 
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    phases = g.phases()
    types = g.types()
    m = []

    while len(candidates) > 0:
        h = candidates.pop()
        if types[h] != VertexType.H_BOX or phases[h] != 1: continue

        for n in g.neighbors(h):
            if g.vertex_degree(n) != 2 or phases[n] != 1: continue # If it turns out to be useful, this rule can be generalised to allow spiders of arbitrary phase here
            v = [v for v in g.neighbors(n) if v != h][0] # The other neighbor of n
            if not g.connected(v,h): continue
            if types[v] != VertexType.Z or g.edge_type(g.edge(h,v)) != EdgeType.SIMPLE: continue
            if g.edge_type(g.edge(h,n)) == EdgeType.SIMPLE and types[n] == VertexType.X: 
                if g.edge_type(g.edge(v,n)) != EdgeType.SIMPLE:
                    continue
            if g.edge_type(g.edge(h,n)) == EdgeType.HADAMARD and types[n] == VertexType.Z: 
                if g.edge_type(g.edge(v,n)) != EdgeType.HADAMARD:
                    continue
            break
        else:
            continue
        # h is connected to both v and n in the appropriate way, and n is a NOT that is connected to v as well
        m.append((h,v,n))
        candidates.difference_update(g.neighbors(h))
    return m

def hbox_parallel_not_remove(g: BaseGraph[VT,ET], 
        matches: List[Tuple[VT,VT,VT]]
        ) -> rules.RewriteOutputType[ET,VT]:
    rem = []
    etab = {}
    types = g.types()
    for h, v, n in matches:
        rem.append(h)
        rem.append(n)
        for w in g.neighbors(h):
            if w == v or w == n: continue
            et = g.edge_type(g.edge(w,h))
            if types[w] == VertexType.Z and et == EdgeType.SIMPLE: continue
            if types[w] == VertexType.X and et == EdgeType.HADAMARD: continue
            q = 0.6*g.qubit(h) + 0.4*g.qubit(w)
            r = 0.6*g.row(h) + 0.4*g.row(w)
            z = g.add_vertex(VertexType.Z,q,r)
            if et == EdgeType.SIMPLE:
                etab[g.edge(z,w)] = [1,0]
            else: etab[g.edge(z,w)] = [0,1]
    return (etab, rem, [], True)


TYPE_MATCH_PAR_HBOX = Tuple[List[VT],List[VT],List[VT]]
def match_par_hbox(
    g: BaseGraph[VT,ET],
    vertexf: Optional[Callable[[VT],bool]] = None
    ) -> List[TYPE_MATCH_PAR_HBOX]:
    """Matches sets of H-boxes that are connected in parallel (via optional NOT gates)
    to the same white spiders."""
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    
    groupings: Dict[Tuple[FrozenSet[VT],FrozenSet[VT]], Tuple[List[VT],List[VT],List[VT]]] = dict()
    ty = g.types()
    for h in candidates:
        if ty[h] != VertexType.H_BOX: continue
        suitable = True
        neighbors_regular = set()
        neighbors_NOT = set()
        NOTs = []
        for v in g.neighbors(h):
            e = g.edge(v,h)
            if g.edge_type(e) == EdgeType.HADAMARD:
                if ty[v] == VertexType.X:
                    neighbors_regular.add(v)
                elif ty[v] == VertexType.Z and g.vertex_degree(v) == 2 and g.phase(v) == 1:
                    w = [w for w in g.neighbors(v) if w!=h][0] # unique other neighbor
                    if ty[w] != VertexType.Z or g.edge_type(g.edge(v,w)) != EdgeType.HADAMARD:
                        suitable = False
                        break
                    neighbors_NOT.add(w)
                    NOTs.append(v)
                else:    
                    suitable = False
                    break
            else: # e == EdgeType.SIMPLE
                if ty[v] == VertexType.Z:
                    neighbors_regular.add(v)
                elif ty[v] == VertexType.X and g.vertex_degree(v) == 2 and g.phase(v) == 1:
                    w = [w for w in g.neighbors(v) if w!=h][0] # unique other neighbor
                    if ty[w] != VertexType.Z or g.edge_type(g.edge(v,w)) != EdgeType.SIMPLE:
                        suitable = False
                        break
                    neighbors_NOT.add(w)
                    NOTs.append(v)
                else:
                    suitable = False
                    break
        if not suitable: continue
        group = (frozenset(neighbors_regular), frozenset(neighbors_NOT))
        if group in groupings: 
            groupings[group][0].append(h)
            groupings[group][2].extend(NOTs)
        else: groupings[group] = ([h],NOTs, [])

    m = []
    for (n_r, n_N), (hs,firstNOTs, NOTs) in groupings.items():
        if len(hs) < 2: continue
        m.append((hs, firstNOTs, NOTs))
    return m

def par_hbox(g: BaseGraph[VT,ET], matches: List[TYPE_MATCH_PAR_HBOX]) -> rules.RewriteOutputType[ET,VT]:
    """Implements the `multiply rule' (M) from https://arxiv.org/abs/1805.02175"""
    rem_verts = []
    for hs, firstNOTs, NOTs in matches:
        p = sum(g.phase(h) for h in hs) % 2
        rem_verts.extend(hs[1:])
        rem_verts.extend(NOTs)
        if p == 0: 
            rem_verts.append(hs[0])
            rem_verts.extend(firstNOTs)
        else: g.set_phase(hs[0], p)
    
    return ({}, rem_verts, [], False)

def match_zero_hbox(g: BaseGraph[VT,ET]) -> List[VT]:
    """Matches H-boxes that have a phase of 2pi==0."""
    types = g.types()
    phases = g.phases()
    return [v for v in g.vertices() if types[v] == VertexType.H_BOX and phases[v] == 0]

def zero_hbox(g: BaseGraph[VT,ET], m: List[VT]) -> None:
    """Removes H-boxes with a phase of 2pi=0.
    Note that this rule is only semantically correct when all its neighbors are white spiders."""
    g.remove_vertices(m)


hpivot_match_output = List[Tuple[
            VT,
            VT,
            VT,
            List[VT],
            List[VT],
            List[List[VT]],
            List[Tuple[FractionLike,List[VT]]]
            ]]

def match_hpivot(
    g: BaseGraph[VT,ET], matchf=None
    ) -> hpivot_match_output:
    """Finds a matching of the hyper-pivot rule. Note this currently assumes
    hboxes don't have phases.

    :param g: An instance of a ZH-graph.
    :param matchf: An optional filtering function for candidate arity-2 hbox, should
       return True if an hbox should considered as a match. Passing None will
       consider all arity-2 hboxes.
    :rtype: List containing 0 or 1 matches.
    """

    types = g.types()
    phases = g.phases()
    m = []

    min_degree = -1

    for h in g.vertices():
        if not (
            (matchf is None or matchf(h)) and
            g.vertex_degree(h) == 2 and
            types[h] == VertexType.H_BOX and
            phases[h] == 1
        ): continue

        v0, v1 = g.neighbors(h)

        v0n = set(g.neighbors(v0))
        v1n = set(g.neighbors(v1))

        if (len(v0n.intersection(v1n)) > 1): continue

        v0b = [v for v in v0n if types[v] == VertexType.BOUNDARY]
        v0h = [v for v in v0n if types[v] == VertexType.H_BOX and v != h]
        v1b = [v for v in v1n if types[v] == VertexType.BOUNDARY]
        v1h = [v for v in v1n if types[v] == VertexType.H_BOX and v != h]

        # check that at least one of v0 or v1 has all pi phases on adjacent
        # hboxes.
        if not (all(phases[v] == 1 for v in v0h)):
            if not (all(phases[v] == 1 for v in v1h)):
                continue
            else:
                # interchange the roles of v0 <-> v1
                v0,v1 = v1,v0
                v0n,v1n = v1n,v0n
                v0b,v1b = v1b,v0b
                v0h,v1h = v1h,v0h

        v0nn = [list(filter(lambda w : w != v0, g.neighbors(v))) for v in v0h]
        v1nn = [
          (phases[v],
           list(filter(lambda w : w != v1, g.neighbors(v))))
          for v in v1h]


        if not (
            all(all(types[v] == VertexType.Z for v in vs) for vs in v0nn) and
            all(all(types[v] == VertexType.Z for v in vs[1]) for vs in v1nn) and
            len(v0b) + len(v1b) <= 1 and
            len(v0b) + len(v0h) + 1 == len(v0n) and
            len(v1b) + len(v1h) + 1 == len(v1n)
        ): continue

        degree = g.vertex_degree(v0) * g.vertex_degree(v1)

        if min_degree == -1 or degree < min_degree:
            m = [(h, v0, v1, v0b, v1b, v0nn, v1nn)]
            min_degree = degree
    return m


def hpivot(g: BaseGraph[VT,ET], m: hpivot_match_output) -> None:
    if len(m) == 0: return None

    types = g.types()

    # # cache hboxes
    # hboxes = dict()
    # for h in g.vertices():
    #     if types[h] != VertexType.H_BOX: continue
    #     nhd = tuple(sorted(g.neighbors(h)))
    #     hboxes[nhd] = h


    h, v0, v1, v0b, v1b, v0nn, v1nn = m[0]
    g.remove_vertices([v for v in g.neighbors(v0) if types[v] == VertexType.H_BOX])
    g.remove_vertices([v for v in g.neighbors(v1) if types[v] == VertexType.H_BOX])
    g.scalar.add_power(2) # Applying a Fourier Hyperpivot adds a scalar of 2
    
    if len(v0b) == 0:
        g.remove_vertex(v0)
    else:
        e = g.edge(v0, v0b[0])
        g.set_edge_type(e, toggle_edge(g.edge_type(e)))
        v0nn.append([v0])
    
    if len(v1b) == 0:
        g.remove_vertex(v1)
    else:
        e = g.edge(v1, v1b[0])
        g.set_edge_type(e, toggle_edge(g.edge_type(e)))
        v1nn.append((Fraction(1,1), [v1]))

    for phase,ws in v1nn:
        for weight in range(1,len(v0nn)+1):
            phase_mult = int((-2)**(weight-1))
            f_phase = (phase * phase_mult) % 2
            if f_phase == 0: continue
            for vvs in combinations(v0nn, weight):
                us = tuple(sorted(sum(vvs, ws)))

                # TODO: check if this is the right thing to do (and update scalar)
                if len(us) == 0: continue

                # if us in hboxes:
                #     h0 = hboxes[us]
                #     print("adding %s to %s" % (f_phase, g.phase(h0)))
                #     g.add_to_phase(h0, f_phase)
                # else:
                h0 = g.add_vertex(VertexType.H_BOX)
                g.set_phase(h0, f_phase)
                q: FloatInt = 0
                r: FloatInt = 0
                for u in us:
                    q += g.qubit(u)
                    r += g.row(u)
                    g.add_edge(g.edge(h0,u))
                g.set_qubit(h0, q / len(us) - 0.4)
                g.set_row(h0, r / len(us) + 0.4)