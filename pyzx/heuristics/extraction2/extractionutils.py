from pyzx.graph.base import ET, VT, BaseGraph, VertexType

from typing import Dict, Set, List

def get_frontier_gadgets2(g: BaseGraph,  frontier: Dict[int, VT]):
    """returns all frontier gadgets additionally checking if phases of frontiers are 0"""
    res = set()
    for v in g.vertices():
        if g.type(v) == VertexType.Z and len(g.neighbors(v)) == 1:
            root = list(g.neighbors(v))[0]
            candidate = True
            for n in g.neighbors(root):
                if g.phase(n) != 0 and n != v:
                    candidate = False
                    break
                elif not n in set(frontier.values()).union(set[v]):
                    candidate = False
                    break
            if candidate:
                res.add((root,v))
    return res
            

def get_frontier_gadgets(g: BaseGraph, frontier: Dict[int, VT]):
    """Given a graph and a frontier set, returns all phase gadget neighbors of the frontier as a set of tuples (root,top) 
    where root is the (phaseless) root spider, and top the 1-ary spider with phase connected to root"""
    res = set()
    for v in frontier.values():
        for n in g.neighbors(v):
            top = None
            for potential_top in g.neighbors(n):
                if g.vertex_degree(potential_top) == 1 and potential_top not in g.inputs() and potential_top not in g.outputs():
                    res.add((n,potential_top))

    return res

def get_neighbors_of_frontier(g: BaseGraph[VT, ET], frontier_values: List[VT]) -> Set[VT]:
    """Returns the set of neighbors of the frontier."""
    neighbor_set = set()

    for vertex in frontier_values:
        non_start_neighbors = [neighbor for neighbor in g.neighbors(vertex) if g.type(neighbor) == VertexType.Z]
        neighbor_set.update(non_start_neighbors)
    return neighbor_set