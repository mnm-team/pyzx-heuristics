
from fractions import Fraction
from pyzx.rules import apply_rule, lcomp, pivot
from .heuristics import get_phase_type, lcomp_heuristic, lcomp_heuristic_neighbor_unfusion, pivot_heuristic, pivot_heuristic_neighbor_unfusion
from pyzx.graph.base import BaseGraph, VT, ET
from typing import Tuple, List
from pyzx.utils import VertexType, EdgeType
from .tools import split_phases, insert_identity
from pyzx.gflow import gflow
from .gflow_calculation import build_focused_gflow_graph, check_gflow, focus_gflow, update_gflow_from_double_insertion, update_gflow_from_lcomp, update_gflow_from_pivot
from .simplify import get_random_match
from pyzx.rules import match_ids_parallel, remove_ids, match_spider_parallel, spider
import math
import random
import copy #TODO: remove when not testing


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
            for neighbor in get_unfusion_neighbors_fggraph(g, v, None, gfl):
                m.append((lcomp_heuristic_neighbor_unfusion(g,v,neighbor),(v,vn),neighbor))
    return m

def lcomp_matcher_all(g: BaseGraph[VT,ET], gfl=None) -> List[MatchLcompHeuristicNeighbourType]:
    candidates = g.vertex_set()
    types = g.types()

    m = []
    while len(candidates) > 0:
        v = candidates.pop()
        vt = types[v]

        if vt != VertexType.Z: continue
                
        vn = list(g.neighbors(v))

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
                for neighbor in get_unfusion_neighbors_fggraph(g, v1, v0, gfl):
                    m.append((pivot_heuristic_neighbor_unfusion(g,e,None,neighbor),(v0,v1),None,neighbor))
        else:
            if get_phase_type(g.phase(v1)) == 2:
                for neighbor in get_unfusion_neighbors_fggraph(g, v0, v1, gfl):
                    m.append((pivot_heuristic_neighbor_unfusion(g,e,neighbor,None),(v0,v1),neighbor,None))
            else:
                for neighbor_v0 in get_unfusion_neighbors_fggraph(g, v0, v1, gfl):
                    for neighbor_v1 in get_unfusion_neighbors_fggraph(g, v1, v0, gfl):
                        m.append((pivot_heuristic_neighbor_unfusion(g,e,neighbor_v0,neighbor_v1),(v0,v1),neighbor_v0,neighbor_v1))

    return m

def get_possible_unfusion_neighbours(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, gfl=None):
    res = []
    for n in g.neighbors(vertex):
        if g.types()[n] != VertexType.BOUNDARY:
            res.append(n)
    return res

def get_unfusion_neighbors_fggraph(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, fggraph=None):
    possible_unfusion_neighbors = []
    # c_set_neighbor = set(g.neighbors(vertex)).intersection(gfl[1][vertex]).pop()
    # if len(set(g.neighbors(vertex)).intersection(gfl[1][vertex])) > 1 :
    #     print("multiple neighboring vertices in cset ",len(set(g.neighbors(vertex)).intersection(gfl[1][vertex])))
    # if not vertex in gfl[1]:
    #     print("fatal")
    #     breakpoint()
    if len(fggraph.neighbors(vertex)) > 2:
        return possible_unfusion_neighbors
    for neighbor in fggraph.neighbors(vertex):
        if not neighbor == exclude_vertex and len(set(g.neighbors(vertex)).intersection(g.neighbors(neighbor))) < 2:
            possible_unfusion_neighbors.append(neighbor)

    return possible_unfusion_neighbors    

def get_connected_neighbors(g, v1, v2):
    v1n = set(g.neighbors(v1))
    v1n.discard(v2)
    v2n = set(g.neighbors(v2))
    v2n.discard(v1)
    res = []
    for n in v1n:
        for n2 in g.neighbors(n):
            if n2 in v2n:
                res.append((n,n2))
    return res

def traverse_fggraph(fggraph, start, goal):
    neighbors = list(fggraph.neighbors(start))
    if len(neighbors) != 2:
        return False
    
    # current = neighbors[0] if direction == 'l' else neighbors[1]
    for current in neighbors:
        previous = start
        while True:
            if current == goal:
                return True
            neighbors = fggraph.neighbors(current)
            # print("get neighbors ",neighbors," of current ",current, "previous is ",previous)
            if len(neighbors) != 2:
                break
            for neighbor in neighbors:
                if neighbor != previous:
                    previous = current
                    current = neighbor
                    # print("set current ",current)
                    break
    return False
        

def get_possible_unfusion_neighbours_orig(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, gfl=None):
    possible_unfusion_neighbours = []
    if len(gfl[vertex]) == 1:
        possible_unfusion_neighbours.append(next(iter(gfl[vertex]))) #get first element of set
    for neighbor in g.neighbors(vertex):
        if vertex in gfl[neighbor] and len(gfl[neighbor]) == 1:
            possible_unfusion_neighbours.append(neighbor)

    if exclude_vertex and exclude_vertex in possible_unfusion_neighbours:
        # print("removed an exclude vertex ",exclude_vertex)
        possible_unfusion_neighbours.remove(exclude_vertex)
    # if len(possible_unfusion_neighbours) > 0:
        # print("found some neighbors for vertex ",vertex, possible_unfusion_neighbours)
    return possible_unfusion_neighbours

def get_possible_unfusion_neighbours_new(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, gfl=None):
    possible_unfusion_neighbors = []
    # c_set_neighbor = set(g.neighbors(vertex)).intersection(gfl[1][vertex]).pop()
    # if len(set(g.neighbors(vertex)).intersection(gfl[1][vertex])) > 1 :
    #     print("multiple neighboring vertices in cset ",len(set(g.neighbors(vertex)).intersection(gfl[1][vertex])))
    # if not vertex in gfl[1]:
    #     print("fatal")
    #     breakpoint()
    if len(gfl.neighbors(vertex)) > 2:
        return possible_unfusion_neighbors
    for neighbor in gfl.neighbors(vertex):
        if len(gfl.neighbors(neighbor)) <= 2 and not neighbor == exclude_vertex:
            possible_unfusion_neighbors.append(neighbor)

    return possible_unfusion_neighbors
    # for c_set_neighbor in set(g.neighbors(vertex)).intersection(gfl[1][vertex]):
    #     # possible_unfusion_neighbors.append(c_set_neighbor)
    #     if abs(gfl[0][c_set_neighbor] - gfl[0][vertex]) == 1:
    #         possible_unfusion_neighbors.append(c_set_neighbor)
    # if len(set(g.neighbors(vertex)).intersection(gfl[1][vertex])) == 1 :
    #     c_set_neighbor = set(g.neighbors(vertex)).intersection(gfl[1][vertex]).pop()
    #     if abs(gfl[0][c_set_neighbor] - gfl[0][vertex]) == 1:
    #         # print("vertex ",vertex," cset neighbor with distance 1: ",c_set_neighbor)
    #         possible_unfusion_neighbors.append(c_set_neighbor)

    # for neighbor in g.neighbors(vertex):
    #     if neighbor in gfl[1] and vertex in set(g.neighbors(neighbor)).intersection(gfl[1][neighbor]): # no output neighbor
    #         if abs(gfl[0][neighbor] - gfl[0][vertex]) == 1 and len(set(g.neighbors(neighbor)).intersection(gfl[1][neighbor])) == 1:
    #             # print("vertex ",vertex," previous neighbor with distance 1: ",neighbor)
    #             possible_unfusion_neighbors.append(neighbor)
    # if exclude_vertex and exclude_vertex in possible_unfusion_neighbors:
    #     # print("removed an exclude vertex ",exclude_vertex)
    #     possible_unfusion_neighbors.remove(exclude_vertex)
    # return possible_unfusion_neighbors

def get_possible_unfusion_neighbors_3(g: BaseGraph[VT,ET], vertex, exclude_vertex=None, gfl=None):
    neighbors = set(g.neighbors(vertex)) #set for iteration
    res_neighbors = set(g.neighbors(vertex)) #resulting set
    for v in set(g.non_outputs()).difference(neighbors):
        n_in_cset = gfl[1][v].intersection(neighbors)
        if len(n_in_cset) > 0 and len(n_in_cset) % 2 == 0:
            res_neighbors.difference_update(n_in_cset)
    if exclude_vertex:
        res_neighbors.difference_update(set(exclude_vertex))
    return res_neighbors


def unfuse_to_neighbor(g,vertex,neighbor,desired_phase):
    # import pdb
    # pdb.set_trace()
    new_phase = split_phases(g.phases()[vertex], desired_phase)
    phase_spider = g.add_vertex(VertexType.Z,-2,g.rows()[vertex],new_phase)
    g.set_phase(vertex, desired_phase)
    g.add_edge((phase_spider,neighbor), EdgeType.HADAMARD)
    g.add_edge((vertex,phase_spider), EdgeType.SIMPLE)
    phaseless_spider = insert_identity(g,vertex,phase_spider)

    g.remove_edge(g.edge(vertex,neighbor))
    # print("unfuse to neighbor ",vertex, neighbor, phaseless_spider, phase_spider)
    # pdb.set_trace()
    return (phaseless_spider, phase_spider)

def apply_lcomp(g: BaseGraph[VT,ET], match, gfl):
    print("apply lcomp match ",match)
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
    print("apply pivot match ",match)
    v1,v2 = match[1]
    unfusion_neighbors = {}
    unfusion_neighbors[v1] = match[2]
    unfusion_neighbors[v2] = match[3]
    for vertex in [v1,v2]:
        if unfusion_neighbors[vertex]:
            phaseless_s, phase_s = unfuse_to_neighbor(g, vertex, unfusion_neighbors[vertex], Fraction(0,1))
            update_gflow_from_double_insertion(gfl, vertex, unfusion_neighbors[vertex], phaseless_s, phase_s)
            
    gfl = update_gflow_from_pivot(g, v1, v2, gfl)
    apply_rule(g, pivot, [(v1,v2,[],[])])

def generate_matches(g, gfl,max_v=None, cap=1):
    fggraph = build_focused_gflow_graph(g, gfl)
    # fggraph = gflow(g)[1]
    lcomp_matches = lcomp_matcher(g, fggraph)
    pivot_matches = pivot_matcher(g, fggraph)
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
    gfl = gflow(g)
    gff = focus_gflow(g, gfl)
    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, gfl=gff, max_v=max_v, cap=cap)
        gf1 = gff[1]
        if apply_best_match(g, lcomp_matches, pivot_matches, gf1):
            iterations += 1
            changes = True
            gff[1] = gf1
            # if not check_gflow(g,gff):
            #     breakpoint()
            gff = focus_gflow(g, gff)
        else:
            continue

    return iterations

def random_wire_reduce_neighbor(g: BaseGraph[VT,ET], max_v=None, cap=1, quiet:bool=False, stats=None):
    changes = True
    iterations = 0
    gfl = gflow(g)
    if not gfl:
        import pdb
        pdb.set_trace()
    # if not check_gflow(g,gfl):
    #     print("before focus")
    #     import pdb
    #     pdb.set_trace()
    gff = focus_gflow(g, gfl)
    # if not check_gflow(g,gff):
    #     print("before start")
    #     import pdb
    #     pdb.set_trace()
    while changes:
        changes = False
        lcomp_matches, pivot_matches = generate_matches(g, gfl=gff, max_v=max_v, cap=cap)
        gf1 = gff[1]
        old_gf = gfl
        old_g = copy.deepcopy(g)
        old_gff = copy.deepcopy(gff)
        if apply_random_match(g, lcomp_matches, pivot_matches, gf1):
            iterations += 1
            changes = True
            gff[1] = gf1
            # if not check_gflow(g,gff):
            #     print("shouldbreakhere")
            #     import pdb
            #     pdb.set_trace()
            gfl = gflow(g)
            if not gfl:
                import pdb
                pdb.set_trace()

            gff = focus_gflow(g, gfl)
        else:
            continue

    return iterations

def sim_annealing_reduce_neighbor(g: BaseGraph[VT,ET], max_v=None, iterations=100, alpha=0.95, cap=-100000, quiet:bool=False, stats=None):
    temperature = iterations
    epsilon = 1
    it = 0
    gfl = gflow(g)[1]
    best = g.copy()
    best_eval = g.num_edges()
    curr_eval = best_eval
    backtrack = False

    while temperature > epsilon:
        it += 1
        lcomp_matches, pivot_matches = generate_matches(g, gfl=gfl, max_v=max_v, cap=cap) #1-int(temperature)
        method, match = get_best_match(lcomp_matches, pivot_matches)
        if match[0] <= 0:
            if backtrack:
                g = best.copy()
                # g = best
                curr_eval = best_eval
                backtrack = False
                gfl = gflow(g)[1]
                # print("reset to best eval")
                continue
            else:
                method, match = get_random_match(lcomp_matches, pivot_matches)
                backtrack = True
                # print("start branch with negative match")

        if method == "none":
            temperature = 0
            break
            # print("best eval ",)
        theexp = math.exp(match[0]/temperature)
        # if match[0] <= 0:
        #     print(match[0], temperature, theexp)
        if match[0] > 0 or theexp > random.random():
            curr_eval -= match[0]

            if method == "pivot":
                apply_pivot(g,match, gfl)
            else:
                apply_lcomp(g,match, gfl)
                
            if curr_eval < best_eval:
                best = g.copy()
                best_eval = curr_eval
            # temperature *=alpha
                # print("do not apply rule with cost ",match[0])
                # continue
            gfl = gflow(g)[1]
            # print("best eval ",best_eval,"curr ",curr_eval,"match ",match[0])
        # else:
        #     print("theexp fail ",match[0], temperature, theexp)

        # h_cost -= match[0]
        temperature *= alpha
        # print("apply rule with cost ",match[0])
        # print(temperature)
        
        # remove_ids(g, match_ids_parallel(g))
        # spider(g, match_spider_parallel(g))
    print("final num edges: ",best.num_edges())
    return best

def apply_random_match(g, lcomp_matches, pivot_matches, gfl):
    method, match = get_random_match(lcomp_matches, pivot_matches)
    if not gfl:
        breakpoint()

    if method == "pivot":
        apply_pivot(g,match, gfl)
    elif method == "lcomp":
        apply_lcomp(g,match, gfl)
    else:
        return False
    return True

# def apply_random_match(g, lcomp_matches, pivot_matches, gfl):
#     # lcomp_matches.sort(key= lambda m: m[0],reverse=True)
#     # pivot_matches.sort(key= lambda m: m[0],reverse=True)
#     method = "pivot"

#     if len(lcomp_matches) > 0:
#         if len(pivot_matches) > 0:
#             method = "lcomp" if random.random() < 0.5 else "pivot"    
#         else:
#             method = "lcomp"
#     else:
#         if len(pivot_matches) == 0:
#             return False

#     if method == "pivot":
#         apply_pivot(g,pivot_matches[0], gfl)
#     else:
#         apply_lcomp(g,lcomp_matches[0], gfl)
#     return True

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

def get_best_match(lcomp_matches, pivot_matches):
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
            return ("none",None)

    if method == "pivot":
        return ("pivot", pivot_matches[0])
    else:
        return ("lcomp", lcomp_matches[0])
