import copy
import itertools
from typing import List, Dict, Set, Tuple
from pyzx.graph.base import BaseGraph, VT, ET, EdgeType
from pyzx.graph.graph_s import GraphS

def calculate_lcomp(g: BaseGraph[VT,ET], u: VT):
  vn = list(g.neighbors(u))
  all_edges = set(itertools.combinations(vn,2))
  remove_edges = set([(s,t) for s,t in all_edges if g.connected(s,t) and s in vn and t in vn])
  add_edges= all_edges - remove_edges
  g.remove_edges(remove_edges)
  g.add_edges(add_edges, EdgeType.HADAMARD)

def get_odd_neighbourhood(g: BaseGraph[VT,ET], vertex_set):
  all_neighbors = set()
  for vertex in vertex_set:
    all_neighbors.update(set(g.neighbors(vertex)))
  odd_neighbors = []
  for neighbor in all_neighbors:
    if len(set(g.neighbors(neighbor)).intersection(vertex_set)) % 2 == 1:
      odd_neighbors.append(neighbor)
  return odd_neighbors

def get_all_v_for_lcomp(g, u, gf):
  matches = []
  for candidate in gf:
    if u in get_odd_neighbourhood(g, gf[candidate]):
      matches.append(candidate)
  return matches

def get_odd_neighbourhood_fast(g: BaseGraph[VT,ET], u: VT, gf):
  candidates = []
  u_neighbors = set(g.neighbors(u))
  for key in gf: #TODO: this has runtime O(n²), simplify this to O(n) by reverse lookup dict
    if len(u_neighbors.intersection(gf[key])) % 2 != 0:
      candidates.append(key)
  return candidates

def update_lcomp_gflow(g: BaseGraph[VT,ET], u: VT, gf, set_difference_u=True):
  candidates = get_all_v_for_lcomp(g, u, gf)

  if set_difference_u:
    gf[u].symmetric_difference_update(set([u]))

  for candidate in candidates:
    if candidate != u:
      gf[candidate].symmetric_difference_update(gf[u])
      gf[candidate].symmetric_difference_update(set([u]))

  return gf

def update_gflow_from_double_insertion(gf, vs: VT, ve: VT, vid: VT, vend: VT):
  if ve in gf[vs]:
    gf[vend] = set([ve])
    gf[vid] = set([vend])
    gf[vs].remove(ve)
    gf[vs].update(set([vid]))
  elif vs in gf[ve]:
    gf[vend] = set([vid])
    gf[vid] = set([vs])
    gf[ve].remove(vs)
    gf[ve].update(set([vend]))
  else:
    print("Fatal: unfusion neighbor not in gflow")
    # breakpoint()
  return gf

def update_gflow_from_pivot(g: BaseGraph[VT,ET], u: VT, v: VT, gf):
  g_copy = GraphS()
  g_copy.graph = copy.deepcopy(g.graph)

  update_lcomp_gflow(g_copy, u, gf)
  calculate_lcomp(g_copy, u)
  update_lcomp_gflow(g_copy, v, gf)
  calculate_lcomp(g_copy, v)
  update_lcomp_gflow(g_copy, u, gf, False) #no set difference at last time because u is YZ vertex
  gf_u_yz = gf[u]
  gf_v_yz = gf[v]
  for key in gf:
    if key == u or key == v:
      continue
    if not u in gf[key] and not v in gf[key]:
      continue
    if u in gf[key] and not v in gf[key].symmetric_difference(gf_u_yz):
      gf[key].symmetric_difference_update(gf_u_yz)
    elif v in gf[key] and not u in gf[key].symmetric_difference(gf_v_yz):
      gf[key].symmetric_difference_update(gf_v_yz)
    elif v in gf[key].symmetric_difference(gf_u_yz) and u in gf[key].symmetric_difference(gf_v_yz):
      gf[key].symmetric_difference_update(gf_u_yz)
      gf[key].symmetric_difference_update(gf_v_yz)
    else:
      print("Fatal: no pivot gflow match!")
  
  for key in gf: 
    if u in gf[key] and key != u:
      gf[key].symmetric_difference_update(gf[u])
  gf.pop(u)

  for key in gf: 
    if v in gf[key] and key != v:
      gf[key].symmetric_difference_update(gf[v])
  gf.pop(v)

  return gf #TODO: return not necessary

def update_gflow_from_lcomp(g: BaseGraph[VT,ET], u: VT, gf):
  g_copy = GraphS()
  g_copy.graph = copy.deepcopy(g.graph)
  update_lcomp_gflow(g_copy, u, gf)
  for key in gf: 
    if u in gf[key] and key != u:
      gf[key].symmetric_difference_update(gf[u])
  gf.pop(u) #cleanup, i.e. remove lcomp vertex

  return gf

def focus_gflow(g: BaseGraph[VT,ET], gf: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  l:     Dict[VT,int]      = {}
  gflow: Dict[VT, Set[VT]] = {}
  for v in g.outputs():
    l[v] = 0
    gflow[v] = set() #usually outputs do not occur in g function but we need this construct as a temporary helper
  processed = set(g.outputs())

  unprocessed = set(g.vertices()).difference(processed)
  k = 1
  while True:
    candidates = []
    for v in unprocessed:
      try:
        odd_n = set(get_odd_neighbourhood(g,gf[1][v]))
      except:
        breakpoint()
      odd_n.discard(v)
      no_candidate = False
      for n in odd_n:
        if not n in gflow:
          no_candidate = True
          break
      if no_candidate:
        continue

      c_set = set(gf[1][v])
      for n in odd_n:
        c_set.symmetric_difference_update(gflow[n])
      gflow[v] = c_set
      l[v] = k
      candidates.append(v)
    if len(candidates) == 0:
      for v in g.vertices():
        if not v in gflow:
          print("fatal: vertex ",v," not in focused gflow ")
          breakpoint()
      for output in g.outputs(): #delete outputs from g function
        gflow.pop(output)
      return [l, gflow, k]
    unprocessed.difference_update(set(candidates))
    k += 1

def restore_ordering(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  vertices = set(g.outputs())
  ordering = {}
  for vertex in g.vertices(): #init everything with order 0
    ordering[vertex] = -1
  for vertex in vertices: #init everything with order 0
    ordering[vertex] = 0
  k = 0
  while True:
    vertex = vertices.pop()
    if not vertex:
      return [ordering, gflow[1], k]
    for neighbor in g.neighbors(vertex):
      if neighbor in gflow[1] and vertex in gflow[1][neighbor] and ordering[neighbor] <= ordering[vertex]:        
        ordering[neighbor] = ordering[vertex]+1
        vertices.add(neighbor)
        if ordering[vertex]+1 > k:
          k = ordering[vertex]+1

def check_gflow_condition1(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  for v in g.non_outputs():
    for w in gflow[1][v]:
      if v!=w and gflow[0][v] < gflow[0][w]:
        print("gflow violates condition 1 because vertex: ",v," has higher or equal ordering than vertex ",w," which is in the correction set of vertex ",v)
        return False
  return True

def check_gflow_condition2(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  for v in g.non_outputs():
    for w in get_odd_neighbourhood(g,gflow[1][v]):
      if v!=w and gflow[0][v] < gflow[0][w]:
        print("gflow violates condition 2 because vertex: ",v," has higher or equal ordering than vertex ",w," which is in the odd neighborhood of the correction set of vertex ",v)
        return False
  return True

# Assuming everything is XY for now
def check_gflow_condition3(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  for v in g.non_outputs():
    if v in gflow[1][v]:
      print("gflow violates condition 2 because vertex: ",v," appears in its own correction set")
      return False
    if not v in get_odd_neighbourhood(g,gflow[1][v]):
      print("gflow violates condition 2 because vertex: ",v," does not appear in the odd neighborhood of its correction set")
      return False
  return True

def check_focused_gflow(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  for v in g.non_outputs():
    cset = set(get_odd_neighbourhood(g, gflow[1][v])).intersection(g.non_outputs())
    if len(cset) != 1 or not v in cset:
      print("no focused gflow because odd neighborhood of the correction set of vertex ",v," is ",cset)
      return False
  return True

def check_gflow(g: BaseGraph[VT,ET], gflow: List[Tuple[Dict[VT,int], Dict[VT,Set[VT]], int]]):
  if not check_gflow_condition1(g, gflow):
    return False
  if not check_gflow_condition2(g, gflow):
    return False
  if not check_gflow_condition3(g, gflow):
    return False
  return True

def build_line(g, vertex, gflow):
  res = [vertex]
  current = vertex
  while True:
    nlist = []
    for n in g.neighbors(current):
      if n in gflow[1] and current in gflow[1][n]:
        nlist.append(n)
    if not nlist:
      return res
    if len(nlist) > 1:
      print("")
    # if b:
    #   return res

'''
Builds focused gflow graph as lists
'''
def build_line_rec(g, vertex, gflow):
  nlist = []
  for n in g.neighbors(vertex):
    if n in gflow[1] and vertex in gflow[1][n]:
      nlist.append(n)
  if not nlist:
    return [vertex]
  if len(nlist) > 1:
    print("branching on vertex ",vertex)
    res = []
    for n in nlist:
      res.append(build_line_rec(g, n, gflow))
    return [vertex] + res
  else:
    return [vertex] + build_line_rec(g, nlist[0], gflow)

'''
Builds a Diagram out of the focused gflow graph for visualization purposes
'''
def build_diagram_from_lines(g, data):
  graph = GraphS()
  for node in range(0,len(data)):
    graph.add_vertex_indexed(data[node][0])
    recursive_diagram_build(graph, data[node][1:], data[node][0])
  graph.set_inputs(g.inputs())
  graph.set_outputs(g.outputs())
  return graph

'''
Recursive helper function for build_diagram_from_lines function 
'''
def recursive_diagram_build(graph, data, start):
  last_vertex = start
  already_visited = False
  for vertex in data:
    if isinstance(vertex, list):
      if not vertex[0] in graph.vertices():
        graph.add_vertex_indexed(vertex[0])
        recursive_diagram_build(graph, vertex[1:], vertex[0])
      # else:
      #   already_visited = True
      graph.add_edges([(last_vertex,vertex[0])])
      # Assumes this is the last vertex in data list
    else:
      if not vertex in graph.vertices():
        graph.add_vertex_indexed(vertex)
      else:
        already_visited = True
      graph.add_edges([(last_vertex,vertex)])
      last_vertex = vertex

    if already_visited:
      break