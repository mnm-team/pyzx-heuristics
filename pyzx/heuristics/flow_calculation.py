import copy
import itertools
from typing import Dict, Set, Tuple
from pyzx.graph.base import BaseGraph, VT, ET, EdgeType
from pyzx.graph.graph_s import GraphS

def calculate_lcomp(graph: BaseGraph[VT,ET], vertex: VT):
  """
  Calculate local complementation for a given vertex in the graph.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  vertex (VT): The vertex to calculate local complementation for.
  """
  vertex_neighbors = list(graph.neighbors(vertex))
  all_possible_edges = set(itertools.combinations(vertex_neighbors, 2))

  edges_to_remove = set([(source, target) for source, target in all_possible_edges if graph.connected(source, target) and source in vertex_neighbors and target in vertex_neighbors])
  edges_to_add = all_possible_edges - edges_to_remove

  graph.remove_edges(edges_to_remove)
  graph.add_edges(edges_to_add, EdgeType.HADAMARD)

def get_odd_neighbourhood(graph: BaseGraph[VT,ET], vertex_set):
  """
  Get the odd neighborhood for a set of vertices in the graph.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  vertex_set (set): The set of vertices to get the odd neighborhood for.

  Returns:
  list: The list of vertices in the odd neighborhood.
  """
  all_neighbors = set()

  for vertex in vertex_set:
    all_neighbors.update(set(graph.neighbors(vertex)))

  odd_neighbors = []

  for neighbor in all_neighbors:
    # If the intersection of the neighbor's neighbors and the vertex set has an odd length
    if len(set(graph.neighbors(neighbor)).intersection(vertex_set)) % 2 == 1:
      odd_neighbors.append(neighbor)

  return odd_neighbors

def get_all_vertices_for_lcomp(graph: BaseGraph[VT,ET], vertex: VT, flow_dict: dict):
  """
  Get all vertices for which local complementation can be applied.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  vertex (VT): The vertex to calculate local complementation for.
  flow_dict (dict): The dictionary representing the gflow of the graph.

  Returns:
  list: The list of vertices for which local complementation can be applied.
  """
  matching_vertices = []
  for candidate_vertex in flow_dict:
    if vertex in get_odd_neighbourhood(graph, flow_dict[candidate_vertex]):
      matching_vertices.append(candidate_vertex)
  return matching_vertices

def get_odd_neighbourhood_fast(graph: BaseGraph[VT,ET], vertex: VT, flow_dict: dict):
  """
  Get the odd neighborhood for a vertex in the graph in a fast way.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  vertex (VT): The vertex to get the odd neighborhood for.
  flow_dict (dict): The dictionary representing the gflow of the graph.

  Returns:
  list: The list of vertices in the odd neighborhood.
  """
  odd_neighbour_vertices = []
  vertex_neighbors = set(graph.neighbors(vertex))
  for key in flow_dict: #TODO: this has runtime O(n²), simplify this to O(n) by reverse lookup dict
    if len(vertex_neighbors.intersection(flow_dict[key])) % 2 != 0:
      odd_neighbour_vertices.append(key)
  return odd_neighbour_vertices

def update_lcomp_gflow(graph: BaseGraph[VT,ET], vertex: VT, flow_dict: dict, set_difference_vertex=True):
  """
  Update the gflow of the graph after applying local complementation.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  vertex (VT): The vertex to calculate local complementation for.
  flow_dict (dict): The dictionary representing the gflow of the graph.
  set_difference_vertex (bool): Whether to update the gflow of the vertex itself.

  Returns:
  dict: The updated gflow dictionary.
  """
  candidate_vertices = get_all_vertices_for_lcomp(graph, vertex, flow_dict)

  if set_difference_vertex:
    flow_dict[vertex].symmetric_difference_update(set([vertex]))

  for candidate_vertex in candidate_vertices:
    if candidate_vertex != vertex:
      flow_dict[candidate_vertex].symmetric_difference_update(flow_dict[vertex])
      flow_dict[candidate_vertex].symmetric_difference_update(set([vertex]))

  return flow_dict

def update_gflow_from_double_insertion(gflow, start_vertex: VT, end_vertex: VT, intermediate_vertex: VT, end_vertex_duplicate: VT):
  """
  Update the gflow after a double insertion operation.

  Parameters:
  gflow (dict): The dictionary representing the gflow of the graph.
  start_vertex (VT): The start vertex of the edge where the insertion is performed.
  end_vertex (VT): The end vertex of the edge where the insertion is performed.
  intermediate_vertex (VT): The first vertex inserted between start_vertex and end_vertex.
  end_vertex_duplicate (VT): The second vertex inserted between start_vertex and end_vertex.

  Returns:
  dict: The updated gflow dictionary.
  """
  if end_vertex in gflow[start_vertex]:
    gflow[end_vertex_duplicate] = set([end_vertex])
    gflow[intermediate_vertex] = set([end_vertex_duplicate])

    gflow[start_vertex].remove(end_vertex)
    gflow[start_vertex].update(set([intermediate_vertex]))

  elif start_vertex in gflow[end_vertex]:
    gflow[end_vertex_duplicate] = set([intermediate_vertex])
    gflow[intermediate_vertex] = set([start_vertex])

    gflow[end_vertex].remove(start_vertex)
    gflow[end_vertex].update(set([end_vertex_duplicate]))
  else:
    print("Fatal: unfusion neighbor not in graph gflow")
    # breakpoint()
  return gflow

def update_gflow_from_pivot(graph: BaseGraph[VT,ET], pivot_vertex_1: VT, pivot_vertex_2: VT, gflow):
  """
  Update the graph gflow after a pivot operation.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  pivot_vertex_1 (VT): The first vertex to pivot around.
  pivot_vertex_2 (VT): The second vertex to pivot around.
  gflow (dict): The dictionary representing the gflow of the graph.

  Returns:
  dict: The updated gflow dictionary.
  """
  graph_copy = GraphS()
  graph_copy.graph = copy.deepcopy(graph.graph)

  update_lcomp_gflow(graph_copy, pivot_vertex_1, gflow)
  calculate_lcomp(graph_copy, pivot_vertex_1)

  update_lcomp_gflow(graph_copy, pivot_vertex_2, gflow)
  calculate_lcomp(graph_copy, pivot_vertex_2)

  update_lcomp_gflow(graph_copy, pivot_vertex_1, gflow, False)

  graph_flow_pivot_vertex_1 = gflow[pivot_vertex_1]
  graph_flow_pivot_vertex_2 = gflow[pivot_vertex_2]

  for key in gflow:
    if key == pivot_vertex_1 or key == pivot_vertex_2:
      continue
    if not pivot_vertex_1 in gflow[key] and not pivot_vertex_2 in gflow[key]:
      continue
    if pivot_vertex_1 in gflow[key] and not pivot_vertex_2 in gflow[key].symmetric_difference(graph_flow_pivot_vertex_1):
      gflow[key].symmetric_difference_update(graph_flow_pivot_vertex_1)
    elif pivot_vertex_2 in gflow[key] and not pivot_vertex_1 in gflow[key].symmetric_difference(graph_flow_pivot_vertex_2):
      gflow[key].symmetric_difference_update(graph_flow_pivot_vertex_2)
    elif pivot_vertex_2 in gflow[key].symmetric_difference(graph_flow_pivot_vertex_1) and pivot_vertex_1 in gflow[key].symmetric_difference(graph_flow_pivot_vertex_2):
      gflow[key].symmetric_difference_update(graph_flow_pivot_vertex_1)
      gflow[key].symmetric_difference_update(graph_flow_pivot_vertex_2)
    else:
      print("Fatal: no pivot gflow match!")

  # Update the gflow for all vertices that contain the first pivot vertex
  for key in gflow: 
    if pivot_vertex_1 in gflow[key] and key != pivot_vertex_1:
      gflow[key].symmetric_difference_update(gflow[pivot_vertex_1])
  gflow.pop(pivot_vertex_1)

  # Update the gflow for all vertices that contain the second pivot vertex
  for key in gflow: 
    if pivot_vertex_2 in gflow[key] and key != pivot_vertex_2:
      gflow[key].symmetric_difference_update(gflow[pivot_vertex_2])
  gflow.pop(pivot_vertex_2)

  return gflow

def update_gflow_from_lcomp(graph: BaseGraph[VT,ET], lcomp_vertex: VT, gflow):
  """
  Update the gflow after a local complementation operation.

  Parameters:
  graph (BaseGraph[VT,ET]): The graph to perform the operation on.
  lcomp_vertex (VT): The vertex to perform local complementation on.
  gflow (dict): The dictionary representing the gflow of the graph.

  Returns:
  dict: The updated gflow dictionary.
  """
  graph_copy = GraphS()
  graph_copy.graph = copy.deepcopy(graph.graph)

  update_lcomp_gflow(graph_copy, lcomp_vertex, gflow)

  # Update the gflow for all vertices that contain the local complementation vertex
  for key in gflow: 
    if lcomp_vertex in gflow[key] and key != lcomp_vertex:
      gflow[key].symmetric_difference_update(gflow[lcomp_vertex])
  gflow.pop(lcomp_vertex)

  return gflow





Flow = Tuple[Dict[VT, Set[VT]], Dict[VT,int]]

def identify_causal_flow(g: BaseGraph[VT, ET]) -> Flow:
    solved = set(g.outputs())
    correctors = set()
    past: Dict[VT, int] = dict()
    res: Flow = (dict(),dict())
    inputs = [list(g.neighbors(i))[0] for i in g.inputs()]

    for o in g.outputs():
        n = list(g.neighbors(o))[0]
        res[0][n] = set()
        res[1][n] = 0
        solved.add(n)
        if not n in inputs:
            past[n] = len(set(g.neighbors(n)).difference(solved))
            if past[n] == 1:
                correctors.add(n)
    
    depth = 1

    while True:
        new_corrections = set()

        for corrector in correctors:
            candidates = set(g.neighbors(corrector)).difference(solved)
            if len(candidates) == 1:
                candidate = candidates.pop()
                res[0][candidate] = set([corrector])
                res[1][candidate] = depth
                solved.add(candidate)
                if not candidate in inputs:
                    past[candidate] = len(set(g.neighbors(candidate)).difference(solved))
                    if past[candidate] == 1:
                        new_corrections.add(candidate)
                for neighbor in g.neighbors(candidate):
                    if neighbor in past:
                        past[neighbor] -= 1
                        if past[neighbor] == 1:
                            new_corrections.add(neighbor)

        if not new_corrections:
            if len(solved) == g.num_vertices() - g.num_inputs():
                return res
            return None
        
        correctors = new_corrections
        depth += 1
