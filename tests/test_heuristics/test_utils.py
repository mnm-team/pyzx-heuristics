from pathlib import Path
import sys

project_path = Path(__file__).parent.parent.parent
if project_path not in sys.path:
    sys.path.append(str(project_path))

import random
import pyzx as zx
from pyzx.simplify import spider_simp, to_gh
from pyzx.graph.base import BaseGraph
from pyzx.heuristics.simplification import FilterFlowFunc, MatchType, apply_lcomp, apply_pivot, lcomp_matcher, pivot_matcher


def load_graphs() -> dict[str, list]:
    path_to_circuits = project_path / 'circuits\\qasm'
    input_data = {"Name": [], "circuit": [], "graph": []}

    for file in Path(path_to_circuits).glob('*.qasm'):
        circuit = zx.Circuit.load(file).to_basic_gates()
        # if circuit.qubits <= 19 and circuit.qubits >= 8 and len(circuit.gates) <= 5000 and len(circuit.gates) >= 100:

        if file.stem == "gf2^5_mult" or file.stem == "gf2^6_mult" or file.stem == "barenco_tof_3" or file.stem == "mod_red_21":
            try:
                circuit = zx.optimize.basic_optimization(circuit)
            except Exception as e:
                pass
            graph = circuit.to_graph()
            graph = graph.copy()

            input_data["Name"].append(file.stem)
            input_data["circuit"].append(circuit)
            input_data["graph"].append(circuit.to_graph())

    return input_data

def deep_tuple(lst):
    return tuple(deep_tuple(i) if isinstance(i, list) or isinstance(i, tuple) else i for i in lst)

def get_circuit_and_fr_graph(num_qubits: int = 5, depth: int = 50, seed=None):
    if seed:
        random.seed(seed)
    g = zx.generate.cliffordT(qubits=num_qubits, depth=depth, p_t=0.3, p_cnot=0.5)
    # c = zx.generate.phase_poly(n_qubits=16, n_phase_layers=20, cnots_per_layer=10)
    c = zx.Circuit.from_graph(g)
    c = zx.optimize.basic_optimization(c.split_phase_gates()).split_phase_gates()

    g = c.to_graph()
    g_tele = g.copy()
    zx.simplify.full_reduce(g_tele)

    return c, g_tele

def generate_graph(num_qubits: int, depth: int) -> BaseGraph:
    g = zx.generate.cliffordT(qubits=num_qubits, depth=depth)
    spider_simp(g)
    to_gh(g)

    return g

def calculate_gflow(graph: BaseGraph) -> bool:
    g_clone = graph.clone()
    flow_function = FilterFlowFunc.G_FLOW_PRESERVING
    flow = flow_function(g_clone)
    flow = flow if flow else None
    return flow is not None

def calculate_gflow_gadget(graph: BaseGraph) -> bool:
    g_clone = graph.clone()
    flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
    flow = flow_function(g_clone)
    flow = flow if flow else None
    return flow is not None

def apply_random_matches(graph: BaseGraph, num_matches: int = 50, flow_function=None, match_filter_func=None):

    applied_matches = []

    for i in range(num_matches):
        lcomp_matches = lcomp_matcher(graph)
        pivot_matches = pivot_matcher(graph)

        if match_filter_func:
            lcomp_matches = match_filter_func(graph, lcomp_matches)
            pivot_matches = match_filter_func(graph, pivot_matches)
        
        if not lcomp_matches and not pivot_matches:
            return []
        
        if random.randint(0, 1) and lcomp_matches:
            matches = lcomp_matches
            match_type = MatchType.LCOMP
        else:
            matches = pivot_matches
            match_type = MatchType.PIVOT
        
        if not matches:
            if match_type == MatchType.PIVOT:
                matches = lcomp_matches
                match_type = MatchType.LCOMP
            else:
                matches = pivot_matches
                match_type = MatchType.PIVOT

        match_items = list(matches.items())
        random.shuffle(match_items)
        for match_key, match_values in match_items:
            was_match_applied = False
            for match_value in match_values:

                graph_copy = graph.clone()

                if match_type == MatchType.LCOMP:
                    match_result_with_time = apply_lcomp(graph_copy, (match_key, match_value))
                elif match_type == MatchType.PIVOT:
                    match_result_with_time = apply_pivot(graph_copy, (match_key, match_value))

                if flow_function:
                    is_flow_preserving = flow_function(graph=graph_copy)
                else:
                    is_flow_preserving = True

                if is_flow_preserving:
                    was_match_applied = True
                    applied_matches.append((match_key, match_value))
                    if match_type == MatchType.LCOMP:
                        apply_lcomp(graph, (match_key, match_value))
                    elif match_type == MatchType.PIVOT:
                        apply_pivot(graph, (match_key, match_value))
                    break
            if was_match_applied:
                break
    return applied_matches

