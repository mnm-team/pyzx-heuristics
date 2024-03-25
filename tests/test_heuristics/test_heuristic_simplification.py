from pathlib import Path
import sys

sys.path.append('..')
sys.path.append('.')

import random
import unittest
import pyzx as zx
from pyzx.heuristics.simplification import MatchType, apply_lcomp, apply_pivot, get_match_type, lcomp_matcher, pivot_matcher, update_matches
from pyzx.extract import extract_architecture_aware_circuit
from pyzx.routing.architecture import create_line_architecture


def load_graphs() -> dict[str, list]:
    path_to_circuits = 'c:\\Users\\wsajk\\Documents\\Arbeit\\MUNIQC-Atoms\\pyzx-heuristics\\circuits\\qasm\\'
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

def get_circuit_and_fr_graph():
    random.seed(1349)
    g = zx.generate.cliffordT(qubits=5, depth=50, p_t=0.3, p_cnot=0.5)
    # c = zx.generate.phase_poly(n_qubits=16, n_phase_layers=20, cnots_per_layer=10)
    c = zx.Circuit.from_graph(g)
    c = zx.optimize.basic_optimization(c.split_phase_gates()).split_phase_gates()

    g = c.to_graph()
    g_tele = g.copy()
    zx.simplify.full_reduce(g_tele)

    return c, g_tele

class TestHeuristics(unittest.TestCase):

    def test_update_matches(self):
        # Create a test graph
        graph = zx.generate.cliffordT(qubits=10, depth=500)
        lcomp_matches = lcomp_matcher(graph)
        pivot_matches = pivot_matcher(graph)

        for _ in range(10):

            match = random.choice(list({**lcomp_matches, **pivot_matches}.items()))
            match_key, match_values = match
            match_value = random.choice(match_values)

            if get_match_type(match) == MatchType.PIVOT:
                vertex_neighbors = set()
                for vertex in match_key:
                    for vertex_neighbor in graph.neighbors(vertex):
                        if vertex_neighbor not in match_key:
                            vertex_neighbors.add(vertex_neighbor)
                match_result_with_time = apply_pivot(graph=graph, match=(match_key, match_value))

            elif get_match_type(match) == MatchType.LCOMP:
                _, vertex_neighbors, _ = match_value
                match_result_with_time = apply_lcomp(graph=graph, match=(match_key, match_value))

            match_result, time_info = match_result_with_time

            if match_result:
                new_verticies, flow = match_result
                vertex_neighbors = set(vertex_neighbors).union(set(new_verticies))

            # Call the update_matches function
            lcomp_matches, pivot_matches = update_matches(graph, vertex_neighbors, match_key, lcomp_matches, pivot_matches)

            # Assert the expected output
            expected_lcomp_matches = lcomp_matcher(graph)
            expected_pivot_matches = pivot_matcher(graph)

            lcomp_diff = set({key : deep_tuple(value) for key, value in lcomp_matches.items()}).symmetric_difference(set({key : deep_tuple(value) for key, value in expected_lcomp_matches.items()}))
            pivot_diff = set({key : deep_tuple(value) for key, value in pivot_matches.items()}).symmetric_difference(set({key : deep_tuple(value) for key, value in expected_pivot_matches.items()}))

            self.assertEqual(lcomp_diff, set())
            self.assertEqual(pivot_diff, set())

    # def test_greedy_simp(self):

    #     for name, circuit, graph in load_graphs():
    #         print(name)
    #         print(circuit.stats())
    #         print(graph.stats())
    #         print()

    #         for la in range(2):

    #             simplified_graph = graph.copy()
    #             # Apply the greedy simplification
    #             zx.simplify.teleport_reduce(simplified_graph, quiet=True)
    #             print(simplified_graph.stats())
    #             print()

    #             # Apply the greedy simplification
    #             zx.simplify.greedy_simp(simplified_graph, lookahead=la, quiet=True)
    #             print(simplified_graph.stats())
    #             print()

    #             self.assertEqual(simplified_graph, graph)

    # def test_greedy_simp_neighbors(self):
            
    #         for name, circuit, graph in load_graphs():
    #             print(name)
    #             print(circuit.stats())
    #             print(graph.stats())
    #             print()

    #             for la in range(2):

    #                 simplified_graph = graph.copy()
    #                 # Apply the greedy simplification
    #                 zx.simplify.teleport_reduce(simplified_graph, quiet=True)
    #                 print(simplified_graph.stats())
    #                 print()
        
    #                 # Apply the greedy simplification
    #                 zx.simplify.greedy_simp_neighbors(simplified_graph, lookahead=la, quiet=True)
    #                 print(simplified_graph.stats())
    #                 print()
        
    #                 self.assertEqual(simplified_graph, graph)

    # def test_sim_anneal(self):

    #     for name, circuit, graph in load_graphs():
    #         print(name)
    #         print(circuit.stats())
    #         print(graph.stats())
    #         print()

    #         simplified_graph = graph.copy()
    #         # Apply the greedy simplification
    #         zx.simplify.teleport_reduce(simplified_graph, quiet=True)
    #         print(simplified_graph.stats())
    #         print()

    #         # Apply the greedy simplification
    #         zx.simplify.sim_anneal_simp(simplified_graph, quiet=True)
    #         print(simplified_graph.stats())
    #         print()

    #         self.assertEqual(simplified_graph, graph)

    # def test_sim_anneal_neighbors(self):
    
    #         for name, circuit, graph in load_graphs():
    #             print(name)
    #             print(circuit.stats())
    #             print(graph.stats())
    #             print()
    
    #             simplified_graph = graph.copy()
    #             # Apply the greedy simplification
    #             zx.simplify.teleport_reduce(simplified_graph, quiet=True)
    #             print(simplified_graph.stats())
    #             print()
    
    #             # Apply the greedy simplification
    #             zx.simplify.sim_anneal_simp_neighbors(simplified_graph, quiet=True)
    #             print(simplified_graph.stats())
    #             print()
    
    #             self.assertEqual(simplified_graph, graph)
            

    def test_architecture_aware_extraction(self):
        
        _, graph = get_circuit_and_fr_graph()

        architecture = create_line_architecture(graph.qubit_count())
        graph_simp = graph.copy()
        new_circuit = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        for gate in new_circuit.gates:
            if gate.name == "CNOT":
                qubit0 = gate.target
                qubit1 = gate.control
                if not architecture.graph.connected(qubit0, qubit1):
                    assert False

        assert True

    def test_architecture_aware_extraction_correctness(self):
        
        circuit, graph = get_circuit_and_fr_graph()

        architecture = create_line_architecture(graph.qubit_count())
        graph_simp = graph.copy()
        new_circuit = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=False, quiet=True)

        assert zx.compare_tensors(circuit, new_circuit)

if __name__ == '__main__':
    unittest.main()