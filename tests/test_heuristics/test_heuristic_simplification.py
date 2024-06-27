from fractions import Fraction
from pathlib import Path
import sys
import time


project_path = Path(__file__).parent.parent.parent
if project_path not in sys.path:
    sys.path.append(str(project_path))

import random
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.heuristics.simplification import FilterFlowFunc, MatchType, apply_lcomp, apply_pivot, get_match_type, lcomp_matcher, pivot_matcher, update_matches


from tests.test_heuristics.test_utils import apply_random_matches, calculate_gflow, calculate_gflow_gadget, deep_tuple, generate_graph, get_circuit_and_fr_graph



class TestHeuristics():

    def test_update_matches(self):
        random.seed(1)
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

            removed_vertices = [key for key in match_key if key not in graph.vertex_set()]

            # Call the update_matches function
            lcomp_matches, pivot_matches = update_matches(graph, vertex_neighbors, removed_vertices, lcomp_matches, pivot_matches)

            # Assert the expected output
            expected_lcomp_matches = lcomp_matcher(graph)
            expected_pivot_matches = pivot_matcher(graph)

            lcomp_diff = set({key : deep_tuple(value) for key, value in lcomp_matches.items()}).symmetric_difference(set({key : deep_tuple(value) for key, value in expected_lcomp_matches.items()}))
            pivot_diff = set({key : deep_tuple(value) for key, value in pivot_matches.items()}).symmetric_difference(set({key : deep_tuple(value) for key, value in expected_pivot_matches.items()}))

            if len(lcomp_diff) > 0 or len(pivot_diff) > 0:
                print("Lcomp diff: ", lcomp_diff)
                print("Pivot diff: ", pivot_diff)
            assert len(lcomp_diff) == 0
            assert len(pivot_diff) == 0

    def test_greedy_simp(self):

        circuit, graph = get_circuit_and_fr_graph(5, 10)

        for la in range(2):

            simplified_graph = graph.copy()
            # Apply the greedy simplification
            zx.simplify.teleport_reduce(simplified_graph, quiet=True)

            # Apply the greedy simplification
            zx.simplify.greedy_simp(simplified_graph, lookahead=la, use_yz_phase_gadgets=True, use_xz_phase_gadgets=True, quiet=True)

            new_circuit = zx.extract_circuit(simplified_graph)

            assert zx.compare_tensors(circuit, new_circuit)

    def test_greedy_simp_neighbors(self):
            
        random.seed(1)
        circuit, graph = get_circuit_and_fr_graph(10, 20)

        for la in range(2):

            simplified_graph = graph.copy()
            # Apply the greedy simplification
            zx.simplify.teleport_reduce(simplified_graph, quiet=True)

            # Apply the greedy simplification
            zx.simplify.greedy_simp_neighbors(simplified_graph, lookahead=la, use_yz_phase_gadgets=True, use_xz_phase_gadgets=True, quiet=True)

            new_circuit = zx.extract_circuit(simplified_graph)

            assert zx.compare_tensors(circuit, new_circuit)

    def test_c_flow(self):

        random.seed(8)
        circuit, graph = get_circuit_and_fr_graph(5, 10)

        for la in range(2):

            simplified_graph = graph.copy()
            # Apply the greedy simplification
            zx.simplify.teleport_reduce(simplified_graph, quiet=True)

            # Apply the greedy simplification
            g_simp = simplified_graph.copy()
            g_simp_nu = simplified_graph.copy()
            zx.simplify.greedy_simp(g_simp, lookahead=la, flow_function=FilterFlowFunc.C_FLOW_PRESERVING, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False, quiet=True)
            zx.simplify.greedy_simp_neighbors(g_simp_nu, lookahead=la, flow_function=FilterFlowFunc.C_FLOW_PRESERVING, use_yz_phase_gadgets=False, use_xz_phase_gadgets=False, quiet=True)

            new_circuit_simp = zx.extract_circuit(g_simp)
            new_circuit_simp_nu = zx.extract_circuit(g_simp_nu)

            assert zx.compare_tensors(circuit, new_circuit_simp)
            assert zx.compare_tensors(circuit, new_circuit_simp_nu)      

    def test_phase_gadget_extraction(self):
        g = generate_graph(5, 30)
        g_init = g.clone()

        pivot_matches = pivot_matcher(g, check_for_unfusions=True, check_for_phase_gadgets=True)

        def get_matches_with_gadget(pivot_matches):
            matches = set()
            for match_key, match_values in pivot_matches.items():
                for match_value in match_values:
                    pivot_heuritstic, unfusion0, unfusion1 = match_value
                    if unfusion0 and unfusion0 == -1:
                        matches.add((match_key, match_value))
                    elif unfusion1 and unfusion1 == -1:
                        matches.add((match_key, match_value))
            return list(matches)
        
        def get_matches_without_gadget(pivot_matches):
            matches = set()
            for match_key, match_values in pivot_matches.items():
                for match_value in match_values:
                    pivot_heuritstic, unfusion0, unfusion1 = match_value
                    # if unfusion0 and unfusion1:
                    #     if unfusion0 != -1 and unfusion1 != -1:
                    #         matches.add((match_key, match_value))
                    # elif unfusion0:
                    #     if unfusion0 != -1:
                    #         matches.add((match_key, match_value))
                    # elif unfusion1:
                    #     if unfusion1 != -1:
                    #         matches.add((match_key, match_value))
                    if unfusion0 is None and unfusion1 is None:
                        matches.add((match_key, match_value))

            return list(matches)

        def calculate_gflow(graph: BaseGraph, edge=None) -> bool:
            g_clone = graph.clone()
            flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
            flow = flow_function(g_clone)
            flow = flow if flow else None
            return flow is not None
        
        # pivots_without_gadget = get_matches_without_gadget(pivot_matches)
        # apply_pivot(g, pivots_without_gadget[0], calculate_gflow)
        
        # is_graph_flow_preserving = calculate_gflow(g)
        pivot_matches = pivot_matcher(g, check_for_unfusions=True, check_for_phase_gadgets=True)
        matches = get_matches_with_gadget(pivot_matches)

        match_to_apply = None
        for match_key, match_value in matches:
            g_try = g.clone()
            unfusion_info, time_info = apply_pivot(g_try, (match_key, match_value))
            if unfusion_info:
                unfused_vertices, flow = unfusion_info
                is_flow_preserving = calculate_gflow(g_try)
                if is_flow_preserving:
                    match_to_apply = (match_key, match_value)
                    break
        
        if match_to_apply:
            apply_pivot(g, match_to_apply)
        else:
            raise Exception("No match to apply")

        assert zx.compare_tensors(g_init, t2=g)

        # architecture = create_line_architecture(g.qubit_count())
        # graph_simp = g.copy()
        # new_circuit_arch = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        new_circuit = zx.extract_circuit(g.copy())

        assert zx.compare_tensors(g_init, new_circuit)
        # assert zx.compare_tensors(g_init, new_circuit_arch)

    def test_boundary_lcomp_matches(self):

        def get_matches_with_boundaries(lcomp_matches):
            matches = []
            for match_key, match_values in lcomp_matches.items():
                for match_value in match_values:
                    lcomp_heuritstic, vertex_neighbors, unfusion_neighbor = match_value
                    if any([g.type(vertex) == zx.VertexType.BOUNDARY for vertex in vertex_neighbors]):
                        matches.append((match_key, match_value))
            return list(matches)
        
        def calculate_gflow(graph: BaseGraph, edge=None) -> bool:
            g_clone = graph.clone()
            flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
            flow = flow_function(g_clone)
            flow = flow if flow else None
            return flow is not None

        random.seed(0)
        for i in range(30):

            g = generate_graph(5, 20)
            neighbor_next_to_boundary = random.choice(list(g.neighbors(g.inputs()[0])))
            g.set_phase(neighbor_next_to_boundary, Fraction(1, 1))

            g_init = g.clone()

            lcomp_matches = lcomp_matcher(g, check_for_xz_phase_gadgets=True)
            
            matches = get_matches_with_boundaries(lcomp_matches)
            if len(matches) == 0:
                break

            match_to_apply = None
            for match_key, match_value in matches:
                g_try = g.clone()
                result = apply_lcomp(g_try, (match_key, match_value), calculate_gflow)
                if result:
                    data, time = result
                
                    if calculate_gflow(g_try):
                        match_to_apply = (match_key, match_value)
                        break

            print(f"Match {i}: {match_to_apply}")

            if match_to_apply:
                apply_lcomp(g, match_to_apply)
                assert zx.compare_tensors(g_init, t2=g)

            
        # architecture = create_line_architecture(g.qubit_count())
        # graph_simp = g.copy()
        # new_circuit_arch = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        new_circuit = zx.extract_circuit(g.copy())

        assert zx.compare_tensors(g_init, new_circuit)
        # assert zx.compare_tensors(g_init, new_circuit_arch)


    def test_gadget_lcomp_matches(self):

        random.seed(1)
        g = generate_graph(5, 20)
        g_init = g.clone()

        def get_matches_with_gadget(lcomp_matches):
            matches = []
            for match_key, match_values in lcomp_matches.items():
                for match_value in match_values:
                    lcomp_heuritstic, vertex_neighbors, unfusion_neighbor = match_value
                    if unfusion_neighbor == -1:
                        matches.append((match_key, match_value))
            return list(matches)
        
        def calculate_gflow(graph: BaseGraph, edge=None) -> bool:
            g_clone = graph.clone()
            flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
            flow = flow_function(g_clone)
            flow = flow if flow else None
            return flow is not None

        for i in range(5):

            lcomp_matches = lcomp_matcher(g, check_for_xz_phase_gadgets=True)
            
            matches = get_matches_with_gadget(lcomp_matches)

            if len(matches) == 0:
                break
            print(f"Match {i}")

            match_to_apply = None
            for match_key, match_value in matches:
                g_try = g.clone()
                unfusion_info, time_info = apply_lcomp(g_try, (match_key, match_value))
                if calculate_gflow(g_try):
                    match_to_apply = (match_key, match_value)
                    break
            
            if match_to_apply:
                apply_lcomp(g, match_to_apply)
            else:
                raise Exception("No match to apply")

            assert zx.compare_tensors(g_init, t2=g)

        # architecture = create_line_architecture(g.qubit_count())
        # graph_simp = g.copy()
        # new_circuit_arch = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        new_circuit = zx.extract_circuit(g.copy())

        assert zx.compare_tensors(g_init, new_circuit)
        # assert zx.compare_tensors(g_init, new_circuit_arch)

    def test_boundary_pivot_matches(self):
        random.seed(0)
        
        def get_matches_with_boundaries(graph, pivot_matches):
            matches = []
            for match_key, match_values in pivot_matches.items():
                for match_value in match_values:
                    pivot_heuritstic, unfusion0, unfusion1 = match_value
                    vertex_neighbors_0 = set(graph.neighbors(match_key[0]))
                    vertex_neighbors_0.remove(match_key[1])
                    vertex_neighbors_1 = set(graph.neighbors(match_key[1]))
                    vertex_neighbors_1.remove(match_key[0])
                    if any([g.type(vertex) == zx.VertexType.BOUNDARY for vertex in vertex_neighbors_0]):
                        matches.append((match_key, match_value))
                    elif any([g.type(vertex) == zx.VertexType.BOUNDARY for vertex in vertex_neighbors_1]):
                        matches.append((match_key, match_value))
            return list(matches)
        
        def calculate_gflow(graph: BaseGraph, edge=None) -> bool:
            g_clone = graph.clone()
            flow_function = FilterFlowFunc.G_FLOW_PRESERVING_GADGET
            flow = flow_function(g_clone)
            flow = flow if flow else None
            return flow is not None

        for i in range(50):

            g = generate_graph(7, 50)
            neighbor_next_to_boundary = random.choice(list(g.neighbors(g.inputs()[0])))
            g.set_phase(neighbor_next_to_boundary, Fraction(1, 1))

            g_init = g.clone()

            pivot_matches = pivot_matcher(g)
            
            matches = get_matches_with_boundaries(g, pivot_matches)
            if len(matches) == 0:
                break

            match_to_apply = None
            for match_key, match_value in matches:
                g_try = g.clone()
                result = apply_pivot(g_try, (match_key, match_value))
                
                if calculate_gflow(g_try):
                    match_to_apply = (match_key, match_value)
                    break

            print(f"Match {i}: {match_to_apply}")
            
            if match_to_apply:
                apply_pivot(g, match_to_apply)
            else:
                raise Exception("No match to apply")

            assert zx.compare_tensors(g_init, t2=g)

        # architecture = create_line_architecture(g.qubit_count())
        # graph_simp = g.copy()
        # new_circuit_arch = extract_architecture_aware_circuit(g=graph_simp, architecture=architecture, up_to_perm=True, quiet=True)

        new_circuit = zx.extract_circuit(g.copy())

        assert zx.compare_tensors(g_init, new_circuit)
        # assert zx.compare_tensors(new_circuit, new_circuit_arch)


    def test_boundary_matches(self):
        random.seed(8)
        og = generate_graph(5, 20)
        g = og.clone()

        def get_matches_with_boundaries(g:BaseGraph, matches):
            found_matches = {}
            for match_key, match_values in matches.items():
                for match_value in match_values:
                    if get_match_type((match_key, match_values[0])) == MatchType.LCOMP:
                        _, vertex_neighbors, _ = match_value
                    elif get_match_type((match_key, match_values[0])) == MatchType.PIVOT:
                        vertex_neighbors = set(g.neighbors(match_key[0])).union(set(g.neighbors(match_key[1])))
                    if any([g.type(vertex) == zx.VertexType.BOUNDARY for vertex in vertex_neighbors]):
                        new_match_values = found_matches.get(match_key, [])
                        new_match_values.append(match_value)
                        found_matches[match_key] = new_match_values
            return found_matches
        
        match_list = apply_random_matches(g, num_matches=20, flow_function=calculate_gflow_gadget, match_filter_func=get_matches_with_boundaries)
        
        assert zx.compare_tensors(og, t2=g)


    def test_gflow_functions(self):
        
        gflow_time = {"gflow": 0, "gflow_gadget": 0}

        for _ in range(10):
            g = generate_graph(10, 500)

            for _ in range(50):
                lcomp_matches = lcomp_matcher(g, check_for_xz_phase_gadgets=False)
                pivot_matches = pivot_matcher(g, check_for_phase_gadgets=False)

                if random.randint(0, 1):
                    matches = lcomp_matches
                    match_type = MatchType.LCOMP
                else:
                    matches = pivot_matches
                    match_type = MatchType.PIVOT

                while len(matches) > 0:
                    match = random.sample(list(matches.items()), 1)
                    match_key, match_values = match[0]

                    graph_copy = g.clone()

                    if match_type == MatchType.LCOMP:
                        match_value = random.sample(match_values, 1)
                        match_result_with_time = apply_lcomp(graph_copy, (match_key, match_value[0]))
                    elif match_type == MatchType.PIVOT:
                        match_value = random.sample(match_values, 1)
                        match_result_with_time = apply_pivot(graph_copy, (match_key, match_value[0]))
                    
                    time_gflow = time.perf_counter()
                    is_gflow = calculate_gflow(graph=graph_copy)
                    time_gflow = time.perf_counter() - time_gflow

                    time_gflow_gadget = time.perf_counter()
                    is_gflow_gadget = calculate_gflow_gadget(graph=graph_copy)
                    time_gflow_gadget = time.perf_counter() - time_gflow_gadget

                    gflow_time["gflow"] += time_gflow
                    gflow_time["gflow_gadget"] += time_gflow_gadget

                    assert is_gflow == is_gflow_gadget

                    if is_gflow and is_gflow_gadget:
                        if match_type == MatchType.LCOMP:
                            apply_lcomp(g, (match_key, match_value[0]))
                        elif match_type == MatchType.PIVOT:
                            apply_pivot(g, (match_key, match_value[0]))
                        break

            print("Gflow time: ", gflow_time)


    def test_heuristic_values(self):
        random.seed(8)
        og = generate_graph(6, 50)
        g = og.clone()
        
        for i in range(300):

            original_eges = g.num_edges()
            match_list = apply_random_matches(g, num_matches=1, flow_function=calculate_gflow_gadget)
            new_edges = g.num_edges()

            match_key, match_value = match_list[0]
            if get_match_type((match_key, match_value)) == MatchType.LCOMP:
                result = (match_value[2])
            elif get_match_type((match_key, match_value)) == MatchType.PIVOT:
                result = (match_value[1], match_value[2])

            heuristic_value = match_value[0]
            # print(f"Match {i}: {match_key}, {match_value}, Edge difference: {original_eges - new_edges}")
            print('Match {:2s}, {:20s} {:7s}/ {:7s} {}'.format(str(i), str(match_key), str(heuristic_value), str(original_eges - new_edges), str(result)))

            assert heuristic_value == original_eges - new_edges



        
# if __name__ == '__main__':
#     # pytest.main()