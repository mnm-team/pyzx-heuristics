import sys
if __name__ == '__main__':
    sys.path.append('..')
    sys.path.append('.')

import random
import unittest
import pyzx as zx
from pyzx.heuristics.neighbor_unfusion_simplification import apply_lcomp, apply_pivot, lcomp_matcher, pivot_matcher, update_matches

def deep_tuple(lst):
    return tuple(deep_tuple(i) if isinstance(i, list) or isinstance(i, tuple) else i for i in lst)

class TestUpdateMatches(unittest.TestCase):

    def test_update_matches(self):
        # Create a test graph
        graph = zx.generate.cliffordT(qubits=10, depth=500)
        lcomp_matches = lcomp_matcher(graph)
        pivot_matches = pivot_matcher(graph)

        for _ in range(10):

            match = random.choice(list({**lcomp_matches, **pivot_matches}.items()))
            match_key, match_value = match

            if len(match_key) == 2:
                vertex_neighbors = set()
                for vertex in match_key:
                    for vertex_neighbor in graph.neighbors(vertex):
                        if vertex_neighbor not in match_key:
                            vertex_neighbors.add(vertex_neighbor)
                match_result = apply_pivot(graph=graph, match=match)

            elif len(match_key) == 1:
                _, vertex_neighbors, _ = match_value
                match_result = apply_lcomp(graph=graph, match=match)

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

if __name__ == '__main__':
    unittest.main()