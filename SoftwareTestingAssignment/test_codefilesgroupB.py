"""
Unit tests for codefilesgroupB.py

Tests cover all major components:
    - WeightedDiGraph  (graph construction and queries)
    - Dijkstra         (shortest-path distances and predecessor array)
    - reconstruct_path (path rebuilding from predecessor array)
    - path_cost        (edge-weight summation)
    - path_to_edges    (path list -> edge list conversion)
    - k_shortest_simple_paths (k-best path enumeration)

Running the tests:
    Built-in runner (verbose):
        python -m unittest test_codefilesgroupB.py -v

    pytest (recommended — colored output, better failure diffs):
        pip install pytest
        pytest test_codefilesgroupB.py -v
"""

import unittest
from codefilesgroupB import (
    WeightedDiGraph,
    Dijkstra,
    reconstruct_path,
    path_cost,
    path_to_edges,
    k_shortest_simple_paths,
)


# -----------------------------------------------------------------------
# Shared helper: the deterministic sparse graph used across many tests
#
#   Component 1 (cities 0-3):
#       0->1 (4), 0->2 (9), 1->2 (3), 1->3 (7), 2->3 (2), 3->0 (8)
#   Component 2 (cities 4-5, isolated from component 1):
#       4->5 (5), 5->4 (5)
# -----------------------------------------------------------------------
def make_sparse_graph():
    g = WeightedDiGraph(6)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 9)
    g.add_edge(1, 2, 3)
    g.add_edge(1, 3, 7)
    g.add_edge(2, 3, 2)
    g.add_edge(3, 0, 8)
    g.add_edge(4, 5, 5)
    g.add_edge(5, 4, 5)
    return g


# =====================================================================
# WeightedDiGraph Tests
# =====================================================================

class TestWeightedDiGraph(unittest.TestCase):

    def test_init_creates_zero_matrix(self):
        """A fresh graph should have all zeros in its adjacency matrix."""
        g = WeightedDiGraph(3)
        self.assertEqual(g.V, 3)
        for i in range(3):
            for j in range(3):
                self.assertEqual(g.graph[i][j], 0)

    def test_add_edge_sets_weight(self):
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 10)
        self.assertEqual(g.graph[0][1], 10)

    def test_add_edge_is_directed(self):
        """Adding edge (0->1) must NOT automatically create the reverse (1->0)."""
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 10)
        self.assertEqual(g.graph[1][0], 0)

    def test_get_edge_returns_weight(self):
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 7)
        self.assertEqual(g.get_edge(0, 1), 7)

    def test_get_edge_returns_none_when_missing(self):
        g = WeightedDiGraph(3)
        self.assertIsNone(g.get_edge(0, 1))

    def test_isEdge_true(self):
        g = WeightedDiGraph(3)
        g.add_edge(0, 2, 5)
        self.assertTrue(g.isEdge(0, 2))

    def test_isEdge_false_no_edge(self):
        g = WeightedDiGraph(3)
        self.assertFalse(g.isEdge(0, 2))

    def test_isEdge_false_out_of_range(self):
        g = WeightedDiGraph(3)
        self.assertFalse(g.isEdge(0, 99))

    def test_isVertex_valid_bounds(self):
        g = WeightedDiGraph(4)
        self.assertTrue(g.isVertex(0))
        self.assertTrue(g.isVertex(3))

    def test_isVertex_out_of_bounds(self):
        g = WeightedDiGraph(4)
        self.assertFalse(g.isVertex(-1))
        self.assertFalse(g.isVertex(4))

    def test_get_neighbors_returns_correct_pairs(self):
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 4)
        g.add_edge(0, 2, 9)
        neighbors = g.get_neighbors(0)
        self.assertIn((1, 4), neighbors)
        self.assertIn((2, 9), neighbors)
        self.assertEqual(len(neighbors), 2)

    def test_get_neighbors_invalid_vertex_returns_empty(self):
        g = WeightedDiGraph(3)
        self.assertEqual(g.get_neighbors(99), [])

    def test_get_neighbors_isolated_node_returns_empty(self):
        g = WeightedDiGraph(3)
        self.assertEqual(g.get_neighbors(1), [])

    def test_get_edges_returns_all_edges(self):
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 4)
        g.add_edge(1, 2, 3)
        edges = g.get_edges()
        self.assertIn((0, 1, 4), edges)
        self.assertIn((1, 2, 3), edges)
        self.assertEqual(len(edges), 2)

    def test_get_edges_empty_graph(self):
        g = WeightedDiGraph(3)
        self.assertEqual(g.get_edges(), [])

    def test_add_node_increments_vertex_count(self):
        g = WeightedDiGraph(3)
        g.add_node()
        self.assertEqual(g.V, 4)

    def test_add_node_expands_matrix_dimensions(self):
        g = WeightedDiGraph(3)
        g.add_node()
        self.assertEqual(len(g.graph), 4)
        for row in g.graph:
            self.assertEqual(len(row), 4)

    def test_connect_all_fills_every_off_diagonal_entry(self):
        """After connect_all, every city pair (i != j) must have a positive weight."""
        g = WeightedDiGraph(5)
        g.connect_all()
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertGreater(g.graph[i][j], 0)
                else:
                    self.assertEqual(g.graph[i][j], 0)  # No self-loops

    def test_connect_all_does_not_overwrite_existing_edges(self):
        """Pre-existing edge weights should be preserved by connect_all."""
        g = WeightedDiGraph(3)
        g.add_edge(0, 1, 99)
        g.connect_all()
        self.assertEqual(g.graph[0][1], 99)


# =====================================================================
# Dijkstra Tests
# =====================================================================

class TestDijkstra(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()

    def test_start_city_distance_is_zero(self):
        dist, _ = Dijkstra(self.g, 0)
        self.assertEqual(dist[0], 0)

    def test_known_shortest_distances_from_city_0(self):
        """
        From city 0 in the sparse graph:
            0->1        = 4
            0->1->2     = 7  (cheaper than the direct 0->2 edge with cost 9)
            0->1->2->3  = 9
        """
        dist, _ = Dijkstra(self.g, 0)
        self.assertEqual(dist[1], 4)
        self.assertEqual(dist[2], 7)
        self.assertEqual(dist[3], 9)

    def test_unreachable_cities_remain_infinite(self):
        """Cities 4 and 5 are isolated from city 0, so distance stays infinity."""
        dist, _ = Dijkstra(self.g, 0)
        self.assertEqual(dist[4], float("inf"))
        self.assertEqual(dist[5], float("inf"))

    def test_predecessor_array_reflects_cheapest_route(self):
        """
        On the cheapest path 0->1->2, city 2's predecessor should be 1,
        and city 1's predecessor should be 0.
        """
        _, prev = Dijkstra(self.g, 0)
        self.assertEqual(prev[2], 1)
        self.assertEqual(prev[1], 0)

    def test_start_predecessor_is_none(self):
        """The starting city has no predecessor."""
        _, prev = Dijkstra(self.g, 0)
        self.assertIsNone(prev[0])

    def test_single_node_graph(self):
        g = WeightedDiGraph(1)
        dist, prev = Dijkstra(g, 0)
        self.assertEqual(dist[0], 0)
        self.assertIsNone(prev[0])


# =====================================================================
# reconstruct_path Tests
# =====================================================================

class TestReconstructPath(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()
        _, self.prev = Dijkstra(self.g, 0)

    def test_same_start_and_end_returns_single_element(self):
        path = reconstruct_path(self.prev, 0, 0)
        self.assertEqual(path, [0])

    def test_multi_hop_path(self):
        """0->2 should be routed as [0, 1, 2] (cheaper via city 1)."""
        path = reconstruct_path(self.prev, 0, 2)
        self.assertEqual(path, [0, 1, 2])

    def test_path_starts_at_source(self):
        path = reconstruct_path(self.prev, 0, 3)
        self.assertEqual(path[0], 0)

    def test_path_ends_at_destination(self):
        path = reconstruct_path(self.prev, 0, 3)
        self.assertEqual(path[-1], 3)

    def test_unreachable_destination_returns_none(self):
        """City 4 is unreachable from city 0."""
        path = reconstruct_path(self.prev, 0, 4)
        self.assertIsNone(path)


# =====================================================================
# path_cost Tests
# =====================================================================

class TestPathCost(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()

    def test_multi_hop_cost(self):
        """[0, 1, 2] on the sparse graph: 4 + 3 = 7."""
        self.assertEqual(path_cost(self.g, [0, 1, 2]), 7)

    def test_single_hop_direct_edge(self):
        """The direct edge [0, 2] has cost 9."""
        self.assertEqual(path_cost(self.g, [0, 2]), 9)

    def test_single_node_path_is_zero_cost(self):
        """A path with only one city uses no edges, so cost is 0."""
        self.assertEqual(path_cost(self.g, [0]), 0)

    def test_longer_path_cost(self):
        """[0, 1, 2, 3]: 4 + 3 + 2 = 9."""
        self.assertEqual(path_cost(self.g, [0, 1, 2, 3]), 9)


# =====================================================================
# path_to_edges Tests
# =====================================================================

class TestPathToEdges(unittest.TestCase):

    def test_multi_hop_produces_correct_edges(self):
        edges = path_to_edges([0, 1, 2, 3])
        self.assertEqual(edges, [(0, 1), (1, 2), (2, 3)])

    def test_single_hop_produces_one_edge(self):
        edges = path_to_edges([0, 1])
        self.assertEqual(edges, [(0, 1)])

    def test_single_node_produces_no_edges(self):
        edges = path_to_edges([0])
        self.assertEqual(edges, [])

    def test_edge_count_equals_path_length_minus_one(self):
        path = [0, 1, 2, 3, 4]
        self.assertEqual(len(path_to_edges(path)), len(path) - 1)


# =====================================================================
# k_shortest_simple_paths Tests
# =====================================================================

class TestKShortestSimplePaths(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()

    def test_cheapest_path_comes_first(self):
        """0->2: cheapest path is [0,1,2] at cost 7."""
        results = k_shortest_simple_paths(self.g, 0, 2, k=5)
        self.assertGreater(len(results), 0)
        cost, path = results[0]
        self.assertEqual(cost, 7)
        self.assertEqual(path, [0, 1, 2])

    def test_second_path_is_more_expensive(self):
        """The second-best path 0->2 is the direct edge [0,2] at cost 9."""
        results = k_shortest_simple_paths(self.g, 0, 2, k=5)
        self.assertGreaterEqual(len(results), 2)
        cost, path = results[1]
        self.assertEqual(cost, 9)
        self.assertEqual(path, [0, 2])

    def test_paths_are_ordered_by_cost(self):
        """Results must be in non-decreasing order of cost."""
        results = k_shortest_simple_paths(self.g, 0, 3, k=10)
        costs = [c for c, _ in results]
        self.assertEqual(costs, sorted(costs))

    def test_k_limits_number_of_results(self):
        results = k_shortest_simple_paths(self.g, 0, 2, k=1)
        self.assertEqual(len(results), 1)

    def test_no_path_returns_empty_list(self):
        """City 0 cannot reach city 4; no paths should be returned."""
        results = k_shortest_simple_paths(self.g, 0, 4, k=5)
        self.assertEqual(results, [])

    def test_all_returned_paths_start_and_end_correctly(self):
        results = k_shortest_simple_paths(self.g, 0, 3, k=10)
        for cost, path in results:
            self.assertEqual(path[0], 0)
            self.assertEqual(path[-1], 3)

    def test_returned_paths_are_simple(self):
        """No city should appear more than once in any returned path."""
        results = k_shortest_simple_paths(self.g, 0, 3, k=10)
        for cost, path in results:
            self.assertEqual(len(path), len(set(path)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
