"""
Unit tests for integrate.py

Tests cover all major components:
    - WeightedDiGraph  (graph construction and queries)
    - Dijkstra         (shortest-path distances and predecessor array)
    - reconstruct_path (path rebuilding from predecessor array)
    - path_cost        (edge-weight summation)
    - path_to_edges    (path list -> edge list conversion)
    - k_shortest_simple_paths (k-best path enumeration)
    - Union_Find       (cycle detection via disjoint sets)
    - Simulation       (end-to-end route planning)

Running the tests:
    Built-in runner (verbose):
        python -m unittest test_integrate.py -v

    pytest (recommended — colored output, better failure diffs):
        pip install pytest
        pytest test_integrate.py -v
"""

import unittest
from integrate import (
    WeightedDiGraph,
    Dijkstra,
    reconstruct_path,
    path_cost,
    path_to_edges,
    k_shortest_simple_paths,
    Union_Find,
    Simulation,
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
        """Boston->Denver should be routed as [0, 1, 2] (via Chicago)."""
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
        """Boston->Denver: cheapest path is [0,1,2] at cost 7."""
        results = k_shortest_simple_paths(self.g, 0, 2, k=5)
        self.assertGreater(len(results), 0)
        cost, path = results[0]
        self.assertEqual(cost, 7)
        self.assertEqual(path, [0, 1, 2])

    def test_second_path_is_more_expensive(self):
        """The second-best path Boston->Denver is the direct edge [0,2] at cost 9."""
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


# =====================================================================
# Union_Find Tests
# =====================================================================

class TestUnionFind(unittest.TestCase):

    def test_initial_roots_are_self(self):
        """Each city starts as its own root — all components are disjoint."""
        uf = Union_Find(5)
        for i in range(5):
            self.assertEqual(uf.find_root(i), i)

    def test_merge_different_sets_returns_true(self):
        uf = Union_Find(4)
        self.assertTrue(uf.merge_nodes(0, 1))

    def test_merge_same_set_returns_false(self):
        """Attempting to merge two already-connected cities signals a cycle."""
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        self.assertFalse(uf.merge_nodes(0, 1))

    def test_merged_nodes_share_root(self):
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        self.assertEqual(uf.find_root(0), uf.find_root(1))

    def test_transitive_connection(self):
        """After merging 0-1 and 1-2, cities 0 and 2 should share a root."""
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        uf.merge_nodes(1, 2)
        self.assertEqual(uf.find_root(0), uf.find_root(2))

    def test_unmerged_nodes_have_different_roots(self):
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        # City 2 was never touched
        self.assertNotEqual(uf.find_root(0), uf.find_root(2))

    def test_clone_is_independent(self):
        """Changes to a clone must not affect the original."""
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        clone = uf.clone()
        clone.merge_nodes(2, 3)
        # Original should not reflect the 2-3 merge done on the clone
        self.assertNotEqual(uf.find_root(2), uf.find_root(3))

    def test_clone_preserves_existing_state(self):
        """The clone should start with the same component assignments."""
        uf = Union_Find(4)
        uf.merge_nodes(0, 1)
        clone = uf.clone()
        self.assertEqual(clone.find_root(0), clone.find_root(1))

    def test_path_compression_does_not_change_logical_structure(self):
        """Path compression is an optimization; roots should stay consistent."""
        uf = Union_Find(5)
        uf.merge_nodes(0, 1)
        uf.merge_nodes(1, 2)
        uf.merge_nodes(2, 3)
        root = uf.find_root(3)
        # After path compression, all nodes should still resolve to the same root
        self.assertEqual(uf.find_root(0), root)
        self.assertEqual(uf.find_root(1), root)
        self.assertEqual(uf.find_root(2), root)


# =====================================================================
# Simulation Tests
# =====================================================================

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()
        self.labels = ["Boston", "Chicago", "Denver", "Phoenix", "Portland", "San Diego"]

    # --- check_acyclic_path ---

    def test_check_acyclic_path_safe_on_fresh_tracker(self):
        """Any path should be cycle-free when no routes have been committed yet."""
        sim = Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        self.assertTrue(sim.check_acyclic_path(tracker, [0, 1, 2]))

    def test_check_acyclic_path_detects_cycle(self):
        """
        After committing edge 0->1, the reverse path [1, 0] closes a cycle
        because both cities are already in the same component.
        """
        sim = Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1])
        self.assertFalse(sim.check_acyclic_path(tracker, [1, 0]))

    def test_check_acyclic_does_not_mutate_tracker(self):
        """
        check_acyclic_path works on a clone internally and must leave the
        original tracker completely untouched.
        """
        sim = Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.check_acyclic_path(tracker, [0, 1, 2])
        # Cities 0 and 1 must still appear as separate components
        self.assertNotEqual(tracker.find_root(0), tracker.find_root(1))

    # --- commit_route ---

    def test_commit_route_merges_all_cities_on_path(self):
        """After committing [0, 1, 2], all three cities must share a component."""
        sim = Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1, 2])
        self.assertEqual(tracker.find_root(0), tracker.find_root(2))

    def test_commit_route_does_not_affect_unrelated_cities(self):
        """Cities outside the committed path must remain in their own components."""
        sim = Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1])
        # City 4 was never touched
        self.assertNotEqual(tracker.find_root(0), tracker.find_root(4))

    # --- format_route ---

    def test_format_route_produces_arrow_separated_names(self):
        sim = Simulation(self.g, [], self.labels)
        self.assertEqual(sim.format_route([0, 1, 2]), "Boston -> Chicago -> Denver")

    def test_format_route_single_city(self):
        sim = Simulation(self.g, [], self.labels)
        self.assertEqual(sim.format_route([3]), "Phoenix")

    # --- run_simulation ---

    def test_run_simulation_total_cost(self):
        """
        Known result for the sparse graph tickets:
            Boston -> Denver   via Chicago:  [0,1,2] cost = 7
            Denver -> Phoenix  direct:       [2,3]   cost = 2
            Portland -> San Diego direct:    [4,5]   cost = 5
            Total = 14
        """
        tickets = [(0, 2), (2, 3), (4, 5)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(result["total_cost"], 14)

    def test_run_simulation_picks_cheapest_valid_path(self):
        """
        Boston->Denver: the direct route (cost 9) must be skipped in favour of
        the cheaper multi-hop route via Chicago (cost 7).
        """
        tickets = [(0, 2)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        _, chosen_path, chosen_cost = result["paths"][0]
        self.assertEqual(chosen_cost, 7)
        self.assertEqual(chosen_path, [0, 1, 2])

    def test_run_simulation_avoids_cycles_across_tickets(self):
        """
        Routing tickets (0->2) then (2->3) must not create a cycle even though
        city 2 appears in both paths.
        """
        tickets = [(0, 2), (2, 3)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(len(result["paths"]), 2)

    def test_run_simulation_returns_correct_path_count(self):
        """One path entry must be returned per ticket."""
        tickets = [(0, 2), (2, 3), (4, 5)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(len(result["paths"]), 3)

    def test_run_simulation_returns_none_for_impossible_ticket(self):
        """If a ticket has no valid path at all, run_simulation must return None."""
        empty_g = WeightedDiGraph(3)  # No edges
        sim = Simulation(empty_g, [(0, 1)])
        self.assertIsNone(sim.run_simulation())

    def test_run_simulation_uses_numeric_labels_when_none_given(self):
        """When no city labels are provided, integer labels should be used."""
        simple_g = WeightedDiGraph(3)
        simple_g.add_edge(0, 1, 5)
        simple_g.add_edge(1, 2, 3)
        sim = Simulation(simple_g, [(0, 2)])
        result = sim.run_simulation()
        self.assertIsNotNone(result)

    def test_run_simulation_result_structure(self):
        """The returned dictionary must contain 'paths' and 'total_cost' keys."""
        tickets = [(4, 5)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIn("paths", result)
        self.assertIn("total_cost", result)

    def test_run_simulation_each_path_entry_structure(self):
        """
        Each entry in result['paths'] should be a 3-tuple:
        ((src, dst), path_list, cost).
        """
        tickets = [(0, 2)]
        sim = Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        entry = result["paths"][0]
        (src, dst), path, cost = entry
        self.assertEqual(src, 0)
        self.assertEqual(dst, 2)
        self.assertIsInstance(path, list)
        self.assertIsInstance(cost, (int, float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
