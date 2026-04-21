"""
Unit tests for groupAfile.py

Tests cover all major components:
    - Union_Find        (disjoint-set cycle detection)
    - Network_Simulation (end-to-end route planning)

Note: groupAfile.py's Network_Simulation references helper functions
(edges_from_path, dijkstra_search, trace_path, calculate_cost,
find_k_simple_paths) that are not defined inside the file. These are
injected into the module's namespace from codefilesgroupB.py before
any simulation tests run, which mirrors how the code would be used in
a combined project.

Running the tests:
    Built-in runner (verbose):
        python -m unittest test_groupAfile.py -v

    pytest (recommended — colored output, better failure diffs):
        pip install pytest
        pytest test_groupAfile.py -v
"""

import unittest
import groupAfile
from groupAfile import Union_Find, Network_Simulation
from codefilesgroupB import (
    WeightedDiGraph,
    Dijkstra,
    reconstruct_path,
    path_cost,
    path_to_edges,
    k_shortest_simple_paths,
)

# -----------------------------------------------------------------------
# Inject helper functions into groupAfile's global namespace so that
# Network_Simulation.run_simulation() (and related methods) can resolve
# the names it calls at runtime.
# -----------------------------------------------------------------------
groupAfile.edges_from_path = path_to_edges
groupAfile.dijkstra_search = Dijkstra
groupAfile.trace_path = reconstruct_path
groupAfile.calculate_cost = path_cost
groupAfile.find_k_simple_paths = k_shortest_simple_paths


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
# Network_Simulation Tests
# =====================================================================

class TestNetworkSimulation(unittest.TestCase):

    def setUp(self):
        self.g = make_sparse_graph()
        self.labels = ["Boston", "Chicago", "Denver", "Phoenix", "Portland", "San Diego"]

    # --- check_acyclic_path ---

    def test_check_acyclic_path_safe_on_fresh_tracker(self):
        """Any path should be cycle-free when no routes have been committed yet."""
        sim = Network_Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        self.assertTrue(sim.check_acyclic_path(tracker, [0, 1, 2]))

    def test_check_acyclic_path_detects_cycle(self):
        """
        After committing edge 0->1, the reverse path [1, 0] closes a cycle
        because both cities are already in the same component.
        """
        sim = Network_Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1])
        self.assertFalse(sim.check_acyclic_path(tracker, [1, 0]))

    def test_check_acyclic_does_not_mutate_tracker(self):
        """
        check_acyclic_path works on a clone internally and must leave the
        original tracker completely untouched.
        """
        sim = Network_Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.check_acyclic_path(tracker, [0, 1, 2])
        # Cities 0 and 1 must still appear as separate components
        self.assertNotEqual(tracker.find_root(0), tracker.find_root(1))

    # --- commit_route ---

    def test_commit_route_merges_all_cities_on_path(self):
        """After committing [0, 1, 2], all three cities must share a component."""
        sim = Network_Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1, 2])
        self.assertEqual(tracker.find_root(0), tracker.find_root(2))

    def test_commit_route_does_not_affect_unrelated_cities(self):
        """Cities outside the committed path must remain in their own components."""
        sim = Network_Simulation(self.g, [], self.labels)
        tracker = Union_Find(self.g.V)
        sim.commit_route(tracker, [0, 1])
        # City 4 was never touched
        self.assertNotEqual(tracker.find_root(0), tracker.find_root(4))

    # --- format_route ---

    def test_format_route_produces_arrow_separated_names(self):
        sim = Network_Simulation(self.g, [], self.labels)
        self.assertEqual(sim.format_route([0, 1, 2]), "Boston -> Chicago -> Denver")

    def test_format_route_single_city(self):
        sim = Network_Simulation(self.g, [], self.labels)
        self.assertEqual(sim.format_route([3]), "Phoenix")

    # --- run_simulation ---

    def test_run_simulation_total_cost(self):
        """
        Known result for the sparse graph tickets:
            0 -> 2   via 1:  [0,1,2] cost = 7
            2 -> 3   direct: [2,3]   cost = 2
            4 -> 5   direct: [4,5]   cost = 5
            Total = 14
        """
        tickets = [(0, 2), (2, 3), (4, 5)]
        sim = Network_Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(result["total_cost"], 14)

    def test_run_simulation_picks_cheapest_valid_path(self):
        """
        0->2: the direct route (cost 9) must be skipped in favour of
        the cheaper multi-hop route via city 1 (cost 7).
        """
        tickets = [(0, 2)]
        sim = Network_Simulation(self.g, tickets, self.labels)
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
        sim = Network_Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(len(result["paths"]), 2)

    def test_run_simulation_returns_correct_path_count(self):
        """One path entry must be returned per ticket."""
        tickets = [(0, 2), (2, 3), (4, 5)]
        sim = Network_Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIsNotNone(result)
        self.assertEqual(len(result["paths"]), 3)

    def test_run_simulation_returns_none_for_impossible_ticket(self):
        """If a ticket has no valid path at all, run_simulation must return None."""
        empty_g = WeightedDiGraph(3)  # No edges at all
        sim = Network_Simulation(empty_g, [(0, 1)])
        self.assertIsNone(sim.run_simulation())

    def test_run_simulation_uses_numeric_labels_when_none_given(self):
        """When no city labels are provided, integer labels should be used."""
        simple_g = WeightedDiGraph(3)
        simple_g.add_edge(0, 1, 5)
        simple_g.add_edge(1, 2, 3)
        sim = Network_Simulation(simple_g, [(0, 2)])
        result = sim.run_simulation()
        self.assertIsNotNone(result)

    def test_run_simulation_result_structure(self):
        """The returned dictionary must contain 'paths' and 'total_cost' keys."""
        tickets = [(4, 5)]
        sim = Network_Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        self.assertIn("paths", result)
        self.assertIn("total_cost", result)

    def test_run_simulation_each_path_entry_structure(self):
        """
        Each entry in result['paths'] should be a 3-tuple:
        ((src, dst), path_list, cost).
        """
        tickets = [(0, 2)]
        sim = Network_Simulation(self.g, tickets, self.labels)
        result = sim.run_simulation()
        entry = result["paths"][0]
        (src, dst), path, cost = entry
        self.assertEqual(src, 0)
        self.assertEqual(dst, 2)
        self.assertIsInstance(path, list)
        self.assertIsInstance(cost, (int, float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
