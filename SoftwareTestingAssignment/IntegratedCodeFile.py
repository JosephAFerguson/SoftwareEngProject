#Done by Joe Ferguson
import heapq
import random

"""
Ticket to Ride Style Railway Network Path Planning (Simplified Version)

Problem Statement:
    You're given a set of tickets representing railway routes that need to be
    connected (source city to destination city). You also have a weighted
    directed graph representing cities and available rail routes with costs.

Objective:
    Find an optimal railway network such that:
    1. Every ticket's source and destination cities are connected
    2. The total cost of all routes is minimized
    3. The final network contains NO CYCLES (must form a forest/acyclic subgraph)

Key Challenge:
    If selecting the shortest path for a ticket would create a cycle in the
    overall network, we must explore alternative paths (2nd shortest, 3rd shortest,
    etc.) until we find one that maintains the acyclic property.

Algorithms Employed:
    - Dijkstra's Algorithm: Finds shortest paths between cities
    - Union-Find (Disjoint Set Union): Detects and prevents cycles
"""

# =====================================================================
# GRAPH DATA STRUCTURE AND DIJKSTRA'S ALGORITHM
# =====================================================================

class WeightedDiGraph:
    """
    Represents a weighted directed graph using an adjacency matrix.
    Cities are represented as numeric indices (0 to V-1).
    A value of 0 in the matrix means no direct edge exists between two cities.
    """

    def __init__(self, vertices):
        """Initialize graph with given number of vertices (cities)."""
        self.V = vertices
        # Adjacency matrix: graph[i][j] = weight from city i to city j
        # (0 means no direct edge exists)
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def connect_all(self):
        """
        Create a fully connected graph by adding random-weighted edges
        between all pairs of cities that don't already have edges.
        Edge weights are randomly chosen between 1 and 50.
        """
        for i in range(self.V):
            for j in range(self.V):
                # Skip self-loops and edges that already exist
                if i != j and self.graph[i][j] == 0:
                    self.graph[i][j] = random.randint(1, 50)

    def add_edge(self, u, v, w):
        """Add a weighted directed edge from city u to city v with cost w."""
        self.graph[u][v] = w

    def add_node(self):
        """Add a new city (vertex) to the graph."""
        self.V += 1
        # Expand each row of the matrix by one column
        for row in self.graph:
            row.append(0)
        # Add a new row for the new vertex
        self.graph.append([0 for _ in range(self.V)])

    def get_edge(self, u, v):
        """Get the weight of edge from u to v, or None if no edge exists."""
        if not self.isEdge(u, v):
            return None
        return self.graph[u][v]

    def get_edges(self):
        """
        Get all edges in the graph.
        Returns a list of tuples: (source, destination, weight)
        """
        edges = []
        for i in range(self.V):
            for j in range(self.V):
                # Only include edges that actually exist (weight != 0)
                if self.graph[i][j] != 0:
                    edges.append((i, j, self.graph[i][j]))
        return edges

    def get_neighbors(self, u):
        """
        Get all cities directly reachable from city u.
        Returns a list of tuples: (destination_city, edge_weight)
        """
        if not self.isVertex(u):
            return []

        neighbors = []
        for v in range(self.V):
            # Include v if there's an edge from u to v
            if self.graph[u][v] != 0:
                neighbors.append((v, self.graph[u][v]))
        return neighbors

    def isVertex(self, u):
        """Check if u is a valid city index."""
        return 0 <= u < self.V

    def isEdge(self, u, v):
        """Check if a directed edge exists from city u to city v."""
        return 0 <= u < self.V and 0 <= v < self.V and self.graph[u][v] != 0

def Dijkstra(graph, start):
    """
    Dijkstra's shortest path algorithm.

    Finds the minimum-cost path from 'start' to all other cities in the graph.

    Returns:
        - dist: Array where dist[i] = minimum cost to reach city i from start
        - prev: Array where prev[i] = predecessor city on the shortest path to i
                (used to reconstruct the actual path)
    """
    # Initialize distances to infinity, except for the starting city
    dist = [float("inf")] * graph.V
    prev = [None] * graph.V
    dist[start] = 0

    # Priority queue stores (distance_from_start, city)
    # This ensures we always process the nearest unvisited city next
    pq = [(0, start)]

    while pq:
        curr_dist, u = heapq.heappop(pq)

        # Skip outdated entries - we may have found a better path to u already
        if curr_dist > dist[u]:
            continue

        # Check all neighbors of current city
        for v, w in graph.get_neighbors(u):
            new_dist = curr_dist + w
            # If we found a better path to v, update it
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev

def reconstruct_path(prev, start, end):
    """
    Reconstruct the shortest path from 'start' to 'end' using the predecessor array.
    The predecessor array is typically generated by Dijkstra's algorithm.

    Returns:
        A list of cities representing the path from start to end,
        or None if no path exists.
    """
    if start == end:
        return [start]

    path = []
    curr = end
    while curr is not None:
        # Walk backward from destination to source using the predecessor links
        path.append(curr)
        if curr == start:
            break
        curr = prev[curr]

    # Reverse to get the path in forward order (start -> ... -> end)
    path.reverse()

    # Verify we actually found a valid path
    if not path or path[0] != start:
        return None
    return path

def path_cost(graph, path):
    """
    Calculate the total cost of a path by summing all edge weights.

    Args:
        graph: The WeightedDiGraph to look up edge weights
        path: List of cities forming the path

    Returns:
        The total cost of the path
    """
    total = 0
    for i in range(len(path) - 1):
        # Add the cost of each edge: (city[i] -> city[i+1])
        total += graph.get_edge(path[i], path[i + 1])
    return total


def path_to_edges(path):
    """
    Convert a path (list of cities) into a list of edges.

    Args:
        path: List of cities [c1, c2, c3, ...]

    Returns:
        List of (source, destination) pairs [(c1, c2), (c2, c3), ...]
    """
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def k_shortest_simple_paths(graph, source, destination, k=10):
    """
    Find up to k shortest paths from source to destination.
    Each path is SIMPLE (no repeated cities) to avoid infinite loops.

    This uses a best-first search strategy:
    - Explore paths in order of total cost (cheapest first)
    - Never include a city twice in the same path
    - Stop once we've found k complete paths

    Args:
        graph: The WeightedDiGraph to search
        source: Starting city
        destination: Target city
        k: Maximum number of paths to find

    Returns:
        List of (total_cost, path) tuples, sorted by cost
    """
    # Priority queue: (cost_so_far, path_as_list)
    # We always explore the cheapest incomplete path next
    pq = [(0, [source])]
    results = []

    while pq and len(results) < k:
        cost, path = heapq.heappop(pq)
        u = path[-1]  # Current city (last city in path)

        # If we reached the destination, add this complete path to results
        if u == destination:
            results.append((cost, path))
            continue

        # Try extending the path by visiting each neighbor
        for v, w in graph.get_neighbors(u):
            # Only extend if v is not already in the path (avoid cycles)
            if v not in path:
                new_path = path + [v]
                heapq.heappush(pq, (cost + w, new_path))

    return results

# =====================================================================
# INTEGRATION WRAPPER FUNCTIONS
# These functions bridge the naming conventions between the two halves
# =====================================================================

def dijkstra_search(graph, start):
    """Wrapper for Dijkstra's algorithm."""
    return Dijkstra(graph, start)


def trace_path(prev, start, end):
    """Wrapper for path reconstruction."""
    return reconstruct_path(prev, start, end)


def calculate_cost(graph, path):
    """Wrapper for path cost calculation."""
    return path_cost(graph, path)


def edges_from_path(path):
    """Wrapper for path-to-edges conversion."""
    return path_to_edges(path)


def find_k_simple_paths(graph, source, destination, k=10):
    """Wrapper for k-shortest paths algorithm."""
    return k_shortest_simple_paths(graph, source, destination, k)


# =====================================================================
# CYCLE DETECTION AND NETWORK SIMULATION
# =====================================================================

class Union_Find:
    """
    Union-Find (Disjoint Set Union) data structure.

    Used to efficiently detect cycles and manage connected components.
    In this problem, we use it to track which cities are already connected
    in the current railway network.

    Two cities are in the same component (connected set) if there's already
    a path between them. Adding an edge between two cities in the same
    component would create a cycle.
    """

    def __init__(self, n):
        """Initialize with n disjoint sets (each city is its own component)."""
        self.parentNode = list(range(n))  # Each city is initially its own parent
        self.nodeRank = [0] * n  # Rank for union by rank optimization

    def find_root(self, u):
        """
        Find the root (representative) of the set containing city u.
        Uses path compression for efficiency.
        """
        if self.parentNode[u] != u:
            # Path compression: make u point directly to the root
            self.parentNode[u] = self.find_root(self.parentNode[u])
        return self.parentNode[u]

    def merge_nodes(self, u, v):
        """
        Merge the sets containing u and v into one connected component.
        Uses union by rank for efficiency.

        Returns:
            True if merge was successful (u and v were in different sets)
            False if u and v were already in the same set (would create a cycle)
        """
        rootA = self.find_root(u)
        rootB = self.find_root(v)

        # If they're already in the same set, merging would create a cycle
        if rootA == rootB:
            return False

        # Union by rank: attach smaller tree under larger tree
        if self.nodeRank[rootA] > self.nodeRank[rootB]:
            self.parentNode[rootB] = rootA
        elif self.nodeRank[rootA] < self.nodeRank[rootB]:
            self.parentNode[rootA] = rootB
        else:
            # Ranks are equal, so choose rootA as the new root
            self.parentNode[rootB] = rootA
            self.nodeRank[rootA] += 1

        return True

    def clone(self):
        """
        Create a deep copy of this Union-Find structure.
        Used for testing if a path would create a cycle without permanently
        committing to it.
        """
        clonedSet = Union_Find(len(self.parentNode))
        clonedSet.parentNode = self.parentNode[:]  # Deep copy the parent array
        clonedSet.nodeRank = self.nodeRank[:]      # Deep copy the rank array
        return clonedSet


class Simulation:
    """
    Main simulation that solves the railway network path planning problem.

    For each ticket, it tries to reserve the shortest path that doesn't
    create a cycle. If that's impossible, it explores alternative paths
    in increasing order of cost until finding one that fits.
    """

    def __init__(self, railGraph, routeTickets, cityLabels=None, maxAlternatives=20):
        """
        Initialize the simulation.

        Args:
            railGraph: WeightedDiGraph representing the railway network
            routeTickets: List of (source, destination) tuples to connect
            cityLabels: Optional list of city names for pretty printing
            maxAlternatives: Max number of alternate paths to try per ticket
        """
        self.railGraph = railGraph
        self.routeTickets = routeTickets
        self.cityLabels = cityLabels if cityLabels else [str(i) for i in range(railGraph.V)]
        self.maxAlternatives = maxAlternatives

    def check_acyclic_path(self, tracker, routePath):
        """
        Check if adding a path would create a cycle in the network.

        This creates a temporary clone of the Union-Find tracker and
        tries to add all edges from the path. If any edge would merge
        two cities in the same component, a cycle would be created.

        Returns:
            True if the path is safe to add (no cycles created)
            False if adding the path would create a cycle
        """
        tempTracker = tracker.clone()

        # Try adding each edge in the path
        for u, v in edges_from_path(routePath):
            # If u and v are already connected, this edge would close a cycle
            if tempTracker.find_root(u) == tempTracker.find_root(v):
                return False
            # Tentatively merge the components
            tempTracker.merge_nodes(u, v)

        # All edges can be added without creating a cycle
        return True

    def commit_route(self, tracker, routePath):
        """
        Permanently add a path to the final network by updating the tracker.
        This merges all cities in the path into connected components.
        """
        for u, v in edges_from_path(routePath):
            tracker.merge_nodes(u, v)

    def format_route(self, routePath):
        """
        Convert a path (list of city indices) to a human-readable string.
        Example: [0, 2, 4] -> "Houston -> New York -> Denver"
        """
        return " -> ".join(self.cityLabels[v] for v in routePath)

    def run_simulation(self):
        """
        Execute the main simulation algorithm.

        For each ticket:
        1. Find the shortest path using Dijkstra
        2. Check if it creates a cycle
        3. If safe, use it; otherwise try alternate paths
        4. Output the selected paths and total cost

        Returns:
            Dictionary with 'paths' and 'total_cost' keys
            Returns None if any ticket cannot be satisfied
        """
        tracker = Union_Find(self.railGraph.V)  # Tracks connected components
        selectedRoutes = []  # Stores the final selected paths
        totalCost = 0

        # Process each ticket in order
        for originCity, destCity in self.routeTickets:
            # Find shortest path using Dijkstra
            distMap, prevNode = dijkstra_search(self.railGraph, originCity)
            shortestRoute = trace_path(prevNode, originCity, destCity)

            selectedPath = None

            # Try the shortest path first
            if shortestRoute is not None and self.check_acyclic_path(tracker, shortestRoute):
                selectedPath = (calculate_cost(self.railGraph, shortestRoute), shortestRoute)
            else:
                # Shortest path either doesn't exist or creates a cycle
                # Try alternate paths in order of cost
                altRoutes = find_k_simple_paths(
                    self.railGraph,
                    originCity,
                    destCity,
                    k=self.maxAlternatives
                )

                # Try each alternate path until we find one that doesn't create a cycle
                for routeCost, candidatePath in altRoutes:
                    if self.check_acyclic_path(tracker, candidatePath):
                        selectedPath = (routeCost, candidatePath)
                        break

            # Check if we found a valid path for this ticket
            if selectedPath is None:
                print(f"No valid acyclic path found for ticket "
                      f"{self.cityLabels[originCity]} -> {self.cityLabels[destCity]}")
                return None

            # Add this path to the final network
            routeCost, routePath = selectedPath
            self.commit_route(tracker, routePath)
            selectedRoutes.append(((originCity, destCity), routePath, routeCost))
            totalCost += routeCost

        # Print results
        print("Chosen railway paths:")
        for (src, dst), routePath, routeCost in selectedRoutes:
            print(f"{self.cityLabels[src]} -> {self.cityLabels[dst]}: "
                  f"{self.format_route(routePath)} | cost = {routeCost}")

        print(f"\nTotal railway cost = {totalCost}")

        return {
            "paths": selectedRoutes,
            "total_cost": totalCost
        }


if __name__ == "__main__":
    # =====================================================================
    # Example Usage: Two Simulations
    # =====================================================================

    # SIMULATION 1: Fully Connected Dense Graph
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SIMULATION 1: Fully Connected Dense Railway Network")
    print("=" * 70)

    DummyGraph = WeightedDiGraph(10)

    Cities = [
        "Houston", "Atlanta", "New York", "Los Angeles", "Kansas City",
        "Miami", "Austin", "Detroit", "Milwaukee", "Seattle"
    ]

    # Create a fully connected graph where every city can reach every other city
    # Edge weights are randomly assigned between 1 and 50
    DummyGraph.connect_all()

    # Define four tickets that need to be connected
    tickets = [
        (0, 2),  # Houston -> New York
        (3, 9),  # Los Angeles -> Seattle
        (6, 5),  # Austin -> Miami
        (1, 7)   # Atlanta -> Detroit
    ]

    print("\nRunning Simulation 1...\n")
    sim = Simulation(DummyGraph, tickets, Cities, maxAlternatives=30)
    result1 = sim.run_simulation()

    # SIMULATION 2: Sparse Graph with Disconnected Components
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIMULATION 2: Sparse Railway Network (Not Fully Connected)")
    print("=" * 70)

    SparseGraph = WeightedDiGraph(6)

    SparseCities = [
        "Boston", "Chicago", "Denver", "Phoenix", "Portland", "San Diego"
    ]

    # Component 1: Cities 0-3 are interconnected
    # (Boston, Chicago, Denver, Phoenix have routes between them)
    SparseGraph.add_edge(0, 1, 4)   # Boston -> Chicago (cost 4)
    SparseGraph.add_edge(0, 2, 9)   # Boston -> Denver (cost 9, expensive)
    SparseGraph.add_edge(1, 2, 3)   # Chicago -> Denver (cost 3, cheaper)
    SparseGraph.add_edge(1, 3, 7)   # Chicago -> Phoenix (cost 7)
    SparseGraph.add_edge(2, 3, 2)   # Denver -> Phoenix (cost 2)
    SparseGraph.add_edge(3, 0, 8)   # Phoenix -> Boston (cost 8)

    # Component 2: Cities 4-5 form a separate component
    # (Portland and San Diego can only reach each other)
    SparseGraph.add_edge(4, 5, 5)   # Portland -> San Diego (cost 5)
    SparseGraph.add_edge(5, 4, 5)   # San Diego -> Portland (cost 5)

    # Define three tickets for the sparse network
    # Note: Boston -> Denver has an expensive direct route (cost 9)
    # but a cheaper alternative through Chicago (cost 4+3=7)
    SparseTickets = [
        (0, 2),  # Boston -> Denver (will use Boston -> Chicago -> Denver)
        (2, 3),  # Denver -> Phoenix
        (4, 5)   # Portland -> San Diego
    ]

    print("\nRunning Simulation 2...\n")
    sparse_sim = Simulation(SparseGraph, SparseTickets, SparseCities, maxAlternatives=15)
    result2 = sparse_sim.run_simulation()

    print("\n" + "=" * 70)
    print("Both simulations completed successfully!")
    print("=" * 70)
