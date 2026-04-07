"""
Ticket to Ride style path planning (Simplified version)

Problem:
Given:
• Set of tickets [(source1, destination1), (source2,destination2), ...]
• Weighted directed graph (adjacency matrix of cities and routes)

Output: Find an optimal railway path such that
(i) the source and destination of each ticket is connected.
(ii) the total path cost is minimized
(iii) there are no cycles in the final railway network.

Goal:
• Find a path for each source-destination ticket
• Shortest valid path should be selected.
  -> A path is valid only if it does not lead to cycles
     in the final railway network.
  -> If cycle is formed, explore alternate paths
     (second shortest path, third shortest path, ...)

Algorithms used: Dijkstra and Union-Find
"""
import random
import heapq

class WeightedDiGraph:
    def __init__(self, vertices):
        self.V = vertices
        # A value of 0 means there is no directed edge between two cities.
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def connect_all(self):
        for i in range(self.V):
            for j in range(self.V):
                if i != j and self.graph[i][j] == 0:
                    # random weight between 1 and 50
                    self.graph[i][j] = random.randint(1, 50)

    def add_edge(self, u, v, w):
        self.graph[u][v] = w

    def add_node(self):
        self.V += 1
        for row in self.graph:
            row.append(0)
        self.graph.append([0 for _ in range(self.V)])

    def get_edge(self, u, v):
        if not self.isEdge(u, v):
            return None
        return self.graph[u][v]

    def get_edges(self):
        edges = []
        for i in range(self.V):
            for j in range(self.V):
                if self.graph[i][j] != 0:
                    edges.append((i, j, self.graph[i][j]))
        return edges

    def get_neighbors(self, u):
        if not self.isVertex(u):
            return []

        neighbors = []
        for v in range(self.V):
            if self.graph[u][v] != 0:
                neighbors.append((v, self.graph[u][v]))
        return neighbors

    def isVertex(self, u):
        return 0 <= u < self.V

    def isEdge(self, u, v):
        return 0 <= u < self.V and 0 <= v < self.V and self.graph[u][v] != 0

"""
Dijkstra's algorithm:
Returns:
- dist: minimum distance from start to every vertex
- prev: predecessor array for path reconstruction
"""
def Dijkstra(graph, start):
    dist = [float("inf")] * graph.V
    prev = [None] * graph.V
    dist[start] = 0

    # Priority queue entries are (distance_so_far, current_vertex).
    pq = [(0, start)]

    while pq:
        curr_dist, u = heapq.heappop(pq)

        if curr_dist > dist[u]:
            # Skip outdated queue entries that were improved later.
            continue

        for v, w in graph.get_neighbors(u):
            new_dist = curr_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev

def reconstruct_path(prev, start, end):
    if start == end:
        return [start]

    path = []
    curr = end
    while curr is not None:
        # Walk backward from the destination until we reach the source.
        path.append(curr)
        if curr == start:
            break
        curr = prev[curr]

    path.reverse()

    if not path or path[0] != start:
        return None
    return path

def path_cost(graph, path):
    total = 0
    for i in range(len(path) - 1):
        # Sum the weights on consecutive edges of the chosen path.
        total += graph.get_edge(path[i], path[i + 1])
    return total

def path_to_edges(path):
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


"""
Generate up to k shortest SIMPLE paths from source to destination.
This is a uniform-cost search over paths, ordered by total cost.
It avoids repeated vertices in a single path, so each candidate path is simple.
"""
def k_shortest_simple_paths(graph, source, destination, k=10):
    # Store candidate paths ordered by total cost seen so far.
    pq = [(0, [source])]
    results = []

    while pq and len(results) < k:
        cost, path = heapq.heappop(pq)
        u = path[-1]

        if u == destination:
            results.append((cost, path))
            continue

        for v, w in graph.get_neighbors(u):
            if v not in path:  # prevent cycles inside the candidate path
                # Extend the current simple path by one edge.
                new_path = path + [v]
                heapq.heappush(pq, (cost + w, new_path))

    return results


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u == root_v:
            return False

        if self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u
        elif self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1

        return True

    def copy(self):
        new_uf = UnionFind(len(self.parent))
        new_uf.parent = self.parent[:]
        new_uf.rank = self.rank[:]
        return new_uf


class Simulation:
    def __init__(self, graph, tickets, city_names=None, max_alternatives=20):
        self.graph = graph
        self.tickets = tickets
        self.city_names = city_names if city_names else [str(i) for i in range(graph.V)]
        self.max_alternatives = max_alternatives

    def can_add_path_without_cycle(self, uf, path):
        """
        Check whether adding all edges in this path to the current railway network
        would create a cycle. We treat the final railway network as undirected for
        cycle detection.
        """
        temp_uf = uf.copy()

        for u, v in path_to_edges(path):
            if temp_uf.find(u) == temp_uf.find(v):
                return False
            temp_uf.union(u, v)

        return True

    def add_path_to_network(self, uf, path):
        for u, v in path_to_edges(path):
            uf.union(u, v)

    def format_path(self, path):
        return " -> ".join(self.city_names[v] for v in path)

    def run(self):
        # Union-Find tracks the undirected structure of the final railway network.
        uf = UnionFind(self.graph.V)
        chosen_paths = []
        total_cost = 0

        for source, destination in self.tickets:
            # Use Dijkstra first for the absolute shortest path
            dist, prev = Dijkstra(self.graph, source)
            shortest_path = reconstruct_path(prev, source, destination)

            chosen = None

            # Try the Dijkstra shortest path first
            if shortest_path is not None and self.can_add_path_without_cycle(uf, shortest_path):
                chosen = (path_cost(self.graph, shortest_path), shortest_path)
            else:
                # If shortest path causes a cycle, explore next best simple paths
                alternatives = k_shortest_simple_paths(
                    self.graph,
                    source,
                    destination,
                    k=self.max_alternatives
                )

                for cost, path in alternatives:
                    if self.can_add_path_without_cycle(uf, path):
                        # Stop at the first alternative that preserves an acyclic network.
                        chosen = (cost, path)
                        break

            if chosen is None:
                print(f"No valid acyclic path found for ticket "
                      f"{self.city_names[source]} -> {self.city_names[destination]}")
                return None

            cost, path = chosen
            # Once accepted, permanently merge this path into the network state.
            self.add_path_to_network(uf, path)
            chosen_paths.append(((source, destination), path, cost))
            total_cost += cost

        print("Chosen railway paths:")
        for (s, d), path, cost in chosen_paths:
            print(f"{self.city_names[s]} -> {self.city_names[d]}: "
                  f"{self.format_path(path)} | cost = {cost}")

        print(f"\nTotal railway cost = {total_cost}")

        return {
            "paths": chosen_paths,
            "total_cost": total_cost
        }


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------

DummyGraph = WeightedDiGraph(10)

Cities = [
    "Houston", "Atlanta", "New York", "Los Angeles", "Kansas City",
    "Miami", "Austin", "Detroit", "Milwaukee", "Seattle"
]

# First simulation: a dense graph where every city can reach every other city.
DummyGraph.connect_all()

# Example tickets
tickets = [
    (0, 2),  # Houston -> New York
    (3, 9),  # Los Angeles -> Seattle
    (6, 5),  # Austin -> Miami
    (1, 7)   # Atlanta -> Detroit
]

print("Simulation 1: fully connected graph")
sim = Simulation(DummyGraph, tickets, Cities, max_alternatives=30)
sim.run()


# Second simulation: a sparse graph with several missing routes.
# This graph is not fully connected because many city pairs have no direct edge.
SparseGraph = WeightedDiGraph(6)

SparseCities = [
    "Boston", "Chicago", "Denver", "Phoenix", "Portland", "San Diego"
]

# Component 1 contains a few possible routes among the first four cities.
SparseGraph.add_edge(0, 1, 4)   # Boston -> Chicago
# The direct Boston -> Denver edge exists, but it is more expensive than
# going through Chicago, so the chosen solution is multi-step.
SparseGraph.add_edge(0, 2, 9)   # Boston -> Denver
SparseGraph.add_edge(1, 2, 3)   # Chicago -> Denver
SparseGraph.add_edge(1, 3, 7)   # Chicago -> Phoenix
SparseGraph.add_edge(2, 3, 2)   # Denver -> Phoenix
SparseGraph.add_edge(3, 0, 8)   # Phoenix -> Boston

# Component 2 is separate from the first component.
SparseGraph.add_edge(4, 5, 5)   # Portland -> San Diego
SparseGraph.add_edge(5, 4, 5)   # San Diego -> Portland

SparseTickets = [
    # Boston -> Denver should be routed as Boston -> Chicago -> Denver,
    # because that costs 7 instead of taking the direct edge with cost 9.
    # The other tickets still fit into the final acyclic network.
    (0, 2),  # Boston -> Denver
    (2, 3),  # Denver -> Phoenix
    (4, 5)   # Portland -> San Diego
]

print("\nSimulation 2: sparse graph that is not fully connected")
sparse_sim = Simulation(SparseGraph, SparseTickets, SparseCities, max_alternatives=15)
sparse_sim.run()