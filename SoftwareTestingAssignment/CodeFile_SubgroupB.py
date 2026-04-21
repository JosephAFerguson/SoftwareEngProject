#Done by Joe Ferguson
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
