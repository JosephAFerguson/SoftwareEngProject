#Done by Jason Bellerjeau
import heapq
import random


class Union_Find:
    def __init__(self, n):
        self.parentNode = list(range(n))
        self.nodeRank = [0] * n

    def find_root(self, u):
        if self.parentNode[u] != u:
            self.parentNode[u] = self.find_root(self.parentNode[u])
        return self.parentNode[u]

    def merge_nodes(self, u, v):
        rootA = self.find_root(u)
        rootB = self.find_root(v)

        if rootA == rootB:
            return False

        if self.nodeRank[rootA] > self.nodeRank[rootB]:
            self.parentNode[rootB] = rootA
        elif self.nodeRank[rootA] < self.nodeRank[rootB]:
            self.parentNode[rootA] = rootB
        else:
            self.parentNode[rootB] = rootA
            self.nodeRank[rootA] += 1

        return True

    def clone(self):
        clonedSet = Union_Find(len(self.parentNode))
        clonedSet.parentNode = self.parentNode[:]
        clonedSet.nodeRank = self.nodeRank[:]
        return clonedSet


class Network_Simulation:
    def __init__(self, railGraph, routeTickets, cityLabels=None, maxAlternatives=20):
        self.railGraph = railGraph
        self.routeTickets = routeTickets
        self.cityLabels = cityLabels if cityLabels else [str(i) for i in range(railGraph.V)]
        self.maxAlternatives = maxAlternatives

    def check_acyclic_path(self, tracker, routePath):
        tempTracker = tracker.clone()

        for u, v in edges_from_path(routePath):
            if tempTracker.find_root(u) == tempTracker.find_root(v):
                return False
            tempTracker.merge_nodes(u, v)

        return True

    def commit_route(self, tracker, routePath):
        for u, v in edges_from_path(routePath):
            tracker.merge_nodes(u, v)

    def format_route(self, routePath):
        return " -> ".join(self.cityLabels[v] for v in routePath)

    def run_simulation(self):
        tracker = Union_Find(self.railGraph.V)
        selectedRoutes = []
        totalCost = 0

        for originCity, destCity in self.routeTickets:
            distMap, prevNode = dijkstra_search(self.railGraph, originCity)
            shortestRoute = trace_path(prevNode, originCity, destCity)

            selectedPath = None

            if shortestRoute is not None and self.check_acyclic_path(tracker, shortestRoute):
                selectedPath = (calculate_cost(self.railGraph, shortestRoute), shortestRoute)
            else:
                altRoutes = find_k_simple_paths(
                    self.railGraph,
                    originCity,
                    destCity,
                    k=self.maxAlternatives
                )

                for routeCost, candidatePath in altRoutes:
                    if self.check_acyclic_path(tracker, candidatePath):
                        selectedPath = (routeCost, candidatePath)
                        break

            if selectedPath is None:
                print(f"No valid acyclic path found for ticket "
                      f"{self.cityLabels[originCity]} -> {self.cityLabels[destCity]}")
                return None

            routeCost, routePath = selectedPath
            self.commit_route(tracker, routePath)
            selectedRoutes.append(((originCity, destCity), routePath, routeCost))
            totalCost += routeCost

        print("Chosen railway paths:")
        for (src, dst), routePath, routeCost in selectedRoutes:
            print(f"{self.cityLabels[src]} -> {self.cityLabels[dst]}: "
                  f"{self.format_route(routePath)} | cost = {routeCost}")

        print(f"\nTotal railway cost = {totalCost}")

        return {
            "paths": selectedRoutes,
            "total_cost": totalCost
        }
