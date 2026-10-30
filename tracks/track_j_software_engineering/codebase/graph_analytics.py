"""
Graph Analytics and Network Flow Algorithms.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

from collections import deque
from typing import Any, Optional

class GraphNode:
    def __init__(self, node_id: str, value: Any = None):
        self.id = node_id
        self.value = value
        self.neighbors: dict['GraphNode', float] = {}

    def add_edge(self, target: 'GraphNode', weight: float = 1.0):
        self.neighbors[target] = weight

class DirectedGraph:
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}

    def get_or_create(self, node_id: str) -> GraphNode:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(node_id)
        return self.nodes[node_id]

    def add_edge(self, u_id: str, v_id: str, weight: float = 1.0):
        u = self.get_or_create(u_id)
        v = self.get_or_create(v_id)
        u.add_edge(v, weight)

    def topological_sort(self) -> list[str]:
        in_degree = {nid: 0 for nid in self.nodes}
        for u in self.nodes.values():
            for v in u.neighbors:
                in_degree[v.id] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr_id = queue.popleft()
            order.append(curr_id)
            for neighbor in self.nodes[curr_id].neighbors:
                in_degree[neighbor.id] -= 1
                if in_degree[neighbor.id] == 0:
                    queue.append(neighbor.id)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle; topological sort not possible")
        return order

    def dijkstra_shortest_path(self, start_id: str) -> dict[str, float]:
        import heapq
        distances = {nid: float('inf') for nid in self.nodes}
        distances[start_id] = 0.0
        pq = [(0.0, start_id)]

        while pq:
            curr_dist, curr_id = heapq.heappop(pq)
            if curr_dist > distances[curr_id]:
                continue
            curr_node = self.nodes[curr_id]
            for neighbor, weight in curr_node.neighbors.items():
                new_dist = curr_dist + weight
                if new_dist < distances[neighbor.id]:
                    distances[neighbor.id] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor.id))
        return distances

    def pagerank(self, damping: float = 0.85, max_iters: int = 50, tol: float = 1e-6) -> dict[str, float]:
        n = len(self.nodes)
        if n == 0:
            return {}
        scores = {nid: 1.0 / n for nid in self.nodes}

        for _ in range(max_iters):
            new_scores = {nid: (1.0 - damping) / n for nid in self.nodes}
            for u in self.nodes.values():
                out_deg = len(u.neighbors)
                if out_deg > 0:
                    share = damping * (scores[u.id] / out_deg)
                    for v in u.neighbors:
                        new_scores[v.id] += share
                else:
                    # Dangling node
                    share = damping * (scores[u.id] / n)
                    for nid in self.nodes:
                        new_scores[nid] += share

            diff = sum(abs(new_scores[nid] - scores[nid]) for nid in self.nodes)
            scores = new_scores
            if diff < tol:
                break
        return scores
