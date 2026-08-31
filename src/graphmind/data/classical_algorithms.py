"""
Classical shortest-path algorithms used as (a) ground-truth label
generators for training data, and (b) the speed/accuracy baseline
that GraphMind's GNN is benchmarked against.

These are deliberately hand-implemented rather than imported from a
library (e.g. networkx.shortest_path), matching the same
from-first-principles approach used in the Meridian project's
routing engine.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass


@dataclass
class Graph:
    """Simple weighted, undirected adjacency-list graph."""

    num_nodes: int
    adjacency: dict[int, list[tuple[int, float]]]  # node -> [(neighbor, weight)]

    @classmethod
    def from_edge_list(cls, num_nodes: int, edges: list[tuple[int, int, float]]) -> Graph:
        adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(num_nodes)}
        for u, v, w in edges:
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
        return cls(num_nodes=num_nodes, adjacency=adjacency)


def dijkstra(graph: Graph, source: int) -> dict[int, float]:
    """Single-source shortest paths via Dijkstra's algorithm.

    Returns a dict of node -> shortest distance from `source`.
    Unreachable nodes are omitted.
    """
    dist: dict[int, float] = {source: 0.0}
    visited = set()
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, w in graph.adjacency.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def dijkstra_pair(graph: Graph, source: int, target: int) -> float:
    """Shortest distance between a single (source, target) pair.

    Early-exits once the target is popped, which is faster than
    computing full single-source distances when only one pair is needed.
    """
    if source == target:
        return 0.0

    dist: dict[int, float] = {source: 0.0}
    visited = set()
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        if u == target:
            return d
        visited.add(u)

        for v, w in graph.adjacency.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return math.inf  # unreachable


def astar_pair(
    graph: Graph,
    source: int,
    target: int,
    heuristic: dict[tuple[int, int], float] | None = None,
) -> float:
    """Shortest distance between (source, target) via A*.

    `heuristic` maps (node, target) -> estimated remaining cost.
    Falls back to Dijkstra behavior (heuristic = 0) if none is given,
    since without real coordinates we can't compute an admissible
    Euclidean heuristic for a synthetic graph.
    """
    if heuristic is None:
        heuristic = {}

    if source == target:
        return 0.0

    g_score: dict[int, float] = {source: 0.0}
    visited = set()
    h0 = heuristic.get((source, target), 0.0)
    pq: list[tuple[float, int]] = [(h0, source)]

    while pq:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        if u == target:
            return g_score[u]
        visited.add(u)

        for v, w in graph.adjacency.get(u, []):
            ng = g_score[u] + w
            if v not in g_score or ng < g_score[v]:
                g_score[v] = ng
                f = ng + heuristic.get((v, target), 0.0)
                heapq.heappush(pq, (f, v))

    return math.inf


def timed_dijkstra_pair(graph: Graph, source: int, target: int) -> tuple[float, float]:
    """Returns (distance, wall_clock_seconds) — used by the benchmark suite."""
    start = time.perf_counter()
    d = dijkstra_pair(graph, source, target)
    elapsed = time.perf_counter() - start
    return d, elapsed
