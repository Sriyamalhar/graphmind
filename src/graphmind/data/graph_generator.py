"""
Generates synthetic road-network-style graphs for training and
benchmarking GraphMind, plus utilities to load real-world graphs
(e.g. SNAP datasets, matching the graphbench data source) for the
final generalization benchmark.

Two graph families are supported:

1. Grid-with-noise graphs: a city-block-like grid topology with
   randomized edge weights (travel times), optionally with a few
   long-range "highway" shortcut edges. This mirrors Meridian's
   simulated road network more closely than a purely random graph.

2. Random geometric graphs: nodes placed in 2D space, edges added
   between nearby nodes, weight = Euclidean distance. Useful for
   giving the GNN spatial coordinate features and for computing an
   admissible A* heuristic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from graphmind.data.classical_algorithms import Graph


@dataclass
class GeneratedGraph:
    graph: Graph
    coordinates: list[tuple[float, float]]  # node_id -> (x, y), for spatial features


def make_grid_graph(
    rows: int,
    cols: int,
    weight_range: tuple[float, float] = (1.0, 10.0),
    num_shortcuts: int = 0,
    seed: int | None = None,
) -> GeneratedGraph:
    """City-block grid graph with randomized travel-time weights.

    Node ids are assigned row-major: node = r * cols + c.
    Optionally adds `num_shortcuts` random long-range edges to mimic
    highways, which is what makes plain BFS/heuristics interesting.
    """
    rng = random.Random(seed)
    num_nodes = rows * cols
    edges: list[tuple[int, int, float]] = []
    coordinates: list[tuple[float, float]] = []

    for r in range(rows):
        for c in range(cols):
            coordinates.append((float(c), float(r)))

    def node_id(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)
            if c + 1 < cols:
                w = rng.uniform(*weight_range)
                edges.append((u, node_id(r, c + 1), w))
            if r + 1 < rows:
                w = rng.uniform(*weight_range)
                edges.append((u, node_id(r + 1, c), w))

    for _ in range(num_shortcuts):
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u != v:
            ux, uy = coordinates[u]
            vx, vy = coordinates[v]
            dist = ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5
            # Shortcuts are faster than the straight-line grid distance
            # would normally cost, to simulate a highway bypass.
            edges.append((u, v, dist * rng.uniform(0.3, 0.6)))

    graph = Graph.from_edge_list(num_nodes, edges)
    return GeneratedGraph(graph=graph, coordinates=coordinates)


def make_random_geometric_graph(
    num_nodes: int,
    connect_radius: float = 0.15,
    seed: int | None = None,
) -> GeneratedGraph:
    """Random points in the unit square, edges between nearby points.

    Edge weight = Euclidean distance, which also serves directly as
    an admissible A* heuristic (straight-line distance <= any path).
    """
    rng = random.Random(seed)
    coordinates = [(rng.random(), rng.random()) for _ in range(num_nodes)]
    edges: list[tuple[int, int, float]] = []

    for i in range(num_nodes):
        xi, yi = coordinates[i]
        for j in range(i + 1, num_nodes):
            xj, yj = coordinates[j]
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if d <= connect_radius:
                edges.append((i, j, d))

    graph = Graph.from_edge_list(num_nodes, edges)
    return GeneratedGraph(graph=graph, coordinates=coordinates)
