"""
Builds (graph, source, target, distance) training examples.

Ground-truth distances come from the hand-implemented Dijkstra in
`classical_algorithms.py` — the GNN never sees the classical
algorithm's internals, only the final label, which keeps the
benchmark honest (the GNN must learn to approximate shortest-path
distance purely from graph structure + node features).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from graphmind.data.classical_algorithms import Graph, dijkstra
from graphmind.data.graph_generator import GeneratedGraph


@dataclass
class PairExample:
    source: int
    target: int
    distance: float  # ground truth from Dijkstra


@dataclass
class GraphDataset:
    generated: GeneratedGraph
    examples: list[PairExample]

    @property
    def graph(self) -> Graph:
        return self.generated.graph


def build_pair_dataset(
    generated: GeneratedGraph,
    num_pairs: int,
    seed: int | None = None,
) -> GraphDataset:
    """Samples `num_pairs` (source, target) pairs and labels each with
    its true shortest-path distance via Dijkstra.

    Runs one Dijkstra per unique source node (not per pair) since
    Dijkstra already computes distances to all reachable nodes at once
    — this makes label generation for many pairs sharing a source
    much cheaper than calling `dijkstra_pair` independently per pair.
    """
    rng = random.Random(seed)
    graph = generated.graph
    n = graph.num_nodes

    pairs: list[tuple[int, int]] = []
    for _ in range(num_pairs):
        s = rng.randrange(n)
        t = rng.randrange(n)
        if s != t:
            pairs.append((s, t))

    pairs_by_source: dict[int, list[int]] = {}
    for s, t in pairs:
        pairs_by_source.setdefault(s, []).append(t)

    examples: list[PairExample] = []
    for s, targets in pairs_by_source.items():
        distances = dijkstra(graph, s)
        for t in targets:
            if t in distances:  # skip unreachable pairs
                examples.append(PairExample(source=s, target=t, distance=distances[t]))

    return GraphDataset(generated=generated, examples=examples)


def train_val_test_split(
    dataset: GraphDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[list[PairExample], list[PairExample], list[PairExample]]:
    """Splits examples (not nodes) into train/val/test.

    Note: this is a transductive setting — the same graph structure is
    visible across splits, only the (source, target) query pairs
    differ. Section 4 of docs/BENCHMARK.md discusses this choice and
    the inductive generalization experiment (unseen graphs) as a
    follow-up.
    """
    rng = random.Random(seed)
    examples = list(dataset.examples)
    rng.shuffle(examples)

    n = len(examples)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val = examples[:n_val]
    test = examples[n_val : n_val + n_test]
    train = examples[n_val + n_test :]
    return train, val, test
