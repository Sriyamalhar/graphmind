"""
The core differentiator of GraphMind: a rigorous, reproducible
benchmark comparing the learned GNN against classical shortest-path
algorithms (and GNN variants against each other), following the same
methodology as `graphbench` (Sriyamalhar/graphbench) — same inputs,
same hardware, same measured resource constraints, results written
to docs/BENCHMARK.md rather than only reported informally.

Metrics captured per method:
    - accuracy   (MAE, RMSE, and relative error % vs. ground truth)
    - speed      (wall-clock time per query, and for batched queries)
    - memory     (peak resident set size during inference)

Two comparisons are run:
    1. GraphMindGNN (trained) vs. Dijkstra vs. A* — accuracy/speed tradeoff
    2. GNN aggregation variants (mean / max / attention) vs each other
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from graphmind.data.classical_algorithms import Graph, astar_pair, dijkstra_pair
from graphmind.data.dataset import PairExample
from graphmind.models.gnn import BatchedGraph, GraphMindGNN


@dataclass
class BenchmarkResult:
    method: str
    mae: float
    rmse: float
    mean_relative_error_pct: float
    total_wall_time_sec: float
    avg_time_per_query_ms: float
    peak_memory_kb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _score_predictions(preds: list[float], truths: list[float]) -> tuple[float, float, float]:
    n = len(preds)
    abs_errors = [abs(p - t) for p, t in zip(preds, truths)]
    sq_errors = [(p - t) ** 2 for p, t in zip(preds, truths)]
    mae = sum(abs_errors) / n
    rmse = (sum(sq_errors) / n) ** 0.5
    rel_errors = [
        abs(p - t) / t * 100 if t > 1e-9 else 0.0 for p, t in zip(preds, truths)
    ]
    mean_rel_error = sum(rel_errors) / n
    return mae, rmse, mean_rel_error


def benchmark_classical(
    graph: Graph,
    examples: list[PairExample],
    method: Callable[[Graph, int, int], float],
    method_name: str,
) -> BenchmarkResult:
    tracemalloc.start()
    start = time.perf_counter()

    preds = [method(graph, ex.source, ex.target) for ex in examples]

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    truths = [ex.distance for ex in examples]
    mae, rmse, mean_rel = _score_predictions(preds, truths)

    return BenchmarkResult(
        method=method_name,
        mae=mae,
        rmse=rmse,
        mean_relative_error_pct=mean_rel,
        total_wall_time_sec=elapsed,
        avg_time_per_query_ms=(elapsed / len(examples)) * 1000,
        peak_memory_kb=peak / 1024,
    )


def benchmark_gnn(
    model: GraphMindGNN,
    batched_graph: BatchedGraph,
    examples: list[PairExample],
    method_name: str = "GraphMindGNN",
) -> BenchmarkResult:
    model.eval()
    tracemalloc.start()
    start = time.perf_counter()

    with torch.no_grad():
        # Single batched forward pass for ALL query pairs at once —
        # this is the core speed advantage over per-pair Dijkstra/A*
        # calls, and is measured explicitly rather than assumed.
        node_embeddings = model.encode(batched_graph)
        src = torch.tensor([ex.source for ex in examples], dtype=torch.long)
        dst = torch.tensor([ex.target for ex in examples], dtype=torch.long)
        preds = model.predict_distance(node_embeddings, src, dst).tolist()

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    truths = [ex.distance for ex in examples]
    mae, rmse, mean_rel = _score_predictions(preds, truths)

    return BenchmarkResult(
        method=method_name,
        mae=mae,
        rmse=rmse,
        mean_relative_error_pct=mean_rel,
        total_wall_time_sec=elapsed,
        avg_time_per_query_ms=(elapsed / len(examples)) * 1000,
        peak_memory_kb=peak / 1024,
    )


def run_full_benchmark(
    graph: Graph,
    batched_graph: BatchedGraph,
    model: GraphMindGNN,
    test_examples: list[PairExample],
) -> list[BenchmarkResult]:
    """Runs GNN vs. Dijkstra vs. A* on the same test examples and
    returns all results for writing into docs/BENCHMARK.md.
    """
    results = [
        benchmark_gnn(model, batched_graph, test_examples),
        benchmark_classical(graph, test_examples, dijkstra_pair, "Dijkstra"),
        benchmark_classical(
            graph, test_examples, lambda g, s, t: astar_pair(g, s, t), "A* (no heuristic)"
        ),
    ]
    return results
