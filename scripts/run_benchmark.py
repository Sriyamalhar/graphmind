"""
End-to-end pipeline: generate a graph, build a training dataset, train
GraphMindGNN, run the classical-vs-GNN benchmark, and write results to
`results/benchmark_<timestamp>.json` plus print a Markdown-ready table.

Usage:
    python scripts/run_benchmark.py --rows 10 --cols 10 --num-pairs 500 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graphmind.benchmark.suite import run_full_benchmark
from graphmind.data.dataset import build_pair_dataset, train_val_test_split
from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.models.gnn import GraphMindGNN
from graphmind.training.train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphMind train + benchmark pipeline")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--num-shortcuts", type=int, default=5)
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--aggregation", choices=["mean", "max", "attention"], default="mean")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generated = make_grid_graph(
        rows=args.rows, cols=args.cols, num_shortcuts=args.num_shortcuts, seed=args.seed
    )
    dataset = build_pair_dataset(generated, num_pairs=args.num_pairs, seed=args.seed + 1)
    train_ex, val_ex, test_ex = train_val_test_split(dataset, seed=args.seed + 2)

    print(f"Graph: {args.rows}x{args.cols} grid, {generated.graph.num_nodes} nodes")
    print(f"Examples: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")

    batch = build_batched_graph(generated)
    model = GraphMindGNN(
        node_feat_dim=batch.node_features.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        aggregation=args.aggregation,
    )

    history = train_model(model, batch, train_ex, val_ex, num_epochs=args.epochs, lr=args.lr)
    results = run_full_benchmark(generated.graph, batch, model, test_ex)

    print("\n| Method | MAE | RMSE | Rel. error % | Avg ms/query | Peak mem (KB) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r.method} | {r.mae:.2f} | {r.rmse:.2f} | "
            f"{r.mean_relative_error_pct:.1f}% | {r.avg_time_per_query_ms:.4f} | "
            f"{r.peak_memory_kb:.1f} |"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    payload = {
        "config": vars(args),
        "graph_num_nodes": generated.graph.num_nodes,
        "training_history": history.to_dict(),
        "benchmark_results": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
