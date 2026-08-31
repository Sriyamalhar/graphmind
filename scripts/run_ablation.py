"""
Ablation study: trains multiple GraphMindGNN configurations on the
*same* graph and data split, varying one factor at a time, and records
results for docs/BENCHMARK.md.

Two ablations:
    1. Aggregation variant: mean vs. max vs. attention (fixed depth=3)
    2. Depth: 1 / 2 / 3 / 4 message-passing layers (fixed aggregation=mean)

Everything uses the same seed for graph generation and data split, so
differences in results are attributable to the factor being varied,
not to randomness in the data itself (only weight initialization and
training dynamics vary between runs).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from graphmind.benchmark.suite import benchmark_gnn
from graphmind.data.dataset import build_pair_dataset, train_val_test_split
from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.models.gnn import GraphMindGNN
from graphmind.training.train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Fixed experimental setup shared across all ablation runs.
GRAPH_ROWS = 20
GRAPH_COLS = 20
NUM_SHORTCUTS = 10
NUM_PAIRS = 1500
NUM_EPOCHS = 150
LR = 3e-3
SEED = 7


def run_config(aggregation: str, num_layers: int, hidden_dim: int = 32) -> dict:
    generated = make_grid_graph(
        rows=GRAPH_ROWS, cols=GRAPH_COLS, num_shortcuts=NUM_SHORTCUTS, seed=SEED
    )
    dataset = build_pair_dataset(generated, num_pairs=NUM_PAIRS, seed=SEED + 1)
    train_ex, val_ex, test_ex = train_val_test_split(dataset, seed=SEED + 2)

    batch = build_batched_graph(generated)
    model = GraphMindGNN(
        node_feat_dim=batch.node_features.size(1),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        aggregation=aggregation,
    )

    label = f"agg={aggregation},layers={num_layers}"
    print(f"\n=== Training: {label} ===")

    start = time.perf_counter()
    history = train_model(
        model, batch, train_ex, val_ex, num_epochs=NUM_EPOCHS, lr=LR, log_every=50
    )
    train_time = time.perf_counter() - start

    result = benchmark_gnn(model, batch, test_ex, method_name=label)

    return {
        "aggregation": aggregation,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "final_train_loss": history.epochs[-1].train_loss,
        "final_val_loss": history.epochs[-1].val_loss,
        "final_val_mae": history.epochs[-1].val_mae,
        "training_time_sec": train_time,
        "test_mae": result.mae,
        "test_rmse": result.rmse,
        "test_mean_relative_error_pct": result.mean_relative_error_pct,
        "avg_inference_ms_per_query": result.avg_time_per_query_ms,
    }


def main() -> None:
    results = []

    print("### Ablation 1: aggregation variant (depth fixed at 3) ###")
    for agg in ("mean", "max", "attention"):
        results.append(run_config(aggregation=agg, num_layers=3))

    print("\n### Ablation 2: depth (aggregation fixed at mean) ###")
    for depth in (1, 2, 4):  # 3 already covered above
        results.append(run_config(aggregation="mean", num_layers=depth))

    print("\n\n=== SUMMARY ===")
    print(
        f"{'Config':<28} {'TrainTime(s)':>12} {'TestMAE':>9} {'RelErr%':>9} {'InferMs':>9}"
    )
    for r in results:
        label = f"agg={r['aggregation']},layers={r['num_layers']}"
        print(
            f"{label:<28} {r['training_time_sec']:>12.1f} {r['test_mae']:>9.2f} "
            f"{r['test_mean_relative_error_pct']:>9.1f} {r['avg_inference_ms_per_query']:>9.4f}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ablation_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "config": {
                    "graph_rows": GRAPH_ROWS,
                    "graph_cols": GRAPH_COLS,
                    "num_shortcuts": NUM_SHORTCUTS,
                    "num_pairs": NUM_PAIRS,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                    "seed": SEED,
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved ablation results to {output_path}")


if __name__ == "__main__":
    main()
