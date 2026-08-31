"""
Trains an attention-aggregation GraphMindGNN on a city-grid graph, then
generates two interpretability visualizations:

    1. An attention-weighted graph plot: edges colored/thickened by how
       much attention the final message-passing layer assigns them,
       overlaid on the actual grid layout.
    2. A t-SNE projection of final-layer node embeddings, colored by
       node degree, to see whether the model organizes nodes by
       structural role rather than just re-encoding raw (x, y).

Outputs PNGs to docs/figures/ for use in the README and BENCHMARK.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from graphmind.data.dataset import build_pair_dataset, train_val_test_split
from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.interpretability.attention import extract_attention, extract_node_embeddings
from graphmind.models.gnn import GraphMindGNN
from graphmind.training.train import train_model

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

ROWS, COLS = 12, 12
NUM_SHORTCUTS = 6
NUM_PAIRS = 800
NUM_EPOCHS = 120
SEED = 3


def plot_attention_graph(generated, report, layer_idx: int, out_path: Path) -> None:
    coords = generated.coordinates
    edges = report.layer_edge_attentions[layer_idx]

    fig, ax = plt.subplots(figsize=(9, 9))

    weights = np.array([e.weight for e in edges])
    # Normalize for line width/opacity scaling (avoid div-by-zero if uniform).
    w_min, w_max = weights.min(), weights.max()
    norm = (weights - w_min) / (w_max - w_min + 1e-9)

    for edge, n in zip(edges, norm):
        x0, y0 = coords[edge.source]
        x1, y1 = coords[edge.target]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="tab:blue",
            linewidth=0.5 + 3.0 * n,
            alpha=0.15 + 0.7 * n,
        )

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax.scatter(xs, ys, s=20, color="black", zorder=3)

    ax.set_title(
        f"Attention weights, message-passing layer {layer_idx}\n"
        f"(thicker/darker edge = higher attention weight)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_tsne(embedding_report, generated, out_path: Path) -> None:
    embeddings = np.array(embedding_report.embeddings)
    degrees = [
        len(generated.graph.adjacency.get(i, [])) for i in embedding_report.node_ids
    ]

    tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca")
    projected = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1], c=degrees, cmap="viridis", s=40
    )
    fig.colorbar(scatter, ax=ax, label="Node degree")
    ax.set_title(
        "t-SNE projection of learned node embeddings\n"
        "(colored by node degree - checks whether the model organizes\n"
        "nodes by structural role, not just spatial position)"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    generated = make_grid_graph(rows=ROWS, cols=COLS, num_shortcuts=NUM_SHORTCUTS, seed=SEED)
    dataset = build_pair_dataset(generated, num_pairs=NUM_PAIRS, seed=SEED + 1)
    train_ex, val_ex, _test_ex = train_val_test_split(dataset, seed=SEED + 2)

    batch = build_batched_graph(generated)
    model = GraphMindGNN(
        node_feat_dim=batch.node_features.size(1),
        hidden_dim=32,
        num_layers=3,
        aggregation="attention",
    )

    print("Training attention-aggregation model for interpretability report...")
    train_model(model, batch, train_ex, val_ex, num_epochs=NUM_EPOCHS, lr=3e-3, log_every=30)

    print("Extracting attention weights...")
    attn_report = extract_attention(model, batch)
    attn_path = FIGURES_DIR / "attention_weights_layer2.png"
    plot_attention_graph(generated, attn_report, layer_idx=2, out_path=attn_path)
    print(f"Saved {attn_path}")

    top_edges = attn_report.top_k_edges(layer_idx=2, k=10)
    print("\nTop 10 highest-attention edges (final layer):")
    for e in top_edges:
        print(f"  ({e.source} -> {e.target}): weight={e.weight:.4f}")

    print("\nExtracting node embeddings for t-SNE...")
    embedding_report = extract_node_embeddings(model, batch, generated)
    tsne_path = FIGURES_DIR / "embedding_tsne.png"
    plot_embedding_tsne(embedding_report, generated, tsne_path)
    print(f"Saved {tsne_path}")


if __name__ == "__main__":
    main()
