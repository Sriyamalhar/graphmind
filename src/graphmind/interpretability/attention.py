"""
Extracts and formats interpretability signals from a trained
GraphMindGNN: per-edge attention weights (for attention-aggregation
models) and node embeddings for dimensionality-reduction plots.

This module only computes and packages data — actual plotting lives
in `scripts/generate_interpretability_report.py`, keeping the
core library free of a hard matplotlib/sklearn dependency at import
time for users who only want to train/benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from graphmind.data.graph_generator import GeneratedGraph
from graphmind.models.gnn import BatchedGraph, GraphMindGNN


@dataclass
class EdgeAttention:
    source: int
    target: int
    weight: float


@dataclass
class AttentionReport:
    """Per-layer attention weights, only populated for layers that use
    attention aggregation (mean/max layers contribute an empty list).
    """

    layer_edge_attentions: list[list[EdgeAttention]]

    def top_k_edges(self, layer_idx: int, k: int = 15) -> list[EdgeAttention]:
        """Returns the k highest-attention edges for a given layer,
        sorted descending by weight. Useful for both a "which roads
        matter most" visualization and a sanity check that attention
        isn't just uniform (which would suggest it learned nothing
        useful beyond what mean aggregation already does).
        """
        edges = self.layer_edge_attentions[layer_idx]
        return sorted(edges, key=lambda e: e.weight, reverse=True)[:k]


def extract_attention(model: GraphMindGNN, batch: BatchedGraph) -> AttentionReport:
    """Runs the model in eval mode and extracts per-edge attention
    weights for every layer that uses attention aggregation.

    Layers using mean/max aggregation contribute an empty list at
    their index (their MessagePassingLayer.forward returns None for
    attention weights when return_attention=True is passed but the
    layer isn't attention-based).
    """
    model.eval()
    with torch.no_grad():
        _, attn_per_layer = model.encode(batch, return_attention=True)

    src = batch.edge_index[0].tolist()
    dst = batch.edge_index[1].tolist()

    layer_reports: list[list[EdgeAttention]] = []
    for attn in attn_per_layer:
        if attn is None:
            layer_reports.append([])
            continue
        weights = attn.tolist()
        edges = [
            EdgeAttention(source=s, target=t, weight=w) for s, t, w in zip(src, dst, weights)
        ]
        layer_reports.append(edges)

    return AttentionReport(layer_edge_attentions=layer_reports)


@dataclass
class EmbeddingReport:
    node_ids: list[int]
    embeddings: list[list[float]]  # raw hidden_dim embeddings, pre-reduction
    coordinates: list[tuple[float, float]]  # original spatial (x, y), for comparison


def extract_node_embeddings(
    model: GraphMindGNN, batch: BatchedGraph, generated: GeneratedGraph
) -> EmbeddingReport:
    """Runs the model's encoder and returns final-layer node embeddings
    alongside each node's original spatial coordinate, so a downstream
    dimensionality-reduction plot can compare "did the model just
    re-learn (x, y), or something richer (e.g. degree, connectivity
    role)?"
    """
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(batch)

    return EmbeddingReport(
        node_ids=list(range(node_embeddings.size(0))),
        embeddings=node_embeddings.tolist(),
        coordinates=generated.coordinates,
    )
