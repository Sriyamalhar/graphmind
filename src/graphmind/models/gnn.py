"""
A Graph Neural Network implemented from first principles in raw
PyTorch — no `torch_geometric`, `dgl`, or other graph-learning
library. The message-passing, neighborhood aggregation, and node
update steps are all written out explicitly so the mechanics are
inspectable and testable, the same way Meridian's Dijkstra/A*/BFS
were hand-implemented instead of imported.

Architecture: a stack of message-passing layers followed by a
readout that combines two node embeddings (source, target) into a
single predicted shortest-path distance.

    for each layer:
        1. MESSAGE:      m_uv = MLP([h_u, h_v, w_uv])   for each edge (u, v)
        2. AGGREGATE:     h_v' = AGG({m_uv : u in N(v)})  (mean / max / attention)
        3. UPDATE:        h_v  = MLP([h_v, h_v'])          (residual update)

    readout:
        dist_hat = MLP([h_source, h_target, |h_source - h_target|])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BatchedGraph:
    """Dense tensor representation of a batch of (typically one) graph.

    For the scale used in this project (graphs of a few hundred to a
    few thousand nodes), a dense/padded edge-index representation is
    simpler to reason about than a sparse CSR format, at some memory
    cost. `docs/ADRs/003-dense-vs-sparse-representation.md` records
    this tradeoff.
    """

    node_features: torch.Tensor  # [num_nodes, feat_dim]
    edge_index: torch.Tensor  # [2, num_edges] (source_idx, target_idx) rows
    edge_weight: torch.Tensor  # [num_edges]


class MessagePassingLayer(nn.Module):
    """A single hand-written message-passing layer.

    aggregation: "mean", "max", or "attention".
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation: Literal["mean", "max", "attention"] = "mean",
    ):
        super().__init__()
        self.aggregation = aggregation

        # MESSAGE: combines sender embedding, receiver embedding, and
        # scalar edge weight into a message vector.
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # UPDATE: combines a node's previous embedding with the
        # aggregated incoming message into a new embedding (residual).
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if aggregation == "attention":
            self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        h:           [num_nodes, hidden_dim]
        edge_index:  [2, num_edges]  (row 0 = source u, row 1 = target v)
        edge_weight: [num_edges]
        return_attention: if True and aggregation="attention", also returns
            the per-edge attention weight tensor [num_edges] used in this
            forward pass (for the interpretability dashboard). Ignored for
            mean/max aggregation (returns None in that slot instead).

        Returns updated node embeddings [num_nodes, hidden_dim], or
        (embeddings, attention_weights) if return_attention=True.
        """
        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # --- 1. MESSAGE ---
        h_src = h[src]  # [num_edges, hidden_dim]
        h_dst = h[dst]  # [num_edges, hidden_dim]
        messages = self.message_mlp(
            torch.cat([h_src, h_dst, edge_weight.unsqueeze(-1)], dim=-1)
        )  # [num_edges, hidden_dim]

        # --- 2. AGGREGATE (messages arriving at each destination node) ---
        attn_weights = None
        if self.aggregation == "mean":
            agg = _scatter_mean(messages, dst, num_nodes)
        elif self.aggregation == "max":
            agg = _scatter_max(messages, dst, num_nodes)
        elif self.aggregation == "attention":
            scores = self.attn_score(messages).squeeze(-1)  # [num_edges]
            attn_weights = _scatter_softmax(scores, dst, num_nodes)
            agg = _scatter_sum(messages * attn_weights.unsqueeze(-1), dst, num_nodes)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # --- 3. UPDATE (residual) ---
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_out = h + h_new  # residual connection stabilizes deeper stacks

        if return_attention:
            return h_out, attn_weights
        return h_out


class GraphMindGNN(nn.Module):
    """Full model: input projection -> K message-passing layers -> readout.

    Predicts shortest-path distance between a (source, target) node
    pair given the full graph as context.
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        aggregation: Literal["mean", "max", "attention"] = "mean",
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, aggregation) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self, batch: BatchedGraph, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Runs message passing and returns final node embeddings.

        If return_attention=True, also returns a list (one entry per
        layer) of that layer's per-edge attention weights — only
        populated (non-None) for layers using attention aggregation.
        Used by the interpretability dashboard to visualize which edges
        the model relies on most.
        """
        h = self.input_proj(batch.node_features)
        attn_per_layer: list[torch.Tensor | None] = []
        for layer in self.layers:
            if return_attention:
                h, attn = layer(h, batch.edge_index, batch.edge_weight, return_attention=True)
                attn_per_layer.append(attn)
            else:
                h = layer(h, batch.edge_index, batch.edge_weight)

        if return_attention:
            return h, attn_per_layer
        return h

    def predict_distance(
        self, node_embeddings: torch.Tensor, source_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """Given node embeddings and batches of (source, target) indices,
        predicts shortest-path distance for each pair.

        source_idx, target_idx: [batch_size] long tensors
        Returns: [batch_size] predicted distances (non-negative via softplus).
        """
        h_s = node_embeddings[source_idx]
        h_t = node_embeddings[target_idx]
        combined = torch.cat([h_s, h_t, torch.abs(h_s - h_t)], dim=-1)
        raw = self.readout(combined).squeeze(-1)
        return F.softplus(raw)  # distances are non-negative

    def forward(
        self, batch: BatchedGraph, source_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        node_embeddings = self.encode(batch)
        return self.predict_distance(node_embeddings, source_idx, target_idx)


# ---------------------------------------------------------------------------
# Scatter helper functions (hand-implemented, no torch_scatter dependency)
# ---------------------------------------------------------------------------


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, src.size(-1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    summed = _scatter_sum(src, index, num_nodes)
    counts = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
    counts.index_add_(0, index, torch.ones(index.size(0), dtype=src.dtype, device=src.device))
    counts = counts.clamp(min=1).unsqueeze(-1)
    return summed / counts


def _scatter_max(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-destination-node max over incoming messages.

    Uses `scatter_reduce` (out-of-place, autograd-friendly) rather than a
    Python loop with in-place tensor writes — the original loop-based
    version broke gradient tracking under backprop (in-place mutation of
    a tensor autograd needs for the backward pass), so this scatter_reduce
    formulation is required, not just a style preference.
    """
    expanded_index = index.unsqueeze(-1).expand(-1, src.size(-1))
    out = torch.full(
        (num_nodes, src.size(-1)), float("-inf"), dtype=src.dtype, device=src.device
    )
    out = out.scatter_reduce(0, expanded_index, src, reduce="amax", include_self=True)
    # Nodes with no incoming edges stay at -inf; zero them out instead.
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` grouped by `index` (i.e. per destination node)."""
    max_per_node = torch.full((num_nodes,), float("-inf"), dtype=scores.dtype, device=scores.device)
    max_per_node.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    shifted = scores - max_per_node[index]
    exp = torch.exp(shifted)
    sum_per_node = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    sum_per_node.index_add_(0, index, exp)
    return exp / sum_per_node[index].clamp(min=1e-12)
