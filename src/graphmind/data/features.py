"""
Converts a `Graph` + `GeneratedGraph` (with coordinates) into the
tensor format `GraphMindGNN` expects: node features, edge index,
edge weights.

Node features used: (x, y) spatial coordinates + node degree.
Keeping features simple and interpretable is intentional — it makes
the embedding-space visualizations in the interpretability dashboard
easier to reason about (e.g. "did the model learn something beyond
raw coordinates?").
"""

from __future__ import annotations

import torch

from graphmind.data.graph_generator import GeneratedGraph
from graphmind.models.gnn import BatchedGraph


def build_batched_graph(generated: GeneratedGraph) -> BatchedGraph:
    graph = generated.graph
    n = graph.num_nodes

    degrees = [len(graph.adjacency.get(i, [])) for i in range(n)]
    coords = generated.coordinates

    node_features = torch.tensor(
        [[coords[i][0], coords[i][1], float(degrees[i])] for i in range(n)],
        dtype=torch.float32,
    )
    # Normalize degree feature so its scale doesn't dominate coordinates.
    if node_features[:, 2].max() > 0:
        node_features[:, 2] = node_features[:, 2] / node_features[:, 2].max()

    src_list, dst_list, weight_list = [], [], []
    for u, neighbors in graph.adjacency.items():
        for v, w in neighbors:
            # Add both directions explicitly since message passing here
            # treats edges as directed messages; the underlying graph
            # is undirected so both (u->v) and (v->u) should exist.
            src_list.append(u)
            dst_list.append(v)
            weight_list.append(w)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    return BatchedGraph(node_features=node_features, edge_index=edge_index, edge_weight=edge_weight)
