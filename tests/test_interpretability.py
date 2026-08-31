from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.interpretability.attention import extract_attention, extract_node_embeddings
from graphmind.models.gnn import GraphMindGNN


def _sample_setup(aggregation: str):
    generated = make_grid_graph(rows=4, cols=4, seed=0)
    batch = build_batched_graph(generated)
    model = GraphMindGNN(
        node_feat_dim=batch.node_features.size(1),
        hidden_dim=8,
        num_layers=2,
        aggregation=aggregation,
    )
    return generated, batch, model


def test_extract_attention_populates_weights_for_attention_aggregation():
    _generated, batch, model = _sample_setup("attention")
    report = extract_attention(model, batch)

    assert len(report.layer_edge_attentions) == 2  # num_layers
    for layer_edges in report.layer_edge_attentions:
        assert len(layer_edges) == batch.edge_index.size(1)
        for edge in layer_edges:
            assert 0.0 <= edge.weight <= 1.0


def test_extract_attention_empty_for_non_attention_aggregation():
    _generated, batch, model = _sample_setup("mean")
    report = extract_attention(model, batch)

    assert len(report.layer_edge_attentions) == 2
    for layer_edges in report.layer_edge_attentions:
        assert layer_edges == []  # mean aggregation has no attention weights


def test_top_k_edges_returns_sorted_descending():
    _generated, batch, model = _sample_setup("attention")
    report = extract_attention(model, batch)

    top5 = report.top_k_edges(layer_idx=0, k=5)
    assert len(top5) == 5
    weights = [e.weight for e in top5]
    assert weights == sorted(weights, reverse=True)


def test_top_k_edges_respects_k_larger_than_available():
    _generated, batch, model = _sample_setup("attention")
    report = extract_attention(model, batch)

    # 4x4 grid, small edge count - request more than exist
    all_edges = report.top_k_edges(layer_idx=0, k=10_000)
    assert len(all_edges) == batch.edge_index.size(1)


def test_extract_node_embeddings_shapes_and_coordinates_match():
    generated, batch, model = _sample_setup("mean")
    report = extract_node_embeddings(model, batch, generated)

    num_nodes = batch.node_features.size(0)
    assert len(report.node_ids) == num_nodes
    assert len(report.embeddings) == num_nodes
    assert len(report.coordinates) == num_nodes
    assert report.coordinates == generated.coordinates

    hidden_dim = 8
    for emb in report.embeddings:
        assert len(emb) == hidden_dim
