import torch

from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.models.gnn import GraphMindGNN, MessagePassingLayer


def _sample_batch():
    generated = make_grid_graph(rows=3, cols=3, seed=0)
    return build_batched_graph(generated)


def test_message_passing_layer_output_shape():
    batch = _sample_batch()
    hidden_dim = 16
    layer = MessagePassingLayer(hidden_dim=hidden_dim, aggregation="mean")
    h = torch.randn(batch.node_features.size(0), hidden_dim)
    out = layer(h, batch.edge_index, batch.edge_weight)
    assert out.shape == h.shape


def test_all_aggregation_variants_run_without_error():
    batch = _sample_batch()
    hidden_dim = 8
    for agg in ("mean", "max", "attention"):
        layer = MessagePassingLayer(hidden_dim=hidden_dim, aggregation=agg)
        h = torch.randn(batch.node_features.size(0), hidden_dim)
        out = layer(h, batch.edge_index, batch.edge_weight)
        assert out.shape == h.shape
        assert torch.isfinite(out).all()


def test_gnn_forward_produces_nonnegative_distances():
    batch = _sample_batch()
    model = GraphMindGNN(node_feat_dim=batch.node_features.size(1), hidden_dim=16, num_layers=2)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([8, 7, 6])
    preds = model(batch, src, dst)
    assert preds.shape == (3,)
    assert (preds >= 0).all()


def test_gnn_gradients_flow_to_all_parameters():
    batch = _sample_batch()
    model = GraphMindGNN(node_feat_dim=batch.node_features.size(1), hidden_dim=16, num_layers=2)
    src = torch.tensor([0, 1])
    dst = torch.tensor([8, 7])
    target = torch.tensor([3.0, 2.0])

    preds = model(batch, src, dst)
    loss = ((preds - target) ** 2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter: {name}"


def test_gnn_same_node_pair_has_small_distance_after_training_step():
    # Sanity check only (not a convergence guarantee): after a single
    # optimizer step toward distance=0 for identical source/target,
    # the loss should not increase.
    batch = _sample_batch()
    model = GraphMindGNN(node_feat_dim=batch.node_features.size(1), hidden_dim=16, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    src = torch.tensor([4])
    dst = torch.tensor([4])
    target = torch.tensor([0.0])

    preds_before = model(batch, src, dst)
    loss_before = ((preds_before - target) ** 2).mean()

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    preds_after = model(batch, src, dst)
    loss_after = ((preds_after - target) ** 2).mean()

    assert loss_after.item() <= loss_before.item() + 1e-4
