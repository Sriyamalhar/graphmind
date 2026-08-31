
from graphmind.benchmark.suite import _score_predictions, benchmark_classical, benchmark_gnn
from graphmind.data.classical_algorithms import dijkstra_pair
from graphmind.data.dataset import build_pair_dataset
from graphmind.data.features import build_batched_graph
from graphmind.data.graph_generator import make_grid_graph
from graphmind.models.gnn import GraphMindGNN


def test_score_predictions_perfect_match_is_zero_error():
    preds = [1.0, 2.0, 3.0]
    truths = [1.0, 2.0, 3.0]
    mae, rmse, rel = _score_predictions(preds, truths)
    assert mae == 0.0
    assert rmse == 0.0
    assert rel == 0.0


def test_score_predictions_known_error():
    preds = [2.0]
    truths = [1.0]
    mae, rmse, rel = _score_predictions(preds, truths)
    assert mae == 1.0
    assert rmse == 1.0
    assert rel == 100.0


def test_benchmark_classical_runs_and_returns_result():
    generated = make_grid_graph(rows=4, cols=4, seed=0)
    dataset = build_pair_dataset(generated, num_pairs=10, seed=1)
    result = benchmark_classical(generated.graph, dataset.examples, dijkstra_pair, "Dijkstra")
    assert result.method == "Dijkstra"
    assert result.mae == 0.0  # Dijkstra vs. its own labels is exact
    assert result.total_wall_time_sec >= 0.0
    assert result.avg_time_per_query_ms >= 0.0


def test_benchmark_gnn_runs_and_returns_result():
    generated = make_grid_graph(rows=4, cols=4, seed=0)
    dataset = build_pair_dataset(generated, num_pairs=10, seed=1)
    batch = build_batched_graph(generated)
    model = GraphMindGNN(node_feat_dim=batch.node_features.size(1), hidden_dim=8, num_layers=2)

    result = benchmark_gnn(model, batch, dataset.examples)
    assert result.method == "GraphMindGNN"
    assert result.mae >= 0.0
    assert result.peak_memory_kb >= 0.0
