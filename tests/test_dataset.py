from graphmind.data.classical_algorithms import dijkstra_pair
from graphmind.data.dataset import build_pair_dataset, train_val_test_split
from graphmind.data.graph_generator import make_grid_graph


def test_build_pair_dataset_labels_match_dijkstra():
    generated = make_grid_graph(rows=4, cols=4, seed=0)
    dataset = build_pair_dataset(generated, num_pairs=20, seed=1)

    assert len(dataset.examples) > 0
    for ex in dataset.examples:
        expected = dijkstra_pair(generated.graph, ex.source, ex.target)
        assert abs(ex.distance - expected) < 1e-9


def test_train_val_test_split_sizes_and_no_overlap():
    generated = make_grid_graph(rows=5, cols=5, seed=0)
    dataset = build_pair_dataset(generated, num_pairs=200, seed=1)

    train, val, test = train_val_test_split(dataset, val_frac=0.2, test_frac=0.2, seed=2)

    total = len(train) + len(val) + len(test)
    assert total == len(dataset.examples)

    # No example object should appear in more than one split.
    train_ids = {id(e) for e in train}
    val_ids = {id(e) for e in val}
    test_ids = {id(e) for e in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_is_reproducible_with_same_seed():
    generated = make_grid_graph(rows=5, cols=5, seed=0)
    dataset = build_pair_dataset(generated, num_pairs=100, seed=1)

    train1, val1, test1 = train_val_test_split(dataset, seed=5)
    train2, val2, test2 = train_val_test_split(dataset, seed=5)

    assert [(e.source, e.target) for e in train1] == [(e.source, e.target) for e in train2]
    assert [(e.source, e.target) for e in val1] == [(e.source, e.target) for e in val2]
    assert [(e.source, e.target) for e in test1] == [(e.source, e.target) for e in test2]
