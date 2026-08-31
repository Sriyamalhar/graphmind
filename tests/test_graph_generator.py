from graphmind.data.graph_generator import make_grid_graph, make_random_geometric_graph


def test_grid_graph_node_count():
    g = make_grid_graph(rows=4, cols=5, seed=0)
    assert g.graph.num_nodes == 20
    assert len(g.coordinates) == 20


def test_grid_graph_is_connected_via_bfs():
    g = make_grid_graph(rows=3, cols=3, seed=0)
    # BFS reachability check from node 0 should reach every node.
    visited = {0}
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for v, _w in g.graph.adjacency[u]:
            if v not in visited:
                visited.add(v)
                frontier.append(v)
    assert visited == set(range(9))


def test_grid_graph_shortcuts_add_edges():
    base = make_grid_graph(rows=4, cols=4, num_shortcuts=0, seed=1)
    with_shortcuts = make_grid_graph(rows=4, cols=4, num_shortcuts=5, seed=1)
    base_edges = sum(len(v) for v in base.graph.adjacency.values())
    shortcut_edges = sum(len(v) for v in with_shortcuts.graph.adjacency.values())
    assert shortcut_edges >= base_edges


def test_random_geometric_graph_node_count_and_bounds():
    g = make_random_geometric_graph(num_nodes=30, connect_radius=0.3, seed=2)
    assert g.graph.num_nodes == 30
    for x, y in g.coordinates:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_random_geometric_graph_is_deterministic_with_seed():
    g1 = make_random_geometric_graph(num_nodes=20, seed=42)
    g2 = make_random_geometric_graph(num_nodes=20, seed=42)
    assert g1.coordinates == g2.coordinates
