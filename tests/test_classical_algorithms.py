import math

from graphmind.data.classical_algorithms import Graph, astar_pair, dijkstra, dijkstra_pair


def make_simple_graph() -> Graph:
    #   0 --1-- 1
    #   |       |
    #   4       2
    #   |       |
    #   3 --1-- 2
    edges = [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 1.0),
        (0, 3, 4.0),
    ]
    return Graph.from_edge_list(num_nodes=4, edges=edges)


def test_dijkstra_single_source_distances():
    graph = make_simple_graph()
    dist = dijkstra(graph, 0)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    assert dist[2] == 3.0  # via 0-1-2
    assert dist[3] == 4.0  # min(0-3 direct=4, 0-1-2-3=4) -> tie, both 4.0


def test_dijkstra_pair_matches_single_source():
    graph = make_simple_graph()
    full = dijkstra(graph, 0)
    for target in range(4):
        assert dijkstra_pair(graph, 0, target) == full[target]


def test_dijkstra_pair_same_node_is_zero():
    graph = make_simple_graph()
    assert dijkstra_pair(graph, 2, 2) == 0.0


def test_dijkstra_unreachable_node_is_infinite():
    edges = [(0, 1, 1.0)]
    graph = Graph.from_edge_list(num_nodes=3, edges=edges)  # node 2 isolated
    assert dijkstra_pair(graph, 0, 2) == math.inf


def test_astar_matches_dijkstra_without_heuristic():
    graph = make_simple_graph()
    for target in range(4):
        d = dijkstra_pair(graph, 0, target)
        a = astar_pair(graph, 0, target)
        assert math.isclose(d, a, rel_tol=1e-9)


def test_astar_with_admissible_heuristic_matches_dijkstra():
    graph = make_simple_graph()
    # A trivially admissible (and useless) heuristic: always 0.
    heuristic = {(u, 3): 0.0 for u in range(4)}
    d = dijkstra_pair(graph, 0, 3)
    a = astar_pair(graph, 0, 3, heuristic=heuristic)
    assert math.isclose(d, a, rel_tol=1e-9)
