# ADR 002: Implement message passing from scratch instead of using torch_geometric / DGL

## Status
Accepted

## Context
Standard graph learning libraries (`torch_geometric`, `dgl`) provide
production-grade, highly optimized message-passing layers. Using them would
get a working GNN running faster.

## Decision
Implement the message-passing layer (message construction, neighborhood
aggregation, node state update) directly in raw PyTorch, including the
scatter-reduce operations (`_scatter_mean`, `_scatter_max`, `_scatter_softmax`
in `src/graphmind/models/gnn.py`) rather than depending on
`torch_scatter`/`torch_geometric`.

## Alternatives considered
- **Use `torch_geometric`**: faster to build, battle-tested, but turns this
  into "I called a GNN library," which does not demonstrate understanding of
  what a GNN actually computes internally — the same reasoning that led to
  hand-implementing Dijkstra/A*/BFS in Meridian instead of using `networkx`.

## Consequences
- Every operation in the forward pass is inspectable and independently
  testable (see `tests/test_gnn.py`), including verifying gradients reach
  every parameter and that all three aggregation variants (mean/max/attention)
  produce finite, correctly-shaped output.
- Some performance is left on the table versus a fused, optimized library
  implementation — acceptable at this project's scale (graphs of hundreds to
  low thousands of nodes), and explicitly out of scope: this project
  optimizes for correctness and explainability, not production throughput at
  the scale where a hand-rolled scatter would become a bottleneck.
- If GraphMind needs to scale to very large graphs in the future, revisiting
  this decision in favor of `torch_geometric`'s optimized sparse ops would be
  the natural next step, tracked as future work.
