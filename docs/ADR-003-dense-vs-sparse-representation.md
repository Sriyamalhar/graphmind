# ADR 003: Dense edge-index tensor representation instead of sparse CSR

## Status
Accepted

## Context
Graph structure for GNN message passing can be represented as a dense
`[2, num_edges]` edge-index tensor (as used by `torch_geometric`) or as a
sparse CSR/CSC adjacency matrix multiplied against node features.

## Decision
Use the dense edge-index + `index_add_`/`scatter_reduce_` representation
(`BatchedGraph` in `src/graphmind/models/gnn.py`) rather than building a
sparse adjacency matrix and using sparse matrix multiplication.

## Alternatives considered
- **Sparse adjacency matrix + `torch.sparse.mm`**: more memory-efficient at
  very large scale, and arguably closer to "textbook" GNN formulations
  (H' = A H W). Rejected for this project's scale (graphs of hundreds to a
  few thousand nodes, matching Meridian's simulated city size and the
  benchmark's test graphs) — the memory savings don't matter yet, and the
  edge-index representation is easier to reason about and debug when writing
  the message/aggregate/update steps out explicitly (see ADR-002).

## Consequences
- Code is more readable and closely mirrors the message-passing pseudocode
  in `src/graphmind/models/gnn.py`'s module docstring.
- Memory usage scales with `num_edges`, not with `num_nodes^2` (unlike a
  dense adjacency matrix), so this remains reasonable well past the graph
  sizes used in this project's benchmarks.
- Revisit if/when GraphMind is applied to graphs with hundreds of thousands
  of nodes or more, where the sparse formulation's memory profile would
  matter more.
