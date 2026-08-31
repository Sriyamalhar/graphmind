# GraphMind Benchmark

Following the same methodology as [`graphbench`](https://github.com/Sriyamalhar/graphbench):
identical inputs, identical hardware, resource constraints stated explicitly,
and results reported even when they are unflattering to the "new" approach.

## Methodology

- **Task**: predict shortest-path distance for a sampled (source, target)
  node pair on a synthetic city-grid graph (see ADR-001 for why this task).
- **Ground truth**: hand-implemented Dijkstra, run once per unique source
  node across all sampled pairs sharing that source.
- **Graph**: `make_grid_graph` — an N x M city-block grid with randomized
  edge weights (simulated travel times) and a small number of long-range
  "highway shortcut" edges, generated with a fixed seed for reproducibility.
- **Split**: transductive — same graph, disjoint (source, target) query
  pairs across train/val/test (see ADR-001, "Consequences").
- **Metrics**:
  - **MAE / RMSE** — absolute prediction error against Dijkstra ground truth.
  - **Mean relative error %** — error normalized by true distance, since
    grid graphs have a wide range of distances.
  - **Avg time per query (ms)** — wall-clock, averaged over the full test
    set. For the GNN, this includes one batched forward pass amortized over
    all test queries; for Dijkstra/A*, it's the sum of independent per-pair
    calls (`timed_dijkstra_pair` / `astar_pair`).
  - **Peak memory (KB)** — measured via `tracemalloc`, isolated per method.

## Initial results (10x10 grid, 500 sampled pairs, 60 training epochs)

| Method              | MAE  | RMSE | Rel. error % | Avg ms/query | Peak mem (KB) |
|---------------------|------|------|--------------|---------------|----------------|
| GraphMindGNN (mean) | 5.26 | 6.57 | 34.3%        | 0.021         | 2.9            |
| Dijkstra            | 0.00 | 0.00 | 0.0%         | 0.133         | 19.2           |
| A* (no heuristic)   | 0.00 | 0.00 | 0.0%         | 0.182         | 19.9           |

*(Numbers above are from an initial smoke-test run on a small 10x10 grid /
60 epochs — not yet a tuned final result. Re-run via
`scripts/run_benchmark.py` after full training and replace this table before
treating it as a final reported number. See "How to reproduce" below.)*

### Honest reading of these numbers

- **Speed**: the GNN is meaningfully faster per query once trained — because
  a single batched forward pass answers all test queries at once, versus
  Dijkstra/A* re-running independent searches per pair. This gap should widen
  further as query volume grows (the GNN's per-query marginal cost drops
  toward zero after the one-time graph encoding pass; Dijkstra/A*'s does not).
- **Accuracy**: Dijkstra and A* are exact by construction — they *are* the
  ground truth generator, so their reported error is trivially zero. The
  real question is whether the GNN's ~34% relative error is an acceptable
  tradeoff for its speed advantage, which depends entirely on the use case
  (e.g., acceptable for coarse batch estimation across thousands of pairs;
  not acceptable if exact distances are required).
- **This is not yet a "GNN wins" result** — it's an honest accuracy/speed
  tradeoff, which is the actual point of the benchmark. Further experiments
  (deeper networks, more training data, attention aggregation, more epochs)
  are needed before drawing a stronger conclusion either way.

## Planned follow-up experiments

1. **Aggregation variant comparison** — mean vs. max vs. attention
   aggregation, same architecture otherwise, same train/val/test split.
2. **Depth ablation** — 1 / 2 / 3 / 4 message-passing layers, measuring
   accuracy vs. training time vs. inference time tradeoff.
3. **Scale study** — repeat the full benchmark at 10x10, 25x25, 50x50 grid
   sizes to see how the GNN's speed advantage (or accuracy gap) changes as
   graph size grows.
4. **Generalization to unseen graphs** (stretch goal, inductive setting) —
   train on one set of generated grids, test on a held-out grid the model
   never saw during training, to measure whether the GNN learns transferable
   structure or overfits to the specific training graph.

## How to reproduce

```bash
pip install -e ".[dev]"
python scripts/run_benchmark.py --rows 10 --cols 10 --num-pairs 500 --epochs 60
```

Results are written to `docs/BENCHMARK.md`-compatible JSON in
`results/benchmark_<timestamp>.json` for tracking benchmark history across
runs/commits.
