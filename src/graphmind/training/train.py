"""
Training loop for GraphMindGNN.

Tracks per-epoch train/val loss and MAE so the interpretability
dashboard (docs, frontend) can plot real training curves rather than
a single final number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from graphmind.data.dataset import PairExample
from graphmind.models.gnn import BatchedGraph, GraphMindGNN


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "epoch": [e.epoch for e in self.epochs],
            "train_loss": [e.train_loss for e in self.epochs],
            "val_loss": [e.val_loss for e in self.epochs],
            "val_mae": [e.val_mae for e in self.epochs],
        }


def _examples_to_tensors(examples: list[PairExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.tensor([e.source for e in examples], dtype=torch.long)
    dst = torch.tensor([e.target for e in examples], dtype=torch.long)
    dist = torch.tensor([e.distance for e in examples], dtype=torch.float32)
    return src, dst, dist


def train_model(
    model: GraphMindGNN,
    batched_graph: BatchedGraph,
    train_examples: list[PairExample],
    val_examples: list[PairExample],
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    log_every: int = 10,
) -> TrainingHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_src, train_dst, train_dist = _examples_to_tensors(train_examples)
    val_src, val_dst, val_dist = _examples_to_tensors(val_examples)

    history = TrainingHistory()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(batched_graph, train_src, train_dst)
        loss = loss_fn(preds, train_dist)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(batched_graph, val_src, val_dst)
            val_loss = loss_fn(val_preds, val_dist).item()
            val_mae = (val_preds - val_dist).abs().mean().item()

        history.epochs.append(
            EpochMetrics(epoch=epoch, train_loss=loss.item(), val_loss=val_loss, val_mae=val_mae)
        )

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | train_loss {loss.item():.4f} "
                f"| val_loss {val_loss:.4f} | val_mae {val_mae:.4f}"
            )

    return history
