"""Loss landscape of the *session* model (Li et al., 2018).

A 2-D slice of the dataset loss around the parameters of the live model or a
stored checkpoint:

    L(alpha, beta) = loss(theta + alpha * d1 + beta * d2)

``d1`` and ``d2`` are random Gaussian directions with *filter normalisation*:
every row of every weight matrix (one neuron's incoming weights) is rescaled
to the norm of the corresponding row of ``theta``; bias directions are zero,
as in the paper.  The directions are drawn from a fixed seed so the same
checkpoint always gives the same slice, and the slice is a real evaluation:
every grid value is one full-dataset forward pass.  It is still only a 2-D
slice of a high-dimensional surface, which the UI says.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from .mlp import DenseParams, Params, forward

GRID = 25
SPAN = 1.0


class LossLandscape(BaseModel):
    checkpoint_epoch: int
    is_latest: bool
    alphas: List[float]
    betas: List[float]
    loss: List[List[float]]  # loss[i][j] at (alpha_i, beta_j)
    center_loss: float
    min_loss: float
    max_loss: float
    seed: int
    note: str = ("2-D slice through two random filter-normalised directions around the model's parameters "
                 "(Li et al., 2018). Each value is the real cross-entropy over the whole dataset.")


def _direction(params: Params, gen: torch.Generator) -> Params:
    out: Params = []
    for p in params:
        d = torch.randn(p.weight.shape, generator=gen)
        row_norm_p = p.weight.norm(dim=1, keepdim=True)
        row_norm_d = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out.append(DenseParams(d / row_norm_d * row_norm_p, torch.zeros_like(p.bias)))
    return out


def loss_landscape(session, checkpoint_epoch: Optional[int], grid: int = GRID, span: float = SPAN,
                   seed: int = 0) -> LossLandscape:
    params = session.params_at(checkpoint_epoch)
    epoch = session.epoch if checkpoint_epoch is None else checkpoint_epoch
    gen = torch.Generator().manual_seed(seed)
    d1, d2 = _direction(params, gen), _direction(params, gen)
    alphas = torch.linspace(-span, span, grid)
    betas = torch.linspace(-span, span, grid)
    values: List[List[float]] = []
    with torch.no_grad():
        for a in alphas:
            row = []
            for b in betas:
                moved = [DenseParams(p.weight + a * u.weight + b * v.weight, p.bias)
                         for p, u, v in zip(params, d1, d2)]
                logits = forward(session.spec, moved, session.X).logits
                row.append(float(F.cross_entropy(logits, session.y)))
            values.append(row)
        center = float(F.cross_entropy(forward(session.spec, params, session.X).logits, session.y))
    flat = [v for r in values for v in r]
    return LossLandscape(
        checkpoint_epoch=epoch, is_latest=epoch == session.epoch,
        alphas=[float(a) for a in alphas], betas=[float(b) for b in betas], loss=values,
        center_loss=center, min_loss=min(flat), max_loss=max(flat), seed=seed,
    )
