"""Model sessions: a real network, its dataset, training history and checkpoints.

A session is the unit the frontend talks to.  It owns *only* model state;
the probe input and the intervention list are sent with each request so
experiments never leave hidden server-side state behind.

Sessions live in a small in-process LRU registry.  This is deliberate for a
locally-run laboratory: no database, bounded memory.  It also means sessions
do not survive a backend restart and are not shared between worker
processes (run uvicorn with a single worker).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoints import CheckpointStore
from .mlp import MLPSpec, Params, clone_params, forward, init_params


@dataclass
class TrainingSettings:
    epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    reg_type: str = "None"
    reg_rate: float = 0.0
    seed: int = 0


@dataclass
class ModelSession:
    id: str
    spec: MLPSpec
    params: Params
    X: torch.Tensor  # [N, input_dim] float32
    y: torch.Tensor  # [N] int64
    dataset_name: str
    seed: int = 0
    epoch: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoints: CheckpointStore = field(default_factory=CheckpointStore)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(self, params: Optional[Params] = None) -> Dict[str, float]:
        with torch.no_grad():
            trace = forward(self.spec, params or self.params, self.X)
            loss = F.cross_entropy(trace.logits, self.y).item()
            acc = (trace.logits.argmax(-1) == self.y).float().mean().item()
        return {"loss": float(loss), "accuracy": float(acc)}

    def params_at(self, checkpoint_epoch: Optional[int]) -> Params:
        if checkpoint_epoch is None:
            return self.params
        ckpt = self.checkpoints.get(checkpoint_epoch)
        if ckpt is None:
            raise KeyError(f"no checkpoint for epoch {checkpoint_epoch}")
        return ckpt.params

    # ── training ─────────────────────────────────────────────────────────
    def record(self) -> None:
        metrics = self.evaluate()
        self.history.append({"epoch": self.epoch, **metrics})
        self.checkpoints.add(self.epoch, self.params, metrics)

    def train(self, settings: TrainingSettings) -> List[Dict[str, float]]:
        """Real mini-batch training with Adam + cross-entropy.

        Training continues from the current parameters; history and
        checkpoints are appended (epochs keep counting up).
        """
        params = clone_params(self.params, requires_grad=True)
        flat = [t for p in params for t in (p.weight, p.bias)]
        opt = torch.optim.Adam(flat, lr=settings.learning_rate)
        gen = torch.Generator().manual_seed(settings.seed + self.epoch)
        n = self.X.shape[0]
        bs = max(1, min(settings.batch_size, n))
        new_rows: List[Dict[str, float]] = []

        for _ in range(settings.epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                trace = forward(self.spec, params, self.X[idx])
                loss = F.cross_entropy(trace.logits, self.y[idx])
                loss = loss + _regularisation(params, settings.reg_type, settings.reg_rate)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.params = clone_params(params)
            self.epoch += 1
            self.record()
            new_rows.append(self.history[-1])
        return new_rows


def _regularisation(params: Params, reg_type: str, rate: float) -> torch.Tensor:
    if reg_type in (None, "None") or rate <= 0:
        return torch.zeros(())
    weights = [p.weight for p in params]
    l1 = sum(w.abs().sum() for w in weights)
    l2 = sum((w ** 2).sum() for w in weights)
    if reg_type == "L1":
        return rate * l1
    if reg_type == "L2":
        return rate * l2
    if reg_type == "L1L2":
        return rate * (l1 + l2)
    return torch.zeros(())


def create_session(
    spec: MLPSpec,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    seed: int = 0,
    checkpoint_capacity: int = 32,
) -> ModelSession:
    session = ModelSession(
        id=uuid.uuid4().hex,
        spec=spec,
        params=init_params(spec, seed),
        X=torch.as_tensor(np.asarray(X), dtype=torch.float32),
        y=torch.as_tensor(np.asarray(y), dtype=torch.long),
        dataset_name=dataset_name,
        seed=seed,
        checkpoints=CheckpointStore(checkpoint_capacity),
    )
    session.record()  # epoch 0 = initialisation
    return session


class SessionRegistry:
    """Thread-safe LRU of live sessions."""

    def __init__(self, max_sessions: int = 8) -> None:
        self.max_sessions = max_sessions
        self._items: "OrderedDict[str, ModelSession]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, session: ModelSession) -> None:
        with self._lock:
            self._items[session.id] = session
            self._items.move_to_end(session.id)
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)

    def get(self, session_id: str) -> Optional[ModelSession]:
        with self._lock:
            session = self._items.get(session_id)
            if session is not None:
                self._items.move_to_end(session_id)
            return session

    def __len__(self) -> int:
        return len(self._items)
