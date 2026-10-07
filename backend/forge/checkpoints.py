"""Bounded in-memory checkpoint store (backbone of the Training Time Machine).

Storage policy
--------------
* Every epoch is offered to the store; the store keeps at most
  ``capacity`` snapshots.
* When full, it evicts the *interior* snapshot whose removal leaves the
  smallest gap on a **balanced time axis** (half linear, half logarithmic
  in the epoch number).  The first (initialisation) and the latest snapshot
  are never evicted.  The result covers the whole run: dense early on, where
  a network changes fastest, but with no long blind spots later.  (The
  previous "drop every second snapshot" policy left e.g. nothing between
  epoch 0 and 224 after 300 epochs.)
* Snapshots are float32 copies of the parameters and are immutable once
  stored: ``Checkpoint`` is frozen, and every consumer clones before doing
  anything in-place (see ``interventions.compile_interventions`` and
  ``mlp.clone_params``).  Memory stays O(capacity * n_params) no matter how
  long training runs; for the MLPs Neural Forge supports (<= ~66k
  parameters) 48 snapshots cost < 13 MB, typical models far less.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .mlp import Params, clone_params

DEFAULT_CAPACITY = 48


@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    params: Params
    metrics: Dict[str, float] = field(default_factory=dict)


def _axis(epoch: int, horizon: int) -> float:
    """Position on the balanced time axis, in [0, 2] for epochs in [0, horizon]."""
    h = max(1, horizon)
    return epoch / h + math.log1p(epoch) / math.log1p(h)


class CheckpointStore:
    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.capacity = capacity
        self._items: List[Checkpoint] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def epochs(self) -> List[int]:
        return [c.epoch for c in self._items]

    @property
    def items(self) -> List[Checkpoint]:
        """Snapshots in epoch order (a copy of the list; the snapshots themselves are shared)."""
        return list(self._items)

    def add(self, epoch: int, params: Params, metrics: Optional[Dict[str, float]] = None) -> None:
        if self._items and epoch <= self._items[-1].epoch:
            # Re-recording an epoch (e.g. after re-initialisation) replaces history.
            self._items = [c for c in self._items if c.epoch < epoch]
        self._items.append(Checkpoint(epoch, clone_params(params), dict(metrics or {})))
        while len(self._items) > self.capacity:
            self._evict()

    def _evict(self) -> None:
        horizon = self._items[-1].epoch
        pos = [_axis(c.epoch, horizon) for c in self._items]
        # Removing item i merges the gaps on both sides of it; drop the one
        # that creates the smallest merged gap (ties: the earlier one).
        best = min(range(1, len(self._items) - 1), key=lambda i: (pos[i + 1] - pos[i - 1], i))
        del self._items[best]

    def get(self, epoch: int) -> Optional[Checkpoint]:
        for c in self._items:
            if c.epoch == epoch:
                return c
        return None

    def previous(self, epoch: int) -> Optional[Checkpoint]:
        """The latest stored snapshot strictly before ``epoch``."""
        prev = None
        for c in self._items:
            if c.epoch >= epoch:
                break
            prev = c
        return prev

    def nearest(self, epoch: int) -> Optional[int]:
        """Stored epoch closest to ``epoch`` (ties: the later one)."""
        if not self._items:
            return None
        return min(self.epochs, key=lambda e: (abs(e - epoch), -e))

    def clear(self) -> None:
        self._items = []

    def summary(self) -> List[Dict]:
        return [{"epoch": c.epoch, **c.metrics} for c in self._items]
