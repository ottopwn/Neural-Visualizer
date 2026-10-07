"""Bounded in-memory checkpoint store (foundation for the Training Time Machine).

Storage policy
--------------
* Every epoch is offered to the store; the store keeps at most
  ``capacity`` snapshots.
* When full, it *thins*: every second snapshot is dropped, except the first
  (initialisation) and the most recent one.  Repeated thinning yields an
  approximately geometric spacing biased towards the latest epochs, so
  memory stays O(capacity * n_params) no matter how long training runs.
* Snapshots are float32 copies of the parameters.  For the MLPs Neural
  Forge supports (<= ~20k parameters) 32 snapshots cost < 3 MB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .mlp import Params, clone_params


@dataclass
class Checkpoint:
    epoch: int
    params: Params
    metrics: Dict[str, float] = field(default_factory=dict)


class CheckpointStore:
    def __init__(self, capacity: int = 32) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.capacity = capacity
        self._items: List[Checkpoint] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def epochs(self) -> List[int]:
        return [c.epoch for c in self._items]

    def add(self, epoch: int, params: Params, metrics: Optional[Dict[str, float]] = None) -> None:
        if self._items and epoch <= self._items[-1].epoch:
            # Re-recording an epoch (e.g. after re-initialisation) replaces history.
            self._items = [c for c in self._items if c.epoch < epoch]
        self._items.append(Checkpoint(epoch, clone_params(params), dict(metrics or {})))
        if len(self._items) > self.capacity:
            self._thin()

    def _thin(self) -> None:
        first, middle, last = self._items[0], self._items[1:-1], self._items[-1]
        # keep every second item of the middle, preferring later ones
        kept = middle[len(middle) % 2::2] if middle else []
        self._items = [first, *kept, last]

    def get(self, epoch: int) -> Optional[Checkpoint]:
        for c in self._items:
            if c.epoch == epoch:
                return c
        return None

    def clear(self) -> None:
        self._items = []

    def summary(self) -> List[Dict]:
        return [{"epoch": c.epoch, **c.metrics} for c in self._items]
