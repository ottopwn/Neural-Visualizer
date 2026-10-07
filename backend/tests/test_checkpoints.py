import torch

from forge.checkpoints import CheckpointStore
from forge.mlp import MLPSpec, init_params


def _params():
    return init_params(MLPSpec(2, (3,), ("ReLU",), 2))


def test_capacity_and_thinning_keep_first_and_last():
    store = CheckpointStore(capacity=8)
    for epoch in range(100):
        store.add(epoch, _params(), {"loss": float(epoch)})
        assert len(store) <= 8
    assert store.epochs[0] == 0
    assert store.epochs[-1] == 99
    assert store.epochs == sorted(store.epochs)


def test_snapshots_are_copies():
    store = CheckpointStore()
    params = _params()
    store.add(0, params)
    params[0].weight.add_(1.0)
    assert not torch.equal(store.get(0).params[0].weight, params[0].weight)


def test_rerecording_an_epoch_truncates_newer_history():
    store = CheckpointStore()
    for e in range(5):
        store.add(e, _params())
    store.add(2, _params())
    assert store.epochs == [0, 1, 2]
    assert store.get(4) is None
