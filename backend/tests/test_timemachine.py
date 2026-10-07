"""Training Time Machine: every number must match an independent computation
on the stored checkpoint, and nothing may ever modify a stored checkpoint."""
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from datasets import generate_dataset
from forge import schema as S
from forge.checkpoints import CheckpointStore
from forge.interventions import AblateNeuron, SetBias, SetWeight
from forge.introspect import IntrospectionError, MLPIntrospector
from forge.mlp import MLPSpec, forward, init_params
from forge.session import TrainingSettings, create_session
from forge.timemachine import TimeMachine
from main import app


@pytest.fixture
def trained(session):
    session.train(TrainingSettings(epochs=12, learning_rate=0.05))
    session.train(TrainingSettings(epochs=8, learning_rate=0.01, batch_size=16))
    return session


@pytest.fixture
def tm(trained):
    return TimeMachine(trained)


def snapshot(session):
    """Deep copy of the live weights and every stored checkpoint."""
    copy = lambda ps: [(p.weight.clone(), p.bias.clone()) for p in ps]  # noqa: E731
    return copy(session.params), {c.epoch: copy(c.params) for c in session.checkpoints.items}


def assert_same(a, b):
    live_a, ck_a = a
    live_b, ck_b = b
    assert ck_a.keys() == ck_b.keys()
    for x, y in zip([live_a, *ck_a.values()], [live_b, *ck_b.values()]):
        for (wa, ba), (wb, bb) in zip(x, y):
            assert torch.equal(wa, wb) and torch.equal(ba, bb)


# ── checkpoint store ────────────────────────────────────────────────────────
def _p():
    return init_params(MLPSpec(2, (3,), ("ReLU",), 2))


@pytest.mark.parametrize("n_epochs", [10, 60, 300, 1500])
def test_retention_is_ordered_bounded_and_covers_the_whole_run(n_epochs):
    store = CheckpointStore(capacity=48)
    for e in range(n_epochs + 1):
        store.add(e, _p())
        assert len(store) <= 48
    epochs = store.epochs
    assert epochs == sorted(set(epochs))
    assert epochs[0] == 0 and epochs[-1] == n_epochs
    if n_epochs + 1 <= 48:
        assert epochs == list(range(n_epochs + 1))
    else:
        assert epochs[:4] == [0, 1, 2, 3]  # early learning is kept
        gaps = np.diff(epochs)
        assert gaps.max() <= 0.07 * n_epochs  # no long blind spot anywhere


def test_checkpoints_are_frozen_copies():
    store = CheckpointStore()
    params = _p()
    store.add(0, params)
    params[0].weight.add_(1.0)
    assert not torch.equal(store.get(0).params[0].weight, params[0].weight)
    with pytest.raises(dataclasses.FrozenInstanceError):
        store.get(0).params = _p()


def test_previous_and_nearest():
    store = CheckpointStore()
    for e in (0, 2, 5, 9):
        store.add(e, _p())
    assert store.previous(5).epoch == 2 and store.previous(0) is None and store.previous(100).epoch == 9
    assert store.nearest(3) == 2 and store.nearest(4) == 5 and store.nearest(6) == 5
    assert store.nearest(7) == 9 and store.nearest(50) == 9  # tie -> later


# ── training statistics ─────────────────────────────────────────────────────
def test_update_norm_matches_consecutive_checkpoints(trained):
    ck = {c.epoch: c.params for c in trained.checkpoints.items}
    for row in trained.history[1:]:
        e = row["epoch"]
        if e - 1 not in ck or e not in ck:
            continue
        per_layer = [float(((a.weight - b.weight) ** 2).sum() + ((a.bias - b.bias) ** 2).sum()) ** 0.5
                     for a, b in zip(ck[e], ck[e - 1])]
        assert row["layer_update_norms"] == pytest.approx(per_layer, rel=1e-5, abs=1e-7)
        assert row["update_norm"] == pytest.approx(sum(v ** 2 for v in per_layer) ** 0.5, rel=1e-5)


def test_grad_norm_is_the_real_training_gradient():
    """With one full batch per epoch the logged norm is the full-data gradient at the epoch start."""
    X, y = generate_dataset("Gaussian", 0.1)
    s = create_session(MLPSpec(2, (5,), ("Tanh",), 2), np.asarray(X), np.asarray(y), "Gaussian", seed=2)
    s.train(TrainingSettings(epochs=3, learning_rate=0.02, batch_size=10_000))
    for e in (1, 2, 3):
        start = [(c.weight.clone().requires_grad_(True), c.bias.clone().requires_grad_(True))
                 for c in s.checkpoints.get(e - 1).params]
        h = s.X
        for k, (w, b) in enumerate(start):
            z = h @ w.T + b
            h = z if k == len(start) - 1 else torch.tanh(z)
        F.cross_entropy(h, s.y).backward()
        expected = [float((w.grad ** 2).sum() + (b.grad ** 2).sum()) ** 0.5 for w, b in start]
        row = s.history[e]
        assert row["layer_grad_norms"] == pytest.approx(expected, rel=1e-4)
        assert row["grad_norm"] == pytest.approx(sum(v ** 2 for v in expected) ** 0.5, rel=1e-4)


# ── timeline ────────────────────────────────────────────────────────────────
def test_timeline_matches_history_and_checkpoints(tm, trained):
    t = tm.timeline()
    assert [r.epoch for r in t.history] == list(range(21))
    assert [c.epoch for c in t.checkpoints] == trained.checkpoints.epochs
    assert t.live_epoch == 20 and len(t.runs) == 2 and t.runs[1].start_epoch == 12
    for c in t.checkpoints:
        assert (c.loss, c.accuracy) == pytest.approx((trained.history[c.epoch]["loss"], trained.history[c.epoch]["accuracy"]))
    assert t.history[0].grad_norm is None and t.history[5].grad_norm > 0

    # layer health at one checkpoint, recomputed independently
    c = trained.checkpoints.get(7)
    with torch.no_grad():
        trace = forward(trained.spec, c.params, trained.X)
    h1 = next(x for x in t.checkpoints if x.epoch == 7).layers[0]
    a = trace.post[0]
    assert h1.weight_norm == pytest.approx(float(c.params[0].weight.norm()))
    assert h1.mean_abs_activation == pytest.approx(float(a.abs().mean()))
    assert h1.saturated_fraction == pytest.approx(float((a.abs() > 0.99).float().mean()))  # Tanh layer
    h2 = next(x for x in t.checkpoints if x.epoch == 7).layers[1]
    assert h2.dead_fraction == pytest.approx(float((trace.post[1].abs() < 1e-6).all(0).float().mean()))
    assert h2.saturated_fraction is None  # ReLU does not saturate
    assert h1.update_ratio == pytest.approx(h1.update_norm / h1.param_norm)


def test_events_are_derived_from_the_log(tm, trained):
    events = {e.kind: e for e in tm.timeline().events}
    assert events["init"].epoch == 0
    hist = trained.history
    best = max(hist[1:], key=lambda r: (r["accuracy"], -r["epoch"]))
    assert events["best_accuracy"].epoch == best["epoch"]
    low = min(hist[1:], key=lambda r: r["loss"])
    assert events["min_loss"].epoch == low["epoch"]
    for e in events.values():
        assert e.checkpoint_epoch in trained.checkpoints.epochs
    first = next((r["epoch"] for r in hist if r["accuracy"] >= 0.9), None)
    if first:
        assert events["acc_threshold"].epoch == first


# ── frames ──────────────────────────────────────────────────────────────────
def test_frame_is_the_stored_checkpoint(tm, trained):
    f = tm.frame(S.FrameRequest(checkpoint_epoch=5, probe=S.Probe(sample_index=3)))
    params = trained.checkpoints.get(5).params
    with torch.no_grad():
        trace = forward(trained.spec, params, trained.X)
    assert f.epoch == 5 and not f.is_latest
    assert f.loss == pytest.approx(trained.history[5]["loss"])
    assert f.accuracy == pytest.approx(trained.history[5]["accuracy"])
    assert f.predictions == trace.logits.argmax(-1).tolist()
    assert f.confidence == pytest.approx(trace.probabilities.max(-1).values.tolist())
    assert f.probe_probabilities == pytest.approx(trace.probabilities[3].tolist())
    # boundary corner = network evaluated at (x_min, y_min)
    corner = torch.tensor([[f.boundary.x_range[0], f.boundary.y_range[0]]])
    with torch.no_grad():
        p1 = forward(trained.spec, params, corner).probabilities[0, 1]
    assert f.boundary.values[0][0] == pytest.approx(float(p1), abs=1e-6)

    prev = trained.checkpoints.get(4).params
    with torch.no_grad():
        prev_pred = forward(trained.spec, prev, trained.X).logits.argmax(-1)
    now_pred = trace.logits.argmax(-1)
    assert f.previous.epoch == 4
    assert f.previous.changed == int((prev_pred != now_pred).sum())
    ok_prev, ok_now = prev_pred == trained.y, now_pred == trained.y
    assert f.previous.fixed == int((~ok_prev & ok_now).sum())
    assert f.previous.broken == int((ok_prev & ~ok_now).sum())


def test_live_frame_and_unknown_epoch(tm):
    live = tm.frame(S.FrameRequest(checkpoint_epoch=None))
    assert live.is_latest and live.epoch == 20
    assert tm.frame(S.FrameRequest(checkpoint_epoch=0)).previous is None
    with pytest.raises(IntrospectionError):
        tm.frame(S.FrameRequest(checkpoint_epoch=999))


# ── one component across time ───────────────────────────────────────────────
def test_neuron_history_tracks_the_stored_parameters(tm, trained):
    ref = S.ComponentRef(kind="neuron", layer=2, index=3)
    h = tm.component_history(S.ComponentHistoryRequest(ref=ref, probe=S.Probe(sample_index=7)))
    assert h.name == "L2N4" and h.epochs == trained.checkpoints.epochs
    series = {s.key: s.values for s in h.series}
    for i, e in enumerate(h.epochs):
        p = trained.checkpoints.get(e).params
        with torch.no_grad():
            trace = forward(trained.spec, p, trained.X)
        assert series["bias"][i] == pytest.approx(float(p[1].bias[3]))
        assert series["w_in_norm"][i] == pytest.approx(float(p[1].weight[3].norm()))
        assert series["a_probe"][i] == pytest.approx(float(trace.post[1][7, 3]), abs=1e-6)
        assert series["act_mean"][i] == pytest.approx(float(trace.post[1][:, 3].mean()), abs=1e-6)

    # the microscope's own historical inspection reports the same numbers
    n = MLPIntrospector(trained).inspect(S.InspectRequest(ref=ref, probe=S.Probe(sample_index=7), checkpoint_epoch=h.epochs[3]))
    assert n.bias == pytest.approx(series["bias"][3])
    assert n.value == pytest.approx(series["a_probe"][3], abs=1e-6)
    assert n.grad_bias == pytest.approx(series["grad_bias"][3], abs=1e-6)


def test_layer_and_connection_history(tm, trained):
    lh = tm.component_history(S.ComponentHistoryRequest(ref=S.ComponentRef(kind="layer", layer=1)))
    s = {x.key: x.values for x in lh.series}
    assert s["train_grad_norm"][0] is None  # epoch 0 was never trained
    assert s["train_grad_norm"][-1] == pytest.approx(trained.history[-1]["layer_grad_norms"][0])
    ch = tm.component_history(S.ComponentHistoryRequest(
        ref=S.ComponentRef(kind="connection", layer=1, source=1, target=4), probe=S.Probe(sample_index=2)))
    s = {x.key: x.values for x in ch.series}
    for i, e in enumerate(ch.epochs):
        w = float(trained.checkpoints.get(e).params[0].weight[4, 1])
        assert s["weight"][i] == pytest.approx(w)
        assert s["contribution"][i] == pytest.approx(w * float(trained.X[2, 1]), abs=1e-6)
    with pytest.raises(IntrospectionError):
        tm.component_history(S.ComponentHistoryRequest(ref=S.ComponentRef(kind="neuron", layer=1, index=99)))


# ── A vs B ──────────────────────────────────────────────────────────────────
def test_epoch_comparison_is_exact(tm, trained):
    ref = S.ComponentRef(kind="neuron", layer=1, index=2)
    c = tm.compare(S.EpochCompareRequest(epoch_a=2, epoch_b=None, probe=S.Probe(sample_index=1), ref=ref))
    ha, hb = trained.history[2], trained.history[20]
    assert c.a.epoch == 2 and c.b.epoch == 20
    assert c.loss_delta == pytest.approx(hb["loss"] - ha["loss"])
    assert c.accuracy_delta == pytest.approx(hb["accuracy"] - ha["accuracy"])

    pa, pb = trained.checkpoints.get(2).params, trained.params
    with torch.no_grad():
        ta, tb = forward(trained.spec, pa, trained.X), forward(trained.spec, pb, trained.X)
    pred_a, pred_b = ta.logits.argmax(-1), tb.logits.argmax(-1)
    n = len(trained.y)
    assert c.changed == int((pred_a != pred_b).sum()) == len(c.changed_indices)
    assert c.changed_fraction == pytest.approx(c.changed / n)
    assert c.fixed - c.broken == round(c.accuracy_delta * n)
    p_true = lambda t: t.probabilities.gather(1, trained.y[:, None])[:, 0]  # noqa: E731
    assert c.confidence_delta == pytest.approx((p_true(tb) - p_true(ta)).tolist(), abs=1e-6)

    for layer in c.layers:
        k = layer.layer - 1
        assert layer.weight_delta_norm == pytest.approx(float((pb[k].weight - pa[k].weight).norm()), rel=1e-5)
        assert layer.bias_delta_norm == pytest.approx(float((pb[k].bias - pa[k].bias).norm()), rel=1e-5)
        per_neuron = ((pb[k].weight - pa[k].weight) ** 2).sum(1) + (pb[k].bias - pa[k].bias) ** 2
        assert layer.top_neurons[0] == int(per_neuron.argmax())
    total = sum(float(((b.weight - a.weight) ** 2).sum() + ((b.bias - a.bias) ** 2).sum()) for a, b in zip(pa, pb)) ** 0.5
    assert c.total_delta_norm == pytest.approx(total, rel=1e-5)

    # component rows agree with the history endpoint, and delta = b - a
    h = tm.component_history(S.ComponentHistoryRequest(ref=ref, probe=S.Probe(sample_index=1)))
    series = {s.key: s.values for s in h.series}
    ia, ib = h.epochs.index(2), h.epochs.index(20)
    for row in c.component.rows:
        assert row.a == pytest.approx(series[row.key][ia], abs=1e-6)
        assert row.b == pytest.approx(series[row.key][ib], abs=1e-6)
        assert row.delta == pytest.approx(row.b - row.a, abs=1e-9)
    assert c.component.response_a is not None and c.component.response_a.values != c.component.response_b.values


def test_comparing_an_epoch_with_itself_changes_nothing(tm):
    c = tm.compare(S.EpochCompareRequest(epoch_a=6, epoch_b=6))
    assert c.changed == c.fixed == c.broken == 0
    assert c.loss_delta == 0 and c.total_delta_norm == 0 and c.boundary_flip_fraction == 0


# ── immutability ────────────────────────────────────────────────────────────
def test_nothing_mutates_stored_checkpoints(trained):
    before = snapshot(trained)
    ivs = [AblateNeuron(layer=1, index=0), SetWeight(layer=2, source=1, target=1, value=7.0),
           SetBias(layer=3, index=1, value=-4.0)]
    intro, tm = MLPIntrospector(trained), TimeMachine(trained)
    for epoch in (0, 5, None):
        req = dict(probe=S.Probe(sample_index=4), interventions=ivs, checkpoint_epoch=epoch)
        intro.graph(S.ExperimentRequest(**req))
        intro.compare(S.ExperimentRequest(**req))
        for ref in ({"kind": "neuron", "layer": 2, "index": 1}, {"kind": "layer", "layer": 1},
                    {"kind": "connection", "layer": 2, "source": 1, "target": 1}):
            intro.inspect(S.InspectRequest(ref=S.ComponentRef(**ref), **req))
    tm.timeline()
    tm.frame(S.FrameRequest(checkpoint_epoch=5))
    tm.component_history(S.ComponentHistoryRequest(ref=S.ComponentRef(kind="neuron", layer=1, index=0)))
    tm.compare(S.EpochCompareRequest(epoch_a=0, epoch_b=5, ref=S.ComponentRef(kind="layer", layer=2)))
    assert_same(before, snapshot(trained))


def test_what_if_on_a_checkpoint_is_a_temporary_overlay(trained):
    intro = MLPIntrospector(trained)
    ref = S.ComponentRef(kind="neuron", layer=1, index=1)
    edited = intro.inspect(S.InspectRequest(ref=ref, checkpoint_epoch=3,
                                            interventions=[SetBias(layer=1, index=1, value=5.0)]))
    assert edited.bias == 5.0 and edited.provenance.checkpoint_epoch == 3
    assert edited.provenance.interventions_applied == 1 and not edited.provenance.is_latest
    clean = intro.inspect(S.InspectRequest(ref=ref, checkpoint_epoch=3))
    assert clean.bias == pytest.approx(float(trained.checkpoints.get(3).params[0].bias[1]))
    assert edited.bias_edit.original == pytest.approx(clean.bias)


# ── HTTP ────────────────────────────────────────────────────────────────────
def test_time_machine_api():
    client = TestClient(app)
    sid = client.post("/api/forge/sessions", json={"neurons": [5], "activations": ["ReLU"]}).json()["session_id"]
    client.post(f"/api/forge/sessions/{sid}/train", json={"epochs": 4})
    t = client.get(f"/api/forge/sessions/{sid}/timeline")
    assert t.status_code == 200 and [c["epoch"] for c in t.json()["checkpoints"]] == [0, 1, 2, 3, 4]
    assert t.json()["history"][2]["grad_norm"] > 0
    assert client.post(f"/api/forge/sessions/{sid}/frame", json={"checkpoint_epoch": 2}).json()["epoch"] == 2
    assert client.post(f"/api/forge/sessions/{sid}/frame", json={"checkpoint_epoch": 7}).status_code == 422
    h = client.post(f"/api/forge/sessions/{sid}/component-history", json={"ref": {"kind": "layer", "layer": 1}})
    assert h.status_code == 200 and h.json()["epochs"] == [0, 1, 2, 3, 4]
    cmp = client.post(f"/api/forge/sessions/{sid}/epoch-compare", json={"epoch_a": 0, "epoch_b": 4})
    assert cmp.status_code == 200 and cmp.json()["component"] is None
    assert client.post(f"/api/forge/sessions/{sid}/epoch-compare",
                       json={"epoch_a": 0, "ref": {"kind": "neuron", "layer": 1, "index": 9}}).status_code == 422
    assert client.get("/api/forge/sessions/nope/timeline").status_code == 404
    summary = client.get(f"/api/forge/sessions/{sid}").json()
    assert summary["history"][1]["update_norm"] > 0
