"""The Pass Explorer trace must be exactly what autograd computes."""
import math

import pytest
import torch
import torch.nn.functional as F

from forge import schema as S
from forge.introspect import IntrospectionError, MLPIntrospector
from forge.mlp import activation_fn
from forge.session import TrainingSettings


@pytest.fixture
def intro(session):
    session.train(TrainingSettings(epochs=4, learning_rate=0.05))
    return MLPIntrospector(session)


def reference(session, params, x, target, masks=None):
    """Independent autograd pass with plain tensors (no forge code)."""
    ps = [(w.clone().requires_grad_(True), b.clone().requires_grad_(True)) for w, b in params]
    xx = x.clone().unsqueeze(0).requires_grad_(True)
    h, zs, hs = xx, [], []
    for k, (w, b) in enumerate(ps):
        z = h @ w.T + b
        z.retain_grad()
        zs.append(z)
        if k == len(ps) - 1:
            h = z
        else:
            h = activation_fn(session.spec.activations[k])(z)
            if masks and masks[k] is not None:
                h = h * masks[k]
            h.retain_grad()
            hs.append(h)
    loss = F.cross_entropy(h, torch.tensor([target]))
    loss.backward()
    return ps, xx, zs, hs, float(loss.detach())


def params_of(session):
    return [(p.weight, p.bias) for p in session.params]


def test_forward_values_and_equations(intro, session):
    t = intro.trace(S.TraceRequest(probe=S.Probe(sample_index=7)))
    assert t.input == pytest.approx(session.X[7].tolist())
    assert len(t.layers) == len(session.spec.layer_sizes) - 1
    prev = t.input
    for L in t.layers:
        assert L.input == pytest.approx(prev, abs=1e-6)
        for i, row in enumerate(L.weight):
            z = sum(w * a for w, a in zip(row, L.input)) + L.bias[i]
            assert L.z[i] == pytest.approx(z, abs=1e-5)
        prev = L.a
    out = t.layers[-1]
    assert out.activation == "Softmax" and out.role == "output"
    assert t.logits == pytest.approx(out.z)
    exp = [math.exp(v) for v in t.logits]
    assert t.probabilities == pytest.approx([e / sum(exp) for e in exp], abs=1e-6)
    assert t.loss == pytest.approx(-math.log(t.probabilities[t.target]), abs=1e-5)
    assert t.predicted_class == max(range(2), key=lambda c: t.logits[c])
    assert t.target == int(session.y[7]) and t.probe.target_source == "label"


def test_gradients_match_independent_autograd(intro, session):
    t = intro.trace(S.TraceRequest(probe=S.Probe(sample_index=19)))
    ps, xx, zs, hs, loss = reference(session, params_of(session), session.X[19], int(session.y[19]))
    assert t.loss == pytest.approx(loss, abs=1e-6)
    assert t.grad_input == pytest.approx(xx.grad[0].tolist(), abs=1e-6)
    for k, L in enumerate(t.layers):
        w, b = ps[k]
        assert sum(L.grad_weight, []) == pytest.approx(w.grad.flatten().tolist(), abs=1e-6)
        assert L.grad_bias == pytest.approx(b.grad.tolist(), abs=1e-6)
        assert L.grad_z == pytest.approx(zs[k].grad[0].tolist(), abs=1e-6)
        if k < len(hs):
            assert L.grad_a == pytest.approx(hs[k].grad[0].tolist(), abs=1e-6)


def test_chain_rule_identities(intro):
    t = intro.trace(S.TraceRequest(probe=S.Probe(sample_index=3)))
    out = t.layers[-1]
    onehot = [1.0 if c == t.target else 0.0 for c in range(len(t.probabilities))]
    # softmax + cross-entropy: dL/dlogits = p - onehot
    assert out.grad_z == pytest.approx([p - o for p, o in zip(t.probabilities, onehot)], abs=1e-6)
    for i, L in enumerate(t.layers):
        # dL/dW = dL/dz (outer) a_prev ; dL/db = dL/dz ; dL/da_prev = W^T dL/dz
        for r, gz in enumerate(L.grad_z):
            assert L.grad_weight[r] == pytest.approx([gz * a for a in L.input], abs=1e-6)
        assert L.grad_bias == pytest.approx(L.grad_z, abs=1e-6)
        back = [sum(L.weight[r][c] * L.grad_z[r] for r in range(len(L.z))) for c in range(len(L.input))]
        assert L.grad_input == pytest.approx(back, abs=1e-5)
        if L.grad_a is not None:
            assert L.grad_z == pytest.approx([g * d for g, d in zip(L.grad_a, L.local_grad)], abs=1e-6)
        if i > 0:
            assert t.layers[i - 1].grad_a == pytest.approx(L.grad_input, abs=1e-6)
    assert t.layers[0].grad_input == pytest.approx(t.grad_input, abs=1e-6)


def test_trace_with_interventions_and_checkpoint(intro, session):
    ivs = [
        {"type": "ablate_neuron", "layer": 1, "index": 2},
        {"type": "set_bias", "layer": 2, "index": 0, "value": 0.7},
        {"type": "set_weight", "layer": 3, "source": 1, "target": 0, "value": -1.5},
    ]
    t = intro.trace(S.TraceRequest(probe=S.Probe(sample_index=5), interventions=ivs, checkpoint_epoch=2))
    assert t.provenance.checkpoint_epoch == 2 and not t.provenance.is_latest
    assert t.provenance.interventions_applied == 3
    L1, L2, L3 = t.layers
    assert L1.ablated == [2] and L1.a[2] == 0.0 and L1.local_grad[2] == 0.0 and L1.grad_z[2] == 0.0
    assert L1.z[2] != 0.0  # the natural pre-activation stays observable
    assert L2.bias[0] == pytest.approx(0.7) and L2.edited_bias == [0]
    assert L3.weight[0][1] == pytest.approx(-1.5) and L3.edited_weights == [[1, 0]]

    ckpt = session.checkpoints.get(2).params
    params = [(p.weight.clone(), p.bias.clone()) for p in ckpt]
    params[1][1][0] = 0.7
    params[2][0][0, 1] = -1.5
    mask = torch.ones(session.spec.layer_sizes[1])
    mask[2] = 0
    _, xx, zs, _, loss = reference(session, params, session.X[5], int(session.y[5]), [mask, None])
    assert t.loss == pytest.approx(loss, abs=1e-6)
    assert t.grad_input == pytest.approx(xx.grad[0].tolist(), abs=1e-6)
    # stored checkpoint untouched
    assert session.checkpoints.get(2).params[1].bias[0] != pytest.approx(0.7)


def test_sgd_preview_is_a_real_step(intro, session):
    lr = 0.1
    t = intro.trace(S.TraceRequest(probe=S.Probe(sample_index=9), learning_rate=lr))
    pv = t.sgd_preview
    assert pv.learning_rate == lr and pv.loss_before == pytest.approx(t.loss)
    stepped = [(torch.tensor(L.weight) - lr * torch.tensor(L.grad_weight),
                torch.tensor(L.bias) - lr * torch.tensor(L.grad_bias)) for L in t.layers]
    _, _, _, _, loss_after = reference(session, stepped, session.X[9], t.target)
    assert pv.loss_after == pytest.approx(loss_after, abs=1e-5)
    assert pv.loss_after < pv.loss_before  # small step along -grad on the same sample
    before = [p.weight.clone() for p in session.params]
    intro.trace(S.TraceRequest(probe=S.Probe(sample_index=9), learning_rate=lr))
    assert all(torch.equal(a, p.weight) for a, p in zip(before, session.params))
    with pytest.raises(IntrospectionError):
        intro.trace(S.TraceRequest(probe=S.Probe(sample_index=9), learning_rate=2.0))


def test_trace_endpoint():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    sid = client.post("/api/forge/sessions", json={"neurons": [4, 3], "activations": ["ReLU", "Sigmoid"]}).json()["session_id"]
    r = client.post(f"/api/forge/sessions/{sid}/trace", json={"probe": {"sample_index": 1}, "learning_rate": 0.01})
    assert r.status_code == 200
    body = r.json()
    assert len(body["layers"]) == 3 and body["sgd_preview"]["learning_rate"] == 0.01
    r = client.post(f"/api/forge/sessions/{sid}/trace", json={"probe": {"sample_index": 10**6}})
    assert r.status_code == 422
    assert client.post("/api/forge/sessions/nope/trace", json={}).status_code == 404


def test_loss_landscape_is_a_real_slice(session):
    from forge.landscape import loss_landscape
    from forge.mlp import forward
    session.train(TrainingSettings(epochs=3, learning_rate=0.05))
    ls = loss_landscape(session, None, grid=5)
    mid = 2
    assert ls.alphas[mid] == pytest.approx(0.0, abs=1e-6) and ls.betas[mid] == pytest.approx(0.0, abs=1e-6)
    direct = float(F.cross_entropy(forward(session.spec, session.params, session.X).logits, session.y))
    assert ls.loss[mid][mid] == pytest.approx(direct, abs=1e-6) == ls.center_loss
    assert ls.min_loss <= ls.center_loss <= ls.max_loss
    again = loss_landscape(session, None, grid=5)
    assert again.loss == ls.loss  # fixed seed: reproducible
    hist = loss_landscape(session, 1, grid=5)
    assert hist.checkpoint_epoch == 1 and not hist.is_latest
    with pytest.raises(KeyError):
        loss_landscape(session, 999, grid=3)
