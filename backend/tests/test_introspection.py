"""The microscope must report exactly what the network computes."""
import pytest
import torch
import torch.nn.functional as F

from forge import schema as S
from forge.introspect import IntrospectionError, MLPIntrospector
from forge.mlp import activation_fn, forward
from forge.session import TrainingSettings


def req(**kw):
    return S.ExperimentRequest(**kw)


def inspect(intro, ref, **kw):
    return intro.inspect(S.InspectRequest(ref=S.ComponentRef(**ref), **kw))


@pytest.fixture
def intro(session):
    session.train(TrainingSettings(epochs=3, learning_rate=0.05))
    return MLPIntrospector(session)


def reference_grads(session, sample_index):
    """Independent autograd pass with plain tensors."""
    params = [(p.weight.clone().requires_grad_(True), p.bias.clone().requires_grad_(True)) for p in session.params]
    x = session.X[sample_index:sample_index + 1].clone().requires_grad_(True)
    h = x
    for k, (w, b) in enumerate(params):
        z = h @ w.T + b
        h = z if k == len(params) - 1 else activation_fn(session.spec.activations[k])(z)
    loss = F.cross_entropy(h, session.y[sample_index:sample_index + 1])
    loss.backward()
    return params, x


def test_neuron_equation_is_exact(intro, session):
    n = inspect(intro, {"kind": "neuron", "layer": 1, "index": 2}, probe=S.Probe(sample_index=5))
    assert sum(n.contributions) + n.bias == pytest.approx(n.pre_activation, abs=1e-5)
    assert n.value == pytest.approx(float(torch.tanh(torch.tensor(n.pre_activation))), abs=1e-6)
    assert n.inputs == pytest.approx(session.X[5].tolist())
    assert n.weights == pytest.approx(session.params[0].weight[2].tolist())
    assert n.probe.target_source == "label"
    assert n.provenance.source == "model" and n.provenance.is_latest


def test_neuron_gradients_match_independent_autograd(intro, session):
    params, x = reference_grads(session, 11)
    n = inspect(intro, {"kind": "neuron", "layer": 2, "index": 1}, probe=S.Probe(sample_index=11))
    w, b = params[1]
    assert n.grad_bias == pytest.approx(float(b.grad[1]), abs=1e-6)
    assert n.grad_weights == pytest.approx(w.grad[1].tolist(), abs=1e-6)
    # chain rule: dL/dw_ij = dL/dz_i * a_j
    assert n.grad_weights == pytest.approx([n.grad_pre_activation * a for a in n.inputs], abs=1e-6)

    inp = inspect(intro, {"kind": "neuron", "layer": 0, "index": 1}, probe=S.Probe(sample_index=11))
    assert inp.grad_value == pytest.approx(float(x.grad[0, 1]), abs=1e-6)


def test_output_neuron(intro, session):
    out = inspect(intro, {"kind": "neuron", "layer": 3, "index": 1}, probe=S.Probe(sample_index=0))
    with torch.no_grad():
        probs = forward(session.spec, session.params, session.X[:1]).probabilities[0]
    assert out.value == pytest.approx(float(probs[1]), abs=1e-6)
    assert out.grad_value is None  # loss is defined on logits
    assert out.grad_pre_activation == pytest.approx(float(probs[1]) - (1.0 if int(session.y[0]) == 1 else 0.0), abs=1e-5)
    assert out.response_map is not None and out.response_map.resolution == 40


def test_dataset_statistics_are_over_whole_dataset(intro, session):
    n = inspect(intro, {"kind": "neuron", "layer": 2, "index": 0})
    with torch.no_grad():
        acts = forward(session.spec, session.params, session.X).post[1][:, 0]
    assert n.dataset_stats.count == session.X.shape[0]
    assert n.dataset_stats.mean == pytest.approx(float(acts.mean()), abs=1e-6)
    assert sum(n.dataset_histogram.counts) == session.X.shape[0]
    assert n.inactive_fraction == pytest.approx(float((acts.abs() < 1e-6).float().mean()))


def test_layer_inspection_consistent_with_neurons(intro, session):
    layer = inspect(intro, {"kind": "layer", "layer": 2}, probe=S.Probe(sample_index=3))
    neuron = inspect(intro, {"kind": "neuron", "layer": 2, "index": 4}, probe=S.Probe(sample_index=3))
    assert layer.activations[4] == pytest.approx(neuron.value)
    assert layer.weights[4] == pytest.approx(neuron.weights)
    assert layer.grad_weights[4] == pytest.approx(neuron.grad_weights, abs=1e-7)
    assert layer.shapes["weight"] == [5, 6]
    assert layer.weight_stats.count == 30


def test_connection_contribution_and_share(intro):
    shares = []
    for s in range(6):
        c = inspect(intro, {"kind": "connection", "layer": 2, "source": s, "target": 3}, probe=S.Probe(sample_index=9))
        assert c.contribution == pytest.approx(c.weight * c.source_value, abs=1e-6)
        shares.append(c.share_of_input)
    assert sum(shares) == pytest.approx(1.0, abs=1e-5)


def test_custom_probe_uses_prediction_as_target(intro):
    n = inspect(intro, {"kind": "neuron", "layer": 1, "index": 0}, probe=S.Probe(x=[0.1, -0.2]))
    assert n.probe.target_source == "prediction" and n.probe.label is None
    n = inspect(intro, {"kind": "neuron", "layer": 1, "index": 0}, probe=S.Probe(x=[0.1, -0.2], target=1))
    assert n.probe.target_source == "user" and n.probe.target == 1


def test_graph_values_are_the_forward_pass(intro, session):
    g = intro.graph(req(probe=S.Probe(sample_index=2)))
    assert len(g.nodes) == sum(session.spec.layer_sizes)
    assert len(g.edges) == 2 * 6 + 6 * 5 + 5 * 2
    with torch.no_grad():
        trace = forward(session.spec, session.params, session.X[2:3])
    hidden2 = [n for n in g.nodes if n.layer == 2]
    assert [n.value for n in hidden2] == pytest.approx(trace.post[1][0].tolist(), abs=1e-6)
    assert len(g.forward_steps) == 4 and len(g.backward_steps) == 4
    assert g.backward_steps[0].gradients  # real gradients on output layer
    e = g.edges[0]
    assert e.weight == pytest.approx(float(session.params[0].weight[e.target_index, e.source_index]))


def test_checkpoint_inspection(intro, session):
    latest = inspect(intro, {"kind": "layer", "layer": 1})
    init = inspect(intro, {"kind": "layer", "layer": 1}, checkpoint_epoch=0)
    assert init.provenance.checkpoint_epoch == 0 and not init.provenance.is_latest
    assert init.weights != latest.weights
    with pytest.raises(IntrospectionError):
        inspect(intro, {"kind": "layer", "layer": 1}, checkpoint_epoch=999)


@pytest.mark.parametrize("ref", [
    {"kind": "neuron", "layer": 9, "index": 0},
    {"kind": "neuron", "layer": 1, "index": 99},
    {"kind": "connection", "layer": 0, "source": 0, "target": 0},
    {"kind": "connection", "layer": 1, "source": 5, "target": 0},
])
def test_bad_refs(intro, ref):
    with pytest.raises(IntrospectionError):
        inspect(intro, ref)


def test_bad_probe(intro):
    with pytest.raises(IntrospectionError):
        intro.graph(req(probe=S.Probe(x=[1.0])))
    with pytest.raises(IntrospectionError):
        intro.graph(req(probe=S.Probe(sample_index=10_000)))
