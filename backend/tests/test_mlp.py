import pytest
import torch
import torch.nn as nn

from forge.mlp import LEAKY_RELU_SLOPE, SUPPORTED_ACTIVATIONS, MLPSpec, forward, init_params, param_count

TORCH_ACTS = {
    "ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(LEAKY_RELU_SLOPE), "ELU": nn.ELU(), "SELU": nn.SELU(),
}


def reference_model(spec, params):
    """An independent nn.Sequential carrying the same weights."""
    layers = []
    sizes = spec.layer_sizes
    for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(i, o)
        with torch.no_grad():
            lin.weight.copy_(params[k].weight)
            lin.bias.copy_(params[k].bias)
        layers.append(lin)
        if k < len(spec.hidden):
            layers.append(TORCH_ACTS[spec.activations[k]])
    return nn.Sequential(*layers)


@pytest.mark.parametrize("act", SUPPORTED_ACTIVATIONS)
def test_forward_matches_reference(act):
    spec = MLPSpec(input_dim=3, hidden=(5, 4), activations=(act, act), n_classes=2)
    params = init_params(spec, seed=1)
    x = torch.randn(7, 3)
    trace = forward(spec, params, x)
    ref = reference_model(spec, params)
    assert torch.allclose(trace.logits, ref(x), atol=1e-6)
    assert torch.allclose(trace.probabilities.sum(-1), torch.ones(7), atol=1e-6)
    assert len(trace.pre) == len(trace.post) == spec.n_dense


def test_init_is_reproducible_and_does_not_touch_global_rng():
    spec = MLPSpec(2, (4,), ("ReLU",), 2)
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a = init_params(spec, seed=9)
    after = torch.rand(1)
    b = init_params(spec, seed=9)
    assert torch.equal(before, after)
    assert all(torch.equal(p.weight, q.weight) for p, q in zip(a, b))


def test_param_count_and_layout():
    spec = MLPSpec(2, (8, 4), ("ReLU", "Tanh"), 2)
    assert spec.layer_sizes == [2, 8, 4, 2]
    assert param_count(spec) == (2 * 8 + 8) + (8 * 4 + 4) + (4 * 2 + 2)
    assert spec.activation_of(0) is None
    assert spec.activation_of(2) == "Tanh"
    assert spec.activation_of(3) == "Softmax"


def test_spec_validation():
    with pytest.raises(ValueError):
        MLPSpec(2, (4,), ("Swish",), 2)
    with pytest.raises(ValueError):
        MLPSpec(2, (4, 4), ("ReLU",), 2)
