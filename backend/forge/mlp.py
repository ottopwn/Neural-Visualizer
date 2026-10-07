"""Functional multilayer perceptron used by Neural Forge.

Why functional (explicit parameter tensors) instead of an ``nn.Module``?

* The same forward pass must run on *different parameter sets*: the live
  weights, a training checkpoint, or the live weights with interventions
  overlaid.  Passing parameters explicitly makes that trivial and keeps the
  stored weights immutable.
* Every intermediate tensor (pre-activation ``z`` and activation ``a`` of
  every layer) is returned, so introspection never has to re-derive values.

Layer indexing convention (shared with the frontend graph):

* graph layer ``0``            -> the input features (no parameters)
* graph layer ``1 .. H``       -> hidden dense layers
* graph layer ``H + 1``        -> output dense layer (logits, softmax)

``params[k]`` holds the weights of graph layer ``k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ACTIVATIONS = ("ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "SELU")
LEAKY_RELU_SLOPE = 0.2  # matches the legacy CustomNN in models.py

_ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
    "LeakyReLU": lambda z: F.leaky_relu(z, LEAKY_RELU_SLOPE),
    "ELU": F.elu,
    "SELU": F.selu,
}


def activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        return _ACTIVATION_FNS[name]
    except KeyError as exc:  # pragma: no cover - guarded by validation
        raise ValueError(f"Unsupported activation '{name}'") from exc


@dataclass(frozen=True)
class MLPSpec:
    """Static description of an MLP: sizes and activation functions."""

    input_dim: int
    hidden: tuple
    activations: tuple
    n_classes: int

    def __post_init__(self) -> None:
        if len(self.hidden) != len(self.activations):
            raise ValueError("hidden and activations must have the same length")
        for act in self.activations:
            if act not in SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Unsupported activation '{act}'")
        if self.input_dim < 1 or self.n_classes < 2:
            raise ValueError("input_dim must be >= 1 and n_classes >= 2")

    @property
    def layer_sizes(self) -> List[int]:
        """Sizes of every graph layer, input first, output last."""
        return [self.input_dim, *self.hidden, self.n_classes]

    @property
    def n_dense(self) -> int:
        return len(self.hidden) + 1

    @property
    def output_layer(self) -> int:
        return len(self.hidden) + 1

    def activation_of(self, graph_layer: int) -> Optional[str]:
        """Activation applied by a graph layer; ``Softmax`` for the output."""
        if graph_layer == 0:
            return None
        if graph_layer == self.output_layer:
            return "Softmax"
        return self.activations[graph_layer - 1]


@dataclass
class DenseParams:
    weight: torch.Tensor  # [out, in]
    bias: torch.Tensor  # [out]


Params = List[DenseParams]


def init_params(spec: MLPSpec, seed: int = 0) -> Params:
    """PyTorch's default ``nn.Linear`` initialisation, made reproducible."""
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        params: Params = []
        sizes = spec.layer_sizes
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(fan_in, fan_out)
            params.append(DenseParams(layer.weight.detach().clone(), layer.bias.detach().clone()))
        return params
    finally:
        torch.random.set_rng_state(gen_state)


def clone_params(params: Params, requires_grad: bool = False) -> Params:
    out = []
    for p in params:
        w = p.weight.detach().clone().requires_grad_(requires_grad)
        b = p.bias.detach().clone().requires_grad_(requires_grad)
        out.append(DenseParams(w, b))
    return out


def param_count(spec: MLPSpec) -> int:
    sizes = spec.layer_sizes
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))


@dataclass
class ForwardTrace:
    """All intermediate tensors of one forward pass (batch-first)."""

    inputs: torch.Tensor  # [B, input_dim]
    pre: List[torch.Tensor] = field(default_factory=list)  # z per dense layer
    post: List[torch.Tensor] = field(default_factory=list)  # a per dense layer (post-mask)

    @property
    def logits(self) -> torch.Tensor:
        return self.pre[-1]

    @property
    def probabilities(self) -> torch.Tensor:
        return self.post[-1]

    def activations_of(self, graph_layer: int) -> torch.Tensor:
        """Output tensor of a graph layer (inputs for layer 0)."""
        return self.inputs if graph_layer == 0 else self.post[graph_layer - 1]

    def preactivations_of(self, graph_layer: int) -> torch.Tensor:
        return self.pre[graph_layer - 1]


def forward(
    spec: MLPSpec,
    params: Params,
    x: torch.Tensor,
    neuron_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    retain_grad: bool = False,
) -> ForwardTrace:
    """Run the network and record every ``z`` and ``a``.

    ``neuron_masks[k]`` (optional, shape ``[out_k]``) multiplies the
    activations of dense layer ``k`` -- this is how neuron ablation is
    applied without touching the stored weights.  Weight/bias overrides are
    expected to be already baked into ``params`` (see ``interventions``).
    """
    trace = ForwardTrace(inputs=x)
    h = x
    last = len(params) - 1
    for k, p in enumerate(params):
        z = h @ p.weight.T + p.bias
        if retain_grad and z.requires_grad:
            z.retain_grad()
        if k == last:
            a = torch.softmax(z, dim=-1)
        else:
            a = activation_fn(spec.activations[k])(z)
            mask = neuron_masks[k] if neuron_masks is not None else None
            if mask is not None:
                a = a * mask
        if retain_grad and a.requires_grad:
            a.retain_grad()
        trace.pre.append(z)
        trace.post.append(a)
        h = a
    return trace
