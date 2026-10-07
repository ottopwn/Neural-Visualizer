"""What-if interventions, applied as a non-destructive overlay.

An intervention list is *data*: it never mutates the stored parameters.
``compile_interventions`` turns the list into

* an overridden copy of the parameters (weight / bias edits), and
* per-layer neuron masks (ablation = force the neuron's output to 0).

Interventions are applied in order, so a later edit of the same element
wins.  The frontend keeps the list (undo/reset are list operations) and
sends it with every request, which keeps the backend stateless with
respect to experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Union

import torch
from pydantic import BaseModel, Field

from .mlp import MLPSpec, Params, clone_params


class AblateNeuron(BaseModel):
    type: Literal["ablate_neuron"] = "ablate_neuron"
    layer: int = Field(..., description="Graph layer of a hidden neuron (1..H)")
    index: int


class SetWeight(BaseModel):
    type: Literal["set_weight"] = "set_weight"
    layer: int = Field(..., description="Graph layer of the *target* neuron (1..H+1)")
    source: int = Field(..., description="Neuron index in layer - 1")
    target: int = Field(..., description="Neuron index in layer")
    value: float


class SetBias(BaseModel):
    type: Literal["set_bias"] = "set_bias"
    layer: int = Field(..., description="Graph layer (1..H+1)")
    index: int
    value: float


Intervention = Union[AblateNeuron, SetWeight, SetBias]


class InterventionError(ValueError):
    """Raised when an intervention references a component that does not exist."""


@dataclass
class CompiledInterventions:
    params: Params
    neuron_masks: List[Optional[torch.Tensor]]
    ablated: set  # {(layer, index)}
    weight_edits: dict  # {(layer, source, target): value}
    bias_edits: dict  # {(layer, index): value}

    @property
    def is_empty(self) -> bool:
        return not (self.ablated or self.weight_edits or self.bias_edits)


def validate_intervention(spec: MLPSpec, iv: Intervention) -> None:
    sizes = spec.layer_sizes
    out_layer = spec.output_layer

    def check_neuron(layer: int, index: int, allow_output: bool) -> None:
        max_layer = out_layer if allow_output else out_layer - 1
        if not 1 <= layer <= max_layer:
            raise InterventionError(f"layer {layer} out of range 1..{max_layer}")
        if not 0 <= index < sizes[layer]:
            raise InterventionError(f"neuron {index} out of range for layer {layer}")

    if isinstance(iv, AblateNeuron):
        check_neuron(iv.layer, iv.index, allow_output=False)
    elif isinstance(iv, SetBias):
        check_neuron(iv.layer, iv.index, allow_output=True)
    elif isinstance(iv, SetWeight):
        check_neuron(iv.layer, iv.target, allow_output=True)
        if not 0 <= iv.source < sizes[iv.layer - 1]:
            raise InterventionError(f"source neuron {iv.source} out of range for layer {iv.layer - 1}")
    else:  # pragma: no cover
        raise InterventionError(f"unknown intervention {iv!r}")
    if isinstance(iv, (SetWeight, SetBias)) and not math.isfinite(iv.value):
        raise InterventionError("value must be finite")


def compile_interventions(
    spec: MLPSpec, params: Params, interventions: Sequence[Intervention]
) -> CompiledInterventions:
    edited = clone_params(params)
    masks: List[Optional[torch.Tensor]] = [None] * len(params)
    ablated: set = set()
    weight_edits: dict = {}
    bias_edits: dict = {}

    for iv in interventions:
        validate_intervention(spec, iv)
        k = iv.layer - 1  # dense layer index
        if isinstance(iv, AblateNeuron):
            if masks[k] is None:
                masks[k] = torch.ones(spec.layer_sizes[iv.layer])
            masks[k][iv.index] = 0.0
            ablated.add((iv.layer, iv.index))
        elif isinstance(iv, SetWeight):
            edited[k].weight[iv.target, iv.source] = iv.value
            weight_edits[(iv.layer, iv.source, iv.target)] = iv.value
        elif isinstance(iv, SetBias):
            edited[k].bias[iv.index] = iv.value
            bias_edits[(iv.layer, iv.index)] = iv.value

    return CompiledInterventions(edited, masks, ablated, weight_edits, bias_edits)
