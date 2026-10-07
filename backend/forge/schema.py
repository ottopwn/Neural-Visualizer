"""Wire contract between the Forge backend and visualisation clients.

Everything here is plain data.  Visualisation code depends on these shapes
(mirrored in ``frontend/src/forge/types.ts``), never on PyTorch objects, so
the ML implementation can change without touching the UI.

Every introspection payload carries a ``Provenance`` block saying exactly
which parameters produced the numbers (live weights or a checkpoint) and
how many interventions were overlaid.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .interventions import Intervention

Role = Literal["input", "hidden", "output"]


# ── requests ────────────────────────────────────────────────────────────────
class Probe(BaseModel):
    """The single input example the network is being inspected on."""

    x: Optional[List[float]] = Field(None, description="Feature vector; defaults to sample_index's features")
    sample_index: Optional[int] = Field(None, description="Dataset row to use as probe (gives a true label)")
    target: Optional[int] = Field(None, description="Class used for the loss/gradients; defaults to the label")


class ExperimentRequest(BaseModel):
    probe: Probe = Field(default_factory=lambda: Probe(sample_index=0))
    interventions: List[Intervention] = Field(default_factory=list)
    checkpoint_epoch: Optional[int] = Field(None, description="Inspect a stored checkpoint instead of live weights")


class ComponentRef(BaseModel):
    kind: Literal["neuron", "layer", "connection"]
    layer: int
    index: Optional[int] = None  # neuron
    source: Optional[int] = None  # connection: neuron index in layer - 1
    target: Optional[int] = None  # connection: neuron index in layer


class InspectRequest(ExperimentRequest):
    ref: ComponentRef


# ── shared blocks ───────────────────────────────────────────────────────────
class ResolvedProbe(BaseModel):
    x: List[float]
    sample_index: Optional[int]
    label: Optional[int] = Field(None, description="True label when the probe is a dataset row")
    target: int
    target_source: Literal["label", "user", "prediction"]


class Provenance(BaseModel):
    source: Literal["model"] = "model"
    checkpoint_epoch: int
    is_latest: bool
    interventions_applied: int
    note: str = "Computed by a PyTorch forward/backward pass of the session model."


class TensorStats(BaseModel):
    shape: List[int]
    count: int
    mean: float
    std: float
    min: float
    max: float
    abs_mean: float
    l2_norm: float
    frac_zero: float


class Histogram(BaseModel):
    edges: List[float]
    counts: List[int]


class ResponseMap(BaseModel):
    """A scalar quantity evaluated on a grid over the 2-D input plane."""

    label: str
    x_range: List[float]
    y_range: List[float]
    resolution: int
    values: List[List[float]]  # values[row][col]; row 0 = y_range[0]
    value_range: List[float]


class LayerInfo(BaseModel):
    layer: int
    role: Role
    label: str
    size: int
    activation: Optional[str]
    fan_in: Optional[int]
    param_count: int
    weight_shape: Optional[List[int]]
    bias_shape: Optional[List[int]]


class ModelStructure(BaseModel):
    model_family: Literal["mlp"] = "mlp"
    input_dim: int
    n_classes: int
    class_names: List[str]
    feature_names: List[str]
    layers: List[LayerInfo]
    param_count: int


class WeightEdit(BaseModel):
    source: int
    target: int
    value: float
    original: float


class BiasEdit(BaseModel):
    value: float
    original: float


# ── inspections ─────────────────────────────────────────────────────────────
class NeuronInspection(BaseModel):
    kind: Literal["neuron"] = "neuron"
    layer: int
    index: int
    name: str
    role: Role
    layer_label: str
    activation_fn: Optional[str]
    probe: ResolvedProbe
    provenance: Provenance

    value: float = Field(..., description="Neuron output on the probe (input value / activation / probability)")
    natural_value: Optional[float] = Field(None, description="Output before ablation, if ablated")
    pre_activation: Optional[float] = None
    bias: Optional[float] = None
    input_names: Optional[List[str]] = None
    inputs: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    contributions: Optional[List[float]] = None
    outgoing_names: Optional[List[str]] = None
    outgoing_weights: Optional[List[float]] = None

    grad_value: Optional[float] = Field(None, description="dLoss/d(output of this neuron)")
    grad_pre_activation: Optional[float] = None
    grad_bias: Optional[float] = None
    grad_weights: Optional[List[float]] = None

    dataset_stats: TensorStats
    dataset_histogram: Histogram
    inactive_fraction: float = Field(..., description="Fraction of dataset samples where |output| < 1e-6")
    response_map: Optional[ResponseMap] = None

    ablated: bool = False
    bias_edit: Optional[BiasEdit] = None
    weight_edits: List[WeightEdit] = Field(default_factory=list)
    shapes: Dict[str, List[int]] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class LayerInspection(BaseModel):
    kind: Literal["layer"] = "layer"
    layer: int
    role: Role
    label: str
    size: int
    activation_fn: Optional[str]
    fan_in: Optional[int]
    param_count: int
    probe: ResolvedProbe
    provenance: Provenance
    neuron_names: List[str]
    input_names: Optional[List[str]] = None

    weights: Optional[List[List[float]]] = None
    bias: Optional[List[float]] = None
    weight_stats: Optional[TensorStats] = None
    bias_stats: Optional[TensorStats] = None
    weight_histogram: Optional[Histogram] = None

    pre_activations: Optional[List[float]] = None
    activations: List[float]
    grad_activations: List[float]
    grad_weights: Optional[List[List[float]]] = None
    grad_weight_norm: Optional[float] = None
    grad_bias_norm: Optional[float] = None

    dataset_stats: TensorStats
    never_active: List[int] = Field(default_factory=list, description="Neurons with |output| < 1e-6 on every sample")
    ablated: List[int] = Field(default_factory=list)
    shapes: Dict[str, List[int]] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class ConnectionInspection(BaseModel):
    kind: Literal["connection"] = "connection"
    layer: int
    source: int
    target: int
    source_name: str
    target_name: str
    probe: ResolvedProbe
    provenance: Provenance

    weight: float
    original_weight: Optional[float] = None
    source_value: float
    contribution: float
    target_pre_activation: float
    target_bias: float
    share_of_input: float = Field(..., description="|contribution| / sum of |contributions| into the target")
    rank: int = Field(..., description="1 = strongest incoming connection of the target on this probe")
    fan_in: int
    grad_weight: float
    dataset_contribution: TensorStats
    notes: List[str] = Field(default_factory=list)


# ── graph / predictions ─────────────────────────────────────────────────────
class GraphNode(BaseModel):
    id: int
    x: float
    y: float
    name: str
    layer: int
    index: int
    layer_type: str
    value: float
    z_val: Optional[float] = None
    activation: Optional[str] = None
    bias: Optional[float] = None
    grad: Optional[float] = None
    ablated: bool = False
    edited: bool = False


class GraphEdge(BaseModel):
    source: int
    target: int
    weight: float
    layer: int
    source_index: int
    target_index: int
    grad: Optional[float] = None
    edited: bool = False


class PropStepOut(BaseModel):
    step: int
    label: str
    active_nodes: List[int]
    active_edges: List[int]
    layer_type: str
    gradients: Optional[Dict[str, float]] = None


class ForgeGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    forward_steps: List[PropStepOut]
    backward_steps: List[PropStepOut]
    probe: ResolvedProbe
    provenance: Provenance


class PredictionSummary(BaseModel):
    logits: List[float]
    probabilities: List[float]
    predicted_class: int
    dataset_accuracy: float
    dataset_loss: float


class BoundaryPair(BaseModel):
    """P(class 1) over the input plane, before and after interventions."""

    x_range: List[float]
    y_range: List[float]
    resolution: int
    baseline: List[List[float]]
    intervened: List[List[float]]


class Comparison(BaseModel):
    probe: ResolvedProbe
    provenance: Provenance
    baseline: PredictionSummary
    intervened: PredictionSummary
    delta: List[float]
    prediction_changed: bool
    dataset_flip_fraction: float = Field(..., description="Fraction of dataset samples whose predicted class changed")
    boundary: Optional[BoundaryPair] = None


# ── sessions ────────────────────────────────────────────────────────────────
class HistoryRow(BaseModel):
    epoch: int
    loss: float
    accuracy: float


class SessionSummary(BaseModel):
    session_id: str
    structure: ModelStructure
    dataset_name: str
    dataset_X: List[List[float]]
    dataset_y: List[int]
    epoch: int
    history: List[HistoryRow]
    checkpoints: List[HistoryRow]
