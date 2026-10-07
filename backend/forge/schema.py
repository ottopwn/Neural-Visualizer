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
    grad_norm: Optional[float] = Field(None, description="Mean L2 norm of the training gradient during this epoch")
    update_norm: Optional[float] = Field(None, description="||theta_end - theta_start|| over this epoch")


class SessionSummary(BaseModel):
    session_id: str
    structure: ModelStructure
    dataset_name: str
    dataset_X: List[List[float]]
    dataset_y: List[int]
    epoch: int
    history: List[HistoryRow]
    checkpoints: List[HistoryRow]


# ── Training Time Machine ───────────────────────────────────────────────────
# Everything below is computed from *stored* parameters (checkpoints are
# immutable) or from statistics logged during real training.  Interventions
# are never applied here: the time machine shows history as it happened.

class TimelineRow(HistoryRow):
    """One epoch of the training log (every epoch, not only stored ones)."""

    layer_grad_norms: Optional[List[float]] = None
    layer_update_norms: Optional[List[float]] = None


class LayerHealth(BaseModel):
    layer: int
    label: str
    activation: Optional[str]
    weight_norm: float
    bias_norm: float
    param_norm: float = Field(..., description="sqrt(||W||^2 + ||b||^2)")
    train_grad_norm: Optional[float] = Field(None, description="Logged training gradient norm for this epoch")
    update_norm: Optional[float] = Field(None, description="Logged parameter change over this epoch")
    update_ratio: Optional[float] = Field(None, description="update_norm / param_norm")
    mean_abs_activation: Optional[float] = Field(None, description="Hidden layers: mean |a| over dataset x neurons")
    zero_fraction: Optional[float] = Field(None, description="Hidden layers: fraction of (sample, neuron) outputs with |a| < 1e-6")
    dead_fraction: Optional[float] = Field(None, description="Hidden layers: fraction of neurons that output 0 on every sample")
    saturated_fraction: Optional[float] = Field(
        None, description="Sigmoid/Tanh layers: fraction of outputs within 1% of an asymptote")


class CheckpointHealth(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    layers: List[LayerHealth]


class TrainingRun(BaseModel):
    start_epoch: int
    end_epoch: int
    learning_rate: float
    batch_size: int
    reg_type: str
    reg_rate: float


class TrainingEvent(BaseModel):
    epoch: int = Field(..., description="Epoch of the event in the training log")
    checkpoint_epoch: int = Field(..., description="Nearest stored checkpoint (what the UI can jump to)")
    kind: Literal["init", "run", "acc_threshold", "best_accuracy", "min_loss", "largest_drop"]
    label: str


class Timeline(BaseModel):
    live_epoch: int
    capacity: int
    history: List[TimelineRow]
    checkpoints: List[CheckpointHealth]
    runs: List[TrainingRun]
    events: List[TrainingEvent]
    majority_rate: float = Field(..., description="Accuracy of always predicting the most common class")


class FrameRequest(BaseModel):
    checkpoint_epoch: Optional[int] = Field(None, description="Stored checkpoint; null = live weights")
    probe: Probe = Field(default_factory=lambda: Probe(sample_index=0))


class FramePrevious(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    changed: int = Field(..., description="Samples whose predicted class differs from the previous checkpoint")
    fixed: int = Field(..., description="Wrong at the previous checkpoint, right now")
    broken: int = Field(..., description="Right at the previous checkpoint, wrong now")
    boundary_flip_fraction: Optional[float] = None


class Frame(BaseModel):
    """What the network looked like at one stored checkpoint."""

    epoch: int
    is_latest: bool
    loss: float
    accuracy: float
    probe: ResolvedProbe
    probe_probabilities: List[float]
    probe_predicted: int
    predictions: List[int] = Field(..., description="Predicted class of every dataset sample")
    confidence: List[float] = Field(..., description="Probability of the predicted class, per sample")
    boundary: Optional[ResponseMap] = Field(None, description="P(class 1) over the input plane (2-D inputs)")
    previous: Optional[FramePrevious] = None


class ComponentHistoryRequest(BaseModel):
    ref: ComponentRef
    probe: Probe = Field(default_factory=lambda: Probe(sample_index=0))


class Series(BaseModel):
    key: str
    label: str
    group: Literal["parameter", "probe", "dataset", "gradient"]
    values: List[Optional[float]]


class ComponentHistory(BaseModel):
    ref: ComponentRef
    name: str
    epochs: List[int]
    series: List[Series]
    notes: List[str] = Field(default_factory=list)


class EpochCompareRequest(BaseModel):
    epoch_a: int
    epoch_b: Optional[int] = Field(None, description="null = live weights")
    probe: Probe = Field(default_factory=lambda: Probe(sample_index=0))
    ref: Optional[ComponentRef] = None


class EpochMetrics(BaseModel):
    epoch: int
    loss: float
    accuracy: float
    mean_confidence: float
    probe_probabilities: List[float]
    probe_predicted: int


class ParamChange(BaseModel):
    layer: int
    label: str
    weight_norm_a: float
    weight_norm_b: float
    bias_norm_a: float
    bias_norm_b: float
    weight_delta_norm: float = Field(..., description="||W_B - W_A||")
    bias_delta_norm: float = Field(..., description="||b_B - b_A||")
    relative_change: float = Field(..., description="||theta_B - theta_A|| / ||theta_A||")
    mean_abs_weight_delta: float
    max_abs_weight_delta: float
    top_neurons: List[int] = Field(..., description="Neurons whose incoming weights + bias changed most")
    top_neuron_deltas: List[float]


class ComponentDeltaRow(BaseModel):
    key: str
    label: str
    group: str
    a: Optional[float]
    b: Optional[float]
    delta: Optional[float]


class ComponentCompare(BaseModel):
    ref: ComponentRef
    name: str
    rows: List[ComponentDeltaRow]
    response_a: Optional[ResponseMap] = None
    response_b: Optional[ResponseMap] = None


class EpochComparison(BaseModel):
    a: EpochMetrics
    b: EpochMetrics
    probe: ResolvedProbe
    loss_delta: float
    accuracy_delta: float
    changed: int
    changed_fraction: float
    fixed: int
    broken: int
    changed_indices: List[int]
    predictions_a: List[int]
    predictions_b: List[int]
    confidence_delta: List[float] = Field(..., description="Per sample: P_B(true label) - P_A(true label)")
    boundary_a: Optional[ResponseMap] = None
    boundary_b: Optional[ResponseMap] = None
    boundary_flip_fraction: Optional[float] = None
    layers: List[ParamChange]
    total_delta_norm: float
    total_relative_change: float
    component: Optional[ComponentCompare] = None
