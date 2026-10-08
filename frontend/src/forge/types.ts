// Mirror of backend/forge/schema.py — the introspection wire contract.
// Visualisation components depend on these types only, never on how the
// backend computes them.

import type { NetworkNode, NetworkEdge, PropStep } from '../types';

export type Role = 'input' | 'hidden' | 'output';
export type ExperienceMode = 'learn' | 'lab';

// ── requests ────────────────────────────────────────────────────────────────
export interface Probe {
  x?: number[] | null;
  sample_index?: number | null;
  target?: number | null;
}

export interface AblateNeuron { type: 'ablate_neuron'; layer: number; index: number }
export interface SetWeight { type: 'set_weight'; layer: number; source: number; target: number; value: number }
export interface SetBias { type: 'set_bias'; layer: number; index: number; value: number }
export type Intervention = AblateNeuron | SetWeight | SetBias;

export type ComponentRef =
  | { kind: 'neuron'; layer: number; index: number }
  | { kind: 'layer'; layer: number }
  | { kind: 'connection'; layer: number; source: number; target: number };

export interface ExperimentRequest {
  probe: Probe;
  interventions: Intervention[];
  checkpoint_epoch: number | null;
}

// ── shared blocks ───────────────────────────────────────────────────────────
export interface ResolvedProbe {
  x: number[];
  sample_index: number | null;
  label: number | null;
  target: number;
  target_source: 'label' | 'user' | 'prediction';
}

export interface Provenance {
  source: 'model';
  checkpoint_epoch: number;
  is_latest: boolean;
  interventions_applied: number;
  note: string;
}

export interface TensorStats {
  shape: number[];
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  abs_mean: number;
  l2_norm: number;
  frac_zero: number;
}

export interface Histogram { edges: number[]; counts: number[] }

export interface ResponseMap {
  label: string;
  x_range: [number, number];
  y_range: [number, number];
  resolution: number;
  values: number[][];
  value_range: [number, number];
}

export interface LayerInfo {
  layer: number;
  role: Role;
  label: string;
  size: number;
  activation: string | null;
  fan_in: number | null;
  param_count: number;
  weight_shape: number[] | null;
  bias_shape: number[] | null;
}

export interface ModelStructure {
  model_family: 'mlp';
  input_dim: number;
  n_classes: number;
  class_names: string[];
  feature_names: string[];
  layers: LayerInfo[];
  param_count: number;
}

export interface WeightEdit { source: number; target: number; value: number; original: number }
export interface BiasEdit { value: number; original: number }

// ── inspections ─────────────────────────────────────────────────────────────
interface InspectionBase {
  probe: ResolvedProbe;
  provenance: Provenance;
}

export interface NeuronInspection extends InspectionBase {
  kind: 'neuron';
  layer: number;
  index: number;
  name: string;
  role: Role;
  layer_label: string;
  activation_fn: string | null;
  value: number;
  natural_value: number | null;
  pre_activation: number | null;
  bias: number | null;
  input_names: string[] | null;
  inputs: number[] | null;
  weights: number[] | null;
  contributions: number[] | null;
  outgoing_names: string[] | null;
  outgoing_weights: number[] | null;
  grad_value: number | null;
  grad_pre_activation: number | null;
  grad_bias: number | null;
  grad_weights: number[] | null;
  dataset_stats: TensorStats;
  dataset_histogram: Histogram;
  inactive_fraction: number;
  response_map: ResponseMap | null;
  ablated: boolean;
  bias_edit: BiasEdit | null;
  weight_edits: WeightEdit[];
  shapes: Record<string, number[]>;
  notes: string[];
}

export interface LayerInspection extends InspectionBase {
  kind: 'layer';
  layer: number;
  role: Role;
  label: string;
  size: number;
  activation_fn: string | null;
  fan_in: number | null;
  param_count: number;
  neuron_names: string[];
  input_names: string[] | null;
  weights: number[][] | null;
  bias: number[] | null;
  weight_stats: TensorStats | null;
  bias_stats: TensorStats | null;
  weight_histogram: Histogram | null;
  pre_activations: number[] | null;
  activations: number[];
  grad_activations: number[];
  grad_weights: number[][] | null;
  grad_weight_norm: number | null;
  grad_bias_norm: number | null;
  dataset_stats: TensorStats;
  never_active: number[];
  ablated: number[];
  shapes: Record<string, number[]>;
  notes: string[];
}

export interface ConnectionInspection extends InspectionBase {
  kind: 'connection';
  layer: number;
  source: number;
  target: number;
  source_name: string;
  target_name: string;
  weight: number;
  original_weight: number | null;
  source_value: number;
  contribution: number;
  target_pre_activation: number;
  target_bias: number;
  share_of_input: number;
  rank: number;
  fan_in: number;
  grad_weight: number;
  dataset_contribution: TensorStats;
  notes: string[];
}

export type Inspection = NeuronInspection | LayerInspection | ConnectionInspection;

// ── graph / predictions ─────────────────────────────────────────────────────
export interface PredictionSummary {
  logits: number[];
  probabilities: number[];
  predicted_class: number;
  dataset_accuracy: number;
  dataset_loss: number;
}

export interface BoundaryPair {
  x_range: [number, number];
  y_range: [number, number];
  resolution: number;
  baseline: number[][];
  intervened: number[][];
}

export interface Comparison {
  probe: ResolvedProbe;
  provenance: Provenance;
  baseline: PredictionSummary;
  intervened: PredictionSummary;
  delta: number[];
  prediction_changed: boolean;
  dataset_flip_fraction: number;
  boundary: BoundaryPair | null;
}

export interface HistoryRow {
  epoch: number;
  loss: number;
  accuracy: number;
  grad_norm?: number | null; // mean L2 norm of the training gradient during the epoch
  update_norm?: number | null; // ||θ_end − θ_start|| over the epoch
}

export interface SessionSummary {
  session_id: string;
  structure: ModelStructure;
  dataset_name: string;
  dataset_X: number[][];
  dataset_y: number[];
  epoch: number;
  history: HistoryRow[];
  checkpoints: HistoryRow[];
}

export interface ForgeGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  forward_steps: PropStep[];
  backward_steps: PropStep[];
  probe: ResolvedProbe;
  provenance: Provenance;
}

// ── Training Time Machine (mirror of the time-machine part of schema.py) ─────
// Always computed from stored, immutable checkpoints or the real training log;
// what-if interventions are never applied to these payloads.

export interface TimelineRow extends HistoryRow {
  layer_grad_norms?: number[] | null;
  layer_update_norms?: number[] | null;
}

export interface LayerHealth {
  layer: number;
  label: string;
  activation: string | null;
  weight_norm: number;
  bias_norm: number;
  param_norm: number;
  train_grad_norm: number | null;
  update_norm: number | null;
  update_ratio: number | null;
  mean_abs_activation: number | null;
  zero_fraction: number | null;
  dead_fraction: number | null;
  saturated_fraction: number | null;
}

export interface CheckpointHealth { epoch: number; loss: number; accuracy: number; layers: LayerHealth[] }

export interface TrainingRun {
  start_epoch: number;
  end_epoch: number;
  learning_rate: number;
  batch_size: number;
  reg_type: string;
  reg_rate: number;
}

export type TrainingEventKind = 'init' | 'run' | 'acc_threshold' | 'best_accuracy' | 'min_loss' | 'largest_drop';

export interface TrainingEvent { epoch: number; checkpoint_epoch: number; kind: TrainingEventKind; label: string }

export interface Timeline {
  live_epoch: number;
  capacity: number;
  history: TimelineRow[];
  checkpoints: CheckpointHealth[];
  runs: TrainingRun[];
  events: TrainingEvent[];
  majority_rate: number;
}

export interface FramePrevious {
  epoch: number;
  loss: number;
  accuracy: number;
  changed: number;
  fixed: number;
  broken: number;
  boundary_flip_fraction: number | null;
}

export interface Frame {
  epoch: number;
  is_latest: boolean;
  loss: number;
  accuracy: number;
  probe: ResolvedProbe;
  probe_probabilities: number[];
  probe_predicted: number;
  predictions: number[];
  confidence: number[];
  boundary: ResponseMap | null;
  previous: FramePrevious | null;
}

export type SeriesGroup = 'parameter' | 'probe' | 'dataset' | 'gradient';

export interface Series { key: string; label: string; group: SeriesGroup; values: (number | null)[] }

export interface ComponentHistory {
  ref: ComponentRef;
  name: string;
  epochs: number[];
  series: Series[];
  notes: string[];
}

export interface EpochMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  mean_confidence: number;
  probe_probabilities: number[];
  probe_predicted: number;
}

export interface ParamChange {
  layer: number;
  label: string;
  weight_norm_a: number;
  weight_norm_b: number;
  bias_norm_a: number;
  bias_norm_b: number;
  weight_delta_norm: number;
  bias_delta_norm: number;
  relative_change: number;
  mean_abs_weight_delta: number;
  max_abs_weight_delta: number;
  top_neurons: number[];
  top_neuron_deltas: number[];
}

export interface ComponentDeltaRow { key: string; label: string; group: SeriesGroup; a: number | null; b: number | null; delta: number | null }

export interface ComponentCompare {
  ref: ComponentRef;
  name: string;
  rows: ComponentDeltaRow[];
  response_a: ResponseMap | null;
  response_b: ResponseMap | null;
}

export interface EpochComparison {
  a: EpochMetrics;
  b: EpochMetrics;
  probe: ResolvedProbe;
  loss_delta: number;
  accuracy_delta: number;
  changed: number;
  changed_fraction: number;
  fixed: number;
  broken: number;
  changed_indices: number[];
  predictions_a: number[];
  predictions_b: number[];
  confidence_delta: number[];
  boundary_a: ResponseMap | null;
  boundary_b: ResponseMap | null;
  boundary_flip_fraction: number | null;
  layers: ParamChange[];
  total_delta_norm: number;
  total_relative_change: number;
  component: ComponentCompare | null;
}

// ── Forward / Backward Pass Explorer (mirror of ComputationTrace) ────────────
// Every tensor of one forward pass and every gradient of its backward pass,
// computed by PyTorch autograd on the probe (checkpoint + what-if applied).

export interface TraceLayer {
  layer: number;
  label: string;
  role: Role;
  activation: string; // 'Softmax' for the output layer
  input_names: string[];
  neuron_names: string[];
  weight: number[][]; // [out][in]
  bias: number[];
  input: number[]; // a_{l-1}
  z: number[];
  a: number[];
  ablated: number[];
  edited_bias: number[];
  edited_weights: [number, number][]; // [source, target]
  grad_a: number[] | null; // hidden only
  local_grad: number[] | null; // da/dz, hidden only (ablation mask included)
  grad_z: number[];
  grad_weight: number[][];
  grad_bias: number[];
  grad_input: number[]; // dL/da_{l-1}
}

export interface SgdPreview {
  learning_rate: number;
  loss_before: number;
  loss_after: number;
  target_prob_before: number;
  target_prob_after: number;
  predicted_after: number;
  update_norm: number;
  note: string;
}

export interface ComputationTrace {
  probe: ResolvedProbe;
  provenance: Provenance;
  feature_names: string[];
  class_names: string[];
  input: number[];
  layers: TraceLayer[];
  logits: number[];
  probabilities: number[];
  predicted_class: number;
  target: number;
  loss: number;
  grad_input: number[];
  sgd_preview: SgdPreview | null;
}
