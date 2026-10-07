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

export interface HistoryRow { epoch: number; loss: number; accuracy: number }

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
