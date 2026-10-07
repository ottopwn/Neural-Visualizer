// Pure helpers for the Forward / Backward Pass Explorer.
//
// The explorer never computes model values itself: every number comes from
// the backend `ComputationTrace` (one PyTorch forward + backward pass).  This
// module only decides *which* numbers a step shows and how they decompose
// (e.g. z = Σ w·a + b), so every displayed identity can be checked against
// the trace in unit tests.

import type { ComputationTrace, TraceLayer } from './types';

export type PassDirection = 'forward' | 'backward';

export type StepKind =
  | 'input' | 'linear' | 'activation' | 'logits' | 'softmax' | 'prediction'
  | 'loss' | 'grad_logits' | 'grad_params' | 'grad_input' | 'grad_activation' | 'saliency';

export interface PassStep {
  id: string;
  dir: PassDirection;
  kind: StepKind;
  /** Graph layer the step produces / is about (0 = input). */
  layer: number;
  /** Graph layer whose incoming connections carry the signal (null = none). */
  edgeLayer: number | null;
  title: string;
  short: string;
}

function layerName(t: ComputationTrace, layer: number): string {
  if (layer === 0) return 'Input';
  return t.layers[layer - 1].label;
}

/** The ordered steps of one direction for a trace. */
export function buildSteps(t: ComputationTrace, dir: PassDirection): PassStep[] {
  const L = t.layers.length; // dense layers; output = graph layer L
  const steps: PassStep[] = [];
  if (dir === 'forward') {
    steps.push({ id: 'f:input', dir, kind: 'input', layer: 0, edgeLayer: null, title: 'Input vector x', short: 'Input' });
    for (let g = 1; g <= L; g++) {
      const out = g === L;
      const name = layerName(t, g);
      steps.push({
        id: `f:linear:${g}`, dir, kind: out ? 'logits' : 'linear', layer: g, edgeLayer: g,
        title: out ? `${name}: linear transform → logits` : `${name}: linear transform z = W·a + b`,
        short: out ? 'Logits' : `${name} · z`,
      });
      if (!out) {
        steps.push({
          id: `f:act:${g}`, dir, kind: 'activation', layer: g, edgeLayer: null,
          title: `${name}: activation a = ${t.layers[g - 1].activation}(z)`, short: `${name} · a`,
        });
      }
    }
    steps.push({ id: 'f:softmax', dir, kind: 'softmax', layer: L, edgeLayer: null, title: 'Softmax → probabilities', short: 'Softmax' });
    steps.push({ id: 'f:pred', dir, kind: 'prediction', layer: L, edgeLayer: null, title: 'Prediction', short: 'Prediction' });
    return steps;
  }
  steps.push({ id: 'b:loss', dir, kind: 'loss', layer: L, edgeLayer: null, title: 'Loss L = −log p(target)', short: 'Loss' });
  steps.push({ id: 'b:dlogits', dir, kind: 'grad_logits', layer: L, edgeLayer: null, title: 'Gradient at the logits: dL/dz = p − y', short: 'dL/dlogits' });
  for (let g = L; g >= 1; g--) {
    const name = layerName(t, g);
    steps.push({
      id: `b:params:${g}`, dir, kind: 'grad_params', layer: g, edgeLayer: g,
      title: `${name}: parameter gradients dL/dW, dL/db`, short: `${name} · dW`,
    });
    if (g === 1) {
      steps.push({ id: 'b:saliency', dir, kind: 'saliency', layer: 0, edgeLayer: 1, title: 'Input saliency dL/dx = Wᵀ·δ', short: 'dL/dx' });
    } else {
      const prev = layerName(t, g - 1);
      steps.push({
        id: `b:input:${g}`, dir, kind: 'grad_input', layer: g - 1, edgeLayer: g,
        title: `Back through ${name}'s weights: dL/da = Wᵀ·δ`, short: `${prev} · dL/da`,
      });
      steps.push({
        id: `b:act:${g - 1}`, dir, kind: 'grad_activation', layer: g - 1, edgeLayer: null,
        title: `${prev}: through the activation δ = dL/da ⊙ f′(z)`, short: `${prev} · δ`,
      });
    }
  }
  return steps;
}

/** Dense layer of the trace for a graph layer (1..L). */
export function denseOf(t: ComputationTrace, layer: number): TraceLayer | null {
  return layer >= 1 ? t.layers[layer - 1] ?? null : null;
}

/** Values drawn on each graph layer's column for a step (forward: outputs, backward: gradients). */
export function columnValues(t: ComputationTrace, step: PassStep): number[][] {
  const cols: number[][] = [t.input, ...t.layers.map((l) => l.a)];
  if (step.dir === 'forward') {
    // Linear steps show z for their layer; later layers have not been computed yet.
    return cols.map((c, g) => {
      if (g === step.layer && (step.kind === 'linear' || step.kind === 'logits')) return t.layers[g - 1].z;
      return c;
    });
  }
  return cols.map((_, g) => {
    if (g === 0) return t.grad_input;
    const d = t.layers[g - 1];
    if (step.kind === 'grad_input' && g === step.layer) return d.grad_a ?? d.grad_z;
    return d.grad_z;
  });
}

/** Which graph layers a step has "reached" (for progressive reveal). */
export function reached(step: PassStep, nLayers: number): boolean[] {
  return Array.from({ length: nLayers + 1 }, (_, g) =>
    step.dir === 'forward' ? g <= step.layer : g >= step.layer);
}

export interface Term { index: number; name: string; a: number; w: number; product: number }

/** z_i = Σ_j W[i][j]·a_j + b_i, terms sorted by |product| (largest first). */
export function contributionTerms(d: TraceLayer, neuron: number): { terms: Term[]; sum: number; bias: number; z: number } {
  const terms = d.input.map((a, j) => ({ index: j, name: d.input_names[j], a, w: d.weight[neuron][j], product: d.weight[neuron][j] * a }));
  const sum = terms.reduce((s, x) => s + x.product, 0);
  terms.sort((x, y) => Math.abs(y.product) - Math.abs(x.product));
  return { terms, sum, bias: d.bias[neuron], z: d.z[neuron] };
}

/** dL/da_j (into layer `d`'s inputs) = Σ_i W[i][j]·δ_i, terms sorted by |product|. */
export function backpropTerms(d: TraceLayer, source: number): { terms: { index: number; name: string; w: number; delta: number; product: number }[]; sum: number; reported: number } {
  const terms = d.grad_z.map((delta, i) => ({ index: i, name: d.neuron_names[i], w: d.weight[i][source], delta, product: d.weight[i][source] * delta }));
  const sum = terms.reduce((s, x) => s + x.product, 0);
  terms.sort((x, y) => Math.abs(y.product) - Math.abs(x.product));
  return { terms, sum, reported: d.grad_input[source] };
}

/** Softmax pieces from the logits (exp shifted by the max for stability, as PyTorch does). */
export function softmaxParts(logits: number[]): { shifted: number[]; exps: number[]; total: number; probs: number[] } {
  const m = Math.max(...logits);
  const shifted = logits.map((z) => z - m);
  const exps = shifted.map(Math.exp);
  const total = exps.reduce((a, b) => a + b, 0);
  return { shifted, exps, total, probs: exps.map((e) => e / total) };
}

/** Neuron a step focuses on: the selected one if it is in the step's layer, else the most active / largest gradient. */
export function focusNeuron(t: ComputationTrace, step: PassStep, selected: { layer: number; index: number } | null): number {
  const g = step.layer;
  if (selected && selected.layer === g) return selected.index;
  const values = columnValues(t, step)[g] ?? [];
  let best = 0;
  values.forEach((v, i) => { if (Math.abs(v) > Math.abs(values[best])) best = i; });
  return best;
}

/** Index of a step id in a list, or 0 when it no longer exists (e.g. after rebuilding a different architecture). */
export function stepIndexOf(steps: PassStep[], id: string | null): number {
  if (!id) return 0;
  const i = steps.findIndex((s) => s.id === id);
  return i < 0 ? 0 : i;
}

/** Element-wise difference b - a when shapes match (epoch comparison), else null. */
export function diff(a: number[] | null | undefined, b: number[] | null | undefined): number[] | null {
  if (!a || !b || a.length !== b.length) return null;
  return b.map((v, i) => v - a[i]);
}
