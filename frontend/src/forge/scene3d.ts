// Pure helpers for the 3D view of the real MLP.
//
// Geometry (where each neuron sits) is a presentation choice; every colour,
// size and edge emphasis is derived from the backend computation trace.
// When not every connection is drawn, `selectEdges` reports how many are
// shown so the UI can disclose the filtering.

import type { ComponentRef, ComputationTrace } from './types';

export type ColorMode = 'signal' | 'weights' | 'gradients';

export const COLOR_MODES: { id: ColorMode; label: string; node: string; edge: string }[] = [
  { id: 'signal', label: 'Signal', node: 'activation a on the probe', edge: 'contribution w·a on the probe' },
  { id: 'weights', label: 'Weights', node: 'bias b', edge: 'weight w' },
  { id: 'gradients', label: 'Gradients', node: 'error signal δ = dL/dz', edge: 'weight gradient dL/dw' },
];

export type Vec3 = [number, number, number];

export interface Layout {
  /** positions[layer][index] */
  positions: Vec3[][];
  layerX: number[];
  /** half extent of each layer in y / z */
  extents: { y: number; z: number }[];
  center: Vec3;
  radius: number;
}

export const LAYER_GAP = 6;
export const NEURON_GAP = 1.15;

/**
 * Layers along x; the neurons of a layer on a near-square grid in the y-z
 * plane (a single column for ≤ 8 neurons), so wide layers stay compact and
 * the depth axis carries structure instead of decoration.
 */
export function layoutNetwork(sizes: number[]): Layout {
  const n = sizes.length;
  const layerX = sizes.map((_, g) => (g - (n - 1) / 2) * LAYER_GAP);
  const positions: Vec3[][] = [];
  const extents: { y: number; z: number }[] = [];
  sizes.forEach((size, g) => {
    const cols = size <= 8 ? 1 : Math.ceil(Math.sqrt(size / 2));
    const rows = Math.ceil(size / cols);
    const pos: Vec3[] = [];
    for (let i = 0; i < size; i++) {
      const c = Math.floor(i / rows);
      const r = i % rows;
      pos.push([layerX[g], ((rows - 1) / 2 - r) * NEURON_GAP, (c - (cols - 1) / 2) * NEURON_GAP]);
    }
    positions.push(pos);
    extents.push({ y: ((rows - 1) / 2) * NEURON_GAP, z: ((cols - 1) / 2) * NEURON_GAP });
  });
  const all = positions.flat();
  const lo = [0, 1, 2].map((k) => Math.min(...all.map((p) => p[k])));
  const hi = [0, 1, 2].map((k) => Math.max(...all.map((p) => p[k])));
  const center: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const radius = Math.max(1, ...all.map((p) => Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2])));
  return { positions, layerX, extents, center, radius };
}

/** Value that colours each neuron, per graph layer, for a colour mode. */
export function nodeValues(t: ComputationTrace, mode: ColorMode): number[][] {
  if (mode === 'signal') return [t.input, ...t.layers.map((l) => l.a)];
  if (mode === 'weights') return [t.input.map(() => 0), ...t.layers.map((l) => l.bias)];
  return [t.grad_input, ...t.layers.map((l) => l.grad_z)];
}

export interface Edge3D {
  layer: number; // graph layer of the target
  source: number;
  target: number;
  value: number; // the quantity the edge encodes (signed)
  edited: boolean;
}

/** Every connection with the quantity it encodes in a colour mode. */
export function edgeList(t: ComputationTrace, mode: ColorMode): Edge3D[] {
  const out: Edge3D[] = [];
  t.layers.forEach((d, k) => {
    const edited = new Set(d.edited_weights.map(([s, tg]) => `${s}:${tg}`));
    d.weight.forEach((row, i) => row.forEach((w, j) => {
      const value = mode === 'signal' ? w * d.input[j] : mode === 'weights' ? w : d.grad_weight[i][j];
      out.push({ layer: k + 1, source: j, target: i, value, edited: edited.has(`${j}:${i}`) });
    }));
  });
  return out;
}

export interface EdgeSelection {
  /** indices into the edge list, drawn as the faint background set */
  shown: number[];
  /** indices drawn emphasised (connections of the selected neuron / the selected connection / edits) */
  emphasised: number[];
  total: number;
  /** true when `shown` ∪ `emphasised` is not every edge */
  filtered: boolean;
}

/**
 * The strongest `limit` edges by |value|, plus every edge touching the
 * selection (always emphasised) and every edited edge.
 */
export function selectEdges(edges: Edge3D[], limit: number, sel: ComponentRef | null): EdgeSelection {
  const touches = (e: Edge3D) => {
    if (!sel) return false;
    if (sel.kind === 'neuron') return (e.layer === sel.layer && e.target === sel.index) || (e.layer === sel.layer + 1 && e.source === sel.index);
    if (sel.kind === 'connection') return e.layer === sel.layer && e.source === sel.source && e.target === sel.target;
    return e.layer === sel.layer;
  };
  const emphasised: number[] = [];
  const rest: number[] = [];
  edges.forEach((e, i) => (touches(e) || e.edited ? emphasised : rest).push(i));
  // A selected layer can have thousands of edges: emphasise only its strongest.
  let emph = emphasised;
  if (sel?.kind === 'layer' && emphasised.length > limit) {
    emph = [...emphasised].sort((a, b) => Math.abs(edges[b].value) - Math.abs(edges[a].value)).slice(0, limit);
  }
  const shown = [...rest].sort((a, b) => Math.abs(edges[b].value) - Math.abs(edges[a].value)).slice(0, Math.max(0, limit));
  return { shown, emphasised: emph, total: edges.length, filtered: shown.length + emph.length < edges.length };
}

/** Indices of the `n` strongest edges into a graph layer (for animated signal pulses). */
export function strongestInto(edges: Edge3D[], layer: number, n: number): number[] {
  return edges.map((_, i) => i).filter((i) => edges[i].layer === layer)
    .sort((a, b) => Math.abs(edges[b].value) - Math.abs(edges[a].value)).slice(0, n);
}

/** Camera distance that fits a sphere of `radius` in a vertical field of view (degrees). */
export function fitDistance(radius: number, fovDeg: number, aspect = 1.6): number {
  const vf = (fovDeg * Math.PI) / 360;
  const hf = Math.atan(Math.tan(vf) * aspect);
  return (radius / Math.sin(Math.min(vf, hf))) * 0.92;
}
