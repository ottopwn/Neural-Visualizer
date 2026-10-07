// Map between graph elements and introspection component references.

import type { NetworkGraph, NetworkNode } from '../types';
import type { ComponentRef, ModelStructure } from './types';

export interface Highlight {
  nodeIds: Set<number>;
  edgeIds: Set<number>;
  layer: number | null;
}

export const EMPTY_HIGHLIGHT: Highlight = { nodeIds: new Set(), edgeIds: new Set(), layer: null };

export function refForNode(node: NetworkNode): ComponentRef | null {
  if (node.index === undefined) return null; // illustrative graph: no model behind it
  return { kind: 'neuron', layer: node.layer, index: node.index };
}

export function highlightFor(graph: NetworkGraph, ref: ComponentRef | null): Highlight {
  if (!ref) return EMPTY_HIGHLIGHT;
  if (ref.kind === 'layer') return { nodeIds: new Set(), edgeIds: new Set(), layer: ref.layer };
  if (ref.kind === 'neuron') {
    const node = graph.nodes.find((n) => n.layer === ref.layer && n.index === ref.index);
    return { nodeIds: new Set(node ? [node.id] : []), edgeIds: new Set(), layer: null };
  }
  const edgeIds = new Set<number>();
  const nodeIds = new Set<number>();
  graph.edges.forEach((e, i) => {
    if (e.layer === ref.layer && e.source_index === ref.source && e.target_index === ref.target) {
      edgeIds.add(i);
      nodeIds.add(e.source);
      nodeIds.add(e.target);
    }
  });
  return { nodeIds, edgeIds, layer: null };
}

export function layerLabels(structure: ModelStructure): Record<number, { title: string; subtitle: string }> {
  const out: Record<number, { title: string; subtitle: string }> = {};
  for (const l of structure.layers) {
    out[l.layer] = {
      title: l.label,
      subtitle: l.role === 'input'
        ? `${l.size} feature${l.size > 1 ? 's' : ''}`
        : `${l.size} · ${l.activation === 'Softmax' ? 'softmax' : l.activation}`,
    };
  }
  return out;
}

/** Index of the dataset row closest to (x, y), or -1 if none is within maxDist. */
export function nearestSample(X: number[][], x: number, y: number, maxDist: number): number {
  let best = -1;
  let bestD = maxDist * maxDist;
  X.forEach((p, i) => {
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

export function architectureString(structure: ModelStructure): string {
  return structure.layers.map((l) => l.size).join(' → ');
}
