// Pure helpers for the what-if intervention list.
//
// Semantics: the list is an append-only log of user actions, applied in
// order by the backend (later edits of the same element win).  `undo` pops
// the last action, `reset` clears the log, and restoring an ablated neuron
// removes its ablation entries.

import type { ComponentRef, Intervention, ModelStructure } from './types';

export function neuronName(structure: ModelStructure, layer: number, index: number): string {
  const info = structure.layers[layer];
  if (!info) return `L${layer}N${index + 1}`;
  if (info.role === 'input') return structure.feature_names[index] ?? `x${index + 1}`;
  if (info.role === 'output') return `C${index}`;
  return `L${layer}N${index + 1}`;
}

export function isAblated(ivs: Intervention[], layer: number, index: number): boolean {
  return ivs.some((iv) => iv.type === 'ablate_neuron' && iv.layer === layer && iv.index === index);
}

export function toggleAblation(ivs: Intervention[], layer: number, index: number): Intervention[] {
  if (isAblated(ivs, layer, index)) {
    return ivs.filter((iv) => !(iv.type === 'ablate_neuron' && iv.layer === layer && iv.index === index));
  }
  return [...ivs, { type: 'ablate_neuron', layer, index }];
}

/** Remove every edit that targets the given component (neuron bias/ablation or one weight). */
export function restoreComponent(ivs: Intervention[], ref: ComponentRef): Intervention[] {
  return ivs.filter((iv) => !touches(iv, ref));
}

export function restoreBias(ivs: Intervention[], layer: number, index: number): Intervention[] {
  return ivs.filter((iv) => !(iv.type === 'set_bias' && iv.layer === layer && iv.index === index));
}

export function touches(iv: Intervention, ref: ComponentRef): boolean {
  if (ref.kind === 'connection') {
    return iv.type === 'set_weight' && iv.layer === ref.layer && iv.source === ref.source && iv.target === ref.target;
  }
  if (ref.kind === 'neuron') {
    return (iv.type === 'ablate_neuron' || iv.type === 'set_bias') && iv.layer === ref.layer && iv.index === ref.index;
  }
  return iv.layer === ref.layer;
}

export function undo(ivs: Intervention[]): Intervention[] {
  return ivs.slice(0, -1);
}

/** Latest effective edit per element, in first-touched order (for chips/summary). */
export function effective(ivs: Intervention[]): Intervention[] {
  const byKey = new Map<string, Intervention>();
  for (const iv of ivs) {
    const key = iv.type === 'set_weight'
      ? `w:${iv.layer}:${iv.source}:${iv.target}`
      : `${iv.type}:${iv.layer}:${iv.index}`;
    byKey.set(key, iv);
  }
  return [...byKey.values()];
}

export function describe(iv: Intervention, structure: ModelStructure | null): string {
  const name = (layer: number, index: number) =>
    structure ? neuronName(structure, layer, index) : `L${layer}#${index}`;
  switch (iv.type) {
    case 'ablate_neuron':
      return `Disable ${name(iv.layer, iv.index)}`;
    case 'set_bias':
      return `bias(${name(iv.layer, iv.index)}) = ${formatEditValue(iv.value)}`;
    case 'set_weight':
      return iv.value === 0
        ? `Cut ${name(iv.layer - 1, iv.source)} → ${name(iv.layer, iv.target)}`
        : `w(${name(iv.layer - 1, iv.source)} → ${name(iv.layer, iv.target)}) = ${formatEditValue(iv.value)}`;
  }
}

function formatEditValue(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}

/** Graph-node id lookup for a component reference (nodes carry layer + index). */
export function refKey(ref: ComponentRef | null): string {
  if (!ref) return '';
  if (ref.kind === 'neuron') return `n:${ref.layer}:${ref.index}`;
  if (ref.kind === 'layer') return `l:${ref.layer}`;
  return `c:${ref.layer}:${ref.source}:${ref.target}`;
}
