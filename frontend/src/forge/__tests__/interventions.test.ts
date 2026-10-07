import { describe, expect, it } from 'vitest';
import * as iv from '../interventions';
import type { Intervention, ModelStructure } from '../types';

const structure: ModelStructure = {
  model_family: 'mlp', input_dim: 2, n_classes: 2, class_names: ['Class 0', 'Class 1'], feature_names: ['x1', 'x2'],
  param_count: 0,
  layers: [
    { layer: 0, role: 'input', label: 'Input', size: 2, activation: null, fan_in: null, param_count: 0, weight_shape: null, bias_shape: null },
    { layer: 1, role: 'hidden', label: 'Dense 1', size: 4, activation: 'ReLU', fan_in: 2, param_count: 12, weight_shape: [4, 2], bias_shape: [4] },
    { layer: 2, role: 'output', label: 'Output', size: 2, activation: 'Softmax', fan_in: 4, param_count: 10, weight_shape: [2, 4], bias_shape: [2] },
  ],
};

describe('intervention log', () => {
  it('toggles ablation on and off', () => {
    let log: Intervention[] = [];
    log = iv.toggleAblation(log, 1, 2);
    expect(iv.isAblated(log, 1, 2)).toBe(true);
    log = iv.toggleAblation(log, 1, 2);
    expect(log).toEqual([]);
  });

  it('undo pops only the last action so earlier values come back', () => {
    const log: Intervention[] = [
      { type: 'set_bias', layer: 1, index: 0, value: 1 },
      { type: 'set_bias', layer: 1, index: 0, value: 5 },
    ];
    const undone = iv.undo(log);
    expect(iv.effective(undone)).toEqual([{ type: 'set_bias', layer: 1, index: 0, value: 1 }]);
    expect(iv.undo([])).toEqual([]);
  });

  it('effective keeps the last edit per element', () => {
    const log: Intervention[] = [
      { type: 'set_weight', layer: 2, source: 1, target: 0, value: 0 },
      { type: 'ablate_neuron', layer: 1, index: 3 },
      { type: 'set_weight', layer: 2, source: 1, target: 0, value: 2 },
    ];
    expect(iv.effective(log)).toEqual([
      { type: 'set_weight', layer: 2, source: 1, target: 0, value: 2 },
      { type: 'ablate_neuron', layer: 1, index: 3 },
    ]);
  });

  it('restores a single component without touching others', () => {
    const log: Intervention[] = [
      { type: 'set_weight', layer: 2, source: 1, target: 0, value: 0 },
      { type: 'set_bias', layer: 1, index: 0, value: 3 },
      { type: 'ablate_neuron', layer: 1, index: 0 },
    ];
    expect(iv.restoreComponent(log, { kind: 'connection', layer: 2, source: 1, target: 0 })).toHaveLength(2);
    expect(iv.restoreBias(log, 1, 0)).toEqual([log[0], log[2]]);
  });

  it('describes interventions with model names', () => {
    expect(iv.describe({ type: 'ablate_neuron', layer: 1, index: 2 }, structure)).toBe('Disable L1N3');
    expect(iv.describe({ type: 'set_weight', layer: 2, source: 3, target: 1, value: 0 }, structure)).toBe('Cut L1N4 → C1');
    expect(iv.describe({ type: 'set_weight', layer: 1, source: 0, target: 1, value: 0.5 }, structure)).toBe('w(x1 → L1N2) = 0.500');
    expect(iv.describe({ type: 'set_bias', layer: 2, index: 0, value: 2 }, structure)).toBe('bias(C0) = 2');
  });
});
