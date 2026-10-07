import { describe, expect, it } from 'vitest';
import { describeComparison, describeFrame, describeSeries, describeSilence } from '../explainTime';
import {
  LruCache, makeAxis, nearestEpoch, playbackInterval, probeKey, resolveEpoch, sortedEpochs, stepEpoch,
} from '../timeline';
import type { ComponentHistory, EpochComparison, Frame } from '../types';

const EPOCHS = [0, 1, 2, 4, 8, 16];

describe('timeline navigation', () => {
  it('sorts and de-duplicates epochs', () => {
    expect(sortedEpochs([8, 0, 2, 2, 1])).toEqual([0, 1, 2, 8]);
  });

  it('steps between stored checkpoints only', () => {
    expect(stepEpoch(EPOCHS, 2, 1)).toBe(4);
    expect(stepEpoch(EPOCHS, 3, 1)).toBe(4); // from an unstored epoch
    expect(stepEpoch(EPOCHS, 16, 1)).toBeNull();
    expect(stepEpoch(EPOCHS, 4, -1)).toBe(2);
    expect(stepEpoch(EPOCHS, 0, -1)).toBeNull();
  });

  it('finds the nearest stored epoch (ties go to the later one)', () => {
    expect(nearestEpoch(EPOCHS, 5)).toBe(4);
    expect(nearestEpoch(EPOCHS, 6)).toBe(8);
    expect(nearestEpoch(EPOCHS, 100)).toBe(16);
    expect(nearestEpoch([], 3)).toBeNull();
  });

  it('resolves the live cursor to the live epoch', () => {
    expect(resolveEpoch(null, 16)).toBe(16);
    expect(resolveEpoch(4, 16)).toBe(4);
  });

  it('playback interval shrinks with speed', () => {
    expect(playbackInterval(2)).toBe(playbackInterval(1) / 2);
    expect(playbackInterval(4)).toBeLessThan(playbackInterval(0.5));
  });
});

describe('time axis', () => {
  it('linear axis is proportional to epochs and invertible', () => {
    const ax = makeAxis(EPOCHS, 'linear', 0, 160);
    expect(ax.toX(8)).toBe(80);
    expect(ax.fromX(40)).toBe(4);
  });

  it('checkpoint axis spaces stored checkpoints evenly and round-trips', () => {
    const ax = makeAxis(EPOCHS, 'checkpoints', 0, 100);
    expect(EPOCHS.map((e) => ax.toX(e))).toEqual([0, 20, 40, 60, 80, 100]);
    expect(ax.toX(12)).toBe(90); // halfway between 8 and 16
    for (const e of [0, 1.5, 3, 12, 16]) expect(ax.fromX(ax.toX(e))).toBeCloseTo(e);
  });
});

describe('LRU cache', () => {
  it('evicts the least recently used key', () => {
    const c = new LruCache<number>(2);
    c.set('a', 1);
    c.set('b', 2);
    c.get('a');
    c.set('c', 3);
    expect(c.has('a')).toBe(true);
    expect(c.has('b')).toBe(false);
    expect(c.size).toBe(2);
  });

  it('probe keys distinguish samples, free points and targets', () => {
    expect(probeKey({ sample_index: 1 })).not.toBe(probeKey({ sample_index: 2 }));
    expect(probeKey({ x: [0.1, 0.2] })).not.toBe(probeKey({ x: [0.1, 0.2], target: 1 }));
  });
});

// ── Learn-mode narration ────────────────────────────────────────────────────
const NAMES = ['Class 0', 'Class 1'];
const probe = { x: [0, 0], sample_index: 0, label: 1, target: 1, target_source: 'label' as const };

function frame(partial: Partial<Frame>): Frame {
  return {
    epoch: 10, is_latest: false, loss: 0.3, accuracy: 0.8, probe, probe_probabilities: [0.3, 0.7], probe_predicted: 1,
    predictions: [0, 1, 1, 0], confidence: [0.9, 0.8, 0.7, 0.6], boundary: null, previous: null, ...partial,
  };
}

describe('describeFrame', () => {
  it('says the network has not separated the classes when it predicts one class everywhere', () => {
    const text = describeFrame(frame({ epoch: 0, predictions: [1, 1, 1, 1], accuracy: 0.5 }), NAMES, 0.5).join(' ');
    expect(text).toContain('before any training');
    expect(text).toContain('predicts Class 1 for every point');
  });

  it('compares with the majority-class rate, using the real numbers', () => {
    const text = describeFrame(frame({ accuracy: 0.51 }), NAMES, 0.5).join(' ');
    expect(text).toContain('no better than always guessing');
    const good = describeFrame(frame({ accuracy: 0.8 }), NAMES, 0.5).join(' ');
    expect(good).toContain('80.0%');
    expect(good).toContain('50.0%');
  });

  it('describes the change since the previous checkpoint', () => {
    const text = describeFrame(frame({
      previous: { epoch: 4, loss: 0.6, accuracy: 0.6, changed: 3, fixed: 2, broken: 1, boundary_flip_fraction: 0.25 },
    }), NAMES, 0.5).join(' ');
    expect(text).toContain('epoch 4');
    expect(text).toContain('rose from 60.0% to 80.0%');
    expect(text).toContain('went down from 0.6000 to 0.3000');
    expect(text).toContain('3 points changed predicted class: 2 became correct and 1 became wrong');
    expect(text).toContain('25.0% of the plotted input plane');
    expect(text).toContain('Class 1 with 70.0% probability (true label: Class 1, correct)');
  });
});

describe('describeComparison', () => {
  const base: EpochComparison = {
    a: { epoch: 10, loss: 0.6, accuracy: 0.55, mean_confidence: 0.6, probe_probabilities: [0.6, 0.4], probe_predicted: 0 },
    b: { epoch: 30, loss: 0.2, accuracy: 0.9, mean_confidence: 0.9, probe_probabilities: [0.1, 0.9], probe_predicted: 1 },
    probe, loss_delta: -0.4, accuracy_delta: 0.35, changed: 40, changed_fraction: 0.2, fixed: 37, broken: 3,
    changed_indices: [], predictions_a: [], predictions_b: [], confidence_delta: [], boundary_a: null, boundary_b: null,
    boundary_flip_fraction: 0.3,
    layers: [
      { layer: 1, label: 'Dense 1', weight_norm_a: 1, weight_norm_b: 2, bias_norm_a: 1, bias_norm_b: 1, weight_delta_norm: 0.3, bias_delta_norm: 0.4, relative_change: 0.5, mean_abs_weight_delta: 0, max_abs_weight_delta: 0, top_neurons: [0], top_neuron_deltas: [0.5] },
      { layer: 2, label: 'Output', weight_norm_a: 1, weight_norm_b: 2, bias_norm_a: 1, bias_norm_b: 1, weight_delta_norm: 0.1, bias_delta_norm: 0, relative_change: 0.1, mean_abs_weight_delta: 0, max_abs_weight_delta: 0, top_neurons: [0], top_neuron_deltas: [0.1] },
    ],
    total_delta_norm: 0.6, total_relative_change: 0.3,
    component: {
      ref: { kind: 'neuron', layer: 1, index: 2 }, name: 'L1N3', response_a: null, response_b: null,
      rows: [
        { key: 'bias', label: 'bias b', group: 'parameter', a: 0.1, b: -0.2, delta: -0.3 },
        { key: 'act_mean', label: 'mean output', group: 'dataset', a: 0.05, b: 0.4, delta: 0.35 },
        { key: 'active_frac', label: 'active', group: 'dataset', a: 0.5, b: 0.75, delta: 0.25 },
      ],
    },
  };

  it('reports real deltas in words', () => {
    const text = describeComparison(base, NAMES).join(' ');
    expect(text).toContain('Between epoch 10 and epoch 30, accuracy increased from 55.0% to 90.0% while the loss decreased from 0.6000 to 0.2000');
    expect(text).toContain('40 points (20.0%) changed predicted class: 37 became correct, 3 became wrong');
    expect(text).toContain('flipped from Class 0 to Class 1');
    expect(text).toContain('Dense 1: they moved by 50%');
    expect(text).toContain("L1N3's average output over the dataset went from 0.050 to 0.400 (+0.350)");
    expect(text).toContain('active (non-zero) for 75% of the points at epoch 30, versus 50% at epoch 10');
  });

  it('does not invent change when nothing changed', () => {
    const same = { ...base, changed: 0, total_delta_norm: 0, boundary_flip_fraction: 0, component: null,
      b: { ...base.a }, layers: base.layers.map((l) => ({ ...l, relative_change: 0 })) };
    const text = describeComparison(same, NAMES).join(' ');
    expect(text).toContain('accuracy stayed at 55.0%');
    expect(text).toContain('Every point keeps the same predicted class');
    expect(text).toContain('identical parameters');
    expect(text).not.toContain('flipped');
  });
});

describe('describeSeries', () => {
  const h: ComponentHistory = {
    ref: { kind: 'neuron', layer: 1, index: 0 }, name: 'L1N1', epochs: [0, 5, 10], notes: [],
    series: [{ key: 'train', label: 'training ‖grad‖', group: 'gradient', values: [null, 0.5, 0.2] }],
  };
  it('starts from the first epoch that has a value', () => {
    expect(describeSeries(h, 'train', 10)).toBe('L1N1: training ‖grad‖ went from 0.500 at epoch 5 to 0.200 at epoch 10 (-0.300).');
    expect(describeSeries(h, 'train', 0)).toBeNull();
    expect(describeSeries(h, 'missing', 5)).toBeNull();
  });
});

describe('describeSilence / unchanged series', () => {
  const dead: ComponentHistory = {
    ref: { kind: 'neuron', layer: 2, index: 2 }, name: 'L2N3', epochs: [0, 1, 4], notes: [],
    series: [
      { key: 'active_frac', label: 'active', group: 'dataset', values: [0, 0, 0] },
      { key: 'bias', label: 'bias b', group: 'parameter', values: [-0.118, -0.118, -0.118] },
    ],
  };
  it('names a neuron that was never active as dead, from the data', () => {
    expect(describeSilence(dead, 4)).toContain('L2N3 has output 0 for every point');
    expect(describeSilence(dead, 4)).not.toContain('gradient');
    expect(describeSilence(dead, 4, 'ReLU')).toContain('no gradient');
    expect(describeSeries(dead, 'bias', 4)).toBe('L2N3: bias b has stayed at -0.118 since epoch 0.');
  });
  it('distinguishes a neuron that died later, and says nothing for an active one', () => {
    const died = { ...dead, series: [{ ...dead.series[0], values: [0.5, 0, 0] }] };
    expect(describeSilence(died, 1)).toContain('has become a "dead" neuron');
    expect(describeSilence(died, 0)).toBeNull();
  });
});
