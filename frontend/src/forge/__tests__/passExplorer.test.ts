import { describe, expect, it } from 'vitest';
import raw from './fixtures/trace.json';
import {
  backpropTerms, buildSteps, columnValues, contributionTerms, diff, focusNeuron, reached, softmaxParts, stepIndexOf,
} from '../passExplorer';
import type { ComputationTrace } from '../types';

// A real trace produced by the backend (MLP 2→3→4→2, Tanh/ReLU, XOR, L2N2 disabled).
const t = raw as unknown as ComputationTrace;

describe('pass explorer · steps', () => {
  it('forward walks input → (z, a) per hidden layer → logits → softmax → prediction', () => {
    const s = buildSteps(t, 'forward');
    expect(s.map((x) => x.kind)).toEqual(['input', 'linear', 'activation', 'linear', 'activation', 'logits', 'softmax', 'prediction']);
    expect(s.map((x) => x.layer)).toEqual([0, 1, 1, 2, 2, 3, 3, 3]);
    expect(s.find((x) => x.kind === 'linear')!.edgeLayer).toBe(1);
    expect(new Set(s.map((x) => x.id)).size).toBe(s.length);
  });

  it('backward walks loss → dL/dlogits → (dW, Wᵀδ, δ) per layer → input saliency', () => {
    const s = buildSteps(t, 'backward');
    expect(s.map((x) => x.kind)).toEqual([
      'loss', 'grad_logits', 'grad_params', 'grad_input', 'grad_activation', 'grad_params', 'grad_input', 'grad_activation', 'grad_params', 'saliency',
    ]);
    expect(s.map((x) => x.layer)).toEqual([3, 3, 3, 2, 2, 2, 1, 1, 1, 0]);
  });

  it('reveals layers progressively in each direction', () => {
    const f = buildSteps(t, 'forward');
    expect(reached(f[1], 3)).toEqual([true, true, false, false]);
    const b = buildSteps(t, 'backward');
    expect(reached(b[3], 3)).toEqual([false, false, true, true]);
  });

  it('keeps the current step across traces by id and falls back to the first', () => {
    const f = buildSteps(t, 'forward');
    expect(stepIndexOf(f, 'f:act:2')).toBe(4);
    expect(stepIndexOf(f, 'f:act:9')).toBe(0);
    expect(stepIndexOf(f, null)).toBe(0);
  });
});

describe('pass explorer · identities on real values', () => {
  it('z = Σ w·a + b for every neuron of every layer', () => {
    for (const d of t.layers) {
      d.z.forEach((z, i) => {
        const c = contributionTerms(d, i);
        expect(c.sum + c.bias).toBeCloseTo(z, 5);
        expect(c.z).toBe(z);
        const mags = c.terms.map((x) => Math.abs(x.product));
        expect(mags).toEqual([...mags].sort((a, b) => b - a));
      });
    }
  });

  it('softmax of the logits reproduces the probabilities and the loss', () => {
    const sm = softmaxParts(t.logits);
    sm.probs.forEach((p, i) => expect(p).toBeCloseTo(t.probabilities[i], 6));
    expect(-Math.log(t.probabilities[t.target])).toBeCloseTo(t.loss, 5);
  });

  it('dL/dlogits = p − onehot(target)', () => {
    const out = t.layers[t.layers.length - 1];
    out.grad_z.forEach((g, i) => expect(g).toBeCloseTo(t.probabilities[i] - (i === t.target ? 1 : 0), 6));
  });

  it('Wᵀ·δ reproduces dL/da of the previous layer; δ = dL/da ⊙ f′(z)', () => {
    t.layers.forEach((d, k) => {
      d.input.forEach((_, j) => {
        const b = backpropTerms(d, j);
        expect(b.sum).toBeCloseTo(b.reported, 5);
        if (k > 0) expect(b.reported).toBeCloseTo(t.layers[k - 1].grad_a![j], 6);
        else expect(b.reported).toBeCloseTo(t.grad_input[j], 6);
      });
      if (d.grad_a && d.local_grad) d.grad_z.forEach((g, i) => expect(g).toBeCloseTo(d.grad_a![i] * d.local_grad![i], 6));
    });
  });

  it('a disabled neuron outputs 0 and passes no gradient back', () => {
    const d = t.layers[1];
    expect(d.ablated).toEqual([1]);
    expect(d.a[1]).toBe(0);
    expect(d.local_grad![1]).toBe(0);
    expect(d.grad_z[1]).toBe(0);
  });

  it('dL/dW = δ ⊗ a_prev', () => {
    for (const d of t.layers) d.grad_weight.forEach((row, i) => row.forEach((g, j) => expect(g).toBeCloseTo(d.grad_z[i] * d.input[j], 6)));
  });

  it('the SGD preview is a real step that lowers this sample’s loss', () => {
    expect(t.sgd_preview!.loss_before).toBeCloseTo(t.loss, 6);
    expect(t.sgd_preview!.loss_after).toBeLessThan(t.sgd_preview!.loss_before);
  });
});

describe('pass explorer · views', () => {
  it('columns show outputs forward, z on the linear step, gradients backward', () => {
    const f = buildSteps(t, 'forward');
    const cols = columnValues(t, f[1]);
    expect(cols[0]).toEqual(t.input);
    expect(cols[1]).toEqual(t.layers[0].z);
    const b = buildSteps(t, 'backward');
    const gi = b.find((x) => x.kind === 'grad_input')!;
    expect(columnValues(t, gi)[gi.layer]).toEqual(t.layers[gi.layer - 1].grad_a);
    expect(columnValues(t, b[0])[0]).toEqual(t.grad_input);
  });

  it('focuses the selected neuron when it is in the step layer, else the strongest', () => {
    const f = buildSteps(t, 'forward');
    expect(focusNeuron(t, f[2], { layer: 1, index: 2 })).toBe(2);
    const a = t.layers[0].a;
    const strongest = a.reduce((best, v, i) => (Math.abs(v) > Math.abs(a[best]) ? i : best), 0);
    expect(focusNeuron(t, f[2], { layer: 2, index: 0 })).toBe(strongest);
  });

  it('diffs vectors of equal shape only', () => {
    expect(diff([1, 2], [3, 5])).toEqual([2, 3]);
    expect(diff([1], [1, 2])).toBeNull();
  });
});
