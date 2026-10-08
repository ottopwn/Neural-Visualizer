import { describe, expect, it } from 'vitest';
import raw from './fixtures/trace.json';
import { edgeList, fitDistance, layoutNetwork, nodeValues, selectEdges, strongestInto } from '../scene3d';
import type { ComputationTrace } from '../types';

const t = raw as unknown as ComputationTrace; // MLP 2→3→4→2

describe('3D layout', () => {
  it('places every neuron once, layers along x, centred', () => {
    const l = layoutNetwork([2, 3, 4, 2]);
    expect(l.positions.map((p) => p.length)).toEqual([2, 3, 4, 2]);
    expect(new Set(l.positions.flat().map((p) => p.join(','))).size).toBe(11);
    l.positions.forEach((p, g) => p.forEach((v) => expect(v[0]).toBe(l.layerX[g])));
    expect(l.center[0]).toBeCloseTo(0);
  });

  it('uses a y-z grid for wide layers', () => {
    const l = layoutNetwork([2, 32, 2]);
    expect(new Set(l.positions[1].map((p) => p[2])).size).toBeGreaterThan(1);
    expect(l.extents[1].z).toBeGreaterThan(0);
    expect(l.extents[0].z).toBe(0);
  });

  it('fit distance grows with the radius', () => {
    expect(fitDistance(10, 45)).toBeGreaterThan(fitDistance(5, 45));
  });
});

describe('3D values come from the trace', () => {
  it('node values per colour mode', () => {
    expect(nodeValues(t, 'signal')[1]).toEqual(t.layers[0].a);
    expect(nodeValues(t, 'weights')[2]).toEqual(t.layers[1].bias);
    expect(nodeValues(t, 'gradients')[0]).toEqual(t.grad_input);
    expect(nodeValues(t, 'gradients')[3]).toEqual(t.layers[2].grad_z);
  });

  it('edge values: w·a, w, dL/dw', () => {
    const n = 2 * 3 + 3 * 4 + 4 * 2;
    const sig = edgeList(t, 'signal');
    expect(sig).toHaveLength(n);
    const e = sig.find((x) => x.layer === 2 && x.source === 1 && x.target === 3)!;
    expect(e.value).toBeCloseTo(t.layers[1].weight[3][1] * t.layers[1].input[1], 10);
    expect(edgeList(t, 'weights').find((x) => x.layer === 3 && x.source === 2 && x.target === 0)!.value).toBe(t.layers[2].weight[0][2]);
    expect(edgeList(t, 'gradients').find((x) => x.layer === 1 && x.source === 0 && x.target === 2)!.value).toBe(t.layers[0].grad_weight[2][0]);
  });
});

describe('edge filtering is disclosed', () => {
  const edges = edgeList(t, 'signal');

  it('keeps the strongest edges and reports the total', () => {
    const s = selectEdges(edges, 5, null);
    expect(s.shown).toHaveLength(5);
    expect(s.total).toBe(edges.length);
    expect(s.filtered).toBe(true);
    const minShown = Math.min(...s.shown.map((i) => Math.abs(edges[i].value)));
    const others = edges.map((_, i) => i).filter((i) => !s.shown.includes(i));
    expect(Math.max(...others.map((i) => Math.abs(edges[i].value)))).toBeLessThanOrEqual(minShown);
  });

  it('shows everything when the limit allows it', () => {
    const s = selectEdges(edges, 1000, null);
    expect(s.filtered).toBe(false);
    expect(s.shown.length).toBe(edges.length);
  });

  it('always emphasises the selected neuron’s incoming and outgoing edges', () => {
    const s = selectEdges(edges, 0, { kind: 'neuron', layer: 2, index: 1 });
    const em = s.emphasised.map((i) => edges[i]);
    expect(em).toHaveLength(3 + 2); // 3 inputs from Dense 1, 2 outputs to the output layer
    expect(em.every((e) => (e.layer === 2 && e.target === 1) || (e.layer === 3 && e.source === 1))).toBe(true);
    expect(s.shown).toHaveLength(0);
  });

  it('emphasises exactly a selected connection', () => {
    const s = selectEdges(edges, 3, { kind: 'connection', layer: 1, source: 1, target: 2 });
    expect(s.emphasised.map((i) => edges[i])).toEqual([expect.objectContaining({ layer: 1, source: 1, target: 2 })]);
  });

  it('strongest edges into a layer', () => {
    const idx = strongestInto(edges, 2, 4);
    expect(idx).toHaveLength(4);
    expect(idx.every((i) => edges[i].layer === 2)).toBe(true);
  });
});
