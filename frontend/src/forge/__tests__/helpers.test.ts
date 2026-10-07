import { describe, expect, it } from 'vitest';
import { applyActivation, sampleCurve } from '../activations';
import { explainConnection, gradientHint } from '../explain';
import { classColor, diverging, fmt, fmtSigned, NEG, NEUTRAL, POS, rankByMagnitude } from '../format';
import { highlightFor, layerLabels, nearestSample, refForNode } from '../selection';
import type { NetworkGraph } from '../../types';
import type { ConnectionInspection, ModelStructure } from '../types';

describe('format', () => {
  it('formats numbers predictably', () => {
    expect(fmt(null)).toBe('—');
    expect(fmt(0.12345)).toBe('0.123');
    expect(fmt(0.00001)).toBe('1.00e-5');
    expect(fmtSigned(0.5, 2)).toBe('+0.50');
    expect(fmtSigned(-0.5, 2)).toBe('-0.50');
  });

  it('diverging colours encode sign', () => {
    expect(diverging(1, 1)).toEqual(POS);
    expect(diverging(-1, 1)).toEqual(NEG);
    expect(diverging(0, 1)).toEqual(NEUTRAL);
    expect(diverging(3, 0)).toEqual(NEUTRAL);
    expect(classColor(0.5)).toEqual(NEUTRAL);
  });

  it('ranks by magnitude', () => {
    expect(rankByMagnitude([0.1, -3, 2])).toEqual([1, 2, 0]);
  });
});

describe('activations mirror the backend', () => {
  it('computes known values', () => {
    expect(applyActivation('ReLU', -2)).toBe(0);
    expect(applyActivation('LeakyReLU', -1)).toBeCloseTo(-0.2);
    expect(applyActivation('Sigmoid', 0)).toBeCloseTo(0.5);
    expect(applyActivation('ELU', -1)).toBeCloseTo(Math.exp(-1) - 1);
    expect(applyActivation('SELU', 1)).toBeCloseTo(1.0507009873554805);
    expect(applyActivation('Softmax', 1)).toBeNull();
    expect(sampleCurve('Tanh', -1, 1, 5)).toHaveLength(5);
  });
});

const graph: NetworkGraph = {
  nodes: [
    { id: 0, x: 0, y: 0, name: 'x1', layer: 0, index: 0, layer_type: 'input', value: 1 },
    { id: 1, x: 2, y: 0, name: 'L1N1', layer: 1, index: 0, layer_type: 'hidden', value: 0.5 },
    { id: 2, x: 2, y: 1, name: 'L1N2', layer: 1, index: 1, layer_type: 'hidden', value: 0 },
  ],
  edges: [
    { source: 0, target: 1, weight: 0.3, layer: 1, source_index: 0, target_index: 0 },
    { source: 0, target: 2, weight: -0.3, layer: 1, source_index: 0, target_index: 1 },
  ],
};

describe('selection mapping', () => {
  it('maps nodes to refs only for model-backed graphs', () => {
    expect(refForNode(graph.nodes[2])).toEqual({ kind: 'neuron', layer: 1, index: 1 });
    expect(refForNode({ id: 9, x: 0, y: 0, name: 'n', layer: 1, layer_type: 'conv', value: 0 })).toBeNull();
  });

  it('highlights neurons, connections and layers', () => {
    expect([...highlightFor(graph, { kind: 'neuron', layer: 1, index: 1 }).nodeIds]).toEqual([2]);
    const c = highlightFor(graph, { kind: 'connection', layer: 1, source: 0, target: 1 });
    expect([...c.edgeIds]).toEqual([1]);
    expect([...c.nodeIds].sort()).toEqual([0, 2]);
    expect(highlightFor(graph, { kind: 'layer', layer: 1 }).layer).toBe(1);
    expect(highlightFor(graph, null).nodeIds.size).toBe(0);
  });

  it('finds the nearest sample within a radius', () => {
    const X = [[0, 0], [1, 1], [0.2, 0.1]];
    expect(nearestSample(X, 0.18, 0.12, 0.1)).toBe(2);
    expect(nearestSample(X, 5, 5, 0.5)).toBe(-1);
  });

  it('labels layers from the structure', () => {
    const s = {
      layers: [
        { layer: 0, role: 'input', label: 'Input', size: 2, activation: null },
        { layer: 1, role: 'output', label: 'Output', size: 2, activation: 'Softmax' },
      ],
    } as unknown as ModelStructure;
    expect(layerLabels(s)).toEqual({
      0: { title: 'Input', subtitle: '2 features' },
      1: { title: 'Output', subtitle: '2 · softmax' },
    });
  });
});

describe('learn-mode explanations use real values', () => {
  it('describes a connection with its numbers', () => {
    const c = {
      source_name: 'L1N1', target_name: 'L2N3', source_value: 0.5, weight: -2, contribution: -1,
      share_of_input: 0.25, rank: 2, fan_in: 8,
    } as ConnectionInspection;
    const [a, b] = explainConnection(c);
    expect(a).toContain('0.500');
    expect(a).toContain('-2.000');
    expect(b).toContain('-1.000');
    expect(b).toContain('25%');
  });

  it('gradient hints follow the descent sign convention', () => {
    expect(gradientHint(-0.3, 'the bias', 'Class 1')).toMatch(/^Increasing/);
    expect(gradientHint(0.3, 'the bias', 'Class 1')).toMatch(/^Decreasing/);
    expect(gradientHint(0, 'the bias', 'Class 1')).toBeNull();
  });
});
