import { useMemo } from 'react';
import { explainLayer } from '../../forge/explain';
import { diverging, fmt, maxAbs, pct } from '../../forge/format';
import { useForgeStore } from '../../forge/store';
import type { LayerInspection } from '../../forge/types';
import { HeatmapCanvas, KV, MiniHistogram, Section } from './charts';

function StatsBlock({ title, s }: { title: string; s: LayerInspection['dataset_stats'] }) {
  return (
    <Section title={title} hint={`shape [${s.shape.join(', ')}]`}>
      <div className="grid grid-cols-2 gap-x-4">
        <KV k="mean" v={fmt(s.mean)} />
        <KV k="std" v={fmt(s.std)} />
        <KV k="min" v={fmt(s.min)} />
        <KV k="max" v={fmt(s.max)} />
        <KV k="‖·‖₂" v={fmt(s.l2_norm)} />
        <KV k="zeros" v={pct(s.frac_zero, 1)} />
      </div>
    </Section>
  );
}

export function LayerPanel({ l }: { l: LayerInspection }) {
  const mode = useForgeStore((s) => s.mode);
  const select = useForgeStore((s) => s.select);
  const lab = mode === 'lab';
  const wMax = useMemo(() => (l.weights ? maxAbs(l.weights.flat()) : 0), [l.weights]);
  const weightColor = useMemo(() => (v: number) => diverging(v, wMax), [wMax]);
  const gMax = useMemo(() => (l.grad_weights ? maxAbs(l.grad_weights.flat()) : 0), [l.grad_weights]);
  const gradColor = useMemo(() => (v: number) => diverging(v, gMax), [gMax]);
  const aMax = maxAbs(l.activations) || 1;

  return (
    <div className="space-y-2.5">
      {!lab && (
        <Section title="What is happening">
          <ul className="space-y-1.5 text-[12px] leading-relaxed">
            {explainLayer(l).map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Section>
      )}

      <Section title={l.role === 'output' ? 'Outputs on this probe' : 'Activations on this probe'} hint="click a neuron">
        <div className="flex flex-wrap gap-1">
          {l.activations.map((a, i) => {
            const ablated = l.ablated.includes(i);
            const dead = l.never_active.includes(i);
            const t = Math.abs(a) / aMax;
            return (
              <button
                key={i}
                type="button"
                onClick={() => select({ kind: 'neuron', layer: l.layer, index: i })}
                title={`${l.neuron_names[i]} = ${fmt(a, 4)}${ablated ? ' (disabled)' : dead ? ' (never active on dataset)' : ''}`}
                className="w-6 h-6 rounded text-[8px] font-mono flex items-center justify-center"
                style={{
                  background: ablated ? '#374151' : a >= 0 ? `rgba(16,185,129,${0.12 + t * 0.85})` : `rgba(239,68,68,${0.12 + t * 0.85})`,
                  border: dead ? '1px dashed #f87171' : '1px solid var(--border)',
                  color: '#e2e8f0',
                }}
              >
                {ablated ? '×' : i + 1}
              </button>
            );
          })}
        </div>
        {l.never_active.length > 0 && (
          <p className="text-[10px] mt-1.5" style={{ color: '#fca5a5' }}>
            Dashed: {l.never_active.length} neuron(s) output 0 for every sample in the dataset.
          </p>
        )}
      </Section>

      {l.weights && l.weight_stats && (
        <Section title="Weight matrix W" hint={`[${l.size} × ${l.fan_in}] · rows = neurons, cols = inputs`}>
          <HeatmapCanvas
            ariaLabel={`${l.label} weight matrix`}
            values={l.weights}
            color={weightColor}
            flipY
            height={Math.min(220, Math.max(60, l.size * 7))}
          />
          <div className="flex justify-between text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>
            <span style={{ color: '#fca5a5' }}>−{fmt(wMax, 2)}</span>
            <span>0</span>
            <span style={{ color: '#6ee7b7' }}>+{fmt(wMax, 2)}</span>
          </div>
          {l.weight_histogram && <MiniHistogram hist={l.weight_histogram} color="#a78bfa" height={40} />}
        </Section>
      )}

      {lab && l.weight_stats && <StatsBlock title="Weight statistics" s={l.weight_stats} />}
      {lab && l.bias_stats && <StatsBlock title="Bias statistics" s={l.bias_stats} />}
      <StatsBlock title={lab ? 'Activation statistics (whole dataset)' : 'Outputs across the dataset'} s={l.dataset_stats} />

      {lab && l.grad_weights && (
        <Section title="Gradient dL/dW" hint={`‖dL/dW‖ = ${fmt(l.grad_weight_norm, 4)} · ‖dL/db‖ = ${fmt(l.grad_bias_norm, 4)}`}>
          <HeatmapCanvas
            ariaLabel={`${l.label} weight gradient`}
            values={l.grad_weights}
            color={gradColor}
            flipY
            height={Math.min(180, Math.max(50, l.size * 6))}
          />
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>
            For this single probe. Gradient descent would move each weight against this sign.
          </p>
        </Section>
      )}

      {lab && (
        <Section title="Tensor shapes" hint={`${l.param_count} parameters`}>
          {Object.entries(l.shapes).map(([k, v]) => <KV key={k} k={k} v={`[${v.join(', ')}]`} />)}
          {l.activation_fn && <KV k="activation" v={l.activation_fn} mono={false} />}
        </Section>
      )}
    </div>
  );
}
