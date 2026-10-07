import { Scissors } from 'lucide-react';
import { explainConnection, gradientHint } from '../../forge/explain';
import { fmt, fmtSigned, pct } from '../../forge/format';
import * as ivs from '../../forge/interventions';
import { useForgeStore } from '../../forge/store';
import type { ConnectionInspection, ModelStructure } from '../../forge/types';
import { KV, Section } from './charts';
import { ValueEditor } from './shared';

export function ConnectionPanel({ c, structure }: { c: ConnectionInspection; structure: ModelStructure }) {
  const mode = useForgeStore((s) => s.mode);
  const interventions = useForgeStore((s) => s.interventions);
  const select = useForgeStore((s) => s.select);
  const setInterventions = useForgeStore((s) => s.setInterventions);
  const addIntervention = useForgeStore((s) => s.addIntervention);
  const lab = mode === 'lab';
  const ref = { kind: 'connection' as const, layer: c.layer, source: c.source, target: c.target };
  const target = structure.class_names[c.probe.target];
  const hint = gradientHint(c.grad_weight, 'this weight', target);

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-2 text-xs">
        <button type="button" className="font-mono px-2 py-1 rounded border hover:bg-white/5" style={{ borderColor: 'var(--border-soft)' }}
          onClick={() => select({ kind: 'neuron', layer: c.layer - 1, index: c.source })}>{c.source_name}</button>
        <span style={{ color: 'var(--text-faint)' }}>── w ──▶</span>
        <button type="button" className="font-mono px-2 py-1 rounded border hover:bg-white/5" style={{ borderColor: 'var(--border-soft)' }}
          onClick={() => select({ kind: 'neuron', layer: c.layer, index: c.target })}>{c.target_name}</button>
      </div>

      {!lab && (
        <Section title="What is happening">
          <ul className="space-y-1.5 text-[12px] leading-relaxed">
            {explainConnection(c).map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Section>
      )}

      <Section title="Signal on this probe">
        <KV k="source output a" v={fmt(c.source_value, 4)} />
        <KV k={c.original_weight !== null ? 'weight w (edited)' : 'weight w'} v={fmtSigned(c.weight, 4)}
          color={c.original_weight !== null ? '#f59e0b' : undefined} />
        <KV k="contribution w·a" v={fmtSigned(c.contribution, 4)} color={c.contribution >= 0 ? '#6ee7b7' : '#fca5a5'} />
        <KV k={`share of ${c.target_name}'s input`} v={pct(c.share_of_input, 1)} />
        <KV k="rank among inputs" v={`${c.rank} / ${c.fan_in}`} />
        <KV k={`${c.target_name} z`} v={fmt(c.target_pre_activation, 4)} color="#93c5fd" />
      </Section>

      {lab && (
        <Section title="Across the dataset" hint="w·a over all samples">
          <div className="grid grid-cols-2 gap-x-4">
            <KV k="mean" v={fmt(c.dataset_contribution.mean)} />
            <KV k="std" v={fmt(c.dataset_contribution.std)} />
            <KV k="min" v={fmt(c.dataset_contribution.min)} />
            <KV k="max" v={fmt(c.dataset_contribution.max)} />
            <KV k="silent" v={pct(c.dataset_contribution.frac_zero, 0)} />
          </div>
        </Section>
      )}

      {lab ? (
        <Section title="Gradient">
          <KV k="dL/dw" v={fmt(c.grad_weight, 6)} />
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>loss = CE(target = {target}, {c.probe.target_source})</p>
        </Section>
      ) : hint ? (
        <Section title="Learning signal"><p className="text-[12px]">{hint}</p></Section>
      ) : null}

      <Section title="What-if" hint="changes the real forward pass">
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => addIntervention({ type: 'set_weight', layer: c.layer, source: c.source, target: c.target, value: 0 })}
            disabled={c.weight === 0}
            className="w-full flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-lg border disabled:opacity-40"
            style={{ borderColor: 'rgba(239,68,68,0.5)', color: '#fca5a5', background: 'rgba(239,68,68,0.08)' }}
          >
            <Scissors size={12} /> Cut this connection (w = 0)
          </button>
          <ValueEditor
            key={`${ivs.refKey(ref)}:${c.weight}`}
            label="weight"
            current={c.weight}
            original={c.original_weight}
            onApply={(value) => addIntervention({ type: 'set_weight', layer: c.layer, source: c.source, target: c.target, value })}
            onRestore={() => setInterventions(ivs.restoreComponent(interventions, ref))}
          />
        </div>
      </Section>

      {c.notes.length > 0 && (
        <ul className="text-[10px] space-y-0.5 px-1" style={{ color: 'var(--text-faint)' }}>
          {c.notes.map((t, i) => <li key={i}>• {t}</li>)}
        </ul>
      )}
    </div>
  );
}
