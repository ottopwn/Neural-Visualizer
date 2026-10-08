import { RotateCcw, Undo2 } from 'lucide-react';
import { useMemo } from 'react';
import { classColor, pct } from '../../forge/format';
import * as ivs from '../../forge/interventions';
import { useForgeStore } from '../../forge/store';
import type { ComponentRef, Intervention } from '../../forge/types';
import { CompareBars, HeatmapCanvas, ProbBars } from './charts';
import { useDatasetPoints } from './hooks';
import { useNeutral } from './TimeMachine/hooks';

function refOf(iv: Intervention): ComponentRef {
  return iv.type === 'set_weight'
    ? { kind: 'connection', layer: iv.layer, source: iv.source, target: iv.target }
    : { kind: 'neuron', layer: iv.layer, index: iv.index };
}

/**
 * The probe's prediction, and -- while what-if edits are active -- the real
 * before/after comparison (probabilities, dataset accuracy, flip rate and
 * decision regions).  Laid out as a column for the inspector.
 */
export function WhatIfPanel() {
  const session = useForgeStore((s) => s.session);
  const comparison = useForgeStore((s) => s.comparison);
  const interventions = useForgeStore((s) => s.interventions);
  const undo = useForgeStore((s) => s.undo);
  const reset = useForgeStore((s) => s.reset);
  const select = useForgeStore((s) => s.select);
  const points = useDatasetPoints();
  const neutral = useNeutral();
  const color = useMemo(() => (v: number) => classColor(v, neutral), [neutral]);

  if (!session || !comparison) return <p className="text-[12px]" style={{ color: 'var(--text-faint)' }}>…</p>;
  const names = session.structure.class_names;
  const { baseline, intervened } = comparison;
  const active = interventions.length > 0;
  const effective = ivs.effective(interventions);
  const accDrop = intervened.dataset_accuracy < baseline.dataset_accuracy;

  if (!active) {
    return (
      <div className="space-y-2">
        <ProbBars probs={baseline.probabilities} names={names} />
        <div className="flex justify-between text-[12px] tnum" style={{ color: 'var(--text-muted)' }}>
          <span>Dataset accuracy <b style={{ color: 'var(--text-primary)' }}>{pct(baseline.dataset_accuracy)}</b></span>
          <span>loss <b className="font-mono" style={{ color: 'var(--text-primary)' }}>{baseline.dataset_loss.toFixed(4)}</b></span>
        </div>
        <p className="text-[11.5px] leading-relaxed" style={{ color: 'var(--text-faint)' }}>
          To intervene, select a neuron or connection and use its <b>What-if</b> controls (disable, cut, set weight or bias).
          The original and modified predictions then appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2.5" data-testid="whatif-result">
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] font-semibold" style={{ color: 'var(--tm-whatif-text)' }}>Before → after</span>
        <div className="ml-auto flex gap-1">
          <button type="button" className="btn-secondary !py-0.5 !px-2 !text-[11px]" onClick={() => void undo()}><Undo2 size={11} />Undo</button>
          <button type="button" className="btn-secondary !py-0.5 !px-2 !text-[11px]" onClick={() => void reset()}><RotateCcw size={11} />Reset</button>
        </div>
      </div>
      <CompareBars before={baseline.probabilities} after={intervened.probabilities} names={names} />
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px] tnum m-0">
        <dt style={{ color: 'var(--text-muted)' }}>Dataset accuracy</dt>
        <dd className="m-0 text-right font-mono">
          <span style={{ color: 'var(--text-primary)' }}>{pct(baseline.dataset_accuracy)}</span>
          <span style={{ color: 'var(--text-faint)' }}> → </span>
          <b style={{ color: accDrop ? 'var(--text-neg)' : 'var(--text-pos)' }}>{pct(intervened.dataset_accuracy)}</b>
        </dd>
        <dt style={{ color: 'var(--text-muted)' }}>Predictions changed</dt>
        <dd className="m-0 text-right font-mono" style={{ color: 'var(--text-primary)' }}>{pct(comparison.dataset_flip_fraction)}</dd>
      </dl>
      {comparison.prediction_changed && (
        <p className="text-[12px] font-medium" style={{ color: 'var(--text-warn)' }} role="status">This probe's predicted class flipped.</p>
      )}
      <div className="flex flex-wrap gap-1">
        {effective.map((iv, i) => (
          <button key={i} type="button" onClick={() => void select(refOf(iv))}
            className="text-[11px] font-mono px-1.5 py-0.5 rounded border hover:bg-[var(--bg-hover)]"
            style={{ borderColor: 'var(--tm-whatif)', color: 'var(--tm-whatif-text)' }}>
            {ivs.describe(iv, session.structure)}
          </button>
        ))}
      </div>
      {comparison.boundary && points && (
        <div className="grid grid-cols-2 gap-2">
          {(['baseline', 'intervened'] as const).map((k) => (
            <div key={k} className="min-w-0">
              <div className="text-[11px] mb-1" style={{ color: 'var(--text-faint)' }}>{k === 'baseline' ? 'Decision regions · before' : 'after'}</div>
              <HeatmapCanvas
                ariaLabel={`Decision regions ${k === 'baseline' ? 'before' : 'after'} the what-if edits`}
                values={comparison.boundary![k]}
                color={color}
                xRange={comparison.boundary!.x_range}
                yRange={comparison.boundary!.y_range}
                points={points}
                marker={comparison.probe.x.length === 2 ? [comparison.probe.x[0], comparison.probe.x[1]] : null}
                height={120}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
