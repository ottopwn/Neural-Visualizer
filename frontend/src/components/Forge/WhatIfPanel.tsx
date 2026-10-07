import { RotateCcw, Undo2, FlaskConical } from 'lucide-react';
import { useMemo } from 'react';
import { classColor, pct } from '../../forge/format';
import * as ivs from '../../forge/interventions';
import { useForgeStore } from '../../forge/store';
import type { ComponentRef, Intervention } from '../../forge/types';
import { CompareBars, HeatmapCanvas, ProbBars } from './charts';
import { useDatasetPoints } from './hooks';

function refOf(iv: Intervention): ComponentRef {
  return iv.type === 'set_weight'
    ? { kind: 'connection', layer: iv.layer, source: iv.source, target: iv.target }
    : { kind: 'neuron', layer: iv.layer, index: iv.index };
}

export function WhatIfPanel() {
  const session = useForgeStore((s) => s.session);
  const comparison = useForgeStore((s) => s.comparison);
  const interventions = useForgeStore((s) => s.interventions);
  const undo = useForgeStore((s) => s.undo);
  const reset = useForgeStore((s) => s.reset);
  const select = useForgeStore((s) => s.select);
  const points = useDatasetPoints();
  const color = useMemo(() => (v: number) => classColor(v), []);

  if (!session || !comparison) return null;
  const names = session.structure.class_names;
  const { baseline, intervened } = comparison;
  const active = interventions.length > 0;
  const effective = ivs.effective(interventions);

  return (
    <div className="h-full flex gap-3 min-w-0">
      <div className="flex-1 min-w-0 flex flex-col gap-1.5">
        <div className="flex items-center gap-1.5">
          <FlaskConical size={12} style={{ color: '#f59e0b' }} />
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {active ? 'What-if: before vs after' : 'Prediction'}
          </span>
          {active && (
            <div className="ml-auto flex gap-1">
              <button type="button" className="btn-secondary !py-0.5 !px-2 text-[11px] flex items-center gap-1" onClick={undo}><Undo2 size={11} />Undo</button>
              <button type="button" className="btn-secondary !py-0.5 !px-2 text-[11px] flex items-center gap-1" onClick={reset}><RotateCcw size={11} />Reset</button>
            </div>
          )}
        </div>

        {active ? (
          <>
            <CompareBars before={baseline.probabilities} after={intervened.probabilities} names={names} />
            <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              <span>Dataset accuracy <b style={{ color: 'var(--text-primary)' }}>{pct(baseline.dataset_accuracy)}</b> → <b style={{ color: intervened.dataset_accuracy < baseline.dataset_accuracy ? '#fca5a5' : '#6ee7b7' }}>{pct(intervened.dataset_accuracy)}</b></span>
              <span>Predictions changed on <b style={{ color: 'var(--text-primary)' }}>{pct(comparison.dataset_flip_fraction)}</b> of samples</span>
              {comparison.prediction_changed && <span style={{ color: '#f59e0b' }}>⚠ this probe's predicted class flipped</span>}
            </div>
            <div className="flex flex-wrap gap-1 overflow-y-auto">
              {effective.map((iv, i) => (
                <button key={i} type="button" onClick={() => select(refOf(iv))}
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded border hover:bg-white/5"
                  style={{ borderColor: 'rgba(245,158,11,0.5)', color: '#fcd34d' }}>
                  {ivs.describe(iv, session.structure)}
                </button>
              ))}
            </div>
          </>
        ) : (
          <>
            <ProbBars probs={baseline.probabilities} names={names} />
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              Dataset accuracy <b style={{ color: 'var(--text-primary)' }}>{pct(baseline.dataset_accuracy)}</b> · loss {baseline.dataset_loss.toFixed(4)}
            </div>
            <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-faint)' }}>
              Select a neuron or connection and use <b>What-if</b> to disable it or change its weight/bias.
              The original and modified predictions will appear here side by side.
            </p>
          </>
        )}
      </div>

      {active && comparison.boundary && points && (
        <div className="flex gap-2 flex-shrink-0" style={{ width: 300 }}>
          {(['baseline', 'intervened'] as const).map((k) => (
            <div key={k} className="flex-1 min-w-0">
              <div className="text-[10px] mb-1" style={{ color: 'var(--text-faint)' }}>{k === 'baseline' ? 'Decision regions · before' : 'after'}</div>
              <HeatmapCanvas
                ariaLabel={`Decision regions ${k}`}
                values={comparison.boundary![k]}
                color={color}
                xRange={comparison.boundary!.x_range}
                yRange={comparison.boundary!.y_range}
                points={points}
                marker={comparison.probe.x.length === 2 ? [comparison.probe.x[0], comparison.probe.x[1]] : null}
                height={128}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
