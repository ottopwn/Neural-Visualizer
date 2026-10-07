import { ChevronLeft, ChevronRight, Crosshair } from 'lucide-react';
import { useMemo } from 'react';
import { classColor } from '../../forge/format';
import { nearestSample } from '../../forge/selection';
import { useForgeStore } from '../../forge/store';
import { HeatmapCanvas } from './charts';
import { useDatasetPoints } from './hooks';
import { useNeutral } from './TimeMachine/hooks';

const PICK_RADIUS_FRACTION = 0.035; // of the plot width, in data units

export function ProbePicker() {
  const session = useForgeStore((s) => s.session);
  const probe = useForgeStore((s) => s.probe);
  const comparison = useForgeStore((s) => s.comparison);
  const setProbe = useForgeStore((s) => s.setProbe);
  const points = useDatasetPoints();
  const boundary = comparison?.boundary ?? null;
  const resolved = comparison?.probe;
  const neutral = useNeutral();
  const color = useMemo(() => (v: number) => classColor(v, neutral), [neutral]);

  if (!session) return null;
  const n = session.dataset_X.length;
  const names = session.structure.class_names;
  const current = probe.sample_index ?? null;

  const step = (d: number) => setProbe({ sample_index: (((current ?? 0) + d) % n + n) % n });

  return (
    <div className="h-full flex flex-col gap-1.5 min-w-0">
      <div className="flex items-center gap-1.5">
        <Crosshair size={13} style={{ color: 'var(--select)' }} aria-hidden="true" />
        <span className="text-[12px]" style={{ color: 'var(--text-muted)' }}>Click a point or anywhere on the map</span>
        <div className="ml-auto flex items-center">
          <button type="button" aria-label="Previous sample" className="btn-ghost p-1" onClick={() => step(-1)}><ChevronLeft size={15} /></button>
          <button type="button" aria-label="Next sample" className="btn-ghost p-1" onClick={() => step(1)}><ChevronRight size={15} /></button>
        </div>
      </div>
      {boundary && points ? (
        <HeatmapCanvas
          ariaLabel="Dataset and decision regions; click to choose the probe input"
          values={boundary.intervened}
          color={color}
          xRange={boundary.x_range}
          yRange={boundary.y_range}
          points={points}
          marker={resolved && resolved.x.length === 2 ? [resolved.x[0], resolved.x[1]] : null}
          height={150}
          onPick={(x, y) => {
            const r = (boundary.x_range[1] - boundary.x_range[0]) * PICK_RADIUS_FRACTION;
            const idx = nearestSample(session.dataset_X, x, y, r);
            setProbe(idx >= 0 ? { sample_index: idx } : { x: [x, y] });
          }}
        />
      ) : (
        <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
          {session.structure.input_dim}-D input — step through samples with the arrows.
        </p>
      )}
      <div className="text-[12px] leading-snug tnum" style={{ color: 'var(--text-muted)' }}>
        {resolved ? (
          resolved.sample_index !== null
            ? <>Sample #{resolved.sample_index} · label <b style={{ color: resolved.label === 1 ? '#93c5fd' : 'var(--text-neg)' }}>{names[resolved.label ?? 0]}</b></>
            : <>Free point ({resolved.x.map((v) => v.toFixed(2)).join(', ')})</>
        ) : '…'}
      </div>
      {resolved && resolved.sample_index === null && (
        <div className="flex items-center gap-1 text-[10px]">
          <span style={{ color: 'var(--text-faint)' }}>gradient target</span>
          {[null, 0, 1].map((t) => (
            <button key={String(t)} type="button"
              onClick={() => setProbe({ ...probe, target: t })}
              className="px-1.5 py-0.5 rounded border"
              style={{
                borderColor: (probe.target ?? null) === t ? 'var(--accent)' : 'var(--border)',
                color: (probe.target ?? null) === t ? 'var(--text-primary)' : 'var(--text-faint)',
              }}>
              {t === null ? 'predicted' : names[t]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
