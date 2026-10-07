import { Loader2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { describeFrame } from '../../../forge/explainTime';
import { classColor, fmt, fmtSigned, pct } from '../../../forge/format';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import type { Frame } from '../../../forge/types';
import { HeatmapCanvas, ProbBars, Section, type ScatterPoint } from '../charts';
import { useFrame, useNeutral, useTimelineData } from './hooks';

function Metric({ label, value, delta, good, color }: {
  label: string; value: string; delta?: string | null; good?: boolean | null; color?: string;
}) {
  return (
    <div className="rounded-lg border px-2.5 py-1.5 min-w-0" style={{ borderColor: 'var(--border)', background: 'var(--tm-panel)' }}>
      <div className="text-[9px] uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>{label}</div>
      <div className="font-mono text-[15px] font-semibold leading-tight" style={{ color: color ?? 'var(--text-primary)' }}>{value}</div>
      {delta && (
        <div className="font-mono text-[10px]" style={{ color: good === null || good === undefined ? 'var(--text-muted)' : good ? 'var(--tm-pos)' : 'var(--tm-neg)' }}>
          {delta}
        </div>
      )}
    </div>
  );
}

/** Points coloured by true class; mistakes of this checkpoint get a ring. */
function usePoints(frame: Frame | null, markMistakes: boolean, ringColor: string): ScatterPoint[] | undefined {
  const session = useForgeStore((s) => s.session);
  return useMemo(() => {
    if (!session || session.structure.input_dim !== 2) return undefined;
    return session.dataset_X.map((p, i) => ({
      x: p[0], y: p[1], cls: session.dataset_y[i],
      ring: markMistakes && frame && frame.predictions[i] !== session.dataset_y[i] ? ringColor : undefined,
    }));
  }, [session, frame, markMistakes, ringColor]);
}

/**
 * "Watch the network learn": the real decision regions of the checkpoint
 * under the playhead, with the dataset and the network's mistakes.
 */
export function BoundaryStage() {
  const session = useForgeStore((s) => s.session);
  const mode = useForgeStore((s) => s.mode);
  const frame = useFrame();
  const timeline = useTimelineData();
  const loading = useTimeMachine((s) => s.frameLoading);
  const cursor = useTimeMachine((s) => s.cursor);
  const [markMistakes, setMarkMistakes] = useState(true);
  const neutral = useNeutral();
  const paper = neutral[0] > 128;
  const color = useMemo(() => (v: number) => classColor(v, neutral), [neutral]);
  const points = usePoints(frame, markMistakes, paper ? '#111827' : '#fde047');
  if (!session) return null;

  const names = session.structure.class_names;
  const prev = frame?.previous ?? null;
  const wrong = frame ? frame.predictions.filter((p, i) => p !== session.dataset_y[i]).length : 0;
  const majority = timeline?.majority_rate ?? null;
  const expectedEpoch = cursor ?? session.epoch;
  const stale = frame !== null && frame.epoch !== expectedEpoch;

  return (
    <div className="h-full min-h-0 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-[11px]">
        <span className="font-semibold uppercase tracking-wider text-[10px]" style={{ color: 'var(--text-muted)' }}>Decision regions</span>
        {frame && <span className="font-mono" style={{ color: cursor === null ? 'var(--tm-live-text)' : 'var(--tm-hist-text)' }}>epoch {frame.epoch}</span>}
        {(loading || stale) && <Loader2 size={12} className="animate-spin" style={{ color: 'var(--text-faint)' }} />}
        <label className="ml-auto flex items-center gap-1 cursor-pointer" style={{ color: 'var(--text-muted)' }}>
          <input type="checkbox" checked={markMistakes} onChange={(e) => setMarkMistakes(e.target.checked)} />
          ring mistakes
        </label>
      </div>

      <div className="relative flex-shrink-0">
        {frame?.boundary ? (
          <HeatmapCanvas
            ariaLabel={`Decision regions of the network at epoch ${frame.epoch}`}
            values={frame.boundary.values}
            color={color}
            xRange={frame.boundary.x_range}
            yRange={frame.boundary.y_range}
            points={points}
            marker={frame.probe.x.length === 2 ? [frame.probe.x[0], frame.probe.x[1]] : null}
            contour={0.5}
            contourColor={paper ? 'rgba(17,24,39,0.85)' : 'rgba(255,255,255,0.9)'}
            pointStroke={paper ? 'rgba(17,24,39,0.5)' : 'rgba(255,255,255,0.55)'}
            height={300}
          />
        ) : frame ? (
          <div className="rounded-md border p-4 text-[11px]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
            The decision boundary can only be drawn for 2-D inputs. This model has {session.structure.input_dim} input features;
            the metrics below are still computed from the stored checkpoint.
          </div>
        ) : (
          <div className="h-[300px] rounded-md border flex items-center justify-center" style={{ borderColor: 'var(--border)' }}>
            <Loader2 size={16} className="animate-spin" style={{ color: 'var(--text-faint)' }} />
          </div>
        )}
        {frame?.boundary && (
          <div className="absolute bottom-1.5 left-1.5 text-[9px] px-1.5 py-0.5 rounded flex gap-2"
            style={{ background: 'var(--tm-overlay)', color: 'var(--text-muted)' }}>
            <span style={{ color: '#f87171' }}>■ {names[0]}</span>
            <span style={{ color: '#60a5fa' }}>■ {names[1]}</span>
            <span>— p = 0.5</span>
            {markMistakes && <span>○ misclassified</span>}
          </div>
        )}
      </div>

      {frame && (
        <div className="grid grid-cols-3 gap-1.5">
          <Metric label="loss" value={fmt(frame.loss, 4)} color="var(--tm-loss)"
            delta={prev ? `${fmtSigned(frame.loss - prev.loss, 4)} vs ep ${prev.epoch}` : null}
            good={prev ? frame.loss < prev.loss : null} />
          <Metric label="accuracy" value={pct(frame.accuracy)} color="var(--tm-acc)"
            delta={prev ? `${fmtSigned((frame.accuracy - prev.accuracy) * 100, 1)} pp` : majority !== null ? `majority class ${pct(majority)}` : null}
            good={prev ? (frame.accuracy === prev.accuracy ? null : frame.accuracy > prev.accuracy) : null} />
          <Metric label="mistakes" value={`${wrong} / ${frame.predictions.length}`}
            delta={prev ? `${prev.changed} changed · +${prev.fixed} −${prev.broken}` : null}
            good={prev ? (prev.fixed === prev.broken ? null : prev.fixed > prev.broken) : null} />
        </div>
      )}

      {frame && mode === 'learn' && (
        <Section title="What the network is doing">
          <ul className="space-y-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            {describeFrame(frame, names, majority ?? 0.5).map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Section>
      )}
      {frame && mode === 'lab' && (
        <Section title="Probe prediction" hint={`epoch ${frame.epoch}`}>
          <ProbBars probs={frame.probe_probabilities} names={names} />
          {prev?.boundary_flip_fraction !== null && prev?.boundary_flip_fraction !== undefined && (
            <p className="text-[10px] mt-1.5" style={{ color: 'var(--text-faint)' }}>
              {pct(prev.boundary_flip_fraction)} of the plotted plane changed class since epoch {prev.epoch}.
            </p>
          )}
        </Section>
      )}
    </div>
  );
}
