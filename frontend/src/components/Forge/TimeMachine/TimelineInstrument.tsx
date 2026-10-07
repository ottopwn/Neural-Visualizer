// The Time Machine's scrubbing instrument.
//
// Curves are the real per-epoch training log; ticks are the stored
// checkpoints, the only epochs the playhead can land on.  Dragging snaps to
// the nearest checkpoint; in compare mode the A / B markers can be dragged.

import {
  ChevronFirst, ChevronLast, Pause, Play, Radio, StepBack, StepForward,
} from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { fmt, pct } from '../../../forge/format';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import {
  PLAYBACK_SPEEDS, checkpointIndex, makeAxis, nearestEpoch, resolveEpoch,
} from '../../../forge/timeline';
import type { HistoryRow, TrainingEvent } from '../../../forge/types';
import { useElementWidth } from '../hooks';
import { useCheckpointEpochs } from './hooks';

const PAD_L = 36;
const PAD_R = 40;
const PAD_COMPACT = 6;
const EVENT_LANE = 14;
const PLOT_H = 74;
const TICK_LANE = 14;
const GRAD_LANE = 30;
const AXIS_LANE = 14;

const EVENT_GLYPH: Record<TrainingEvent['kind'], string> = {
  init: '◆', run: '▸', acc_threshold: '★', best_accuracy: '▲', min_loss: '▼', largest_drop: '⬇',
};

function path(points: [number, number][]): string {
  return points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
}

type Drag = 'playhead' | 'A' | 'B' | null;

export function TimelineInstrument({ lab, compact = false }: { lab: boolean; compact?: boolean }) {
  const session = useForgeStore((s) => s.session);
  const epochs = useCheckpointEpochs();
  const cursor = useTimeMachine((s) => s.cursor);
  const axisMode = useTimeMachine((s) => s.axisMode);
  const compareMode = useTimeMachine((s) => s.compareMode);
  const compareA = useTimeMachine((s) => s.compareA);
  const compareB = useTimeMachine((s) => s.compareB);
  const timeline = useTimeMachine((s) => s.timeline);
  const goTo = useTimeMachine((s) => s.goTo);
  const setCompareEpochs = useTimeMachine((s) => s.setCompareEpochs);
  const [boxRef, width] = useElementWidth<HTMLDivElement>(800);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const drag = useRef<Drag>(null);

  const history: HistoryRow[] = useMemo(() => session?.history ?? [], [session]);
  const live = session?.epoch ?? 0;
  const current = resolveEpoch(cursor, live);
  const showGrad = lab && !compact && history.some((r) => (r.grad_norm ?? 0) > 0);

  const plotTop = compact ? 2 : EVENT_LANE;
  const plotH = compact ? 34 : PLOT_H;
  const tickY = plotTop + plotH;
  const gradTop = tickY + TICK_LANE;
  const axisTop = showGrad ? gradTop + GRAD_LANE : tickY + TICK_LANE;
  const height = compact ? tickY + TICK_LANE : axisTop + AXIS_LANE;

  const padL = compact ? PAD_COMPACT : PAD_L;
  const padR = compact ? PAD_COMPACT : PAD_R;
  const axis = useMemo(() => makeAxis(epochs, axisMode, padL, width - padR), [epochs, axisMode, width, padL, padR]);

  const curves = useMemo(() => {
    if (history.length < 1) return null;
    const maxLoss = Math.max(...history.map((r) => r.loss), 1e-9);
    const yLoss = (v: number) => plotTop + plotH - (v / maxLoss) * (plotH - 4);
    const yAcc = (v: number) => plotTop + plotH - v * (plotH - 4);
    const lossPts = history.map((r) => [axis.toX(r.epoch), yLoss(r.loss)] as [number, number]);
    const accPts = history.map((r) => [axis.toX(r.epoch), yAcc(r.accuracy)] as [number, number]);
    const lossArea = `${path(lossPts)}L${lossPts[lossPts.length - 1][0].toFixed(1)},${plotTop + plotH}L${lossPts[0][0].toFixed(1)},${plotTop + plotH}Z`;
    let grad: string | null = null;
    const g = history.filter((r) => (r.grad_norm ?? 0) > 0);
    if (g.length > 1) {
      const lo = Math.log10(Math.min(...g.map((r) => r.grad_norm!)));
      const hi = Math.log10(Math.max(...g.map((r) => r.grad_norm!)));
      const yG = (v: number) => gradTop + GRAD_LANE - 4 - ((Math.log10(v) - lo) / (hi - lo || 1)) * (GRAD_LANE - 8);
      grad = path(g.map((r) => [axis.toX(r.epoch), yG(r.grad_norm!)]));
    }
    return { maxLoss, lossPath: path(lossPts), lossArea, accPath: path(accPts), grad };
  }, [history, axis, plotTop, plotH, gradTop]);

  if (!session || epochs.length === 0) return null;

  const epochAt = (clientX: number, el: SVGSVGElement) => {
    const rect = el.getBoundingClientRect();
    return nearestEpoch(epochs, axis.fromX(clientX - rect.left));
  };

  const xA = compareA !== null ? axis.toX(compareA) : null;
  const xB = compareMode ? axis.toX(compareB ?? live) : null;

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    drag.current = 'playhead';
    if (compareMode && xA !== null && xB !== null) {
      if (Math.abs(x - xA) <= 8) drag.current = 'A';
      else if (Math.abs(x - xB) <= 8) drag.current = 'B';
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    applyDrag(e);
  };

  const applyDrag = (e: React.PointerEvent<SVGSVGElement>) => {
    const ep = epochAt(e.clientX, e.currentTarget);
    if (ep === null) return;
    if (drag.current === 'playhead' && ep !== current) void goTo(ep);
    if (drag.current === 'A' && ep !== compareA) void setCompareEpochs(ep, compareB);
    if (drag.current === 'B' && ep !== (compareB ?? live)) void setCompareEpochs(compareA ?? epochs[0], ep);
  };

  const hoverRow = hoverEpoch !== null ? history.find((r) => r.epoch === hoverEpoch) : null;
  const historical = cursor !== null;
  const headColor = historical ? 'var(--tm-hist)' : 'var(--tm-live)';
  const xHead = axis.toX(current);
  const events = compact ? [] : timeline?.events ?? [];
  const runs = compact ? [] : (timeline?.runs ?? []).slice(1);
  const axisTicks = [epochs[0], ...epochs.filter((_, i) => i > 0 && i < epochs.length - 1 && i % Math.max(1, Math.ceil(epochs.length / Math.max(2, width / 70))) === 0), epochs[epochs.length - 1]];

  return (
    <div ref={boxRef} className="relative w-full select-none">
      <svg
        width={width}
        height={height}
        role="slider"
        aria-label="Training timeline: drag to travel between stored checkpoints"
        aria-valuemin={epochs[0]}
        aria-valuemax={live}
        aria-valuenow={current}
        aria-valuetext={`epoch ${current}${historical ? ' (historical checkpoint)' : ' (live)'}`}
        tabIndex={-1}
        className="block touch-none"
        style={{ cursor: 'ew-resize' }}
        onPointerDown={onPointerDown}
        onPointerMove={(e) => {
          const ep = epochAt(e.clientX, e.currentTarget);
          setHoverEpoch(ep);
          if (drag.current && e.buttons) applyDrag(e);
        }}
        onPointerUp={() => { drag.current = null; }}
        onPointerLeave={() => setHoverEpoch(null)}
      >
        {/* plot frame */}
        <rect x={padL} y={plotTop} width={width - padL - padR} height={plotH} rx={4}
          fill="var(--tm-panel)" stroke="var(--border)" />
        {[0.5].map((f) => (
          <line key={f} x1={padL} x2={width - padR} y1={plotTop + plotH * f} y2={plotTop + plotH * f}
            stroke="var(--border)" strokeDasharray="2,4" />
        ))}

        {/* compare span */}
        {compareMode && xA !== null && xB !== null && (
          <rect x={Math.min(xA, xB)} y={plotTop} width={Math.abs(xB - xA)} height={plotH}
            fill="color-mix(in srgb, var(--tm-b) 8%, transparent)" />
        )}

        {/* training runs */}
        {runs.map((r) => (
          <g key={r.start_epoch}>
            <line x1={axis.toX(r.start_epoch)} x2={axis.toX(r.start_epoch)} y1={plotTop} y2={plotTop + plotH}
              stroke="var(--border-soft)" strokeDasharray="3,3" />
          </g>
        ))}

        {/* curves: the real training log */}
        {curves && (
          <>
            <path d={curves.lossArea} fill="color-mix(in srgb, var(--tm-loss) 14%, transparent)" />
            <path d={curves.lossPath} fill="none" stroke="var(--tm-loss)" strokeWidth={1.5} />
            <path d={curves.accPath} fill="none" stroke="var(--tm-acc)" strokeWidth={1.75} />
            {!compact && (
              <>
                <text x={padL - 4} y={plotTop + 8} fontSize={9} textAnchor="end" fill="var(--tm-loss)">{fmt(curves.maxLoss, 2)}</text>
                <text x={padL - 4} y={plotTop + plotH} fontSize={9} textAnchor="end" fill="var(--tm-loss)">0</text>
                <text x={width - padR + 4} y={plotTop + 8} fontSize={9} fill="var(--tm-acc)">100%</text>
                <text x={width - padR + 4} y={plotTop + plotH} fontSize={9} fill="var(--tm-acc)">0%</text>
              </>
            )}
          </>
        )}

        {/* gradient lane (Lab) */}
        {showGrad && curves?.grad && (
          <g>
            <rect x={padL} y={gradTop} width={width - padL - padR} height={GRAD_LANE - 2} rx={3}
              fill="var(--tm-panel)" stroke="var(--border)" />
            <path d={curves.grad} fill="none" stroke="var(--tm-grad)" strokeWidth={1.25} />
            <text x={padL - 4} y={gradTop + GRAD_LANE / 2 + 3} fontSize={9} textAnchor="end" fill="var(--tm-grad)">‖∇‖</text>
          </g>
        )}

        {/* checkpoint ticks: the only epochs you can visit */}
        {epochs.map((e) => {
          const x = axis.toX(e);
          const on = e === current;
          return (
            <line key={e} x1={x} x2={x} y1={tickY + 2} y2={tickY + (on ? 13 : 9)}
              stroke={on ? headColor : 'var(--text-muted)'} strokeWidth={on ? 2.5 : 1.25} strokeLinecap="round"
              opacity={on ? 1 : 0.7} />
          );
        })}

        {/* events */}
        {events.map((ev, i) => {
          const x = axis.toX(ev.epoch);
          return (
            <g key={`${ev.kind}-${i}`} style={{ cursor: 'pointer' }}
              onPointerDown={(e) => { e.stopPropagation(); void goTo(ev.checkpoint_epoch); }}>
              <title>{`${ev.label} · epoch ${ev.epoch}${ev.checkpoint_epoch !== ev.epoch ? ` (nearest stored checkpoint: ${ev.checkpoint_epoch})` : ''}`}</title>
              <text x={x} y={10} fontSize={10} textAnchor="middle"
                fill={ev.kind === 'run' ? 'var(--text-muted)' : ev.kind === 'best_accuracy' || ev.kind === 'acc_threshold' ? 'var(--tm-acc)' : ev.kind === 'init' ? 'var(--text-faint)' : 'var(--tm-loss)'}>
                {EVENT_GLYPH[ev.kind]}
              </text>
            </g>
          );
        })}

        {/* compare markers */}
        {compareMode && xA !== null && xB !== null && (['A', 'B'] as const).map((m) => {
          const x = m === 'A' ? xA : xB;
          const color = m === 'A' ? 'var(--tm-a)' : 'var(--tm-b)';
          return (
            <g key={m} style={{ cursor: 'grab' }}>
              <line x1={x} x2={x} y1={plotTop} y2={tickY + 12} stroke={color} strokeWidth={2} strokeDasharray="4,3" />
              <rect x={x - 7} y={plotTop - (compact ? 0 : 12)} width={14} height={12} rx={3} fill={color} />
              <text x={x} y={plotTop - (compact ? -9 : 3)} fontSize={9} fontWeight={700} textAnchor="middle" fill="#0b1020">{m}</text>
            </g>
          );
        })}

        {/* hover */}
        {hoverEpoch !== null && hoverEpoch !== current && (
          <line x1={axis.toX(hoverEpoch)} x2={axis.toX(hoverEpoch)} y1={plotTop} y2={tickY + 12}
            stroke="var(--text-muted)" strokeWidth={1} opacity={0.5} />
        )}

        {/* playhead */}
        <g pointerEvents="none">
          <line x1={xHead} x2={xHead} y1={plotTop} y2={tickY + 13} stroke={headColor} strokeWidth={2} />
          <circle cx={xHead} cy={plotTop} r={4} fill={headColor} />
          {history.find((r) => r.epoch === current) && curves && (
            <>
              <circle cx={xHead} cy={plotTop + plotH - (history.find((r) => r.epoch === current)!.loss / curves.maxLoss) * (plotH - 4)} r={3} fill="var(--tm-loss)" stroke="var(--bg-card)" />
              <circle cx={xHead} cy={plotTop + plotH - history.find((r) => r.epoch === current)!.accuracy * (plotH - 4)} r={3} fill="var(--tm-acc)" stroke="var(--bg-card)" />
            </>
          )}
        </g>

        {/* epoch axis */}
        {!compact && axisTicks.filter((v, i, a) => a.indexOf(v) === i).map((e) => (
          <text key={e} x={axis.toX(e)} y={axisTop + 10} fontSize={9} textAnchor="middle" fill="var(--text-faint)">{e}</text>
        ))}
      </svg>

      {/* hover readout */}
      {hoverRow && !compact && (
        <div className="absolute pointer-events-none text-[10px] px-2 py-1 rounded-md border font-mono whitespace-nowrap"
          style={{
            left: Math.min(width - 190, Math.max(0, axis.toX(hoverRow.epoch) - 80)), top: plotTop + plotH + TICK_LANE + 2,
            background: 'var(--bg-card)', borderColor: 'var(--border-soft)', color: 'var(--text-primary)', zIndex: 5,
          }}>
          epoch {hoverRow.epoch}{checkpointIndex(epochs, hoverRow.epoch) < 0 ? ' · log only' : ' · stored'}
          {' · '}<span style={{ color: 'var(--tm-loss)' }}>loss {fmt(hoverRow.loss, 4)}</span>
          {' · '}<span style={{ color: 'var(--tm-acc)' }}>acc {pct(hoverRow.accuracy)}</span>
          {lab && hoverRow.grad_norm ? <> · <span style={{ color: 'var(--tm-grad)' }}>‖∇‖ {fmt(hoverRow.grad_norm, 3)}</span></> : null}
        </div>
      )}
    </div>
  );
}

const btn = 'p-1.5 rounded-md border transition-colors disabled:opacity-35 disabled:cursor-not-allowed hover:bg-white/5';

/** Transport: first / back / play-pause / forward / live, speed and axis mode. */
export function Transport({ compact = false }: { compact?: boolean }) {
  const session = useForgeStore((s) => s.session);
  const epochs = useCheckpointEpochs();
  const cursor = useTimeMachine((s) => s.cursor);
  const playing = useTimeMachine((s) => s.playing);
  const speed = useTimeMachine((s) => s.speed);
  const axisMode = useTimeMachine((s) => s.axisMode);
  const frameLoading = useTimeMachine((s) => s.frameLoading);
  const tm = useTimeMachine.getState;
  if (!session) return null;
  const live = session.epoch;
  const current = resolveEpoch(cursor, live);
  const idx = checkpointIndex(epochs, current);
  const atStart = current <= epochs[0];
  const atEnd = cursor === null;
  const border = { borderColor: 'var(--border-soft)', color: 'var(--text-primary)' };

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <button type="button" className={btn} style={border} aria-label="First checkpoint" title="First checkpoint (Home)"
        disabled={atStart} onClick={() => void tm().first()}><ChevronFirst size={14} /></button>
      <button type="button" className={btn} style={border} aria-label="Previous checkpoint" title="Previous checkpoint (←)"
        disabled={atStart} onClick={() => void tm().step(-1)}><StepBack size={14} /></button>
      <button type="button" aria-label={playing ? 'Pause' : 'Play training history'} title="Play / pause (Space)"
        disabled={epochs.length < 2} onClick={() => tm().togglePlay()}
        className="px-2.5 py-1.5 rounded-md border flex items-center gap-1 text-xs font-semibold disabled:opacity-35"
        style={{ borderColor: 'var(--accent)', background: playing ? 'var(--accent)' : 'transparent', color: playing ? '#fff' : 'var(--text-primary)' }}>
        {playing ? <Pause size={14} /> : <Play size={14} />}{!compact && (playing ? 'Pause' : 'Play')}
      </button>
      <button type="button" className={btn} style={border} aria-label="Next checkpoint" title="Next checkpoint (→)"
        disabled={atEnd} onClick={() => void tm().step(1)}><StepForward size={14} /></button>
      <button type="button" className={btn} style={border} aria-label="Latest checkpoint" title="Latest checkpoint (End)"
        disabled={atEnd} onClick={() => void tm().last()}><ChevronLast size={14} /></button>

      {!compact && (
        <div className="flex items-center ml-1 rounded-md border p-0.5" style={{ borderColor: 'var(--border)' }} role="group" aria-label="Playback speed">
          {PLAYBACK_SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s} onClick={() => tm().setSpeed(s)}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{ background: speed === s ? 'var(--border-soft)' : 'transparent', color: speed === s ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              {s}×
            </button>
          ))}
        </div>
      )}

      <div className="flex items-baseline gap-1.5 ml-2 font-mono" aria-live="polite">
        <span className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-faint)' }}>epoch</span>
        <span className={compact ? 'text-sm font-bold' : 'text-xl font-bold leading-none'}
          style={{ color: cursor === null ? 'var(--tm-live-text)' : 'var(--tm-hist-text)' }}>{current}</span>
        <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>/ {live}</span>
        {!compact && idx >= 0 && (
          <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>· checkpoint {idx + 1} of {epochs.length}</span>
        )}
        {frameLoading && <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>· loading</span>}
      </div>

      <button type="button" onClick={() => void tm().goLive()} disabled={cursor === null}
        aria-label="Return to the live model"
        title={cursor === null ? 'Showing the live (latest) model' : 'Return to the live (latest) model'}
        className="ml-auto flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-semibold"
        style={cursor === null
          ? { borderColor: 'var(--tm-live)', color: 'var(--tm-live-text)', background: 'color-mix(in srgb, var(--tm-live) 12%, transparent)', cursor: 'default' }
          : { borderColor: 'var(--tm-live)', color: '#fff', background: 'var(--tm-live)' }}>
        <Radio size={12} />{cursor === null ? 'LIVE' : 'Back to live'}
      </button>

      {!compact && (
        <div className="flex items-center rounded-md border p-0.5" style={{ borderColor: 'var(--border)' }} role="group" aria-label="Time axis">
          {(['linear', 'checkpoints'] as const).map((m) => (
            <button key={m} type="button" aria-pressed={axisMode === m} onClick={() => tm().setAxisMode(m)}
              title={m === 'linear' ? 'x proportional to epochs' : 'stored checkpoints evenly spaced'}
              className="px-1.5 py-0.5 rounded text-[10px]"
              style={{ background: axisMode === m ? 'var(--border-soft)' : 'transparent', color: axisMode === m ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              {m === 'linear' ? 'linear' : 'even'}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
