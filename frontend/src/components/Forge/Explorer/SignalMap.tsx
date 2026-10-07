import { useMemo } from 'react';
import { diverging, maxAbs, rgb } from '../../../forge/format';
import { columnValues, reached, type PassStep } from '../../../forge/passExplorer';
import type { ComputationTrace } from '../../../forge/types';
import { useElementWidth } from '../hooks';
import { useNeutral } from '../TimeMachine/hooks';

/** How many of the strongest connections of the active layer are drawn as signal lines. */
export const SIGNAL_LINES = 24;

interface Props {
  trace: ComputationTrace;
  step: PassStep;
  focus: number;
  onPick: (layer: number, index: number) => void;
  height: number;
}

/**
 * Every graph layer as a column of cells coloured by the real value of the
 * current step (forward: outputs / z; backward: gradients).  The connections
 * that carry the largest real |w·a| (forward) or |w·δ| (backward) into the
 * active layer are drawn as moving lines; layers the pass has not reached
 * yet are dimmed.
 */
export function SignalMap({ trace, step, focus, onPick, height }: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>(600);
  const neutral = useNeutral();
  const cols = useMemo(() => columnValues(trace, step), [trace, step]);
  const seen = reached(step, trace.layers.length);
  const names = [trace.feature_names, ...trace.layers.map((l) => l.neuron_names)];
  const labels = ['Input', ...trace.layers.map((l) => l.label)];

  const n = cols.length;
  const padX = 56;
  const top = 34;
  const bottom = 26;
  const x = (g: number) => padX + (g * (width - 2 * padX)) / Math.max(1, n - 1);
  const maxCells = Math.max(...cols.map((c) => c.length));
  const cell = Math.max(3, Math.min(30, (height - top - bottom) / maxCells - 2));
  const gap = cell > 8 ? 2 : 1;
  const y = (g: number, i: number) => {
    const total = cols[g].length * (cell + gap) - gap;
    const y0 = top + (height - top - bottom - total) / 2;
    return y0 + i * (cell + gap);
  };
  const cellW = Math.min(56, Math.max(14, cell * 2.1));

  // Strongest real connections into the active layer.
  const lines = useMemo(() => {
    const g = step.edgeLayer;
    if (g === null) return [];
    const d = trace.layers[g - 1];
    const out: { s: number; t: number; v: number }[] = [];
    d.weight.forEach((row, i) => row.forEach((w, j) => {
      const v = step.dir === 'forward' ? w * d.input[j] : w * d.grad_z[i];
      out.push({ s: j, t: i, v });
    }));
    out.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    return out.slice(0, SIGNAL_LINES);
  }, [trace, step]);
  const lineMax = maxAbs(lines.map((l) => l.v)) || 1;

  return (
    <div ref={ref} className="w-full" style={{ height }}>
      <svg width={width} height={height} role="img"
        aria-label={`Signal map for step "${step.title}": each column is a layer, each cell a neuron coloured by its real value`}>
        <defs>
          <style>{`
            @keyframes nf-flow-f { to { stroke-dashoffset: -24; } }
            @keyframes nf-flow-b { to { stroke-dashoffset: 24; } }
            .nf-flow-f { stroke-dasharray: 6 6; animation: nf-flow-f 0.9s linear infinite; }
            .nf-flow-b { stroke-dasharray: 6 6; animation: nf-flow-b 0.9s linear infinite; }
          `}</style>
        </defs>

        {/* faint full bundles between consecutive layers */}
        {cols.slice(1).map((_, k) => (
          <rect key={k} x={x(k) + cellW / 2} width={Math.max(0, x(k + 1) - x(k) - cellW)} y={top} height={height - top - bottom}
            fill="var(--border)" opacity={step.edgeLayer === k + 1 ? 0.5 : 0.18} rx={4} />
        ))}

        {/* strongest connections of the active layer */}
        {step.edgeLayer !== null && lines.map((l, i) => {
          const g = step.edgeLayer!;
          const t = Math.abs(l.v) / lineMax;
          return (
            <line key={i}
              x1={x(g - 1) + cellW / 2} y1={y(g - 1, l.s) + cell / 2} x2={x(g) - cellW / 2} y2={y(g, l.t) + cell / 2}
              stroke={l.v >= 0 ? 'var(--pos)' : 'var(--neg)'} strokeWidth={0.6 + 2.6 * t} strokeOpacity={0.25 + 0.7 * t}
              className={step.dir === 'forward' ? 'nf-flow-f' : 'nf-flow-b'}>
              <title>{`${names[g - 1][l.s]} → ${names[g][l.t]}: ${step.dir === 'forward' ? 'w·a' : 'w·δ'} = ${l.v.toFixed(4)}`}</title>
            </line>
          );
        })}

        {cols.map((col, g) => {
          const m = maxAbs(col) || 1;
          const active = g === step.layer;
          return (
            <g key={g} opacity={seen[g] ? 1 : 0.28}>
              <text x={x(g)} y={16} textAnchor="middle" fontSize={11.5} fontWeight={600}
                style={{ fill: active ? 'var(--text-primary)' : 'var(--text-muted)' }}>{labels[g]}</text>
              <text x={x(g)} y={height - 8} textAnchor="middle" fontSize={10} className="tnum" style={{ fill: 'var(--text-faint)' }}>
                [{col.length}]
              </text>
              {active && (
                <rect x={x(g) - cellW / 2 - 5} y={top - 6} width={cellW + 10} height={height - top - bottom + 12} rx={7}
                  fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} />
              )}
              {col.map((v, i) => {
                const isFocus = active && i === focus;
                return (
                  <g key={i} role="button" tabIndex={g === 0 ? -1 : 0} aria-label={`${names[g][i]} = ${v.toFixed(4)}`}
                    style={{ cursor: g === 0 ? 'default' : 'pointer' }}
                    onClick={() => g > 0 && onPick(g, i)}
                    onKeyDown={(e) => { if (g > 0 && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onPick(g, i); } }}>
                    <rect x={x(g) - cellW / 2} y={y(g, i)} width={cellW} height={cell} rx={Math.min(4, cell / 3)}
                      fill={rgb(diverging(v, m, neutral))}
                      style={{ stroke: isFocus ? 'var(--select)' : 'var(--border-soft)' }} strokeWidth={isFocus ? 2 : 0.6} />
                    {cell >= 14 && (
                      <text x={x(g)} y={y(g, i) + cell / 2 + 3.5} textAnchor="middle" fontSize={cell >= 22 ? 11 : 9.5} className="tnum"
                        style={{ fill: 'var(--text-primary)', pointerEvents: 'none' }}>
                        {Math.abs(v) >= 100 ? v.toExponential(0) : v.toFixed(Math.abs(v) >= 10 ? 1 : 2)}
                      </text>
                    )}
                    <title>{`${names[g][i]}: ${v.toFixed(5)}`}</title>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
