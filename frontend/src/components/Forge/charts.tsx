// Small, dependency-free chart primitives for the Neural Microscope.

import { useEffect, useMemo, useRef } from 'react';
import { sampleCurve } from '../../forge/activations';
import { CLASS0, CLASS1, fmt, fmtSigned, maxAbs, rankByMagnitude, rgb, type RGB } from '../../forge/format';
import type { Histogram } from '../../forge/types';

// ── Heatmap on canvas ───────────────────────────────────────────────────────
export interface ScatterPoint { x: number; y: number; cls: number }

interface HeatmapProps {
  values: number[][]; // values[row][col], row 0 = bottom (y_range[0])
  color: (v: number) => RGB;
  xRange?: [number, number];
  yRange?: [number, number];
  points?: ScatterPoint[];
  marker?: [number, number] | null;
  height?: number;
  onPick?: (x: number, y: number) => void;
  ariaLabel: string;
  flipY?: boolean; // weight matrices: row 0 at the top
}

export function HeatmapCanvas({
  values, color, xRange, yRange, points, marker, height = 160, onPick, ariaLabel, flipY = false,
}: HeatmapProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rows = values.length;
  const cols = values[0]?.length ?? 0;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rows || !cols) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cw = w / cols;
    const ch = h / rows;
    for (let r = 0; r < rows; r++) {
      const py = flipY ? r * ch : h - (r + 1) * ch;
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = rgb(color(values[r][c]));
        ctx.fillRect(c * cw, py, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
      }
    }
    if (!xRange || !yRange) return;
    const px = (x: number) => ((x - xRange[0]) / (xRange[1] - xRange[0])) * w;
    const py = (y: number) => h - ((y - yRange[0]) / (yRange[1] - yRange[0])) * h;
    points?.forEach((p) => {
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 2.2, 0, Math.PI * 2);
      ctx.fillStyle = rgb(p.cls === 1 ? CLASS1 : CLASS0, 0.9);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth = 0.6;
      ctx.stroke();
    });
    if (marker) {
      const [mx, my] = [px(marker[0]), py(marker[1])];
      ctx.strokeStyle = '#fde047';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(mx, my, 6, 0, Math.PI * 2);
      ctx.moveTo(mx - 10, my); ctx.lineTo(mx - 3, my);
      ctx.moveTo(mx + 3, my); ctx.lineTo(mx + 10, my);
      ctx.moveTo(mx, my - 10); ctx.lineTo(mx, my - 3);
      ctx.moveTo(mx, my + 3); ctx.lineTo(mx, my + 10);
      ctx.stroke();
    }
  }, [values, color, xRange, yRange, points, marker, rows, cols, flipY]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPick || !xRange || !yRange) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = 1 - (e.clientY - rect.top) / rect.height;
    onPick(xRange[0] + fx * (xRange[1] - xRange[0]), yRange[0] + fy * (yRange[1] - yRange[0]));
  };

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      onClick={handleClick}
      className="w-full rounded-md"
      style={{ height, cursor: onPick ? 'crosshair' : 'default', imageRendering: 'pixelated', border: '1px solid var(--border)' }}
    />
  );
}

// ── Histogram ───────────────────────────────────────────────────────────────
export function MiniHistogram({ hist, marker, color = '#60a5fa', height = 56 }: {
  hist: Histogram; marker?: number | null; color?: string; height?: number;
}) {
  const W = 260;
  const max = Math.max(1, ...hist.counts);
  const lo = hist.edges[0];
  const hi = hist.edges[hist.edges.length - 1];
  const bw = W / hist.counts.length;
  const mx = marker !== null && marker !== undefined ? ((marker - lo) / (hi - lo || 1)) * W : null;
  return (
    <svg viewBox={`0 0 ${W} ${height + 14}`} className="w-full" role="img" aria-label="Distribution histogram">
      {hist.counts.map((c, i) => {
        const h = (c / max) * height;
        return <rect key={i} x={i * bw + 0.5} y={height - h} width={Math.max(0.5, bw - 1)} height={h} fill={color} opacity={0.75} rx={1} />;
      })}
      <line x1={0} x2={W} y1={height} y2={height} stroke="var(--border-soft)" />
      {mx !== null && mx >= 0 && mx <= W && (
        <g>
          <line x1={mx} x2={mx} y1={0} y2={height} stroke="#fde047" strokeWidth={1.5} />
          <text x={Math.min(W - 30, Math.max(0, mx - 15))} y={8} fontSize={8} fill="#fde047">probe</text>
        </g>
      )}
      <text x={0} y={height + 11} fontSize={8} fill="var(--text-muted)">{fmt(lo, 2)}</text>
      <text x={W} y={height + 11} fontSize={8} fill="var(--text-muted)" textAnchor="end">{fmt(hi, 2)}</text>
    </svg>
  );
}

// ── Activation function with the operating point ───────────────────────────
export function ActivationCurve({ fn, z, a, natural }: { fn: string; z: number; a: number; natural?: number | null }) {
  const span = Math.max(4, Math.abs(z) * 1.4);
  const pts = useMemo(() => sampleCurve(fn, -span, span), [fn, span]);
  if (!pts.length) return null;
  const W = 260, H = 110, P = 14;
  const ys = pts.map((p) => p[1]).concat([a, natural ?? a]);
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const sx = (v: number) => P + ((v + span) / (2 * span)) * (W - 2 * P);
  const sy = (v: number) => H - P - ((v - yLo) / (yHi - yLo || 1)) * (H - 2 * P);
  const path = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('');
  const dotY = natural !== null && natural !== undefined ? natural : a;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${fn} activation curve`}>
      <line x1={P} x2={W - P} y1={sy(0)} y2={sy(0)} stroke="var(--border-soft)" />
      <line x1={sx(0)} x2={sx(0)} y1={P / 2} y2={H - P} stroke="var(--border-soft)" />
      <path d={path} fill="none" stroke="#93c5fd" strokeWidth={2} />
      <line x1={sx(z)} x2={sx(z)} y1={sy(0)} y2={sy(dotY)} stroke="#fde047" strokeDasharray="3,3" />
      <circle cx={sx(z)} cy={sy(dotY)} r={4.5} fill="#fde047" stroke="#111827" strokeWidth={1.5} />
      <text x={W - P} y={H - 2} fontSize={9} fill="var(--text-muted)" textAnchor="end">z →</text>
      <text x={sx(z) + 6} y={Math.max(10, sy(dotY) - 6)} fontSize={9} fill="#fde047">
        f({fmt(z, 2)}) = {fmt(dotY, 2)}
      </text>
    </svg>
  );
}

// ── Weighted-sum breakdown ─────────────────────────────────────────────────
export function ContributionBars({
  names, contributions, weights, inputs, maxRows, onSelect, highlight,
}: {
  names: string[]; contributions: number[]; weights: number[]; inputs: number[];
  maxRows: number; onSelect?: (index: number) => void; highlight?: number | null;
}) {
  const order = rankByMagnitude(contributions);
  const shown = order.slice(0, maxRows);
  const rest = order.slice(maxRows);
  const restSum = rest.reduce((s, i) => s + contributions[i], 0);
  const m = maxAbs(contributions) || 1;
  return (
    <div className="space-y-0.5">
      {shown.map((i) => {
        const c = contributions[i];
        const w = (Math.abs(c) / m) * 50;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSelect?.(i)}
            className="w-full grid items-center gap-2 text-[11px] rounded px-1 py-0.5 text-left transition-colors hover:bg-white/5"
            style={{ gridTemplateColumns: '52px 1fr 64px', outline: highlight === i ? '1px solid #fde047' : 'none' }}
            title={`${names[i]}: weight ${fmt(weights[i], 4)} × input ${fmt(inputs[i], 4)} = ${fmt(c, 4)}`}
          >
            <span className="font-mono truncate" style={{ color: 'var(--text-muted)' }}>{names[i]}</span>
            <span className="relative h-3">
              <span className="absolute top-0 bottom-0 left-1/2 w-px" style={{ background: 'var(--border-soft)' }} />
              <span
                className="absolute top-0.5 bottom-0.5 rounded-sm"
                style={{
                  left: c >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`,
                  background: c >= 0 ? 'rgba(16,185,129,0.85)' : 'rgba(239,68,68,0.85)',
                }}
              />
            </span>
            <span className="font-mono text-right" style={{ color: c >= 0 ? '#6ee7b7' : '#fca5a5' }}>{fmtSigned(c, 3)}</span>
          </button>
        );
      })}
      {rest.length > 0 && (
        <div className="grid text-[11px] px-1 pt-0.5" style={{ gridTemplateColumns: '1fr 64px', color: 'var(--text-faint)' }}>
          <span>+ {rest.length} smaller inputs</span>
          <span className="font-mono text-right">{fmtSigned(restSum, 3)}</span>
        </div>
      )}
    </div>
  );
}

// ── Probability bars ────────────────────────────────────────────────────────
export function ProbBars({ probs, names }: { probs: number[]; names: string[] }) {
  return (
    <div className="space-y-1">
      {probs.map((p, i) => {
        const col = i === 1 ? CLASS1 : CLASS0;
        return (
          <div key={i} className="grid items-center gap-2 text-[11px]" style={{ gridTemplateColumns: '54px 1fr 56px' }}>
            <span style={{ color: rgb(col) }}>{names[i]}</span>
            <span className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
              <span className="absolute inset-y-0 left-0 rounded-full transition-all duration-300" style={{ width: `${p * 100}%`, background: rgb(col, 0.9) }} />
            </span>
            <span className="font-mono text-right" style={{ color: 'var(--text-primary)' }}>{(p * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Before / after probabilities ───────────────────────────────────────────
export function CompareBars({ before, after, names }: { before: number[]; after: number[]; names: string[] }) {
  return (
    <div className="space-y-1">
      <div className="grid text-[10px] gap-2" style={{ gridTemplateColumns: '54px 46px 1fr 96px', color: 'var(--text-faint)' }}>
        <span /><span className="text-right">original</span><span /><span className="text-right">after</span>
      </div>
      {after.map((p, i) => {
        const col = i === 1 ? CLASS1 : CLASS0;
        const d = p - before[i];
        return (
          <div key={i} className="grid items-center gap-2 text-[11px]" style={{ gridTemplateColumns: '54px 46px 1fr 96px' }}>
            <span style={{ color: rgb(col) }}>{names[i]}</span>
            <span className="font-mono text-right" style={{ color: 'var(--text-muted)' }}>{(before[i] * 100).toFixed(1)}%</span>
            <span className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
              <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${before[i] * 100}%`, background: rgb(col, 0.25) }} />
              <span className="absolute inset-y-0 left-0 rounded-full transition-all duration-300" style={{ width: `${p * 100}%`, background: rgb(col, 0.9) }} />
            </span>
            <span className="font-mono text-right" style={{ color: 'var(--text-primary)' }}>
              {(p * 100).toFixed(1)}%
              {Math.abs(d) > 0.0005 && <span style={{ color: d > 0 ? '#6ee7b7' : '#fca5a5' }}> {d > 0 ? '▲' : '▼'}{Math.abs(d * 100).toFixed(1)}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Layout helpers ──────────────────────────────────────────────────────────
export function Section({ title, hint, children, right }: { title: string; hint?: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-lg border p-2.5" style={{ borderColor: 'var(--border)', background: 'rgba(255,255,255,0.015)' }}>
      <div className="flex items-baseline gap-2 mb-1.5">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{title}</h4>
        {hint && <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>{hint}</span>}
        {right && <span className="ml-auto">{right}</span>}
      </div>
      {children}
    </section>
  );
}

export function KV({ k, v, mono = true, color }: { k: string; v: React.ReactNode; mono?: boolean; color?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-[11px] py-0.5">
      <span style={{ color: 'var(--text-muted)' }}>{k}</span>
      <span className={mono ? 'font-mono' : ''} style={{ color: color ?? 'var(--text-primary)' }}>{v}</span>
    </div>
  );
}
