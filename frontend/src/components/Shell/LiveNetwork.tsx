// The home page's hero: a real PyTorch network, initialised for this page.
//
// A small dedicated session is created once per page load (it never replaces
// the user's experiment).  Node fill = real activation, edge colour and
// thickness = real signal w·a on one input.  The dash animation along the
// edges is decorative and the caption says so.

import { useEffect, useState } from 'react';
import * as api from '../../forge/api';
import type { ComputationTrace } from '../../forge/types';
import { useT } from '../../i18n';

let cached: Promise<{ trace: ComputationTrace; params: number }> | null = null;

function loadLive() {
  if (!cached) {
    cached = (async () => {
      const s = await api.createSession({ neurons: [6, 6], activations: ['Tanh', 'Tanh'], dataset: 'Circle', noise: 10, custom_dataset: null });
      const trace = await api.fetchTrace(s.session_id, { probe: { sample_index: 0 }, interventions: [], checkpoint_epoch: null });
      return { trace, params: s.structure.param_count };
    })();
    cached.catch(() => { cached = null; });
  }
  return cached;
}

const W = 520;
const H = 360;

export function LiveNetwork() {
  const t = useT();
  const [data, setData] = useState<{ trace: ComputationTrace; params: number } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    loadLive().then((d) => { if (alive) setData(d); }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  if (!data) {
    return (
      <div className="h-full min-h-[300px] flex items-center justify-center text-center px-6 text-[13px]" style={{ color: 'var(--text-faint)' }}>
        {failed ? t.home.liveOffline : t.home.liveLoading}
      </div>
    );
  }

  const tr = data.trace;
  const layers = [tr.input, ...tr.layers.map((l) => l.a)];
  const xs = layers.map((_, g) => 50 + (g * (W - 100)) / (layers.length - 1));
  const pos = layers.map((vals, g) => vals.map((_, i) => [xs[g], H / 2 + (i - (vals.length - 1) / 2) * Math.min(48, (H - 60) / Math.max(1, vals.length - 1))] as const));
  const edges = tr.layers.flatMap((d, k) => d.weight.flatMap((row, i) => row.map((w, j) => ({ k, i, j, v: w * d.input[j] }))));
  const maxV = Math.max(1e-9, ...edges.map((e) => Math.abs(e.v)));
  const maxA = layers.map((vals) => Math.max(1e-9, ...vals.map(Math.abs)));

  return (
    <figure className="m-0 h-full flex flex-col">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full flex-1 live-net" role="img" aria-label={t.home.live(data.params)}>
        <defs>
          <radialGradient id="nf-node-glow">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </radialGradient>
        </defs>
        {edges.map((e, n) => {
          const a = pos[e.k][e.j];
          const b = pos[e.k + 1][e.i];
          const s = Math.abs(e.v) / maxV;
          return (
            <line key={n} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
              stroke={e.v >= 0 ? 'var(--pos)' : 'var(--neg)'} strokeOpacity={0.15 + 0.75 * s} strokeWidth={0.6 + 3 * s}
              className="live-flow" style={{ animationDuration: `${2.4 - 1.4 * s}s` }} />
          );
        })}
        {layers.map((vals, g) => vals.map((v, i) => {
          const [x, y] = pos[g][i];
          const s = Math.abs(v) / maxA[g];
          return (
            <g key={`${g}-${i}`}>
              <circle cx={x} cy={y} r={10 + 16 * s} fill="url(#nf-node-glow)" opacity={0.35 + 0.65 * s} />
              <circle cx={x} cy={y} r={8} fill={v >= 0 ? 'var(--pos)' : 'var(--neg)'} fillOpacity={0.2 + 0.8 * s} stroke="var(--border-soft)" strokeWidth={1} />
            </g>
          );
        }))}
      </svg>
      <figcaption className="text-[11.5px] leading-relaxed mt-2" style={{ color: 'var(--text-faint)' }}>
        <span className="badge-green mr-1.5">REAL</span>{t.home.live(data.params)} {t.home.decorative}
      </figcaption>
    </figure>
  );
}
