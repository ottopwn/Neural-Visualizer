import { Canvas } from '@react-three/fiber';
import { Box, Crosshair, Focus, Maximize2, Pause, Play, RotateCcw, StepBack, StepForward } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { useTheme } from '../../../contexts/theme';
import { useExplorer } from '../../../forge/explorer';
import { fmt, fmtSigned } from '../../../forge/format';
import { buildSteps, reached, stepIndexOf } from '../../../forge/passExplorer';
import {
  COLOR_MODES, edgeList, fitDistance, layoutNetwork, nodeValues, selectEdges, strongestInto,
  type ColorMode, type Edge3D, type Vec3,
} from '../../../forge/scene3d';
import { useForgeStore } from '../../../forge/store';
import { NeedsModel } from '../../Workspaces/EmptyState';
import { ModelStrip } from '../../Workspaces/NetworkWorkspace';
import { Scene, type CameraGoal, type Hover, type Palette } from './Scene';

const FOV = 42;
const LIMITS = [100, 300, 600, 1500, 5000, Infinity];
const PULSES = 36;

function cssColor(name: string, fallback: string): THREE.Color {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  try { return new THREE.Color(v || fallback); } catch { return new THREE.Color(fallback); }
}

function usePalette(): Palette {
  const { theme } = useTheme();
  return useMemo(() => {
    void theme; // recompute when the theme (and so the CSS tokens) changes
    const paper = document.documentElement.getAttribute('data-theme') === 'paper';
    return {
      bg: cssColor('--bg-base', '#0a0d12'),
      pos: cssColor('--pos', '#10b981'),
      neg: cssColor('--neg', '#ef4444'),
      neutral: new THREE.Color(paper ? '#c9c6bb' : '#566173'),
      warn: cssColor('--warn', '#f59e0b'),
      select: cssColor('--select', '#fde047'),
      text: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() || '#e7eaf0',
    };
  }, [theme]);
}

function useReducedMotion(): boolean {
  const [r, setR] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!mq) return;
    const on = () => setR(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return !!r;
}

type PassMode = 'off' | 'forward' | 'backward';

export function Forge3DView() {
  const session = useForgeStore((s) => s.session);
  const trace = useForgeStore((s) => s.trace);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const nIv = useForgeStore((s) => s.interventions.length);
  const explorer = useExplorer();
  const palette = usePalette();
  const reduceMotion = useReducedMotion();
  const [mode, setMode] = useState<ColorMode>('signal');
  const [limit, setLimit] = useState(600);
  const [pass, setPass] = useState<PassMode>('off');
  const [hover, setHover] = useState<Hover | null>(null);

  const sizes = useMemo(() => session?.structure.layers.map((l) => l.size) ?? [], [session]);
  const layout = useMemo(() => layoutNetwork(sizes.length ? sizes : [1]), [sizes]);

  // Following the Pass Explorer: its direction decides what the colours mean.
  const effMode: ColorMode = pass === 'forward' ? 'signal' : pass === 'backward' ? 'gradients' : mode;
  const steps = useMemo(() => (trace && pass !== 'off' ? buildSteps(trace, pass) : []), [trace, pass]);
  const step = steps.length ? steps[stepIndexOf(steps, explorer.stepId[pass as 'forward' | 'backward'])] : null;

  const edges: Edge3D[] = useMemo(() => (trace ? edgeList(trace, effMode) : []), [trace, effMode]);
  const nodeVals = useMemo(() => (trace ? nodeValues(trace, effMode) : []), [trace, effMode]);
  const sel = useMemo(() => selectEdges(edges, limit === Infinity ? edges.length : limit, selection), [edges, limit, selection]);
  const ablated = useMemo(() => new Set(trace?.layers.flatMap((l) => l.ablated.map((i) => `${l.layer}:${i}`)) ?? []), [trace]);
  const dimmed = useMemo(() => (step ? reached(step, sizes.length - 1).map((r) => !r) : sizes.map(() => false)), [step, sizes]);

  const pulses = useMemo(() => {
    if (!step || step.edgeLayer === null || !trace) return [];
    // Forward pulses follow w·a, backward pulses follow w·δ (the real terms of Wᵀ·δ).
    const d = trace.layers[step.edgeLayer - 1];
    const strength: Edge3D[] = edges.map((e) => (e.layer !== step.edgeLayer ? e
      : { ...e, value: pass === 'forward' ? d.weight[e.target][e.source] * d.input[e.source] : d.weight[e.target][e.source] * d.grad_z[e.target] }));
    const idx = strongestInto(strength, step.edgeLayer, PULSES);
    const m = Math.max(1e-12, ...idx.map((i) => Math.abs(strength[i].value)));
    return idx.map((i) => ({ edge: i, strength: Math.abs(strength[i].value) / m, positive: strength[i].value >= 0 }));
  }, [step, trace, edges, pass]);

  // Camera goals.
  const fitGoal = useCallback((key: number): CameraGoal => {
    const dist = fitDistance(layout.radius, FOV);
    const dir = new THREE.Vector3(0.42, 0.32, 1).normalize().multiplyScalar(dist);
    const c = layout.center;
    return { position: [c[0] + dir.x, c[1] + dir.y, c[2] + dir.z], target: c, key };
  }, [layout]);
  const [goal, setGoal] = useState<CameraGoal>(() => fitGoal(0));
  // A new architecture: re-fit the camera (derived state, updated during render).
  const [fittedFor, setFittedFor] = useState(layout);
  if (fittedFor !== layout) {
    setFittedFor(layout);
    setGoal(fitGoal(goal.key + 1));
  }

  const focusOn = useCallback((target: Vec3, radius: number) => {
    const dist = Math.max(4.5, fitDistance(radius, FOV) * 1.15);
    const dir = new THREE.Vector3(0.55, 0.3, 1).normalize().multiplyScalar(dist);
    setGoal((g) => ({ position: [target[0] + dir.x, target[1] + dir.y, target[2] + dir.z], target, key: g.key + 1 }));
  }, []);
  const focusLayer = useCallback((g: number) => {
    const e = layout.extents[g];
    focusOn([layout.layerX[g], layout.center[1], layout.center[2]], Math.max(1.5, Math.hypot(e.y, e.z) + 0.8));
  }, [layout, focusOn]);
  const focusSelection = useCallback(() => {
    if (!selection) return;
    if (selection.kind === 'neuron') focusOn(layout.positions[selection.layer][selection.index], 2);
    else if (selection.kind === 'layer') focusLayer(selection.layer);
    else {
      const a = layout.positions[selection.layer - 1][selection.source];
      const b = layout.positions[selection.layer][selection.target];
      focusOn([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / 2 + 1);
    }
  }, [selection, layout, focusOn, focusLayer]);
  const frontView = () => {
    const dist = fitDistance(layout.radius, FOV);
    setGoal((g) => ({ position: [layout.center[0], layout.center[1], layout.center[2] + dist], target: layout.center, key: g.key + 1 }));
  };

  const onSelect = useCallback((ref: Parameters<typeof select>[0]) => void select(ref), [select]);

  if (!session) {
    return (
      <NeedsModel icon={<Box size={20} />} title="See the real network in 3D"
        body={<>Every sphere is a real neuron, every line a real weight. Colours come from the model's activations, weights or gradients on the probe input.</>} />
    );
  }

  const info = COLOR_MODES.find((m) => m.id === effMode)!;
  const shownCount = sel.shown.length + sel.emphasised.length;
  const names = trace ? [trace.feature_names, ...trace.layers.map((l) => l.neuron_names)] : [];
  const labels = session.structure.layers.map((l) => l.label);

  let hoverText: string | null = null;
  if (hover && trace) {
    if (hover.kind === 'neuron') {
      const v = nodeVals[hover.layer]?.[hover.index];
      hoverText = `${names[hover.layer][hover.index]} · ${effMode === 'weights' && hover.layer === 0 ? 'input (no bias)' : `${info.node} = ${fmtSigned(v, 4)}`}${ablated.has(`${hover.layer}:${hover.index}`) ? ' · disabled' : ''}`;
    } else {
      const e = edges.find((x) => x.layer === hover.layer && x.source === hover.source && x.target === hover.index);
      if (e) hoverText = `${names[e.layer - 1][e.source]} → ${names[e.layer][e.target]} · ${info.edge} = ${fmtSigned(e.value, 4)}`;
    }
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      <ModelStrip />
      <div className="flex items-center gap-2 flex-wrap px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
        <div className="seg" role="group" aria-label="Colour by">
          {COLOR_MODES.map((m) => (
            <button key={m.id} type="button" aria-pressed={effMode === m.id} disabled={pass !== 'off'} onClick={() => setMode(m.id)}
              title={`Nodes: ${m.node} · edges: ${m.edge}`}>{m.label}</button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Follow the pass explorer">
          {(['off', 'forward', 'backward'] as PassMode[]).map((p) => (
            <button key={p} type="button" aria-pressed={pass === p}
              onClick={() => { setPass(p); if (p !== 'off') explorer.setDir(p); }}
              title={p === 'off' ? 'Static view' : `Follow the ${p} pass step by step (shared with the Forward / Backward workspace)`}>
              {p === 'off' ? 'Static' : p === 'forward' ? 'Forward pass' : 'Backward pass'}
            </button>
          ))}
        </div>
        {pass !== 'off' && step && (
          <div className="flex items-center gap-1">
            <button type="button" className="btn-ghost p-1.5" aria-label="Previous pass step" onClick={() => explorer.step(-1)}><StepBack size={15} /></button>
            <button type="button" className="btn-ghost p-1.5" aria-label={explorer.playing ? 'Pause pass' : 'Play pass'} onClick={() => explorer.togglePlay()}>
              {explorer.playing ? <Pause size={15} /> : <Play size={15} />}
            </button>
            <button type="button" className="btn-ghost p-1.5" aria-label="Next pass step" onClick={() => explorer.step(1)}><StepForward size={15} /></button>
            <span className="text-[12px]" style={{ color: 'var(--text-primary)' }} aria-live="polite">{step.title}</span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button type="button" className="btn-ghost px-2 py-1 text-[12px]" onClick={() => setGoal(fitGoal(goal.key + 1))} title="Fit the whole network"><Maximize2 size={14} />Fit</button>
          <button type="button" className="btn-ghost px-2 py-1 text-[12px]" onClick={frontView} title="Look along the depth axis"><Crosshair size={14} />Front</button>
          <button type="button" className="btn-ghost px-2 py-1 text-[12px]" onClick={focusSelection} disabled={!selection} title="Fly to the selected neuron, layer or connection"><Focus size={14} />Focus selection</button>
          <button type="button" className="btn-ghost px-2 py-1 text-[12px]" onClick={() => setGoal(fitGoal(goal.key + 1))} title="Reset camera"><RotateCcw size={14} />Reset</button>
        </div>
      </div>

      <div className="flex-1 min-h-0 relative" data-testid="forge-3d">
        {trace && (
          <Canvas camera={{ fov: FOV, position: goal.position, near: 0.1, far: 1000 }} dpr={[1, 2]}
            raycaster={{ params: { Line: { threshold: 0.12 } } as THREE.Raycaster['params'] }}
            onPointerMissed={() => setHover(null)} aria-label="3D view of the real network">
            <Scene layout={layout} sizes={sizes} names={names} layerLabels={labels} nodeVals={nodeVals} dimmed={dimmed}
              ablated={ablated} edges={edges} shown={sel.shown} emphasised={sel.emphasised} pulses={pulses}
              pulseDir={pass === 'backward' ? -1 : 1} palette={palette} selection={selection} goal={goal}
              reduceMotion={reduceMotion} onSelect={onSelect} onFocusLayer={focusLayer} onHover={setHover} />
          </Canvas>
        )}

        {hover && hoverText && (
          <div className="overlay absolute pointer-events-none px-2 py-1 text-[12px] tnum" style={{ left: hover.x + 14, top: hover.y + 10, color: 'var(--text-primary)' }}>
            {hoverText}
          </div>
        )}

        <div className="overlay absolute top-3 left-3 px-3 py-2 text-[12px] space-y-1 max-w-[360px]" style={{ color: 'var(--text-muted)' }}>
          <div className="flex items-center gap-2">
            <span className="badge-green">REAL</span>
            <span>epoch {trace?.provenance.checkpoint_epoch ?? '…'}{trace && !trace.provenance.is_latest ? ' (stored checkpoint)' : ' (live)'}{nIv ? ` · what-if ×${nIv}` : ''}</span>
          </div>
          <div>Spheres: <b style={{ color: 'var(--text-primary)' }}>{info.node}</b></div>
          <div>Lines: <b style={{ color: 'var(--text-primary)' }}>{info.edge}</b></div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: 'var(--pos)' }} />positive</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: 'var(--neg)' }} />negative</span>
            <span style={{ color: 'var(--text-faint)' }}>brightness = |value| / layer max</span>
          </div>
          {ablated.size > 0 && <div style={{ color: 'var(--text-warn)' }}>Orange ring = neuron disabled by what-if</div>}
        </div>

        <div className="overlay absolute bottom-3 left-3 px-3 py-2 text-[12px] flex items-center gap-3 flex-wrap max-w-[calc(100%-24px)]" style={{ color: 'var(--text-muted)' }} data-testid="edge-disclosure">
          <span className="tnum">
            {sel.filtered
              ? <>Showing <b style={{ color: 'var(--text-primary)' }}>{shownCount.toLocaleString()}</b> of {sel.total.toLocaleString()} connections — the strongest by |{effMode === 'signal' ? 'w·a' : effMode === 'weights' ? 'w' : 'dL/dw'}|{sel.emphasised.length ? ', plus those of the selection' : ''}</>
              : <>All <b style={{ color: 'var(--text-primary)' }}>{sel.total.toLocaleString()}</b> connections shown</>}
          </span>
          <label className="flex items-center gap-1.5">
            max
            <select className="select-base !w-auto !py-0.5 !text-xs tnum" value={String(limit)} aria-label="Maximum connections drawn"
              onChange={(e) => setLimit(e.target.value === 'Infinity' ? Infinity : +e.target.value)}>
              {LIMITS.map((l) => <option key={String(l)} value={String(l)}>{l === Infinity ? 'all' : l.toLocaleString()}</option>)}
            </select>
          </label>
          <span style={{ color: 'var(--text-faint)' }}>drag rotate · right-drag pan · scroll zoom · click select · double-click focus</span>
        </div>

        {selection && trace && selection.kind === 'neuron' && (
          <div className="overlay absolute top-3 right-3 px-3 py-2 text-[12px] tnum" style={{ color: 'var(--text-muted)' }}>
            <b style={{ color: 'var(--text-primary)' }}>{names[selection.layer]?.[selection.index]}</b>
            {selection.layer > 0 && <> · z {fmt(trace.layers[selection.layer - 1].z[selection.index], 3)} · a {fmt(trace.layers[selection.layer - 1].a[selection.index], 3)} · δ {fmtSigned(trace.layers[selection.layer - 1].grad_z[selection.index], 3)}</>}
            {selection.layer === 0 && <> · x {fmt(trace.input[selection.index], 3)} · dL/dx {fmtSigned(trace.grad_input[selection.index], 3)}</>}
          </div>
        )}
      </div>
    </div>
  );
}
