import { ArrowLeftRight, Loader2 } from 'lucide-react';
import { useMemo } from 'react';
import { describeComparison } from '../../../forge/explainTime';
import { CLASS0, CLASS1, classColor, fmt, fmtSigned, histogramOf, mean, pct, type RGB } from '../../../forge/format';
import { neuronName } from '../../../forge/interventions';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { resolveEpoch } from '../../../forge/timeline';
import type { EpochComparison, ResponseMap } from '../../../forge/types';
import { CompareBars, HeatmapCanvas, MiniHistogram, Section, type ScatterPoint } from '../charts';
import { responseColor } from '../hooks';
import { useCheckpointEpochs, useEpochComparison, useNeutral } from './hooks';

const CHANGED: RGB = [232, 121, 249];

function mixRGB(a: RGB, b: RGB, t: number): RGB {
  return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as RGB;
}

function EpochSelect({ label, value, epochs, live, color, onChange }: {
  label: string; value: number | null; epochs: number[]; live: number; color: string; onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
      <span className="w-5 h-5 rounded flex items-center justify-center font-bold text-[11px]" style={{ background: color, color: '#0b1020' }}>{label}</span>
      <select className="select-base !w-auto !py-1 !px-2 !text-xs font-mono" aria-label={`Epoch ${label}`}
        value={value ?? live} onChange={(e) => onChange(Number(e.target.value))}>
        {epochs.map((e) => <option key={e} value={e}>epoch {e}{e === live ? ' · live' : ''}</option>)}
      </select>
    </label>
  );
}

function DeltaCard({ label, a, b, delta, good, note }: {
  label: string; a: string; b: string; delta: string; good: boolean | null; note?: string;
}) {
  return (
    <div className="rounded-lg border px-2.5 py-2 min-w-0" style={{ borderColor: 'var(--border)', background: 'var(--tm-panel)' }}>
      <div className="text-[9px] uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>{label}</div>
      {(a || b) && (
        <div className="flex items-baseline gap-1.5 font-mono text-[12px]">
          <span style={{ color: 'var(--tm-a)' }}>{a}</span>
          <span style={{ color: 'var(--text-faint)' }}>→</span>
          <span style={{ color: 'var(--tm-b)' }}>{b}</span>
        </div>
      )}
      <div className="font-mono text-[13px] font-semibold" style={{ color: good === null ? 'var(--text-primary)' : good ? 'var(--tm-pos)' : 'var(--tm-neg)' }}>{delta}</div>
      {note && <div className="text-[10px]" style={{ color: 'var(--text-faint)' }}>{note}</div>}
    </div>
  );
}

function usePoints(c: EpochComparison | null): ScatterPoint[] | undefined {
  const session = useForgeStore((s) => s.session);
  return useMemo(() => {
    if (!session || !c || session.structure.input_dim !== 2) return undefined;
    const y = session.dataset_y;
    return session.dataset_X.map((p, i) => {
      const okA = c.predictions_a[i] === y[i];
      const okB = c.predictions_b[i] === y[i];
      return { x: p[0], y: p[1], cls: y[i], ring: okA === okB ? undefined : okB ? '#10b981' : '#ef4444' };
    });
  }, [session, c]);
}

function MapCard({ title, color, map, points, colorFn, marker, contour = 0.5, contourColor, pointStroke }: {
  title: React.ReactNode; color: string; map: ResponseMap; points?: ScatterPoint[]; colorFn: (v: number) => RGB;
  marker: [number, number] | null; contour?: number | null; contourColor: string; pointStroke: string;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] mb-1 font-semibold" style={{ color }}>{title}</div>
      <HeatmapCanvas ariaLabel={String(title)} values={map.values} color={colorFn} xRange={map.x_range} yRange={map.y_range}
        points={points} marker={marker} contour={contour} contourColor={contourColor} pointStroke={pointStroke} height={170} />
    </div>
  );
}

export function EpochCompareView() {
  const session = useForgeStore((s) => s.session);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const mode = useForgeStore((s) => s.mode);
  const epochs = useCheckpointEpochs();
  const comparison = useEpochComparison();
  const comparing = useTimeMachine((s) => s.comparing);
  const compareA = useTimeMachine((s) => s.compareA);
  const compareB = useTimeMachine((s) => s.compareB);
  const cursor = useTimeMachine((s) => s.cursor);
  const setCompareEpochs = useTimeMachine((s) => s.setCompareEpochs);
  const swapCompare = useTimeMachine((s) => s.swapCompare);
  const neutral = useNeutral();
  const paper = neutral[0] > 128;
  const points = usePoints(comparison);
  const classFn = useMemo(() => (v: number) => classColor(v, neutral), [neutral]);
  const diff = useMemo(() => {
    const c = comparison;
    if (!c?.boundary_a || !c.boundary_b) return null;
    const values = c.boundary_a.values.map((row, r) => row.map((va, k) => {
      const vb = c.boundary_b!.values[r][k];
      return (va > 0.5) === (vb > 0.5) ? (vb > 0.5 ? 1 : 0) : 2;
    }));
    return { ...c.boundary_b, values };
  }, [comparison]);
  const diffFn = useMemo(() => (v: number): RGB => (v === 2 ? CHANGED : mixRGB(neutral, v === 1 ? CLASS1 : CLASS0, 0.22)), [neutral]);
  const confHist = useMemo(() => (comparison ? histogramOf(comparison.confidence_delta, 24) : null), [comparison]);

  if (!session) return null;
  const live = session.epoch;
  const names = session.structure.class_names;
  const c = comparison;
  const stale = c && (c.a.epoch !== compareA || c.b.epoch !== (compareB ?? live));
  const playhead = resolveEpoch(cursor, live);
  const contourColor = paper ? 'rgba(17,24,39,0.85)' : 'rgba(255,255,255,0.9)';
  const pointStroke = paper ? 'rgba(17,24,39,0.5)' : 'rgba(255,255,255,0.55)';
  const marker: [number, number] | null = c && c.probe.x.length === 2 ? [c.probe.x[0], c.probe.x[1]] : null;
  const maxRel = c ? Math.max(1e-12, ...c.layers.map((l) => l.relative_change)) : 1;

  return (
    <div className="h-full min-h-0 overflow-y-auto pr-1 space-y-2.5">
      {/* selectors */}
      <div className="flex items-center gap-2 flex-wrap">
        <EpochSelect label="A" value={compareA} epochs={epochs} live={live} color="var(--tm-a)"
          onChange={(v) => void setCompareEpochs(v, compareB)} />
        <button type="button" className="btn-secondary !py-1 !px-2 text-[11px] flex items-center gap-1" onClick={() => void swapCompare()}
          aria-label="Swap A and B"><ArrowLeftRight size={12} />swap</button>
        <EpochSelect label="B" value={compareB} epochs={epochs} live={live} color="var(--tm-b)"
          onChange={(v) => void setCompareEpochs(compareA ?? epochs[0], v)} />
        <button type="button" className="btn-secondary !py-1 !px-2 text-[11px]" onClick={() => void setCompareEpochs(playhead, compareB)}>A = playhead</button>
        <button type="button" className="btn-secondary !py-1 !px-2 text-[11px]" onClick={() => void setCompareEpochs(compareA ?? epochs[0], playhead)}>B = playhead</button>
        {(comparing || stale) && <Loader2 size={13} className="animate-spin" style={{ color: 'var(--text-faint)' }} />}
        <span className="text-[10px] ml-auto" style={{ color: 'var(--text-faint)' }}>Both epochs are stored checkpoints · no what-if overlay · drag A/B on the timeline</span>
      </div>

      {!c ? (
        <div className="h-40 flex items-center justify-center"><Loader2 size={16} className="animate-spin" style={{ color: 'var(--text-faint)' }} /></div>
      ) : (
        <div className="space-y-2.5" style={{ opacity: stale ? 0.6 : 1, transition: 'opacity 0.15s' }}>
          {mode === 'learn' && (
            <Section title="What changed between the two epochs">
              <ul className="space-y-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                {describeComparison(c, names).map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            </Section>
          )}

          {/* decision boundary before / after / difference */}
          {c.boundary_a && c.boundary_b && diff && (
            <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))' }}>
              <MapCard title={`A · epoch ${c.a.epoch}`} color="var(--tm-a)" map={c.boundary_a} colorFn={classFn} points={points}
                marker={marker} contourColor={contourColor} pointStroke={pointStroke} />
              <MapCard title={`B · epoch ${c.b.epoch}`} color="var(--tm-b)" map={c.boundary_b} colorFn={classFn} points={points}
                marker={marker} contourColor={contourColor} pointStroke={pointStroke} />
              <div className="min-w-0">
                <MapCard title={<>Where the class changed · {pct(c.boundary_flip_fraction ?? 0)} of plane</>} color="#e879f9"
                  map={diff} colorFn={diffFn} points={points} marker={marker} contour={null} contourColor={contourColor} pointStroke={pointStroke} />
                <div className="text-[9.5px] mt-1 flex gap-2 flex-wrap" style={{ color: 'var(--text-faint)' }}>
                  <span style={{ color: '#e879f9' }}>■ class switched</span>
                  <span style={{ color: '#10b981' }}>○ became correct</span>
                  <span style={{ color: '#ef4444' }}>○ became wrong</span>
                </div>
              </div>
            </div>
          )}

          {/* headline deltas */}
          <div className="grid gap-1.5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
            <DeltaCard label="loss" a={fmt(c.a.loss, 4)} b={fmt(c.b.loss, 4)} delta={fmtSigned(c.loss_delta, 4)}
              good={c.loss_delta === 0 ? null : c.loss_delta < 0} />
            <DeltaCard label="accuracy" a={pct(c.a.accuracy)} b={pct(c.b.accuracy)} delta={`${fmtSigned(c.accuracy_delta * 100, 1)} pp`}
              good={c.accuracy_delta === 0 ? null : c.accuracy_delta > 0} />
            <DeltaCard label="predictions changed" a="" b="" delta={`${c.changed} · ${pct(c.changed_fraction)}`}
              good={c.fixed === c.broken ? null : c.fixed > c.broken} note={`${c.fixed} became correct · ${c.broken} became wrong`} />
            <DeltaCard label="mean confidence" a={pct(c.a.mean_confidence)} b={pct(c.b.mean_confidence)}
              delta={`${fmtSigned((c.b.mean_confidence - c.a.mean_confidence) * 100, 1)} pp`} good={null}
              note="probability of the predicted class" />
            <DeltaCard label="parameters moved" a="" b="" delta={`‖Δθ‖ ${fmt(c.total_delta_norm, 3)}`} good={null}
              note={`${pct(c.total_relative_change, 1)} of ‖θ_A‖`} />
          </div>

          <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
            <Section title="Probe prediction" hint={`A ${c.a.epoch} → B ${c.b.epoch}`}>
              <CompareBars before={c.a.probe_probabilities} after={c.b.probe_probabilities} names={names} />
              {c.a.probe_predicted !== c.b.probe_predicted && (
                <p className="text-[10px] mt-1" style={{ color: 'var(--tm-whatif-text)' }}>The probe's predicted class flipped.</p>
              )}
            </Section>
            {confHist && (
              <Section title="Confidence in the true class" hint={`mean ${fmtSigned(mean(c.confidence_delta) * 100, 1)} pp`}>
                <div className="max-w-[420px]"><MiniHistogram hist={confHist} marker={0} markerLabel="no change" color="#e879f9" height={44} /></div>
                <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>Per sample: P_B(true class) − P_A(true class). Right of the line = more confident and correct.</p>
              </Section>
            )}
          </div>

          {/* per-layer parameter change */}
          <Section title="Parameter change per layer" hint="‖θ_B − θ_A‖ relative to ‖θ_A‖ · click a neuron to inspect it">
            <div className="space-y-1.5">
              {c.layers.map((l) => (
                <div key={l.layer} className="grid items-center gap-2 text-[11px]" style={{ gridTemplateColumns: '70px 1fr 64px' }}>
                  <button type="button" className="text-left hover:underline" style={{ color: 'var(--text-muted)' }}
                    onClick={() => void select({ kind: 'layer', layer: l.layer })}>{l.label}</button>
                  <div>
                    <div className="h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                      <div className="h-full rounded-full" style={{ width: `${(l.relative_change / maxRel) * 100}%`, background: 'linear-gradient(90deg, var(--tm-a), var(--tm-b))' }} />
                    </div>
                    <div className="flex gap-3 text-[9.5px] font-mono mt-0.5 flex-wrap" style={{ color: 'var(--text-faint)' }}>
                      <span>‖ΔW‖ {fmt(l.weight_delta_norm, 3)}</span>
                      <span>‖Δb‖ {fmt(l.bias_delta_norm, 3)}</span>
                      <span>max|Δw| {fmt(l.max_abs_weight_delta, 3)}</span>
                      <span>‖W‖ {fmt(l.weight_norm_a, 2)}→{fmt(l.weight_norm_b, 2)}</span>
                      <span className="flex gap-1">most changed:
                        {l.top_neurons.map((n) => (
                          <button key={n} type="button" className="underline" style={{ color: 'var(--text-muted)' }}
                            onClick={() => void select({ kind: 'neuron', layer: l.layer, index: n })}>
                            {neuronName(session.structure, l.layer, n)}
                          </button>
                        ))}
                      </span>
                    </div>
                  </div>
                  <span className="font-mono text-right" style={{ color: 'var(--text-primary)' }}>{pct(l.relative_change, 1)}</span>
                </div>
              ))}
            </div>
          </Section>

          {/* selected component */}
          <Section title={c.component ? `${c.component.name}: epoch ${c.a.epoch} vs ${c.b.epoch}` : 'Selected component'}
            hint={c.component ? 'same component, two checkpoints' : undefined}>
            {!c.component ? (
              <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                {selection ? 'Loading…' : 'Select a neuron, layer or connection (in the network or the list above) to compare it in detail.'}
              </p>
            ) : (
              <div className="grid gap-3" style={{ gridTemplateColumns: c.component.response_a ? 'minmax(220px, 1.3fr) minmax(200px, 1fr)' : '1fr' }}>
                <table className="w-full text-[11px] font-mono">
                  <thead>
                    <tr style={{ color: 'var(--text-faint)' }}>
                      <th className="text-left font-normal" />
                      <th className="text-right font-normal" style={{ color: 'var(--tm-a)' }}>A · {c.a.epoch}</th>
                      <th className="text-right font-normal" style={{ color: 'var(--tm-b)' }}>B · {c.b.epoch}</th>
                      <th className="text-right font-normal">Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.component.rows.filter((r) => mode === 'lab' || r.group !== 'gradient').map((r) => (
                      <tr key={r.key}>
                        <td className="py-0.5 font-sans" style={{ color: 'var(--text-muted)' }}>{r.label}</td>
                        <td className="text-right">{fmt(r.a, 4)}</td>
                        <td className="text-right">{fmt(r.b, 4)}</td>
                        <td className="text-right" style={{ color: r.delta === null || r.delta === 0 ? 'var(--text-muted)' : r.delta > 0 ? 'var(--tm-pos)' : 'var(--tm-neg)' }}>
                          {fmtSigned(r.delta, 4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {c.component.response_a && c.component.response_b && (
                  <div className="grid grid-cols-2 gap-1.5">
                    {(['a', 'b'] as const).map((k) => {
                      const m = (k === 'a' ? c.component!.response_a : c.component!.response_b)!;
                      const ra = c.component!.response_a!.value_range;
                      const rb = c.component!.response_b!.value_range;
                      const shared: ResponseMap = { ...m, value_range: [Math.min(ra[0], rb[0]), Math.max(ra[1], rb[1])] };
                      const ref = c.component!.ref;
                      const isOut = ref.kind === 'neuron' && ref.layer === session.structure.layers.length - 1;
                      return (
                        <div key={k}>
                          <div className="text-[10px] mb-0.5" style={{ color: k === 'a' ? 'var(--tm-a)' : 'var(--tm-b)' }}>
                            where it fires · {k === 'a' ? c.a.epoch : c.b.epoch}
                          </div>
                          <HeatmapCanvas ariaLabel={`${c.component!.name} response at epoch ${k === 'a' ? c.a.epoch : c.b.epoch}`}
                            values={m.values} color={responseColor(shared, isOut && ref.kind === 'neuron' ? ref.index : null)}
                            xRange={m.x_range} yRange={m.y_range} height={110} pointStroke={pointStroke} />
                        </div>
                      );
                    })}
                    <p className="col-span-2 text-[9.5px]" style={{ color: 'var(--text-faint)' }}>Both maps share one colour scale, so brighter means a stronger output.</p>
                  </div>
                )}
              </div>
            )}
          </Section>
        </div>
      )}
    </div>
  );
}
