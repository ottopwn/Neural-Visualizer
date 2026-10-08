// Explore: the beginner path through the same real model the Lab inspects.
//
// Build → Train → Understand, in plain language.  Every number and every map
// comes from the PyTorch session (checkpoints, frames, what-if comparisons);
// the only things this view decides are layout and wording.

import { Box, Check, Clock, FlaskConical, Loader2, MonitorPlay, Play, Power, RotateCcw, Search, Sparkles, Zap } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { mostDamagingNeuron, usePresentation } from '../../app/demos';
import { useExperiment } from '../../app/experiment';
import { useWorkspace } from '../../app/workspace';
import { classColor } from '../../forge/format';
import { useForgeStore } from '../../forge/store';
import { useTimeMachine } from '../../forge/timeMachine';
import type { ComponentRef } from '../../forge/types';
import { useI18n, useT } from '../../i18n';
import { useNetworkStore } from '../../store/networkStore';
import type { DatasetType } from '../../types';
import { DatasetPreview } from '../DatasetPreview';
import { HeatmapCanvas, type ScatterPoint } from '../Forge/charts';
import { useCheckpointEpochs, useFrame, useNeutral } from '../Forge/TimeMachine/hooks';

const DATASETS: DatasetType[] = ['Circle', 'Gaussian', 'XOR', 'Spiral'];
type Size = 'small' | 'medium' | 'large';
const SIZES: Record<Size, number[]> = { small: [4], medium: [8, 8], large: [16, 16, 16] };
const TRAIN_EPOCHS = 30;

function Card({ n, title, lead, done, active, children }: {
  n: number; title: string; lead: string; done: boolean; active: boolean; children: ReactNode;
}) {
  return (
    <section className="explore-card" data-active={active || undefined} aria-labelledby={`explore-step-${n}`}>
      <div className="flex items-start gap-3">
        <span className="explore-num" data-done={done || undefined} aria-hidden="true">{done ? <Check size={15} /> : n}</span>
        <div className="flex-1 min-w-0">
          <h2 id={`explore-step-${n}`} className="text-[17px] font-semibold tracking-tight m-0" style={{ color: 'var(--text-primary)' }}>{title}</h2>
          <p className="mt-1 text-[13.5px] leading-relaxed m-0" style={{ color: 'var(--text-muted)' }}>{lead}</p>
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Tile({ selected, onClick, title, hint, disabled }: { selected: boolean; onClick: () => void; title: string; hint: string; disabled?: boolean }) {
  return (
    <button type="button" className="explore-tile" aria-pressed={selected} onClick={onClick} disabled={disabled}>
      <span className="block text-[13.5px] font-semibold" style={{ color: 'var(--text-primary)' }}>{title}</span>
      <span className="block text-[11.5px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{hint}</span>
    </button>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-lg border px-3 py-2" style={{ borderColor: 'var(--border)', background: 'var(--bg-elevated, var(--bg-card))' }}>
      <div className="text-[10.5px] uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>{label}</div>
      <div className="font-mono tnum text-[18px] font-semibold leading-tight" style={{ color: tone ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}


export function ExploreView() {
  const t = useT();
  const lang = useI18n((s) => s.lang);
  const pctS = (v: number) => `${(v * 100).toFixed(1).replace('.', lang === 'it' ? ',' : '.')}%`;
  const session = useForgeStore((s) => s.session);
  const comparison = useForgeStore((s) => s.comparison);
  const interventions = useForgeStore((s) => s.interventions);
  const dataset = useNetworkStore((s) => s.trainingConfig.dataset);
  const custom = useNetworkStore((s) => s.customDataset);
  const graphSource = useNetworkStore((s) => s.graphSource);
  const { build, train, buildStatus, trainStatus } = useExperiment();
  const setMode = useWorkspace((s) => s.setMode);
  const startTour = usePresentation((s) => s.start);
  const cursor = useTimeMachine((s) => s.cursor);
  const epochs = useCheckpointEpochs();
  const frame = useFrame();
  const neutral = useNeutral();
  const paper = neutral[0] > 128;

  const [size, setSize] = useState<Size>('medium');
  const [search, setSearch] = useState<{ ref: ComponentRef; before: number; after: number; tried: number; key: string } | null>(null);
  const [searching, setSearching] = useState(false);

  const real = !!session && graphSource === 'model';
  const trained = real && session!.epoch > 0;
  const building = buildStatus.type === 'loading';
  const training = trainStatus.type === 'loading';
  const sessionKey = session ? `${session.session_id}@${session.epoch}` : '';
  const found = search && search.key === sessionKey ? search : null;
  const ablated = interventions.length > 0;
  const hiddenCount = session ? session.structure.layers.filter((l) => l.role === 'hidden').reduce((a, l) => a + l.size, 0) : 0;

  const create = async () => {
    const ns = useNetworkStore.getState();
    const neurons = SIZES[size];
    ns.setCustomDataset(null);
    ns.setNetworkConfig({
      model_type: 'ANN', n_layers: neurons.length,
      neurons: [...neurons, 8, 8, 8, 8].slice(0, 5), activations: ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'],
    });
    ns.setTrainingConfig({ noise: 10, learning_rate: dataset === 'Spiral' ? 0.03 : 0.01, batch_size: 32, reg_type: 'None' });
    setSearch(null);
    await useTimeMachine.getState().goLive();
    await build();
  };

  const runTraining = async () => {
    await useForgeStore.getState().reset();
    await useTimeMachine.getState().goLive();
    await train(TRAIN_EPOCHS);
  };

  const findKey = async () => {
    setSearching(true);
    try {
      await useTimeMachine.getState().goLive();
      await useForgeStore.getState().reset();
      const r = await mostDamagingNeuron();
      if (r) setSearch({ ...r, key: sessionKey });
    } finally {
      setSearching(false);
    }
  };

  const toggleOff = async () => {
    const f = useForgeStore.getState();
    if (ablated) { await f.reset(); return; }
    if (found?.ref.kind !== 'neuron') return;
    await useTimeMachine.getState().goLive();
    await f.addIntervention({ type: 'ablate_neuron', layer: found.ref.layer, index: found.ref.index });
  };

  const openLab = async () => {
    if (found) await useForgeStore.getState().select(found.ref);
    useWorkspace.getState().setRightOpen(true);
    setMode('network');
  };

  // ── the map: the checkpoint under the slider, or the live model with the what-if edit ──
  const color = useMemo(() => (v: number) => classColor(v, neutral), [neutral]);
  const showAblated = ablated && cursor === null && comparison?.boundary;
  const map = showAblated ? { values: comparison!.boundary!.intervened, x_range: comparison!.boundary!.x_range, y_range: comparison!.boundary!.y_range } : frame?.boundary ?? null;
  const predictions = showAblated ? null : frame?.predictions;
  const points: ScatterPoint[] | undefined = useMemo(() => {
    if (!session || session.structure.input_dim !== 2) return undefined;
    return session.dataset_X.map((p, i) => ({
      x: p[0], y: p[1], cls: session.dataset_y[i],
      ring: predictions && predictions[i] !== session.dataset_y[i] ? (paper ? '#111827' : '#fde047') : undefined,
    }));
  }, [session, predictions, paper]);

  const history = session?.history ?? [];
  const first = history[0];
  const last = history[history.length - 1];
  const sliderIndex = cursor === null ? epochs.length - 1 : Math.max(0, epochs.indexOf(cursor));
  const shownEpoch = cursor ?? session?.epoch ?? 0;
  const wrong = frame && session ? frame.predictions.filter((p, i) => p !== session.dataset_y[i]).length : 0;
  const accNow = showAblated ? comparison!.intervened.dataset_accuracy : frame?.accuracy;
  const names = session?.structure.class_names ?? ['0', '1'];
  const step = !real ? 1 : !trained ? 2 : 3;

  return (
    <div className="h-full overflow-y-auto explore-bg" role="region" aria-label={t.explore.region}>
      <div className="max-w-[1180px] mx-auto px-6 py-6">
        <ol className="explore-stepper" aria-label={t.explore.stepsLabel}>
          {[t.explore.step1, t.explore.step2, t.explore.step3].map((label, i) => (
            <li key={label} data-state={i + 1 < step ? 'done' : i + 1 === step ? 'active' : 'todo'} aria-current={i + 1 === step ? 'step' : undefined}>
              <span className="explore-dot">{i + 1 < step ? <Check size={12} /> : i + 1}</span>{label}
            </li>
          ))}
        </ol>

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] items-start">
          <div className="space-y-4 min-w-0">
            <Card n={1} title={t.explore.buildTitle} lead={t.explore.buildLead} done={real} active={step === 1}>
              <div className="text-[12px] font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>{t.explore.pickData}</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {DATASETS.map((d) => (
                  <Tile key={d} selected={!custom && dataset === d} title={t.datasets[d].name} hint={t.datasets[d].hint} disabled={building}
                    onClick={() => { useNetworkStore.getState().setCustomDataset(null); useNetworkStore.getState().setTrainingConfig({ dataset: d, noise: 10 }); }} />
                ))}
              </div>
              <div className="text-[12px] font-semibold mt-4 mb-2" style={{ color: 'var(--text-muted)' }}>{t.explore.pickSize}</div>
              <div className="grid grid-cols-3 gap-2">
                {(Object.keys(SIZES) as Size[]).map((k) => (
                  <Tile key={k} selected={size === k} title={t.explore.sizes[k].name} hint={t.explore.sizes[k].hint} disabled={building} onClick={() => setSize(k)} />
                ))}
              </div>
              <div className="mt-4 flex items-center gap-3 flex-wrap">
                <button type="button" className="btn-primary" onClick={() => void create()} disabled={building || training}>
                  {building ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
                  {building ? t.explore.creating : t.explore.create}
                </button>
              </div>
              {buildStatus.type === 'error' && <p role="alert" className="mt-3 text-[13px]" style={{ color: 'var(--text-neg)' }}>{buildStatus.message}</p>}
              {real && (
                <p className="mt-3 text-[13px] leading-relaxed m-0" style={{ color: 'var(--text-primary)' }}>
                  {t.explore.created(session!.structure.param_count, session!.structure.layers.map((l) => l.size).join(' → '))}
                </p>
              )}
            </Card>

            <Card n={2} title={t.explore.trainTitle} lead={t.explore.trainLead} done={trained} active={step === 2}>
              <button type="button" className={trained ? 'btn-secondary' : 'btn-primary'} onClick={() => void runTraining()} disabled={!real || training || building}>
                {training ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
                {training ? t.explore.trainingNow : trained ? t.explore.trainMore(TRAIN_EPOCHS) : t.explore.trainBtn(TRAIN_EPOCHS)}
              </button>
              {trainStatus.type === 'error' && <p role="alert" className="mt-3 text-[13px]" style={{ color: 'var(--text-neg)' }}>{trainStatus.message}</p>}
              {real && !trained && <p className="mt-3 text-[13px] m-0" style={{ color: 'var(--text-muted)' }}>{t.explore.untrained}</p>}
              {trained && first && last && (
                <>
                  <p className="mt-3 text-[13.5px] leading-relaxed m-0" style={{ color: 'var(--text-primary)' }}>
                    {t.explore.progress(first.accuracy, last.accuracy, last.epoch)}
                  </p>
                  {epochs.length > 1 && (
                    <div className="mt-4">
                      <div className="flex items-baseline justify-between text-[12px] mb-1.5" style={{ color: 'var(--text-muted)' }}>
                        <span>{t.explore.sliderLabel}</span>
                        <span className="font-mono tnum" style={{ color: cursor === null ? 'var(--tm-live-text)' : 'var(--tm-hist-text)' }}>
                          {t.explore.atEpoch(shownEpoch)}{cursor === null ? ` · ${t.explore.live}` : ''}
                        </span>
                      </div>
                      <input type="range" min={0} max={epochs.length - 1} step={1} value={sliderIndex} aria-label={t.explore.sliderLabel}
                        aria-valuetext={t.explore.atEpoch(shownEpoch)}
                        onChange={(e) => {
                          const i = +e.target.value;
                          void useTimeMachine.getState().goTo(i >= epochs.length - 1 ? null : epochs[i]);
                        }} />
                      <p className="mt-1 text-[11.5px] m-0" style={{ color: 'var(--text-faint)' }}>{t.explore.sliderHint}</p>
                    </div>
                  )}
                </>
              )}
            </Card>

            <Card n={3} title={t.explore.understandTitle} lead={t.explore.understandLead} done={!!found} active={step === 3}>
              {!trained ? (
                <p className="text-[13px] m-0" style={{ color: 'var(--text-faint)' }}>{t.explore.needTrain}</p>
              ) : (
                <>
                  <button type="button" className={found ? 'btn-secondary' : 'btn-primary'} onClick={() => void findKey()} disabled={searching || training}>
                    {searching ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
                    {searching ? t.explore.searching(hiddenCount) : t.explore.findKey}
                  </button>
                  {found && found.ref.kind === 'neuron' && (
                    <div className="mt-3 space-y-3">
                      <p className="text-[13.5px] leading-relaxed m-0" style={{ color: 'var(--text-primary)' }} data-testid="explore-key-result">
                        {found.before - found.after < 0.01
                          ? t.explore.keyNone
                          : t.explore.keyResult(t.explore.neuronName(found.ref.layer, found.ref.index), found.before, found.after, found.tried)}
                      </p>
                      <button type="button" className="btn-secondary" onClick={() => void toggleOff()}>
                        {ablated ? <RotateCcw size={15} /> : <Power size={15} />}{ablated ? t.explore.restore : t.explore.showOff}
                      </button>
                      {ablated && <p className="text-[12px] m-0" style={{ color: 'var(--tm-whatif-text)' }}>{t.explore.offNote}</p>}
                    </div>
                  )}
                </>
              )}
            </Card>

            <section className="explore-card" aria-label={t.explore.nextSteps}>
              <div className="eyebrow mb-2.5">{t.explore.nextSteps}</div>
              <div className="flex flex-wrap gap-2">
                <button type="button" className="btn-secondary" onClick={() => void openLab()} disabled={!real}><FlaskConical size={15} />{t.explore.openLab}</button>
                <button type="button" className="btn-secondary" onClick={() => setMode('3d')} disabled={!real}><Box size={15} />{t.explore.open3d}</button>
                <button type="button" className="btn-secondary" onClick={() => setMode('timemachine')} disabled={!trained}><Clock size={15} />{t.explore.openTime}</button>
                <button type="button" className="btn-ghost" onClick={() => void startTour()}><MonitorPlay size={15} />{t.explore.openTour}</button>
              </div>
            </section>
          </div>

          <aside className="explore-card lg:sticky lg:top-0" aria-label={t.explore.mapTitle}>
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={15} style={{ color: 'var(--accent)' }} aria-hidden="true" />
              <h2 className="text-[15px] font-semibold m-0" style={{ color: 'var(--text-primary)' }}>{t.explore.mapTitle}</h2>
              {real && (
                <span className={`ml-auto ${showAblated ? 'badge-orange' : cursor === null ? 'badge-green' : 'badge-blue'}`}>
                  {showAblated ? 'WHAT-IF' : t.explore.atEpoch(shownEpoch)}
                </span>
              )}
            </div>
            {!real ? (
              <DatasetPreview />
            ) : map ? (
              <HeatmapCanvas ariaLabel={t.explore.mapAria(shownEpoch)} values={map.values} color={color}
                xRange={map.x_range} yRange={map.y_range} points={points} contour={0.5}
                contourColor={paper ? 'rgba(17,24,39,0.85)' : 'rgba(255,255,255,0.9)'}
                pointStroke={paper ? 'rgba(17,24,39,0.5)' : 'rgba(255,255,255,0.55)'} height={420} />
            ) : session && session.structure.input_dim !== 2 ? (
              <p className="text-[13px]" style={{ color: 'var(--text-muted)' }}>{t.explore.notTwoD}</p>
            ) : (
              <div className="h-[420px] flex items-center justify-center"><Loader2 size={18} className="animate-spin" style={{ color: 'var(--text-faint)' }} /></div>
            )}
            {real && (
              <p className="mt-2 text-[11.5px] leading-relaxed m-0" style={{ color: 'var(--text-faint)' }}>
                {t.explore.mapCaption(names[0], names[1])}
              </p>
            )}
            {real && frame && accNow !== undefined && (
              <>
                <div className="mt-3 grid grid-cols-3 gap-2">
                  <Stat label={t.explore.accuracy} value={pctS(accNow)} tone="var(--tm-acc)" />
                  <Stat label={t.explore.mistakes} value={showAblated ? '—' : `${wrong}/${frame.predictions.length}`} />
                  <Stat label={t.explore.loss} value={showAblated ? '—' : frame.loss.toFixed(3)} tone="var(--tm-loss)" />
                </div>
                {!showAblated && (
                  <p className="mt-3 text-[13px] leading-relaxed m-0" style={{ color: 'var(--text-primary)' }}>
                    {t.explore.frameSentence(frame.accuracy, wrong, frame.predictions.length)}
                  </p>
                )}
              </>
            )}
            <p className="mt-3 text-[11px] m-0" style={{ color: 'var(--text-faint)' }}>{t.explore.honest}</p>
          </aside>
        </div>
      </div>
    </div>
  );
}
