import { ArrowLeftRight, ChevronFirst, ChevronLast, Loader2, Pause, Play, Route, StepBack, StepForward } from 'lucide-react';
import { useCallback, useEffect, useMemo } from 'react';
import { useWorkspace } from '../../../app/workspace';
import { useExplorer } from '../../../forge/explorer';
import { buildSteps, focusNeuron, stepIndexOf, type PassDirection } from '../../../forge/passExplorer';
import { useForgeStore } from '../../../forge/store';
import { ModelStrip } from '../../Workspaces/NetworkWorkspace';
import { NeedsModel } from '../../Workspaces/EmptyState';
import { useElementSize, useElementWidth } from '../hooks';
import { SignalMap } from './SignalMap';
import { StepDetail } from './StepDetail';

const LR_CHOICES = [0.01, 0.05, 0.1, 0.5];

function useExplorerKeys(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const ex = useExplorer.getState();
      if (e.key === 'ArrowRight') { e.preventDefault(); ex.step(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); ex.step(-1); }
      else if (e.key === ' ' && (!t || t.tagName !== 'BUTTON')) { e.preventDefault(); ex.togglePlay(); }
      else if (e.key === 'Home') { e.preventDefault(); ex.first(); }
      else if (e.key === 'End') { e.preventDefault(); ex.last(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [enabled]);
}

export function PassExplorer() {
  const session = useForgeStore((s) => s.session);
  const trace = useForgeStore((s) => s.trace);
  const loading = useForgeStore((s) => s.loading);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const lab = useForgeStore((s) => s.mode === 'lab');
  const previewLr = useForgeStore((s) => s.previewLr);
  const setPreviewLr = useForgeStore((s) => s.setPreviewLr);
  const checkpointEpoch = useForgeStore((s) => s.checkpointEpoch);
  const nIv = useForgeStore((s) => s.interventions.length);
  const { dir, stepId, playing, compareEpoch, compareTrace, compareLoading, compareError } = useExplorer();
  const ex = useExplorer.getState;
  const active = useWorkspace((s) => s.mode === 'explorer');
  const [rootRef, width] = useElementWidth<HTMLDivElement>(900);
  const [mapRef, mapSize] = useElementSize<HTMLDivElement>({ w: 600, h: 420 });
  useExplorerKeys(active && !!trace);

  const steps = useMemo(() => (trace ? buildSteps(trace, dir) : []), [trace, dir]);
  const idx = stepIndexOf(steps, stepId[dir]);
  const step = steps[idx];
  const selNeuron = selection && selection.kind === 'neuron' ? selection : null;
  const focus = trace && step ? focusNeuron(trace, step, selNeuron) : 0;
  const onPick = useCallback((layer: number, index: number) => void select({ kind: 'neuron', layer, index }), [select]);
  const wide = width >= 980;

  if (!session) {
    return (
      <NeedsModel icon={<Route size={20} />} view='explorer' />
    );
  }

  const epochs = session.checkpoints.map((c) => c.epoch);
  const viewing = checkpointEpoch ?? session.epoch;
  const validCompare = compareEpoch !== null && compareEpoch !== viewing ? compareEpoch : null;

  return (
    <div className="h-full flex flex-col min-h-0" ref={rootRef}>
      <ModelStrip />

      {/* transport */}
      <div className="flex items-center gap-2 flex-wrap px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
        <div className="seg" role="group" aria-label="Pass direction">
          {(['forward', 'backward'] as PassDirection[]).map((d) => (
            <button key={d} type="button" aria-pressed={dir === d} onClick={() => ex().setDir(d)}>
              {d === 'forward' ? 'Forward pass →' : '← Backward pass'}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1" role="group" aria-label="Step controls">
          <button type="button" className="btn-ghost p-1.5" aria-label="First step" onClick={() => ex().first()} disabled={idx === 0}><ChevronFirst size={16} /></button>
          <button type="button" className="btn-ghost p-1.5" aria-label="Previous step" onClick={() => ex().step(-1)} disabled={idx === 0}><StepBack size={16} /></button>
          <button type="button" className="btn-secondary !py-1 !px-2.5 !text-xs" onClick={() => ex().togglePlay()} aria-label={playing ? 'Pause' : 'Play the pass'}>
            {playing ? <Pause size={13} /> : <Play size={13} />}{playing ? 'Pause' : 'Play'}
          </button>
          <button type="button" className="btn-ghost p-1.5" aria-label="Next step" onClick={() => ex().step(1)} disabled={idx >= steps.length - 1}><StepForward size={16} /></button>
          <button type="button" className="btn-ghost p-1.5" aria-label="Last step" onClick={() => ex().last()} disabled={idx >= steps.length - 1}><ChevronLast size={16} /></button>
        </div>
        <span className="text-[12px] tnum" style={{ color: 'var(--text-muted)' }} aria-live="polite">
          Step <b style={{ color: 'var(--text-primary)' }}>{idx + 1}</b> / {steps.length}
        </span>
        {loading && <Loader2 size={13} className="animate-spin" style={{ color: 'var(--text-faint)' }} />}

        <div className="ml-auto flex items-center gap-2 text-[12px]" style={{ color: 'var(--text-muted)' }}>
          <label className="flex items-center gap-1.5" title="Show the same computation at another stored checkpoint, with differences">
            <ArrowLeftRight size={13} />compare with
            <select className="select-base !w-auto !py-1 !text-xs tnum" value={compareEpoch ?? ''} aria-label="Compare with epoch"
              onChange={(e) => void ex().setCompareEpoch(e.target.value === '' ? null : +e.target.value)}>
              <option value="">— none —</option>
              {epochs.filter((e) => e !== viewing).map((e) => <option key={e} value={e}>epoch {e}</option>)}
            </select>
          </label>
          {dir === 'backward' && (
            <label className="flex items-center gap-1.5" title="Learning rate used to show −η·gradient and the one-step SGD preview">
              η
              <select className="select-base !w-auto !py-1 !text-xs tnum" value={previewLr} aria-label="Learning rate for the update preview"
                onChange={(e) => void setPreviewLr(+e.target.value)}>
                {LR_CHOICES.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          )}
        </div>
      </div>

      {/* pipeline */}
      <ol className="flex items-center gap-1 px-3 py-2 overflow-x-auto flex-shrink-0 m-0 list-none border-b" style={{ borderColor: 'var(--border)' }} aria-label="Steps">
        {steps.map((s, i) => (
          <li key={s.id} className="flex items-center gap-1 flex-shrink-0">
            {i > 0 && <span aria-hidden="true" style={{ color: 'var(--text-faint)' }}>{dir === 'forward' ? '›' : '‹'}</span>}
            <button type="button" onClick={() => ex().goTo(s.id)} aria-current={i === idx ? 'step' : undefined}
              className="px-2 py-1 rounded-md text-[12px] whitespace-nowrap transition-colors border"
              style={i === idx
                ? { background: 'var(--accent)', color: '#fff', borderColor: 'var(--accent)' }
                : { color: i < idx ? 'var(--text-primary)' : 'var(--text-muted)', borderColor: 'var(--border)', background: i < idx ? 'var(--bg-raised)' : 'transparent' }}>
              {s.short}
            </button>
          </li>
        ))}
      </ol>

      {trace && step ? (
        <div className={`flex-1 min-h-0 ${wide ? 'grid grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]' : 'flex flex-col overflow-y-auto'}`}>
          <section className={`${wide ? 'border-r overflow-hidden' : 'border-b flex-shrink-0'} flex flex-col min-h-0`} style={{ borderColor: 'var(--border)' }}>
            <div className="px-3 pt-2.5 flex items-baseline gap-2 flex-wrap">
              <h3 className="text-[14px] font-semibold m-0" style={{ color: 'var(--text-primary)' }}>{step.title}</h3>
              <span className="text-[11.5px]" style={{ color: 'var(--text-faint)' }}>
                {dir === 'forward' ? 'cells = real values' : 'cells = real gradients'} · epoch {trace.provenance.checkpoint_epoch}
                {nIv ? ` · what-if ×${nIv}` : ''} · probe {trace.probe.sample_index !== null ? `#${trace.probe.sample_index}` : 'free point'}
              </span>
            </div>
            <div className="flex-1 min-h-[300px] px-2" ref={mapRef} style={wide ? undefined : { height: 300 }}>
              <SignalMap trace={trace} step={step} focus={focus} onPick={onPick} height={Math.max(280, mapSize.h)} />
            </div>
            <p className="px-3 pb-2 text-[11.5px]" style={{ color: 'var(--text-faint)' }}>
              Moving lines: the {dir === 'forward' ? 'strongest real contributions w·a' : 'strongest real backpropagated terms w·δ'} into the active layer
              (top 24). Green = positive, red = negative. Click a cell to focus that neuron (also selects it in the Microscope and 3D).
            </p>
          </section>
          <section className={`${wide ? 'overflow-y-auto' : ''} p-3 space-y-3`} aria-live="polite">
            {validCompare !== null && (
              <div className="text-[12px] px-2.5 py-1.5 rounded-md border flex items-center gap-2"
                style={{ borderColor: 'var(--tm-hist)', color: 'var(--tm-hist-text)' }}>
                <ArrowLeftRight size={13} />
                {compareLoading ? 'Loading the same pass at epoch ' + validCompare + '…'
                  : compareError ? compareError
                    : <>Comparing with the same probe at epoch {validCompare}: extra columns show its values and the change.</>}
              </div>
            )}
            <StepDetail t={trace} step={step} focus={focus} lab={lab} onPick={onPick}
              cmp={validCompare !== null ? compareTrace : null} cmpEpoch={validCompare} lr={previewLr} />
          </section>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center" style={{ color: 'var(--text-faint)' }}><Loader2 size={18} className="animate-spin" /></div>
      )}
    </div>
  );
}
