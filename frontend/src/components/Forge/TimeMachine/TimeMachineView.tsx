import { GitCompareArrows, HeartPulse, History, Loader2, RotateCcw, Spline } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { highlightFor, layerLabels as buildLabels, refForNode } from '../../../forge/selection';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { resolveEpoch } from '../../../forge/timeline';
import { useWorkspace } from '../../../app/workspace';
import type { NetworkNode } from '../../../types';
import { NetworkGraph } from '../../Visualizations/NetworkGraph';
import { NeedsModel } from '../../Workspaces/EmptyState';
import { useElementWidth } from '../hooks';
import { BoundaryStage } from './BoundaryStage';
import { EpochCompareView } from './EpochCompareView';
import { HealthPanel } from './HealthPanel';
import { StatePill } from './StatePill';
import { ThroughTime } from './ThroughTime';
import { TimelineInstrument, Transport } from './TimelineInstrument';
import { useTimelineData, useTransportKeys } from './hooks';

type SideTab = 'time' | 'health';

function NetworkPane() {
  const session = useForgeStore((s) => s.session);
  const graph = useForgeStore((s) => s.graph);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const loading = useForgeStore((s) => s.loading);
  const cursor = useTimeMachine((s) => s.cursor);
  const labels = useMemo(() => (session ? buildLabels(session.structure) : undefined), [session]);
  const highlight = useMemo(() => (graph ? highlightFor(graph, selection) : null), [graph, selection]);
  const onNodeClick = useCallback((node: NetworkNode) => {
    const ref = refForNode(node);
    if (ref) void select(ref);
  }, [select]);
  const onLayerClick = useCallback((layer: number) => void select({ kind: 'layer', layer }), [select]);
  if (!session) return null;
  const shown = graph?.provenance.checkpoint_epoch;
  const syncing = loading || (graph !== null && shown !== resolveEpoch(cursor, session.epoch));

  return (
    <div className="h-full min-h-[260px] rounded-xl overflow-hidden border relative" style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}>
      {graph && (
        <NetworkGraph
          graph={graph}
          mode="signal"
          onNodeClick={onNodeClick}
          onLayerClick={onLayerClick}
          selectedNodeIds={highlight?.nodeIds}
          selectedEdgeIds={highlight?.edgeIds}
          selectedLayer={highlight?.layer ?? null}
          layerLabels={labels}
          showHint={false}
        />
      )}
      <div className="absolute bottom-2 left-2 text-[10px] px-2 py-1 rounded-md border flex items-center gap-1.5"
        style={{ background: 'var(--tm-overlay)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
        {syncing && <Loader2 size={10} className="animate-spin" />}
        network at epoch {shown ?? '…'} · edge = w·a on the probe · click a neuron to follow it
      </div>
    </div>
  );
}

export function TimeMachineView() {
  const session = useForgeStore((s) => s.session);
  const mode = useForgeStore((s) => s.mode);
  const interventions = useForgeStore((s) => s.interventions);
  const resetInterventions = useForgeStore((s) => s.reset);
  const workspaceMode = useWorkspace((s) => s.mode);
  const cursor = useTimeMachine((s) => s.cursor);
  const compareMode = useTimeMachine((s) => s.compareMode);
  const setCompareMode = useTimeMachine((s) => s.setCompareMode);
  const error = useTimeMachine((s) => s.error);
  const [side, setSide] = useState<SideTab>('time');
  const [rootRef, width] = useElementWidth<HTMLDivElement>(900);
  useTimelineData();
  useTransportKeys(!!session && workspaceMode === 'timemachine');
  const wide = width >= 860;

  if (!session) {
    return (
      <NeedsModel icon={<History size={20} />} title="Watch a network learn"
        body={<>Every training epoch stores an immutable checkpoint of the real model. Build and train a network, then travel through its learning process here.</>} />
    );
  }
  const lab = mode === 'lab';
  const epoch = resolveEpoch(cursor, session.epoch);
  const untrained = session.epoch === 0;
  const sideTab = side === 'health' && !lab ? 'time' : side;

  return (
    <div ref={rootRef} className="h-full min-h-0 overflow-y-auto">
      <div className="flex flex-col gap-3 p-3">
        {/* status row */}
        <div className="flex items-center gap-2 flex-wrap">
          <StatePill epoch={epoch} historical={cursor !== null} size="md" />
          <span className="text-[12px] tnum" style={{ color: 'var(--text-faint)' }}>
            {session.checkpoints.length} stored checkpoints · {session.history.length} logged epochs · {session.dataset_name}
          </span>
          {error && <span className="text-[12px]" role="alert" style={{ color: 'var(--text-neg)' }}>{error}</span>}
          <button type="button" onClick={() => void setCompareMode(!compareMode)} aria-pressed={compareMode} disabled={untrained}
            className="ml-auto btn-secondary !py-1 !px-2.5 !text-xs"
            style={compareMode ? { borderColor: 'var(--tm-b)', color: '#fff', background: 'var(--tm-b)' } : undefined}>
            <GitCompareArrows size={13} />Compare A ↔ B
          </button>
        </div>

        {interventions.length > 0 && (
          <div className="flex items-center gap-2 text-[12px] px-3 py-2 rounded-md border"
            style={{ borderColor: 'var(--tm-whatif)', color: 'var(--tm-whatif-text)', background: 'color-mix(in srgb, var(--tm-whatif) 8%, transparent)' }}>
            <span>
              What-if overlay active ({interventions.length} edit{interventions.length > 1 ? 's' : ''}): applied temporarily to the graph, Microscope,
              Pass Explorer and 3D view at whichever epoch you view. Stored checkpoints, the timeline, decision regions and comparisons are unaffected.
            </span>
            <button type="button" className="ml-auto btn-secondary !py-1 !px-2 !text-[11px] flex-shrink-0" onClick={() => void resetInterventions()}>
              <RotateCcw size={11} />Reset what-if
            </button>
          </div>
        )}

        {/* the instrument */}
        <div className="card px-3 pt-2.5 pb-3 space-y-2">
          {untrained ? (
            <p className="text-[13px] py-3 px-1" style={{ color: 'var(--text-muted)' }}>
              Only epoch 0 (random initialisation) exists so far. Click <b>Train</b> in the Experiment panel: every epoch is stored and appears here as a checkpoint you can visit.
            </p>
          ) : (
            <>
              <TimelineInstrument lab={lab} />
              <Transport />
              <div className="flex gap-x-3 gap-y-1 text-[11px] flex-wrap" style={{ color: 'var(--text-faint)' }}>
                <span style={{ color: 'var(--tm-loss)' }}>━ loss</span>
                <span style={{ color: 'var(--tm-acc)' }}>━ accuracy</span>
                {lab && <span style={{ color: 'var(--tm-grad)' }}>━ training ‖∇‖ (log)</span>}
                <span>| ticks = stored checkpoints (curves = full training log)</span>
                <span>◆ init · ▸ run · ★ ≥90% · ▲ best acc · ▼ min loss · ⬇ biggest drop</span>
                <span className="ml-auto"><kbd>Space</kbd> play · <kbd>←</kbd>/<kbd>→</kbd> step · <kbd>Home</kbd>/<kbd>End</kbd></span>
              </div>
            </>
          )}
        </div>

        {/* stage */}
        {compareMode ? (
          <div className="min-h-[520px]"><EpochCompareView /></div>
        ) : (
          <>
            <div className="grid gap-3" style={{ gridTemplateColumns: wide ? 'minmax(0,1fr) minmax(0,1fr)' : 'minmax(0,1fr)' }}>
              <div className="min-w-0"><BoundaryStage /></div>
              <div className="min-w-0" style={{ height: wide ? 'auto' : 380, minHeight: 360 }}><NetworkPane /></div>
            </div>
            <section className="card overflow-hidden">
              <div className="flex items-center gap-0.5 px-2 pt-1.5 border-b" style={{ borderColor: 'var(--border)' }} role="tablist" aria-label="Time Machine details">
                {([
                  ['time', 'Selected component through time', Spline],
                  ...(lab ? [['health', 'Training health', HeartPulse] as const] : []),
                ] as const).map(([id, label, Icon]) => (
                  <button key={id} type="button" role="tab" aria-selected={sideTab === id} onClick={() => setSide(id)}
                    className="flex items-center gap-1.5 px-2.5 py-2 text-[12px] -mb-px border-b-2"
                    style={{ borderColor: sideTab === id ? 'var(--accent)' : 'transparent', color: sideTab === id ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    <Icon size={13} />{label}
                  </button>
                ))}
                {!lab && <span className="ml-auto text-[11px] pr-1" style={{ color: 'var(--text-faint)' }}>Lab mode adds training health</span>}
              </div>
              <div className="p-3">{sideTab === 'time' ? <ThroughTime /> : <HealthPanel />}</div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
