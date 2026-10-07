import { FlaskRound, GitCompareArrows, GraduationCap, HeartPulse, Info, Loader2, Microscope, RotateCcw, Spline } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { highlightFor, layerLabels as buildLabels, refForNode } from '../../../forge/selection';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { resolveEpoch } from '../../../forge/timeline';
import { useNetworkStore } from '../../../store/networkStore';
import type { NetworkNode } from '../../../types';
import { NetworkGraph } from '../../Visualizations/NetworkGraph';
import { Inspector } from '../Inspector';
import { BoundaryStage } from './BoundaryStage';
import { EpochCompareView } from './EpochCompareView';
import { HealthPanel } from './HealthPanel';
import { StatePill } from './StatePill';
import { ThroughTime } from './ThroughTime';
import { TimelineInstrument, Transport } from './TimelineInstrument';
import { useTimelineData, useTransportKeys } from './hooks';

type SideTab = 'time' | 'microscope' | 'health';

function Empty({ modelType }: { modelType: string }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="max-w-md text-center space-y-3 px-6">
        <div className="w-12 h-12 mx-auto rounded-xl flex items-center justify-center" style={{ background: 'color-mix(in srgb, var(--tm-hist) 15%, transparent)' }}>
          <Info size={22} style={{ color: 'var(--tm-hist)' }} />
        </div>
        {modelType !== 'ANN' ? (
          <>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>The Training Time Machine works with ANN (MLP) models</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              {modelType} graphs are illustrative diagrams with no real training history to replay. Switch to <b>ANN</b>, click
              <b> Build Network</b>, then <b>Train Model</b>.
            </p>
          </>
        ) : (
          <>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Build and train a network first</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Every training epoch stores a checkpoint of the real model. Click <b>Build Network</b> and <b>Train Model</b>, then come
              back here to travel through its learning process.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

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
  const setMode = useForgeStore((s) => s.setMode);
  const interventions = useForgeStore((s) => s.interventions);
  const resetInterventions = useForgeStore((s) => s.reset);
  const modelType = useNetworkStore((s) => s.networkConfig.model_type);
  const activeTab = useNetworkStore((s) => s.activeTab);
  const cursor = useTimeMachine((s) => s.cursor);
  const compareMode = useTimeMachine((s) => s.compareMode);
  const setCompareMode = useTimeMachine((s) => s.setCompareMode);
  const error = useTimeMachine((s) => s.error);
  const [side, setSide] = useState<SideTab>('time');
  useTimelineData();
  useTransportKeys(!!session && activeTab === 'timemachine');

  if (!session || modelType !== 'ANN') return <Empty modelType={modelType} />;
  const lab = mode === 'lab';
  const epoch = resolveEpoch(cursor, session.epoch);
  const untrained = session.epoch === 0;
  const sideTab = side === 'health' && !lab ? 'time' : side;

  return (
    <div className="h-full min-h-0 flex flex-col gap-2 overflow-y-auto 2xl:overflow-hidden">
      {/* status row */}
      <div className="flex items-center gap-2 flex-wrap flex-shrink-0">
        <StatePill epoch={epoch} historical={cursor !== null} size="md" />
        <span className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
          {session.checkpoints.length} stored checkpoints · {session.history.length} logged epochs · {session.dataset_name}
        </span>
        {error && <span className="text-[11px]" style={{ color: 'var(--tm-neg)' }}>{error}</span>}
        <button type="button" onClick={() => void setCompareMode(!compareMode)} aria-pressed={compareMode} disabled={untrained}
          className="ml-auto flex items-center gap-1 px-2 py-1 rounded-md border text-[11px] font-medium disabled:opacity-40"
          style={compareMode
            ? { borderColor: 'var(--tm-b)', color: '#fff', background: 'var(--tm-b)' }
            : { borderColor: 'var(--border-soft)', color: 'var(--text-muted)' }}>
          <GitCompareArrows size={12} />Compare A ↔ B
        </button>
        <div className="flex items-center p-0.5 rounded-lg border" style={{ borderColor: 'var(--border)' }} role="group" aria-label="Experience mode">
          {([['learn', 'Learn', GraduationCap], ['lab', 'Lab', FlaskRound]] as const).map(([m, label, Icon]) => (
            <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
              className="flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors"
              style={{ background: mode === m ? 'var(--accent)' : 'transparent', color: mode === m ? '#fff' : 'var(--text-muted)' }}>
              <Icon size={12} />{label}
            </button>
          ))}
        </div>
      </div>

      {interventions.length > 0 && (
        <div className="flex items-center gap-2 text-[11px] px-2.5 py-1.5 rounded-lg border flex-shrink-0"
          style={{ borderColor: 'var(--tm-whatif)', color: 'var(--tm-whatif-text)', background: 'color-mix(in srgb, var(--tm-whatif) 8%, transparent)' }}>
          <span>
            What-if overlay active ({interventions.length} edit{interventions.length > 1 ? 's' : ''}): it is applied temporarily to the network graph
            and Microscope at whichever epoch you view. Stored checkpoints, the timeline, decision regions and comparisons are unaffected.
          </span>
          <button type="button" className="ml-auto btn-secondary !py-0.5 !px-2 text-[11px] flex items-center gap-1 flex-shrink-0" onClick={() => void resetInterventions()}>
            <RotateCcw size={11} />Reset what-if
          </button>
        </div>
      )}

      {/* the instrument */}
      <div className="rounded-xl border px-2.5 pt-2 pb-2.5 flex-shrink-0 space-y-1.5" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        {untrained ? (
          <p className="text-[12px] py-3 px-1" style={{ color: 'var(--text-muted)' }}>
            Only epoch 0 (random initialisation) exists so far. Click <b>Train Model</b> in the sidebar: every epoch is stored and appears here as a checkpoint you can visit.
          </p>
        ) : (
          <>
            <TimelineInstrument lab={lab} />
            <Transport />
            <div className="flex gap-3 text-[9.5px] flex-wrap" style={{ color: 'var(--text-faint)' }}>
              <span style={{ color: 'var(--tm-loss)' }}>━ loss</span>
              <span style={{ color: 'var(--tm-acc)' }}>━ accuracy</span>
              {lab && <span style={{ color: 'var(--tm-grad)' }}>━ training ‖∇‖ (log)</span>}
              <span>| ticks = stored checkpoints (curves = full training log)</span>
              <span>◆ init · ▸ run · ★ ≥90% · ▲ best acc · ▼ min loss · ⬇ biggest drop</span>
              <span className="ml-auto">Space play/pause · ←/→ step · Home/End</span>
            </div>
          </>
        )}
      </div>

      {/* stage */}
      {compareMode ? (
        <div className="flex-1 flex-shrink-0 min-h-[480px] 2xl:flex-shrink 2xl:min-h-0"><EpochCompareView /></div>
      ) : (
        <div className="flex-shrink-0 min-h-0 grid gap-2 grid-cols-1 lg:grid-cols-2 2xl:flex-1 2xl:flex-shrink 2xl:grid-cols-[minmax(320px,1fr)_minmax(280px,1fr)_minmax(320px,380px)]">
          <div className="min-h-0 2xl:overflow-y-auto pr-0.5"><BoundaryStage /></div>
          <NetworkPane />
          <aside className="h-[520px] 2xl:h-auto 2xl:min-h-0 flex flex-col rounded-xl border overflow-hidden lg:col-span-2 2xl:col-span-1"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
            <div className="flex items-center gap-0.5 px-1.5 pt-1.5 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }} role="tablist">
              {([
                ['time', 'Through time', Spline],
                ['microscope', 'Microscope', Microscope],
                ...(lab ? [['health', 'Health', HeartPulse] as const] : []),
              ] as const).map(([id, label, Icon]) => (
                <button key={id} type="button" role="tab" aria-selected={sideTab === id} onClick={() => setSide(id)}
                  className="flex items-center gap-1 px-2 py-1.5 text-[11px] rounded-t-md -mb-px border-b-2"
                  style={{ borderColor: sideTab === id ? 'var(--accent)' : 'transparent', color: sideTab === id ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  <Icon size={12} />{label}
                </button>
              ))}
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto">
              {sideTab === 'microscope'
                ? <Inspector structure={session.structure} />
                : <div className="p-3">{sideTab === 'time' ? <ThroughTime /> : <HealthPanel />}</div>}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
