import { Box, Film, Loader2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import * as forgeApi from '../../forge/api';
import { useForgeStore } from '../../forge/store';
import { useNetworkStore } from '../../store/networkStore';
import type { LossLandscapeData, TabId } from '../../types';
import { AttentionHeatmap } from '../Visualizations/AttentionHeatmap';
import { CompareView } from '../Visualizations/CompareView';
import { CustomActivation } from '../Visualizations/CustomActivation';
import { DecisionBoundary } from '../Visualizations/DecisionBoundary';
import { ExportCode } from '../Visualizations/ExportCode';
import { HyperparamSweep } from '../Visualizations/HyperparamSweep';
import { LayerActivationHeatmap } from '../Visualizations/LayerActivationHeatmap';
import { LossLandscape } from '../Visualizations/LossLandscape';
import { Network3DView } from '../Visualizations/Network3DView';
import { NetworkGraph } from '../Visualizations/NetworkGraph';
import { PropagationView } from '../Visualizations/PropagationView';
import { PruningView } from '../Visualizations/PruningView';
import { RealTraining } from '../Visualizations/RealTraining';
import { TrainingCurve } from '../Visualizations/TrainingCurve';
import { WeightHistogram } from '../Visualizations/WeightHistogram';

/**
 * Provenance classes used throughout the audit (docs/AUDIT_AND_ROADMAP.md):
 *   REAL         computed by a real model (the Forge session, or a TF.js model in the browser)
 *   ILLUSTRATIVE a diagram / placeholder values, not computed from the user's model
 *   SYNTHETIC    a formula made to look like model output
 */
type Kind = 'REAL' | 'ILLUSTRATIVE' | 'SYNTHETIC' | 'TOOL';

interface Provenance { kind: Kind; note: string }

const KIND_CLASS: Record<Kind, string> = {
  REAL: 'badge-green', ILLUSTRATIVE: 'badge-orange', SYNTHETIC: 'badge-pink', TOOL: 'badge-gray',
};

interface Item { id: TabId; label: string; group: string }

const ITEMS: Item[] = [
  { id: 'training', label: 'Training curves', group: 'Model analysis' },
  { id: 'decision', label: 'Decision boundary', group: 'Model analysis' },
  { id: 'loss', label: 'Loss landscape', group: 'Model analysis' },
  { id: 'weights', label: 'Weight distribution', group: 'Model analysis' },
  { id: 'layer-act', label: 'Layer activations', group: 'Model analysis' },
  { id: 'pruning', label: 'Activation threshold view', group: 'Model analysis' },
  { id: 'architecture', label: 'Architecture diagram', group: 'Diagrams' },
  { id: 'forward', label: 'Forward step animation', group: 'Diagrams' },
  { id: 'backward', label: 'Backprop step animation', group: 'Diagrams' },
  { id: 'attention', label: 'Attention pattern', group: 'Diagrams' },
  { id: 'live-train', label: 'Live training (TF.js)', group: 'In-browser lab' },
  { id: 'sweep', label: 'Learning-rate sweep', group: 'In-browser lab' },
  { id: 'custom-act', label: 'Custom activation', group: 'In-browser lab' },
  { id: 'compare', label: 'Compare architectures', group: 'Tools' },
  { id: 'export', label: 'Export code', group: 'Tools' },
];
const GROUPS = ['Model analysis', 'Diagrams', 'In-browser lab', 'Tools'];

function useProvenance(tab: TabId): Provenance {
  const graphSource = useNetworkStore((s) => s.graphSource);
  const built = useNetworkStore((s) => s.networkBuilt);
  const training = useNetworkStore((s) => s.trainingResult);
  const decision = useNetworkStore((s) => s.decisionBoundary);
  const session = useForgeStore((s) => s.session);
  const realGraph = built && graphSource === 'model';
  switch (tab) {
    case 'architecture': case 'forward': case 'backward': case 'weights': case 'layer-act': case 'pruning':
      return realGraph
        ? { kind: 'REAL', note: 'Values come from the session model (probe, checkpoint and what-if as selected).' }
        : { kind: 'ILLUSTRATIVE', note: 'Placeholder values for a diagram; no model computed them.' };
    case 'training':
      return training?.source === 'pytorch'
        ? { kind: 'REAL', note: 'Loss and accuracy logged by real PyTorch training of the session model.' }
        : { kind: 'SYNTHETIC', note: 'A closed-form curve with noise. No model was trained.' };
    case 'decision':
      return decision?.source === 'trained-session'
        ? { kind: 'REAL', note: 'P(class 1) of the session model after its last training run.' }
        : { kind: 'ILLUSTRATIVE', note: 'A freshly initialised, untrained model of this architecture — not your model.' };
    case 'loss':
      return session
        ? { kind: 'REAL', note: 'A 2-D slice of the real dataset loss around the session weights (random filter-normalised directions).' }
        : { kind: 'ILLUSTRATIVE', note: 'Around a randomly initialised model of the architecture, on the Circle data — not your model.' };
    case 'attention':
      return { kind: 'SYNTHETIC', note: 'A formula-generated pattern, not computed by any model. The real attention matrices are in Transformer Lab.' };
    case 'live-train': case 'sweep': case 'custom-act':
      return { kind: 'REAL', note: 'A separate TensorFlow.js model trained in your browser — not the Forge session model.' };
    default:
      return { kind: 'TOOL', note: 'Generated from the current configuration.' };
  }
}

function SessionLandscape() {
  const session = useForgeStore((s) => s.session);
  const checkpoint = useForgeStore((s) => s.checkpointEpoch);
  const [state, setState] = useState<{ key: string; data?: forgeApi.SessionLossLandscape; error?: string } | null>(null);
  const key = session ? `${session.session_id}@${session.epoch}:${checkpoint}` : '';

  useEffect(() => {
    if (!session) return;
    const ctrl = new AbortController();
    forgeApi.fetchLossLandscape(session.session_id, checkpoint, ctrl.signal)
      .then((data) => setState({ key, data }))
      .catch((err) => { if (!forgeApi.isCancel(err)) setState({ key, error: forgeApi.errorMessage(err) }); });
    return () => ctrl.abort();
  }, [session, checkpoint, key]);

  const data: LossLandscapeData | null = useMemo(() => {
    const d = state?.key === key ? state.data : undefined;
    if (!d) return null;
    // LossLandscape draws z[i][j] at (x[i][j], y[i][j]) with rows along y.
    return {
      w1: d.betas.map(() => d.alphas),
      w2: d.betas.map((b) => d.alphas.map(() => b)),
      loss: d.betas.map((_, j) => d.alphas.map((__, i) => d.loss[i][j])),
    };
  }, [state, key]);

  if (state?.key === key && state.error) return <p className="text-sm" style={{ color: 'var(--text-neg)' }}>{state.error}</p>;
  const d = state?.key === key ? state.data : undefined;
  return (
    <div className="h-full flex flex-col gap-2">
      <div className="text-[12px] tnum flex flex-wrap gap-x-4" style={{ color: 'var(--text-muted)' }}>
        {d ? (
          <>
            <span>{d.is_latest ? 'Live weights' : 'Stored checkpoint'} · epoch {d.checkpoint_epoch}</span>
            <span>loss at centre <b className="font-mono" style={{ color: 'var(--text-primary)' }}>{d.center_loss.toFixed(4)}</b></span>
            <span>range {d.min_loss.toFixed(3)} – {d.max_loss.toFixed(3)}</span>
          </>
        ) : <span className="flex items-center gap-1.5"><Loader2 size={13} className="animate-spin" />Evaluating 625 perturbed models…</span>}
      </div>
      <div className="flex-1 min-h-0"><LossLandscape data={data} axisTitles={AXES} /></div>
    </div>
  );
}
const AXES: [string, string] = ['α (direction 1)', 'β (direction 2)'];

// Module level so it keeps its identity across renders (no remounts).
function LegacyNetworkView(p: { activeNodeIds?: Set<number>; activeEdgeIds?: Set<number>; gradients?: Record<string, number>; mode?: 'architecture' | 'forward' | 'backward' }) {
  const g = useNetworkStore((s) => s.graph);
  const view3D = useNetworkStore((s) => s.view3D);
  return view3D
    ? <Network3DView graph={g} activeNodeIds={p.activeNodeIds} activeEdgeIds={p.activeEdgeIds} mode={p.mode} />
    : <NetworkGraph graph={g} activeNodeIds={p.activeNodeIds} activeEdgeIds={p.activeEdgeIds} gradients={p.gradients} mode={p.mode} />;
}

function DiagramView({ mode }: { mode: 'architecture' | 'forward' | 'backward' }) {
  const graph = useNetworkStore((s) => s.graph);
  const view3D = useNetworkStore((s) => s.view3D);
  const setView3D = useNetworkStore((s) => s.setView3D);
  const fwd = useNetworkStore((s) => s.forwardSteps);
  const bwd = useNetworkStore((s) => s.backwardSteps);
  return (
    <div className="h-full flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div className="seg" role="group" aria-label="Diagram rendering">
          <button type="button" aria-pressed={!view3D} onClick={() => setView3D(false)}>2D</button>
          <button type="button" aria-pressed={view3D} onClick={() => setView3D(true)}><span className="inline-flex items-center gap-1"><Box size={12} />3D</span></button>
        </div>
        <span className="text-[12px]" style={{ color: 'var(--text-faint)' }}>
          {mode === 'architecture' ? 'Layout of the configured architecture.' : 'Step highlighting only. For the real per-step numbers use Forward / Backward.'}
        </span>
      </div>
      <div className="flex-1 min-h-0 rounded-md border overflow-hidden" style={{ borderColor: 'var(--border)' }}>
        {mode === 'architecture'
          ? (view3D ? <Network3DView graph={graph} mode="architecture" /> : <NetworkGraph graph={graph} mode="architecture" />)
          : <PropagationView steps={mode === 'forward' ? fwd : bwd} mode={mode} NetworkViewComponent={LegacyNetworkView} />}
      </div>
    </div>
  );
}

function Content({ tab, onCinema }: { tab: TabId; onCinema: () => void }) {
  const graph = useNetworkStore((s) => s.graph);
  const fwd = useNetworkStore((s) => s.forwardSteps);
  const propStep = useNetworkStore((s) => s.propStep);
  const modelType = useNetworkStore((s) => s.networkConfig.model_type);
  const training = useNetworkStore((s) => s.trainingResult);
  const decision = useNetworkStore((s) => s.decisionBoundary);
  const legacyLandscape = useNetworkStore((s) => s.lossLandscape);
  const session = useForgeStore((s) => s.session);
  const built = useNetworkStore((s) => s.networkBuilt);

  switch (tab) {
    case 'training': return <TrainingCurve data={training} />;
    case 'decision': return <DecisionBoundary data={decision} />;
    case 'loss': return session ? <SessionLandscape /> : <LossLandscape data={legacyLandscape} />;
    case 'weights': return <WeightHistogram graph={graph} />;
    case 'layer-act': return <LayerActivationHeatmap graph={graph} />;
    case 'pruning': return <PruningView graph={graph} />;
    case 'architecture': return <DiagramView mode="architecture" />;
    case 'forward': return (
      <div className="h-full flex flex-col gap-2">
        {built && <button type="button" className="btn-secondary !py-1 !text-xs self-start" onClick={onCinema}><Film size={13} />Cinema walkthrough</button>}
        <div className="flex-1 min-h-0"><DiagramView mode="forward" /></div>
      </div>
    );
    case 'backward': return <DiagramView mode="backward" />;
    case 'attention': return <AttentionHeatmap graph={graph} currentStep={fwd[propStep] ?? null} modelType={modelType} />;
    case 'live-train': return <RealTraining />;
    case 'sweep': return <HyperparamSweep />;
    case 'custom-act': return <CustomActivation />;
    case 'compare': return <CompareView />;
    case 'export': return <ExportCode />;
    default: return null;
  }
}

function Tag({ p }: { p: Provenance }) {
  return <span className={KIND_CLASS[p.kind]} title={p.note}>{p.kind}</span>;
}

function NavItem({ item, active, onClick }: { item: Item; active: boolean; onClick: () => void }) {
  const p = useProvenance(item.id);
  return (
    <button type="button" role="tab" aria-selected={active} onClick={onClick}
      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-left text-[13px] transition-colors"
      style={{ background: active ? 'var(--bg-active)' : 'transparent', color: active ? 'var(--text-primary)' : 'var(--text-muted)' }}>
      <span className="flex-1 truncate">{item.label}</span>
      <span className="scale-90 origin-right"><Tag p={p} /></span>
    </button>
  );
}

export function AnalysisWorkspace({ onCinema }: { onCinema: () => void }) {
  const active = useNetworkStore((s) => s.activeTab);
  const setActive = useNetworkStore((s) => s.setActiveTab);
  const tab = ITEMS.some((i) => i.id === active) ? active : 'training';
  const item = ITEMS.find((i) => i.id === tab)!;
  const p = useProvenance(tab);
  const nodes = useNetworkStore((s) => s.graph.nodes.length);
  const edges = useNetworkStore((s) => s.graph.edges.length);

  return (
    <div className="h-full flex min-h-0">
      <nav className="w-[232px] flex-shrink-0 border-r overflow-y-auto p-2 space-y-3" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}
        role="tablist" aria-label="Analysis views" aria-orientation="vertical">
        {GROUPS.map((g) => (
          <div key={g}>
            <div className="eyebrow px-2.5 pb-1">{g}</div>
            {ITEMS.filter((i) => i.group === g).map((i) => (
              <NavItem key={i.id} item={i} active={i.id === tab} onClick={() => setActive(i.id)} />
            ))}
          </div>
        ))}
      </nav>
      <section className="flex-1 min-w-0 flex flex-col min-h-0">
        <header className="flex items-center gap-2.5 px-4 py-2.5 border-b flex-shrink-0" style={{ borderColor: 'var(--border)' }}>
          <h2 className="text-[14px] font-semibold" style={{ color: 'var(--text-primary)' }}>{item.label}</h2>
          <Tag p={p} />
          <span className="text-[12px] truncate" style={{ color: 'var(--text-muted)' }}>{p.note}</span>
          {nodes > 0 && ['architecture', 'forward', 'backward', 'weights', 'layer-act', 'pruning'].includes(tab) && (
            <span className="ml-auto text-[11.5px] tnum flex-shrink-0" style={{ color: 'var(--text-faint)' }}>{nodes} nodes · {edges} edges</span>
          )}
        </header>
        <div className="flex-1 min-h-0 p-3 overflow-auto"><Content tab={tab} onCinema={onCinema} /></div>
      </section>
    </div>
  );
}

