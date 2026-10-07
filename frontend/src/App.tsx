import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Network, Play, ArrowRight, ArrowLeft, Target, Mountain, TrendingUp,
  Loader2, AlertCircle, CheckCircle2, Zap, Box, Cpu, GitCompare, Focus,
  BarChart2, Layers, Sliders, Scissors, PenTool, Code2, Microscope,
} from 'lucide-react';

import { Header } from './components/Layout/Header';
import { NetworkConfig } from './components/Sidebar/NetworkConfig';
import { TrainingConfig } from './components/Sidebar/TrainingConfig';
import { DatasetUpload } from './components/DatasetUpload';
import { DatasetPreview } from './components/DatasetPreview';
import { NetworkGraph } from './components/Visualizations/NetworkGraph';
import { Network3DView } from './components/Visualizations/Network3DView';
import { PropagationView } from './components/Visualizations/PropagationView';
import { DecisionBoundary } from './components/Visualizations/DecisionBoundary';
import { LossLandscape } from './components/Visualizations/LossLandscape';
import { TrainingCurve } from './components/Visualizations/TrainingCurve';
import { RealTraining } from './components/Visualizations/RealTraining';
import { AttentionHeatmap } from './components/Visualizations/AttentionHeatmap';
import { CompareView } from './components/Visualizations/CompareView';
import { WeightHistogram } from './components/Visualizations/WeightHistogram';
import { LayerActivationHeatmap } from './components/Visualizations/LayerActivationHeatmap';
import { HyperparamSweep } from './components/Visualizations/HyperparamSweep';
import { PruningView } from './components/Visualizations/PruningView';
import { CustomActivation } from './components/Visualizations/CustomActivation';
import { ExportCode } from './components/Visualizations/ExportCode';
import { CinemaMode } from './components/CinemaMode';
import { Tutorial } from './components/Tutorial';
import { MicroscopeView } from './components/Forge/MicroscopeView';

import { useNetworkStore } from './store/networkStore';
import * as api from './api/client';
import * as forgeApi from './forge/api';
import { onForgeGraph, useForgeStore } from './forge/store';
import type { BoundaryPair, SessionSummary } from './forge/types';
import type { DecisionBoundaryData, TabId, TrainingResult } from './types';

const TABS: { id: TabId; label: string; icon: React.ReactNode; group: string }[] = [
  { id: 'microscope',   label: 'Microscope',   icon: <Microscope size={12} />, group: 'Forge' },
  { id: 'architecture', label: 'Architecture', icon: <Network size={12} />,    group: 'Network' },
  { id: 'forward',      label: 'Forward',      icon: <ArrowRight size={12} />, group: 'Network' },
  { id: 'backward',     label: 'Backprop',     icon: <ArrowLeft size={12} />,  group: 'Network' },
  { id: 'weights',      label: 'Weights',      icon: <BarChart2 size={12} />,  group: 'Network' },
  { id: 'layer-act',    label: 'Activations',  icon: <Layers size={12} />,     group: 'Network' },
  { id: 'pruning',      label: 'Pruning',      icon: <Scissors size={12} />,   group: 'Network' },
  { id: 'attention',    label: 'Attention',    icon: <Focus size={12} />,      group: 'Analysis' },
  { id: 'decision',     label: 'Decision',     icon: <Target size={12} />,     group: 'Analysis' },
  { id: 'loss',         label: 'Loss Surface', icon: <Mountain size={12} />,   group: 'Analysis' },
  { id: 'training',     label: 'Training',     icon: <TrendingUp size={12} />, group: 'Train' },
  { id: 'live-train',   label: 'Live Train',   icon: <Cpu size={12} />,        group: 'Train' },
  { id: 'sweep',        label: 'LR Sweep',     icon: <Sliders size={12} />,    group: 'Train' },
  { id: 'custom-act',   label: 'Custom Act',   icon: <PenTool size={12} />,    group: 'Train' },
  { id: 'compare',      label: 'Compare',      icon: <GitCompare size={12} />, group: 'Tools' },
  { id: 'export',       label: 'Export',       icon: <Code2 size={12} />,      group: 'Tools' },
];

const TAB_GROUPS = ['Forge', 'Network', 'Analysis', 'Train', 'Tools'];

/** Model types whose graph is backed by a real Forge model session. */
const FORGE_MODELS = new Set(['ANN']);

function historyToResult(summary: SessionSummary): TrainingResult {
  return {
    epochs: summary.history.map((r) => r.epoch),
    loss_history: summary.history.map((r) => r.loss),
    accuracy_history: summary.history.map((r) => r.accuracy),
    source: 'pytorch',
  };
}

function boundaryToData(b: BoundaryPair, summary: SessionSummary): DecisionBoundaryData {
  const r = b.resolution;
  const xs = Array.from({ length: r }, (_, i) => b.x_range[0] + ((b.x_range[1] - b.x_range[0]) * i) / (r - 1));
  const ys = Array.from({ length: r }, (_, i) => b.y_range[0] + ((b.y_range[1] - b.y_range[0]) * i) / (r - 1));
  return {
    xx: ys.map(() => xs),
    yy: ys.map((y) => xs.map(() => y)),
    zz: b.baseline,
    X: summary.dataset_X,
    y: summary.dataset_y,
    source: 'trained-session',
  };
}

type Status = { type: 'idle' | 'loading' | 'success' | 'error'; message?: string };

export default function App() {
  const store = useNetworkStore();
  const [buildStatus, setBuildStatus] = useState<Status>({ type: 'idle' });
  const [trainStatus, setTrainStatus] = useState<Status>({ type: 'idle' });
  const [cinemaOpen, setCinemaOpen] = useState(false);
  const [tutorialOpen, setTutorialOpen] = useState(() => !localStorage.getItem('nv_tutorial_seen'));

  // Keep the classic tabs (Architecture, Forward, Backprop, Weights, ...) in sync
  // with the real model graph whenever the Forge session refreshes it.
  useEffect(() => onForgeGraph((g) => {
    const s = useNetworkStore.getState();
    s.setGraph({ nodes: g.nodes, edges: g.edges });
    s.setForwardSteps(g.forward_steps);
    s.setBackwardSteps(g.backward_steps);
  }), []);

  // Auto-open tutorial on first visit (initial state above), then remember it
  useEffect(() => {
    localStorage.setItem('nv_tutorial_seen', '1');
  }, []);

  const handleBuildNetwork = useCallback(async () => {
    setBuildStatus({ type: 'loading' });
    try {
      const config = {
        ...store.networkConfig,
        neurons: store.networkConfig.neurons.slice(0, store.networkConfig.n_layers),
        activations: store.networkConfig.activations.slice(0, store.networkConfig.n_layers),
      };
      store.setTrainingResult(null);
      store.setDecisionBoundary(null);
      store.setPropStep(0);
      let message: string;

      if (FORGE_MODELS.has(config.model_type)) {
        // Real model: a PyTorch MLP the microscope and all graph tabs read from.
        const custom = store.customDataset;
        const summary = await forgeApi.createSession({
          neurons: config.neurons,
          activations: config.activations,
          dataset: store.trainingConfig.dataset,
          noise: store.trainingConfig.noise,
          custom_dataset: custom ? { X: custom.X, y: custom.y } : null,
        });
        await useForgeStore.getState().setSession(summary);
        const err = useForgeStore.getState().error;
        if (err) throw new Error(err);
        store.setGraphSource('model');
        message = `Real model · ${summary.structure.param_count} parameters (untrained)`;
      } else {
        await useForgeStore.getState().setSession(null);
        const graph = await api.buildNetwork(config);
        store.setGraph(graph);
        const [fwd, bwd] = await Promise.all([
          api.getForwardProp(config),
          api.getBackwardProp(config),
        ]);
        store.setForwardSteps(fwd.steps);
        store.setBackwardSteps(bwd.steps);
        store.setGraphSource('illustrative');
        message = `${graph.nodes.length} nodes · ${graph.edges.length} edges (illustrative)`;
      }
      store.setNetworkBuilt(true);

      const landscape = await api.getLossLandscape(config);
      store.setLossLandscape(landscape);

      setBuildStatus({ type: 'success', message });
      setTimeout(() => setBuildStatus({ type: 'idle' }), 3000);
    } catch (err) {
      setBuildStatus({ type: 'error', message: forgeApi.errorMessage(err) });
    }
  }, [store]);

  const handleTraining = useCallback(async () => {
    setTrainStatus({ type: 'loading' });
    const forge = useForgeStore.getState();
    if (forge.session && store.graphSource === 'model') {
      try {
        const t = store.trainingConfig;
        const { summary, new_rows } = await forgeApi.trainSession(forge.session.session_id, {
          epochs: t.epochs, learning_rate: t.learning_rate, batch_size: t.batch_size,
          reg_type: t.reg_type, reg_rate: t.reg_rate,
        });
        await useForgeStore.getState().updateSession(summary);
        store.setTrainingResult(historyToResult(summary));
        const boundary = useForgeStore.getState().comparison?.boundary;
        if (boundary) store.setDecisionBoundary(boundaryToData(boundary, summary));
        const last = new_rows[new_rows.length - 1];
        setTrainStatus({ type: 'success', message: `Epoch ${summary.epoch} · accuracy ${(last.accuracy * 100).toFixed(1)}%` });
        setTimeout(() => setTrainStatus({ type: 'idle' }), 3000);
      } catch (err) {
        setTrainStatus({ type: 'error', message: forgeApi.errorMessage(err) });
      }
      return;
    }
    try {
      const netConfig = {
        ...store.networkConfig,
        neurons: store.networkConfig.neurons.slice(0, store.networkConfig.n_layers),
        activations: store.networkConfig.activations.slice(0, store.networkConfig.n_layers),
      };
      const [result, boundary] = await Promise.all([
        api.simulateTraining(netConfig, store.trainingConfig),
        api.getDecisionBoundary(netConfig, store.trainingConfig),
      ]);
      store.setTrainingResult(result);
      store.setDecisionBoundary(boundary);
      store.setActiveTab('training');

      setTrainStatus({ type: 'success', message: `${result.epochs.length} epochs complete` });
      setTimeout(() => setTrainStatus({ type: 'idle' }), 3000);
    } catch {
      setTrainStatus({ type: 'error', message: 'Training failed — is backend running?' });
    }
  }, [store]);

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      <Header
        onCinema={() => setCinemaOpen(true)}
        cinemaDisabled={!store.networkBuilt}
        onTutorial={() => setTutorialOpen(true)}
      />

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[280px] flex-shrink-0 overflow-y-auto flex flex-col gap-3 p-3"
          style={{ borderRight: '1px solid var(--border)', background: 'var(--bg-sidebar)' }}>
          <NetworkConfig />
          <TrainingConfig />
          <DatasetPreview />

          {/* CSV Upload */}
          <div className="rounded-xl p-3 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
            <p className="label-base mb-2">Custom Dataset</p>
            <DatasetUpload
              loaded={!!store.customDataset}
              onLoad={(ds) => store.setCustomDataset(ds)}
              onClear={() => store.setCustomDataset(null)}
            />
          </div>

          {/* Action buttons */}
          <div className="space-y-2 pb-3">
            <button data-tour="build-btn" onClick={handleBuildNetwork} disabled={buildStatus.type === 'loading'} className="btn-primary w-full flex items-center justify-center gap-2">
              {buildStatus.type === 'loading' ? <><Loader2 size={15} className="animate-spin" />Building…</> : <><Zap size={15} />Build Network</>}
            </button>
            <button onClick={handleTraining} disabled={trainStatus.type === 'loading' || !store.networkBuilt} className="btn-secondary w-full flex items-center justify-center gap-2 text-sm">
              {trainStatus.type === 'loading'
                ? <><Loader2 size={15} className="animate-spin" />Training…</>
                : store.graphSource === 'model'
                  ? <><Play size={15} />Train Model ({store.trainingConfig.epochs} epochs)</>
                  : <><Play size={15} />Simulate Training</>}
            </button>
            {store.graphSource === 'model' && (
              <p className="text-[10px] leading-snug px-1" style={{ color: 'var(--text-faint)' }}>
                Real PyTorch training (Adam, cross-entropy). Each click continues from the current weights; checkpoints are kept for the microscope.
              </p>
            )}
          </div>

          <AnimatePresence>
            {(buildStatus.type !== 'idle' || trainStatus.type !== 'idle') && (
              <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} className="space-y-1.5">
                {buildStatus.type !== 'idle' && buildStatus.type !== 'loading' && <StatusBadge status={buildStatus} label="Build" />}
                {trainStatus.type !== 'idle' && trainStatus.type !== 'loading' && <StatusBadge status={trainStatus} label="Train" />}
              </motion.div>
            )}
          </AnimatePresence>
        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Grouped tab bar */}
          <div className="flex items-center gap-0 px-2 py-1.5 overflow-x-auto flex-shrink-0 flex-wrap"
            style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-sidebar)' }}>
            {TAB_GROUPS.map((group, gi) => (
              <div key={group} className="flex items-center">
                {gi > 0 && <div className="w-px h-4 mx-1.5 flex-shrink-0" style={{ background: 'var(--border-soft)' }} />}
                <span className="text-xs mr-1 flex-shrink-0" style={{ color: 'var(--text-faint)' }}>{group}</span>
                {TABS.filter(t => t.group === group).map(tab => (
                  <button
                    key={tab.id}
                    data-tour={`tab-${tab.id}`}
                    onClick={() => store.setActiveTab(tab.id)}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-all duration-150 whitespace-nowrap mr-0.5"
                    style={{
                      background: store.activeTab === tab.id ? 'var(--accent)' : 'transparent',
                      color: store.activeTab === tab.id ? '#fff' : 'var(--text-muted)',
                      boxShadow: store.activeTab === tab.id ? `0 0 10px var(--accent-glow)` : 'none',
                    }}
                  >
                    {tab.icon}{tab.label}
                  </button>
                ))}
              </div>
            ))}

            {/* 2D/3D toggle */}
            {['architecture','forward','backward','pruning'].includes(store.activeTab) && (
              <button
                data-tour="toggle-3d"
                onClick={() => store.setView3D(!store.view3D)}
                className="ml-auto flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-all flex-shrink-0"
                style={{
                  background: store.view3D ? 'rgba(139,92,246,0.15)' : 'transparent',
                  borderColor: store.view3D ? '#8b5cf6' : 'var(--border-soft)',
                  color: store.view3D ? '#c4b5fd' : 'var(--text-muted)',
                }}
              >
                <Box size={11} />{store.view3D ? '3D' : '2D'}
              </button>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-h-0 p-3 overflow-hidden">
            <AnimatePresence mode="wait">
              <motion.div key={store.activeTab} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.12 }} className="h-full">
                <TabContent tab={store.activeTab} />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>

      <AnimatePresence>
        {cinemaOpen && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
            <CinemaMode onClose={() => setCinemaOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>

      {tutorialOpen && <Tutorial onClose={() => setTutorialOpen(false)} />}
    </div>
  );
}

function TabContent({ tab }: { tab: TabId }) {
  const store = useNetworkStore();

  const panel = (title: string, badge: string, badgeClass: string, children: React.ReactNode) => (
    <div className="card h-full flex flex-col">
      <div className="card-header flex-shrink-0">
        <span className={`badge ${badgeClass}`}>{badge}</span>
        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{title}</span>
        {store.graph.nodes.length > 0 && GRAPH_TABS.includes(tab) && <SourceBadge source={store.graphSource} />}
        {store.graph.nodes.length > 0 && !['compare','export','sweep','custom-act'].includes(tab) && (
          <span className="ml-auto text-xs" style={{ color: 'var(--text-faint)' }}>
            {store.graph.nodes.length} nodes · {store.graph.edges.length} edges
          </span>
        )}
      </div>
      <div className="flex-1 min-h-0 p-3">{children}</div>
    </div>
  );

  switch (tab) {
    case 'microscope':   return (
      <div className="card h-full flex flex-col">
        <div className="card-header flex-shrink-0">
          <span className="badge badge-green">Neural Microscope</span>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Inspect &amp; intervene on a real model</span>
        </div>
        <div className="flex-1 min-h-0 p-3"><MicroscopeView /></div>
      </div>
    );
    case 'architecture': return panel('Network Architecture', '3D/2D', 'badge-blue',   <NetworkView mode="architecture" />);
    case 'forward':      return panel('Forward Propagation',  'Forward', 'badge-green',  <PropagationView steps={store.forwardSteps}  mode="forward"   NetworkViewComponent={NetworkView} />);
    case 'backward':     return panel('Backpropagation',      'Backward','badge-orange', <PropagationView steps={store.backwardSteps} mode="backward"  NetworkViewComponent={NetworkView} />);
    case 'weights':      return panel('Weight Distribution',  'Histogram','badge-blue',  <WeightHistogram graph={store.graph} />);
    case 'layer-act':    return panel('Layer Activations',    'Heatmap', 'badge-purple', <LayerActivationHeatmap graph={store.graph} />);
    case 'pruning':      return panel('Network Pruning',      'Interactive','badge-orange', <PruningView graph={store.graph} />);
    case 'attention':    return panel('Attention Weights',    'Synthetic pattern', 'badge-orange', <AttentionHeatmap graph={store.graph} currentStep={store.forwardSteps[store.propStep]??null} modelType={store.networkConfig.model_type} />);
    case 'decision':     return panel('Decision Boundary',    store.decisionBoundary?.source === 'trained-session' ? 'Trained model' : 'Untrained model', 'badge-purple', <DecisionBoundary data={store.decisionBoundary} />);
    case 'loss':         return panel('Loss Landscape',       'Random-init model','badge-blue', <LossLandscape data={store.lossLandscape} />);
    case 'training':     return panel('Training Curves',      store.trainingResult?.source === 'pytorch' ? 'Real · PyTorch' : 'Synthetic curve','badge-green',  <TrainingCurve data={store.trainingResult} />);
    case 'live-train':   return panel('Live Training',        'TF.js',   'badge-pink',   <RealTraining />);
    case 'sweep':        return panel('LR Sweep',             'TF.js×5', 'badge-purple', <HyperparamSweep />);
    case 'custom-act':   return panel('Custom Activation',    'Draw',    'badge-green',  <CustomActivation />);
    case 'compare':      return (
      <div className="card h-full flex flex-col">
        <div className="card-header flex-shrink-0"><span className="badge badge-purple">Compare</span><span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Architecture Comparison</span></div>
        <div className="flex-1 min-h-0 p-3"><CompareView /></div>
      </div>
    );
    case 'export': return panel('Export Code', 'PyTorch / Keras', 'badge-blue', <ExportCode />);
    default: return null;
  }
}

// Defined at module level so it keeps its identity across renders (no remounts).
function NetworkView({ activeNodeIds, activeEdgeIds, gradients, mode }: {
  activeNodeIds?: Set<number>; activeEdgeIds?: Set<number>; gradients?: Record<string, number>; mode?: 'architecture' | 'forward' | 'backward';
}) {
  const graph = useNetworkStore((s) => s.graph);
  const view3D = useNetworkStore((s) => s.view3D);
  return view3D
    ? <Network3DView graph={graph} activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} mode={mode} />
    : <NetworkGraph graph={graph} activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} gradients={gradients} mode={mode} />;
}

// Tabs that display the graph's node/edge values (and so inherit its provenance).
// 'attention' is excluded: it always draws a synthetic pattern.
const GRAPH_TABS: TabId[] = ['architecture', 'forward', 'backward', 'weights', 'layer-act', 'pruning'];

function SourceBadge({ source }: { source: 'model' | 'illustrative' }) {
  const real = source === 'model';
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded border"
      title={real
        ? 'Values are computed by a PyTorch forward/backward pass of the session model.'
        : 'Diagram only: node values, weights and gradients are random placeholders, not computed from a model.'}
      style={real
        ? { borderColor: 'rgba(16,185,129,0.4)', color: '#6ee7b7', background: 'rgba(16,185,129,0.08)' }
        : { borderColor: 'rgba(245,158,11,0.4)', color: '#fcd34d', background: 'rgba(245,158,11,0.08)' }}
    >
      {real ? 'REAL MODEL VALUES' : 'ILLUSTRATIVE VALUES'}
    </span>
  );
}

function StatusBadge({ status, label }: { status: Status; label: string }) {
  const isError = status.type === 'error';
  return (
    <div className="flex items-start gap-2 p-2.5 rounded-lg text-xs border"
      style={{ background: isError ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)', borderColor: isError ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)', color: isError ? '#fca5a5' : '#86efac' }}>
      {isError ? <AlertCircle size={13} className="flex-shrink-0 mt-0.5" /> : <CheckCircle2 size={13} className="flex-shrink-0 mt-0.5" />}
      <span>{label}: {status.message}</span>
    </div>
  );
}
