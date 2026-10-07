import { AlertCircle, GraduationCap, FlaskRound, Loader2, Info } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { architectureString, highlightFor, layerLabels as buildLabels, refForNode } from '../../forge/selection';
import { useForgeStore } from '../../forge/store';
import { useNetworkStore } from '../../store/networkStore';
import type { NetworkNode } from '../../types';
import { NetworkGraph } from '../Visualizations/NetworkGraph';
import { Network3DView } from '../Visualizations/Network3DView';
import { Inspector } from './Inspector';
import { ProbePicker } from './ProbePicker';
import { WhatIfPanel } from './WhatIfPanel';

function Unsupported({ modelType, built }: { modelType: string; built: boolean }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="max-w-md text-center space-y-3 px-6">
        <div className="w-12 h-12 mx-auto rounded-xl flex items-center justify-center" style={{ background: 'rgba(245,158,11,0.1)' }}>
          <Info size={22} style={{ color: '#f59e0b' }} />
        </div>
        {modelType !== 'ANN' ? (
          <>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>The Neural Microscope currently supports ANN (MLP) models</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              The {modelType} diagram in the other tabs is illustrative: its node values are not computed from a real model,
              so there is nothing truthful to inspect yet. Switch the model type to <b>ANN</b> and click <b>Build Network</b>.
            </p>
          </>
        ) : (
          <>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{built ? 'No model session' : 'Build a network first'}</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Click <b>Build Network</b> to create a real PyTorch model, then <b>Train Model</b> to train it on the selected dataset.
              Everything shown here is computed from that model.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export function MicroscopeView() {
  const session = useForgeStore((s) => s.session);
  const graph = useForgeStore((s) => s.graph);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const mode = useForgeStore((s) => s.mode);
  const setMode = useForgeStore((s) => s.setMode);
  const loading = useForgeStore((s) => s.loading);
  const error = useForgeStore((s) => s.error);
  const checkpointEpoch = useForgeStore((s) => s.checkpointEpoch);
  const setCheckpoint = useForgeStore((s) => s.setCheckpoint);
  const modelType = useNetworkStore((s) => s.networkConfig.model_type);
  const built = useNetworkStore((s) => s.networkBuilt);
  const view3D = useNetworkStore((s) => s.view3D);

  const labels = useMemo(() => (session ? buildLabels(session.structure) : undefined), [session]);
  const highlight = useMemo(() => (graph ? highlightFor(graph, selection) : null), [graph, selection]);
  const onNodeClick = useCallback((node: NetworkNode) => {
    const ref = refForNode(node);
    if (ref) select(ref);
  }, [select]);
  const onLayerClick = useCallback((layer: number) => select({ kind: 'layer', layer }), [select]);

  if (!session || modelType !== 'ANN') return <Unsupported modelType={modelType} built={built} />;

  const structure = session.structure;
  const selectedNodeId = highlight && highlight.nodeIds.size === 1 ? [...highlight.nodeIds][0] : null;

  return (
    <div className="h-full flex gap-3 min-h-0">
      <div className="flex-1 min-w-0 flex flex-col gap-2 min-h-0">
        {/* Toolbar */}
        <div className="flex items-center gap-2 flex-wrap text-xs flex-shrink-0">
          <span className="font-mono px-2 py-1 rounded border" style={{ borderColor: 'var(--border)', color: 'var(--text-primary)' }}
            title="Layer sizes: input → hidden layers → output classes">
            MLP {architectureString(structure)}
          </span>
          <span style={{ color: 'var(--text-faint)' }}>{structure.param_count} params · {session.dataset_name}</span>

          <label className="flex items-center gap-1.5 ml-2" style={{ color: 'var(--text-muted)' }}>
            Weights
            <select
              className="select-base !w-auto !py-1 !px-2 !text-xs"
              value={checkpointEpoch === null ? 'live' : String(checkpointEpoch)}
              onChange={(e) => setCheckpoint(e.target.value === 'live' ? null : Number(e.target.value))}
            >
              <option value="live">live · epoch {session.epoch}</option>
              {session.checkpoints.filter((c) => c.epoch !== session.epoch).slice().reverse().map((c) => (
                <option key={c.epoch} value={c.epoch}>checkpoint · epoch {c.epoch} · acc {(c.accuracy * 100).toFixed(0)}%</option>
              ))}
            </select>
          </label>

          {loading && <Loader2 size={13} className="animate-spin" style={{ color: 'var(--text-faint)' }} />}
          {error && (
            <span className="flex items-center gap-1" style={{ color: '#fca5a5' }}><AlertCircle size={12} />{error}</span>
          )}

          <div className="ml-auto flex items-center p-0.5 rounded-lg border" style={{ borderColor: 'var(--border)' }} role="group" aria-label="Experience mode">
            {([['learn', 'Learn', GraduationCap], ['lab', 'Lab', FlaskRound]] as const).map(([m, label, Icon]) => (
              <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
                className="flex items-center gap-1 px-2 py-1 rounded-md transition-colors"
                style={{ background: mode === m ? 'var(--accent)' : 'transparent', color: mode === m ? '#fff' : 'var(--text-muted)' }}>
                <Icon size={12} />{label}
              </button>
            ))}
          </div>
        </div>

        {/* Network */}
        <div className="flex-1 min-h-0 rounded-xl overflow-hidden border relative" style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}>
          {graph && (view3D ? (
            <Network3DView graph={graph} onNodeClick={onNodeClick} selectedNodeId={selectedNodeId} />
          ) : (
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
          ))}
          {!view3D && (
            <div className="absolute bottom-2 left-2 text-[10px] px-2 py-1 rounded-md border flex gap-3"
              style={{ background: 'rgba(9,13,19,0.85)', borderColor: 'var(--border)', color: 'var(--text-faint)' }}>
              <span>edge = real signal w·a on this input</span>
              <span style={{ color: '#6ee7b7' }}>━ pushes up</span>
              <span style={{ color: '#fca5a5' }}>━ pushes down</span>
              <span>node brightness = |activation|</span>
            </div>
          )}
        </div>

        {/* Probe + prediction / what-if */}
        <div className="flex gap-3 flex-shrink-0 rounded-xl border p-2.5" style={{ borderColor: 'var(--border)', height: 210 }}>
          <div className="flex-shrink-0" style={{ width: 200 }}><ProbePicker /></div>
          <div className="w-px" style={{ background: 'var(--border)' }} />
          <div className="flex-1 min-w-0"><WhatIfPanel /></div>
        </div>
      </div>

      {/* Inspector */}
      <aside className="flex-shrink-0 rounded-xl border overflow-hidden" style={{ width: 380, borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <Inspector structure={structure} />
      </aside>
    </div>
  );
}
