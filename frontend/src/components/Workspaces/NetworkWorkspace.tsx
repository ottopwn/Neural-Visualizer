import { AlertCircle, Loader2, Network } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { architectureString, highlightFor, layerLabels as buildLabels, refForNode } from '../../forge/selection';
import { useForgeStore } from '../../forge/store';
import { useNetworkStore } from '../../store/networkStore';
import type { NetworkNode } from '../../types';
import { TimelineInstrument, Transport } from '../Forge/TimeMachine/TimelineInstrument';
import { NetworkGraph } from '../Visualizations/NetworkGraph';
import { NeedsModel } from './EmptyState';

/** Header strip shared by the Forge workspaces: what model, which epoch. */
export function ModelStrip({ children }: { children?: React.ReactNode }) {
  const session = useForgeStore((s) => s.session);
  const loading = useForgeStore((s) => s.loading);
  const error = useForgeStore((s) => s.error);
  if (!session) return null;
  const trained = session.epoch > 0;
  return (
    <div className="flex items-center gap-3 flex-wrap px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
      <div className="flex items-baseline gap-2 min-w-0">
        <span className="font-mono text-[12.5px]" style={{ color: 'var(--text-primary)' }}>MLP {architectureString(session.structure)}</span>
        <span className="text-[12px] tnum" style={{ color: 'var(--text-faint)' }}>{session.structure.param_count.toLocaleString()} params · {session.dataset_name}</span>
      </div>
      {trained ? (
        <div className="flex items-center gap-2 flex-1 min-w-[320px] max-w-[760px]" title="Which stored checkpoint every instrument shows">
          <div className="flex-1 min-w-[120px]"><TimelineInstrument lab={false} compact /></div>
          <Transport compact />
        </div>
      ) : (
        <span className="text-[12px]" style={{ color: 'var(--text-faint)' }}>Untrained (epoch 0) — Train to create checkpoints</span>
      )}
      {children}
      <div className="ml-auto flex items-center gap-2">
        {loading && <Loader2 size={14} className="animate-spin" style={{ color: 'var(--text-faint)' }} aria-label="Updating" />}
        {error && <span className="flex items-center gap-1 text-[12px]" role="alert" style={{ color: 'var(--text-neg)' }}><AlertCircle size={13} />{error}</span>}
      </div>
    </div>
  );
}

function IllustrativeNetwork() {
  const graph = useNetworkStore((s) => s.graph);
  const modelType = useNetworkStore((s) => s.networkConfig.model_type);
  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex items-center gap-2 px-3 py-2 border-b flex-shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
        <span className="badge-orange">ILLUSTRATIVE VALUES</span>
        <span className="text-[12px]" style={{ color: 'var(--text-muted)' }}>
          {modelType} diagram — node values and weights are placeholders, not computed by a model.
        </span>
      </div>
      <div className="flex-1 min-h-0"><NetworkGraph graph={graph} mode="architecture" /></div>
    </div>
  );
}

export function NetworkWorkspace() {
  const session = useForgeStore((s) => s.session);
  const graph = useForgeStore((s) => s.graph);
  const selection = useForgeStore((s) => s.selection);
  const select = useForgeStore((s) => s.select);
  const graphSource = useNetworkStore((s) => s.graphSource);
  const built = useNetworkStore((s) => s.networkBuilt);

  const labels = useMemo(() => (session ? buildLabels(session.structure) : undefined), [session]);
  const highlight = useMemo(() => (graph ? highlightFor(graph, selection) : null), [graph, selection]);
  const onNodeClick = useCallback((node: NetworkNode) => {
    const ref = refForNode(node);
    if (ref) void select(ref);
  }, [select]);
  const onLayerClick = useCallback((layer: number) => void select({ kind: 'layer', layer }), [select]);

  if (!session && built && graphSource === 'illustrative') return <IllustrativeNetwork />;
  if (!session) {
    return (
      <NeedsModel icon={<Network size={20} />} view='network' />
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      <ModelStrip />
      <div className="flex-1 min-h-0 relative" style={{ background: 'var(--bg-base)' }}>
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
      </div>
      <div className="flex-shrink-0 border-t text-[11.5px] px-3 py-1.5 flex flex-wrap gap-x-4 gap-y-1" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
          <span>Edge = real signal <span className="font-mono">w·a</span> on the probe</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-[3px] rounded" style={{ background: 'var(--pos)' }} />pushes up</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-[3px] rounded" style={{ background: 'var(--neg)' }} />pushes down</span>
          <span>Node fill = |activation|</span>
          <span className="ml-auto" style={{ color: 'var(--text-faint)' }}>Click (or Tab + Enter) a node or layer header · scroll to zoom · drag to pan</span>
      </div>
    </div>
  );
}
