import { Suspense, useEffect, useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';

import { useWorkspace, type WorkspaceMode } from './app/workspace';
import { CinemaMode } from './components/CinemaMode';
import { PassExplorer } from './components/Forge/Explorer/PassExplorer';
import { TimeMachineView } from './components/Forge/TimeMachine/TimeMachineView';
import { ExperimentPanel } from './components/Shell/ExperimentPanel';
import { InspectorPanel } from './components/Shell/InspectorPanel';
import { TopBar } from './components/Shell/TopBar';
import { AnalysisWorkspace } from './components/Workspaces/AnalysisWorkspace';
import { NetworkWorkspace } from './components/Workspaces/NetworkWorkspace';
import { onForgeGraph } from './forge/store';
import { useNetworkStore } from './store/networkStore';

/** Workspaces that are always available. Others register below when they exist. */
const AVAILABLE: Set<WorkspaceMode> = new Set(['network', 'timemachine', 'explorer', 'analysis']);

function Loading() {
  return (
    <div className="h-full flex items-center justify-center" style={{ color: 'var(--text-faint)' }}>
      <Loader2 size={18} className="animate-spin" />
    </div>
  );
}

function Workspace({ mode, onCinema }: { mode: WorkspaceMode; onCinema: () => void }) {
  switch (mode) {
    case 'network': return <NetworkWorkspace />;
    case 'timemachine': return <TimeMachineView />;
    case 'explorer': return <PassExplorer />;
    case 'analysis': return <AnalysisWorkspace onCinema={onCinema} />;
    default: return <NetworkWorkspace />;
  }
}

export default function App() {
  const mode = useWorkspace((s) => s.mode);
  const leftOpen = useWorkspace((s) => s.leftOpen);
  const rightOpen = useWorkspace((s) => s.rightOpen);
  const setWelcomeOpen = useWorkspace((s) => s.setWelcomeOpen);
  const [cinemaOpen, setCinemaOpen] = useState(false);
  const available = useMemo(() => AVAILABLE, []);

  // Keep the classic analysis views (weights, activations, step animations, ...)
  // in sync with the real model graph whenever the Forge session refreshes it.
  useEffect(() => onForgeGraph((g) => {
    const s = useNetworkStore.getState();
    s.setGraph({ nodes: g.nodes, edges: g.edges });
    s.setForwardSteps(g.forward_steps);
    s.setBackwardSteps(g.backward_steps);
  }), []);

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      <a href="#workspace" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:px-3 focus:py-2 focus:rounded-md"
        style={{ background: 'var(--bg-card)', color: 'var(--text-primary)' }}>Skip to workspace</a>
      <TopBar onDemo={() => setWelcomeOpen(true)} onPresent={() => setWelcomeOpen(true)} availableModes={available} />

      <div className="flex flex-1 min-h-0">
        {leftOpen && (
          <aside className="w-[272px] flex-shrink-0 border-r min-h-0" aria-label="Experiment"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
            <ExperimentPanel />
          </aside>
        )}

        <main id="workspace" className="flex-1 min-w-0 min-h-0" tabIndex={-1}>
          <Suspense fallback={<Loading />}>
            <Workspace mode={mode} onCinema={() => setCinemaOpen(true)} />
          </Suspense>
        </main>

        {rightOpen && (
          <aside className="w-[352px] flex-shrink-0 border-l min-h-0" aria-label="Inspector"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
            <InspectorPanel />
          </aside>
        )}
      </div>

      {cinemaOpen && <CinemaMode onClose={() => setCinemaOpen(false)} />}
    </div>
  );
}


