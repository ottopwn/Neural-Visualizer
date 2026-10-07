// Neural Forge application state.
//
// The backend is stateless with respect to experiments: every request
// carries the probe, the intervention log and the checkpoint being viewed.
// This store owns those three inputs; whenever one changes it refetches the
// graph, the before/after comparison and the current inspection.  Responses
// are tagged with a sequence number so a slow, stale response can never
// overwrite a newer one.

import { create } from 'zustand';
import * as api from './api';
import * as iv from './interventions';
import type {
  Comparison, ComponentRef, ExperienceMode, ExperimentRequest, ForgeGraph, Inspection,
  Intervention, Probe, SessionSummary,
} from './types';

export interface ForgeState {
  session: SessionSummary | null;
  probe: Probe;
  interventions: Intervention[];
  checkpointEpoch: number | null; // null = live weights
  selection: ComponentRef | null;
  mode: ExperienceMode;

  graph: ForgeGraph | null;
  comparison: Comparison | null;
  inspection: Inspection | null;
  loading: boolean;
  inspecting: boolean;
  error: string | null;

  setSession: (s: SessionSummary | null) => Promise<void>;
  updateSession: (s: SessionSummary) => Promise<void>;
  setProbe: (p: Probe) => Promise<void>;
  setCheckpoint: (epoch: number | null) => Promise<void>;
  select: (ref: ComponentRef | null) => Promise<void>;
  setMode: (m: ExperienceMode) => void;
  setInterventions: (ivs: Intervention[]) => Promise<void>;
  addIntervention: (i: Intervention) => Promise<void>;
  undo: () => Promise<void>;
  reset: () => Promise<void>;
  refresh: () => Promise<void>;
}

/** Listeners notified with each freshly fetched graph (used to sync legacy tabs). */
type GraphListener = (g: ForgeGraph) => void;
const graphListeners = new Set<GraphListener>();
export function onForgeGraph(fn: GraphListener): () => void {
  graphListeners.add(fn);
  return () => graphListeners.delete(fn);
}

let refreshSeq = 0;
let inspectSeq = 0;

function request(s: Pick<ForgeState, 'probe' | 'interventions' | 'checkpointEpoch'>): ExperimentRequest {
  return { probe: s.probe, interventions: s.interventions, checkpoint_epoch: s.checkpointEpoch };
}

const DEFAULT_PROBE: Probe = { sample_index: 0 };

export const useForgeStore = create<ForgeState>((set, get) => {
  const loadInspection = async () => {
    const { session, selection } = get();
    const seq = ++inspectSeq;
    if (!session || !selection) {
      set({ inspection: null, inspecting: false });
      return;
    }
    set({ inspecting: true });
    try {
      const result = await api.inspect(session.session_id, selection, request(get()));
      if (seq === inspectSeq) set({ inspection: result, inspecting: false, error: null });
    } catch (err) {
      if (seq === inspectSeq) set({ inspecting: false, error: api.errorMessage(err) });
    }
  };

  const refresh = async () => {
    const { session } = get();
    const seq = ++refreshSeq;
    if (!session) {
      set({ graph: null, comparison: null, loading: false });
      return;
    }
    set({ loading: true });
    const req = request(get());
    const inspection = loadInspection();
    try {
      const [graph, comparison] = await Promise.all([
        api.fetchGraph(session.session_id, req),
        api.compare(session.session_id, req),
      ]);
      if (seq === refreshSeq) {
        set({ graph, comparison, loading: false, error: null });
        graphListeners.forEach((fn) => fn(graph));
      }
    } catch (err) {
      if (seq === refreshSeq) set({ loading: false, error: api.errorMessage(err) });
    }
    await inspection;
  };

  const changeInterventions = (next: Intervention[]) => {
    set({ interventions: next });
    return refresh();
  };

  return {
    session: null,
    probe: DEFAULT_PROBE,
    interventions: [],
    checkpointEpoch: null,
    selection: null,
    mode: 'learn',
    graph: null,
    comparison: null,
    inspection: null,
    loading: false,
    inspecting: false,
    error: null,

    setSession: (session) => {
      set({
        session, probe: DEFAULT_PROBE, interventions: [], checkpointEpoch: null,
        selection: null, inspection: null, graph: null, comparison: null, error: null,
      });
      return refresh();
    },
    // After training: keep probe/selection, drop interventions (they referred to old weights).
    updateSession: (session) => {
      set({ session, interventions: [], checkpointEpoch: null });
      return refresh();
    },
    setProbe: (probe) => {
      set({ probe });
      return refresh();
    },
    setCheckpoint: (checkpointEpoch) => {
      set({ checkpointEpoch });
      return refresh();
    },
    select: (selection) => {
      set({ selection, inspection: selection ? get().inspection : null });
      return loadInspection();
    },
    setMode: (mode) => set({ mode }),
    setInterventions: changeInterventions,
    addIntervention: (i) => changeInterventions([...get().interventions, i]),
    undo: () => changeInterventions(iv.undo(get().interventions)),
    reset: () => changeInterventions([]),
    refresh,
  };
});
