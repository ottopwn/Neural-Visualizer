// Pass Explorer state: direction, current step, playback, and an optional
// comparison trace at another stored checkpoint.
//
// The trace itself lives in the Forge store (it follows the same probe,
// checkpoint and what-if list as every other instrument).  Steps are kept by
// id, so moving to another epoch keeps you on "Dense 2 · z" and the numbers
// simply change underneath -- that is the "how did the computation change"
// workflow.

import { create } from 'zustand';
import * as api from './api';
import { buildSteps, stepIndexOf, type PassDirection } from './passExplorer';
import { useForgeStore } from './store';
import { probeKey } from './timeline';
import type { ComputationTrace } from './types';

export const PLAY_INTERVAL_MS = 1400;

interface ExplorerState {
  dir: PassDirection;
  stepId: Record<PassDirection, string | null>;
  playing: boolean;
  compareEpoch: number | null;
  compareTrace: ComputationTrace | null;
  compareLoading: boolean;
  compareError: string | null;

  setDir: (d: PassDirection) => void;
  goTo: (id: string) => void;
  step: (delta: 1 | -1) => void;
  first: () => void;
  last: () => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setCompareEpoch: (epoch: number | null) => Promise<void>;
}

let timer: ReturnType<typeof setTimeout> | null = null;
let compareSeq = 0;
let compareAbort: AbortController | null = null;

function currentSteps(dir: PassDirection) {
  const t = useForgeStore.getState().trace;
  return t ? buildSteps(t, dir) : [];
}

export const useExplorer = create<ExplorerState>((set, get) => {
  const clear = () => { if (timer) clearTimeout(timer); timer = null; };

  const move = (delta: number, keepPlaying = false) => {
    const { dir, stepId } = get();
    const steps = currentSteps(dir);
    if (!steps.length) return false;
    const i = stepIndexOf(steps, stepId[dir]) + delta;
    if (i < 0 || i >= steps.length) return false;
    set({ stepId: { ...stepId, [dir]: steps[i].id }, ...(keepPlaying ? {} : { playing: false }) });
    return true;
  };

  const tick = () => {
    timer = null;
    if (!get().playing) return;
    if (!move(1, true)) { set({ playing: false }); return; }
    timer = setTimeout(tick, PLAY_INTERVAL_MS);
  };

  const loadCompare = async () => {
    const seq = ++compareSeq;
    compareAbort?.abort();
    compareAbort = null;
    const { compareEpoch } = get();
    const f = useForgeStore.getState();
    if (compareEpoch === null || !f.session) {
      set({ compareTrace: null, compareLoading: false, compareError: null });
      return;
    }
    const ctrl = new AbortController();
    compareAbort = ctrl;
    set({ compareLoading: true });
    try {
      const trace = await api.fetchTrace(f.session.session_id,
        { probe: f.probe, interventions: f.interventions, checkpoint_epoch: compareEpoch }, ctrl.signal);
      if (seq === compareSeq) set({ compareTrace: trace, compareLoading: false, compareError: null });
    } catch (err) {
      if (api.isCancel(err)) return;
      if (seq === compareSeq) set({ compareLoading: false, compareError: api.errorMessage(err) });
    }
  };

  // The comparison must describe the same experiment: refetch when the probe
  // or the what-if list changes; drop it when the model is replaced/retrained.
  useForgeStore.subscribe((s, prev) => {
    if (s.session !== prev.session) {
      const epochs = new Set(s.session?.checkpoints.map((c) => c.epoch) ?? []);
      const keep = s.session && prev.session && s.session.session_id === prev.session.session_id && get().compareEpoch !== null && epochs.has(get().compareEpoch!);
      if (!keep) set({ compareEpoch: null });
      void loadCompare();
      return;
    }
    if (get().compareEpoch === null) return;
    if (probeKey(s.probe) !== probeKey(prev.probe) || s.interventions !== prev.interventions) void loadCompare();
  });

  return {
    dir: 'forward',
    stepId: { forward: null, backward: null },
    playing: false,
    compareEpoch: null,
    compareTrace: null,
    compareLoading: false,
    compareError: null,

    setDir: (dir) => { clear(); set({ dir, playing: false }); },
    goTo: (id) => { clear(); set({ stepId: { ...get().stepId, [get().dir]: id }, playing: false }); },
    step: (delta) => { clear(); move(delta); },
    first: () => {
      clear();
      const steps = currentSteps(get().dir);
      if (steps.length) set({ stepId: { ...get().stepId, [get().dir]: steps[0].id }, playing: false });
    },
    last: () => {
      clear();
      const steps = currentSteps(get().dir);
      if (steps.length) set({ stepId: { ...get().stepId, [get().dir]: steps[steps.length - 1].id }, playing: false });
    },
    play: () => {
      if (get().playing) return;
      const { dir, stepId } = get();
      const steps = currentSteps(dir);
      if (steps.length < 2) return;
      // At the end: restart from the first step.
      if (stepIndexOf(steps, stepId[dir]) >= steps.length - 1) set({ stepId: { ...stepId, [dir]: steps[0].id } });
      set({ playing: true });
      clear();
      timer = setTimeout(tick, PLAY_INTERVAL_MS);
    },
    pause: () => { clear(); set({ playing: false }); },
    togglePlay: () => (get().playing ? get().pause() : get().play()),
    setCompareEpoch: (compareEpoch) => {
      set({ compareEpoch });
      return loadCompare();
    },
  };
});
