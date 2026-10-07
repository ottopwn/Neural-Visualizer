// Training Time Machine state.
//
// The playhead (`cursor`) is the epoch of a *stored* checkpoint, or null for
// the live / latest weights.  It drives two things:
//
// * a lightweight `Frame` (metrics, predictions, decision boundary) fetched
//   per checkpoint and cached -- what the Time Machine draws;
// * the existing Forge experiment state (`useForgeStore.checkpointEpoch`),
//   synced after a short debounce, so the Microscope, What-if panel and the
//   classic graph tabs all follow the playhead with the *same* selection.
//
// Frames, component histories and A/B comparisons describe immutable
// checkpoints, so they are cached by (session, live epoch, epoch(s), probe,
// component).  Every request is tagged with a sequence number; histories and
// comparisons are also cancelled when superseded.  Stale responses can never
// overwrite newer ones.

import { create } from 'zustand';
import * as api from './api';
import { refKey } from './interventions';
import { useForgeStore } from './store';
import {
  LruCache, type AxisMode, type PlaybackSpeed, playbackInterval, probeKey, resolveEpoch, sortedEpochs, stepEpoch, nearestEpoch,
} from './timeline';
import type { ComponentHistory, EpochComparison, Frame, Timeline } from './types';

/** Delay before the playhead's epoch is pushed to the Microscope (debounces scrubbing). */
export const SYNC_DELAY_MS = 120;

export interface TimeMachineState {
  cursor: number | null; // stored epoch on the playhead; null = live
  playing: boolean;
  speed: PlaybackSpeed;
  axisMode: AxisMode;

  timeline: Timeline | null;
  timelineLoading: boolean;
  frame: Frame | null;
  frameLoading: boolean;
  history: ComponentHistory | null;
  historyLoading: boolean;

  compareMode: boolean;
  compareA: number | null; // stored epoch
  compareB: number | null; // stored epoch; null = live
  comparison: EpochComparison | null;
  comparing: boolean;

  error: string | null;

  goTo: (epoch: number | null) => Promise<void>;
  step: (dir: 1 | -1) => Promise<void>;
  first: () => Promise<void>;
  last: () => Promise<void>;
  goLive: () => Promise<void>;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setSpeed: (s: PlaybackSpeed) => void;
  setAxisMode: (m: AxisMode) => void;

  loadTimeline: () => Promise<void>;
  loadFrame: () => Promise<void>;
  loadHistory: () => Promise<void>;

  setCompareMode: (on: boolean) => Promise<void>;
  setCompareEpochs: (a: number, b: number | null) => Promise<void>;
  swapCompare: () => Promise<void>;
  loadComparison: () => Promise<void>;
}

const frameCache = new LruCache<Frame>(96);
const historyCache = new LruCache<ComponentHistory>(32);
const comparisonCache = new LruCache<EpochComparison>(16);
const inflightFrames = new Map<string, Promise<Frame>>();

let frameSeq = 0;
let timelineSeq = 0;
let historySeq = 0;
let compareSeq = 0;
let historyAbort: AbortController | null = null;
let compareAbort: AbortController | null = null;
let playTimer: ReturnType<typeof setTimeout> | null = null;
let syncTimer: ReturnType<typeof setTimeout> | null = null;

const forge = () => useForgeStore.getState();
const epochsOf = () => sortedEpochs(forge().session?.checkpoints.map((c) => c.epoch) ?? []);
const liveEpoch = () => forge().session?.epoch ?? 0;
const sessionKey = () => {
  const s = forge().session;
  return s ? `${s.session_id}@${s.epoch}` : '';
};

function frameKey(epoch: number): string {
  return `${sessionKey()}:${epoch}:${probeKey(forge().probe)}`;
}

/** Cached / in-flight / new frame request (frames are shared with prefetching). */
function getFrame(epoch: number): Promise<Frame> {
  const key = frameKey(epoch);
  const cached = frameCache.get(key);
  if (cached) return Promise.resolve(cached);
  const pending = inflightFrames.get(key);
  if (pending) return pending;
  const { session, probe } = forge();
  const p = api.fetchFrame(session!.session_id, epoch, probe)
    .then((f) => {
      frameCache.set(key, f);
      return f;
    })
    .finally(() => inflightFrames.delete(key));
  inflightFrames.set(key, p);
  return p;
}

function errorOf(err: unknown): string {
  return api.errorMessage(err);
}

export const useTimeMachine = create<TimeMachineState>((set, get) => {
  const clearPlayTimer = () => {
    if (playTimer) clearTimeout(playTimer);
    playTimer = null;
  };

  const scheduleAdvance = () => {
    clearPlayTimer();
    playTimer = setTimeout(advance, playbackInterval(get().speed));
  };

  // One playback step: called when the current frame has been on screen for
  // one interval.  The next step is scheduled only once its frame arrived, so
  // slow responses slow playback down instead of skipping checkpoints.
  const advance = () => {
    playTimer = null;
    if (!get().playing) return;
    const next = stepEpoch(epochsOf(), resolveEpoch(get().cursor, liveEpoch()), 1);
    if (next === null) {
      set({ playing: false });
      return;
    }
    void moveTo(next);
  };

  const scheduleSync = () => {
    if (syncTimer) clearTimeout(syncTimer);
    syncTimer = setTimeout(() => {
      syncTimer = null;
      const target = get().cursor;
      if (forge().session && forge().checkpointEpoch !== target) void forge().setCheckpoint(target);
    }, SYNC_DELAY_MS);
  };

  const prefetch = (epoch: number | null) => {
    if (epoch === null || !forge().session) return;
    getFrame(epoch).catch(() => undefined);
  };

  /** Move the playhead without touching the playback state. */
  const moveTo = (epoch: number | null) => {
    const cursor = epoch === null || epoch === liveEpoch() ? null : epoch;
    set({ cursor });
    scheduleSync();
    return loadFrame();
  };

  const loadFrame = async () => {
    const { session } = forge();
    const seq = ++frameSeq;
    if (!session) {
      set({ frame: null, frameLoading: false });
      return;
    }
    const epoch = resolveEpoch(get().cursor, session.epoch);
    set({ frameLoading: true });
    try {
      const frame = await getFrame(epoch);
      if (seq !== frameSeq) return;
      set({ frame, frameLoading: false, error: null });
      if (get().playing) {
        scheduleAdvance();
        prefetch(stepEpoch(epochsOf(), epoch, 1));
      }
    } catch (err) {
      if (seq === frameSeq) {
        clearPlayTimer();
        set({ frameLoading: false, playing: false, error: errorOf(err) });
      }
    }
  };

  const manual = (epoch: number | null) => {
    get().pause();
    return moveTo(epoch);
  };

  // Reset everything that described the previous model whenever the session
  // is replaced or retrained (new epochs, possibly evicted checkpoints).
  useForgeStore.subscribe((s, prev) => {
    if (s.session === prev.session) return;
    clearPlayTimer();
    if (syncTimer) clearTimeout(syncTimer);
    syncTimer = null;
    frameCache.clear();
    historyCache.clear();
    comparisonCache.clear();
    historyAbort?.abort();
    compareAbort?.abort();
    frameSeq++; timelineSeq++; historySeq++; compareSeq++;
    const sameModel = !!s.session && s.session.session_id === prev.session?.session_id;
    const eps = sortedEpochs(s.session?.checkpoints.map((c) => c.epoch) ?? []);
    const snap = (e: number | null) => (e === null ? null : nearestEpoch(eps, e));
    const { compareMode, compareA, compareB } = get();
    set({
      cursor: null, playing: false, frame: null, frameLoading: false, timeline: null, timelineLoading: false,
      history: null, historyLoading: false, comparison: null, comparing: false, error: null,
      ...(sameModel
        ? { compareA: snap(compareA), compareB: compareB === null || compareB === s.session!.epoch ? null : snap(compareB) }
        : { compareMode: false, compareA: null, compareB: null }),
    });
    if (sameModel && compareMode) void get().loadComparison();
  });

  return {
    cursor: null,
    playing: false,
    speed: 1,
    axisMode: 'linear',
    timeline: null,
    timelineLoading: false,
    frame: null,
    frameLoading: false,
    history: null,
    historyLoading: false,
    compareMode: false,
    compareA: null,
    compareB: null,
    comparison: null,
    comparing: false,
    error: null,

    goTo: (epoch) => manual(epoch),
    step: (dir) => {
      const next = stepEpoch(epochsOf(), resolveEpoch(get().cursor, liveEpoch()), dir);
      return next === null ? Promise.resolve() : manual(next);
    },
    first: () => {
      const eps = epochsOf();
      return eps.length ? manual(eps[0]) : Promise.resolve();
    },
    last: () => manual(null),
    goLive: () => manual(null),

    play: () => {
      const eps = epochsOf();
      if (eps.length < 2 || get().playing) return;
      set({ playing: true });
      const current = resolveEpoch(get().cursor, liveEpoch());
      if (current >= eps[eps.length - 1]) {
        void moveTo(eps[0]); // at the end: replay from the first checkpoint
      } else if (get().frame && !get().frameLoading && get().frame!.epoch === current) {
        scheduleAdvance();
        prefetch(stepEpoch(eps, current, 1));
      } else {
        void loadFrame(); // advances once the current frame is on screen
      }
    },
    pause: () => {
      clearPlayTimer();
      if (get().playing) set({ playing: false });
    },
    togglePlay: () => (get().playing ? get().pause() : get().play()),
    setSpeed: (speed) => {
      set({ speed });
      if (get().playing && playTimer) scheduleAdvance();
    },
    setAxisMode: (axisMode) => set({ axisMode }),

    loadTimeline: async () => {
      const { session } = forge();
      const seq = ++timelineSeq;
      if (!session) {
        set({ timeline: null, timelineLoading: false });
        return;
      }
      set({ timelineLoading: true });
      try {
        const timeline = await api.fetchTimeline(session.session_id);
        if (seq === timelineSeq) set({ timeline, timelineLoading: false });
      } catch (err) {
        if (seq === timelineSeq) set({ timelineLoading: false, error: errorOf(err) });
      }
    },

    loadFrame,

    loadHistory: async () => {
      const { session, selection, probe } = forge();
      const seq = ++historySeq;
      historyAbort?.abort();
      historyAbort = null;
      if (!session || !selection) {
        set({ history: null, historyLoading: false });
        return;
      }
      const key = `${sessionKey()}:${refKey(selection)}:${probeKey(probe)}`;
      const cached = historyCache.get(key);
      if (cached) {
        set({ history: cached, historyLoading: false });
        return;
      }
      const ctrl = new AbortController();
      historyAbort = ctrl;
      set({ historyLoading: true });
      try {
        const history = await api.fetchComponentHistory(session.session_id, selection, probe, ctrl.signal);
        historyCache.set(key, history);
        if (seq === historySeq) set({ history, historyLoading: false });
      } catch (err) {
        if (api.isCancel(err)) return;
        if (seq === historySeq) set({ historyLoading: false, error: errorOf(err) });
      }
    },

    setCompareMode: async (on) => {
      if (!on) {
        compareAbort?.abort();
        compareSeq++;
        set({ compareMode: false, comparing: false });
        return;
      }
      const eps = epochsOf();
      if (!eps.length) return;
      let { compareA, compareB } = get();
      if (compareA === null) {
        const cursor = get().cursor;
        compareA = eps[0];
        compareB = cursor !== null && cursor !== eps[0] ? cursor : null;
      }
      get().pause();
      set({ compareMode: true, compareA, compareB });
      return get().loadComparison();
    },

    setCompareEpochs: (a, b) => {
      set({ compareA: a, compareB: b === null || b === liveEpoch() ? null : b });
      return get().loadComparison();
    },

    swapCompare: () => {
      const { compareA, compareB } = get();
      if (compareA === null) return Promise.resolve();
      return get().setCompareEpochs(compareB ?? liveEpoch(), compareA);
    },

    loadComparison: async () => {
      const { session, selection, probe } = forge();
      const { compareMode, compareA, compareB } = get();
      const seq = ++compareSeq;
      compareAbort?.abort();
      compareAbort = null;
      if (!session || !compareMode || compareA === null) {
        set({ comparison: null, comparing: false });
        return;
      }
      const key = `${sessionKey()}:${compareA}:${compareB}:${refKey(selection)}:${probeKey(probe)}`;
      const cached = comparisonCache.get(key);
      if (cached) {
        set({ comparison: cached, comparing: false });
        return;
      }
      const ctrl = new AbortController();
      compareAbort = ctrl;
      set({ comparing: true });
      try {
        const comparison = await api.fetchEpochComparison(session.session_id, {
          epoch_a: compareA, epoch_b: compareB, probe, ref: selection,
        }, ctrl.signal);
        comparisonCache.set(key, comparison);
        if (seq === compareSeq) set({ comparison, comparing: false, error: null });
      } catch (err) {
        if (api.isCancel(err)) return;
        if (seq === compareSeq) set({ comparing: false, error: errorOf(err) });
      }
    },
  };
});
