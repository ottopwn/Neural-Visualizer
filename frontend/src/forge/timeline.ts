// Pure helpers for the Training Time Machine: navigation between stored
// checkpoints, the timeline's time axis, playback timing and a small cache.
// Nothing here talks to the backend or invents values.

/** Playback speeds offered by the transport, in checkpoints per BASE interval. */
export const PLAYBACK_SPEEDS = [0.5, 1, 2, 4] as const;
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];
const BASE_FRAME_MS = 700;

/** How long each checkpoint stays on screen during playback. */
export function playbackInterval(speed: number): number {
  return Math.round(BASE_FRAME_MS / (speed > 0 ? speed : 1));
}

/** Sorted, de-duplicated list of stored checkpoint epochs. */
export function sortedEpochs(epochs: number[]): number[] {
  return [...new Set(epochs)].sort((a, b) => a - b);
}

/** The epoch a cursor refers to (`null` = live = latest stored checkpoint). */
export function resolveEpoch(cursor: number | null, liveEpoch: number): number {
  return cursor ?? liveEpoch;
}

/** Stored epoch closest to `epoch` (ties: the later one); null when there are none. */
export function nearestEpoch(epochs: number[], epoch: number): number | null {
  let best: number | null = null;
  for (const e of epochs) {
    if (best === null) { best = e; continue; }
    const d = Math.abs(e - epoch);
    const bd = Math.abs(best - epoch);
    if (d < bd || (d === bd && e > best)) best = e;
  }
  return best;
}

/** Next (dir = 1) or previous (dir = -1) stored epoch, or null at either end. */
export function stepEpoch(epochs: number[], current: number, dir: 1 | -1): number | null {
  if (dir > 0) {
    for (const e of epochs) if (e > current) return e;
    return null;
  }
  for (let i = epochs.length - 1; i >= 0; i--) if (epochs[i] < current) return epochs[i];
  return null;
}

/** Index of `epoch` among the stored epochs (for "checkpoint 4 of 21"), or -1. */
export function checkpointIndex(epochs: number[], epoch: number): number {
  return epochs.indexOf(epoch);
}

// ── time axis ──────────────────────────────────────────────────────────────
// 'linear'      x ∝ epoch                  (honest durations)
// 'checkpoints' stored checkpoints evenly spaced, piecewise-linear in between
//               (keeps densely stored early epochs clickable on long runs)
export type AxisMode = 'linear' | 'checkpoints';

export interface TimeAxis {
  toX: (epoch: number) => number;
  fromX: (x: number) => number;
}

export function makeAxis(epochs: number[], mode: AxisMode, x0: number, x1: number): TimeAxis {
  const n = epochs.length;
  const lo = n ? epochs[0] : 0;
  const hi = n ? epochs[n - 1] : 1;
  const w = x1 - x0;
  if (mode === 'linear' || n < 2) {
    const span = hi - lo || 1;
    return {
      toX: (e) => x0 + ((e - lo) / span) * w,
      fromX: (x) => lo + ((x - x0) / (w || 1)) * span,
    };
  }
  const step = w / (n - 1);
  return {
    toX: (e) => {
      if (e <= lo) return x0;
      if (e >= hi) return x1;
      let i = 0;
      while (epochs[i + 1] < e) i++;
      const t = (e - epochs[i]) / (epochs[i + 1] - epochs[i]);
      return x0 + (i + t) * step;
    },
    fromX: (x) => {
      const f = Math.max(0, Math.min(n - 1, (x - x0) / (step || 1)));
      const i = Math.min(n - 2, Math.floor(f));
      return epochs[i] + (f - i) * (epochs[i + 1] - epochs[i]);
    },
  };
}

// ── bounded cache ──────────────────────────────────────────────────────────
/** Tiny LRU map: frames and histories are immutable per key, so caching is safe. */
export class LruCache<V> {
  private map = new Map<string, V>();
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }

  set(key: string, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.capacity) this.map.delete(this.map.keys().next().value as string);
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}

/** Stable key for a probe (part of every probe-dependent cache key). */
export function probeKey(probe: { x?: number[] | null; sample_index?: number | null; target?: number | null }): string {
  return `${probe.sample_index ?? ''}|${probe.x ? probe.x.join(',') : ''}|${probe.target ?? ''}`;
}
