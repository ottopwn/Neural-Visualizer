import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ComponentRef, Frame, Probe, SessionSummary } from '../types';

vi.mock('../api', () => ({
  fetchGraph: vi.fn(),
  compare: vi.fn(),
  fetchTrace: vi.fn(async (_id: string, req: { interventions: unknown[]; checkpoint_epoch: number | null }) => ({ tag: req.interventions.length, epoch: req.checkpoint_epoch })),
  inspect: vi.fn(),
  fetchTimeline: vi.fn(),
  fetchFrame: vi.fn(),
  fetchComponentHistory: vi.fn(),
  fetchEpochComparison: vi.fn(),
  errorMessage: (e: unknown) => String(e),
  isCancel: (e: unknown) => (e as Error)?.name === 'CanceledError',
}));

const api = await import('../api');
const { useForgeStore } = await import('../store');
const { useTimeMachine, SYNC_DELAY_MS } = await import('../timeMachine');
const { playbackInterval } = await import('../timeline');

const EPOCHS = [0, 1, 2, 4, 8];
const row = (epoch: number) => ({ epoch, loss: 1 / (epoch + 1), accuracy: 0.5 + epoch / 20 });
const session = (id = 's1', epochs = EPOCHS) => ({
  session_id: id, structure: {}, epoch: epochs[epochs.length - 1],
  history: [], checkpoints: epochs.map(row),
}) as unknown as SessionSummary;

const fakeFrame = (epoch: number, probe?: Probe) => ({ epoch, probe_tag: probe?.sample_index } as unknown as Frame);

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

const tm = () => useTimeMachine.getState();
const flush = () => vi.advanceTimersByTimeAsync(0);

beforeEach(async () => {
  vi.useFakeTimers();
  vi.mocked(api.fetchGraph).mockReset().mockImplementation(async (_id, req) => ({ epoch: req.checkpoint_epoch } as never));
  vi.mocked(api.compare).mockReset().mockImplementation(async () => ({} as never));
  vi.mocked(api.inspect).mockReset().mockImplementation(async (_id, ref: ComponentRef, req) =>
    ({ kind: ref.kind, epoch: req.checkpoint_epoch } as never));
  vi.mocked(api.fetchFrame).mockReset().mockImplementation(async (_id, epoch, probe) => fakeFrame(epoch!, probe));
  vi.mocked(api.fetchTimeline).mockReset().mockImplementation(async () => ({ live_epoch: 8 } as never));
  vi.mocked(api.fetchComponentHistory).mockReset().mockImplementation(async (_id, ref) => ({ ref } as never));
  vi.mocked(api.fetchEpochComparison).mockReset().mockImplementation(async (_id, body) => ({ body } as never));
  useTimeMachine.setState({ speed: 1, compareMode: false, compareA: null, compareB: null });
  useForgeStore.setState({ session: null, interventions: [], selection: null, checkpointEpoch: null });
  await useForgeStore.getState().setSession(session());
  await flush();
});

afterEach(() => {
  tm().pause();
  vi.useRealTimers();
});

describe('timeline selection', () => {
  it('goTo moves the playhead, loads the real frame and syncs the microscope after a debounce', async () => {
    await tm().goTo(2);
    expect(tm().cursor).toBe(2);
    expect(tm().frame).toEqual(fakeFrame(2, { sample_index: 0 }));
    expect(useForgeStore.getState().checkpointEpoch).toBeNull(); // not yet: debounced
    await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS);
    expect(useForgeStore.getState().checkpointEpoch).toBe(2);
    expect(vi.mocked(api.fetchGraph).mock.calls.at(-1)![1].checkpoint_epoch).toBe(2);
  });

  it('scrubbing quickly only syncs the last position', async () => {
    vi.mocked(api.fetchGraph).mockClear();
    for (const e of [0, 1, 2, 4]) await tm().goTo(e);
    await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS);
    expect(vi.mocked(api.fetchGraph)).toHaveBeenCalledTimes(1);
    expect(useForgeStore.getState().checkpointEpoch).toBe(4);
  });

  it('selecting the latest checkpoint means live (null)', async () => {
    await tm().goTo(8);
    expect(tm().cursor).toBeNull();
    await tm().goTo(1);
    await tm().goLive();
    await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS);
    expect(tm().cursor).toBeNull();
    expect(useForgeStore.getState().checkpointEpoch).toBeNull();
    expect(tm().frame!.epoch).toBe(8);
  });

  it('step / first / last walk the stored checkpoints and clamp at the ends', async () => {
    await tm().first();
    expect(tm().cursor).toBe(0);
    await tm().step(-1);
    expect(tm().cursor).toBe(0);
    await tm().step(1);
    await tm().step(1);
    await tm().step(1);
    expect(tm().cursor).toBe(4);
    await tm().step(-1);
    expect(tm().cursor).toBe(2);
    await tm().last();
    expect(tm().cursor).toBeNull();
    await tm().step(1);
    expect(tm().cursor).toBeNull();
  });

  it('frames are cached: revisiting an epoch makes no request', async () => {
    await tm().goTo(1);
    await tm().goTo(2);
    vi.mocked(api.fetchFrame).mockClear();
    await tm().goTo(1);
    expect(api.fetchFrame).not.toHaveBeenCalled();
    expect(tm().frame!.epoch).toBe(1);
  });

  it('ignores frames that arrive out of order', async () => {
    const slow = deferred<Frame>();
    vi.mocked(api.fetchFrame).mockImplementationOnce(() => slow.promise);
    const first = tm().goTo(1);
    await tm().goTo(2);
    expect(tm().frame!.epoch).toBe(2);
    slow.resolve(fakeFrame(1));
    await first;
    expect(tm().frame!.epoch).toBe(2);
    expect(tm().cursor).toBe(2);
  });

  it('frames never carry what-if interventions', async () => {
    await useForgeStore.getState().addIntervention({ type: 'ablate_neuron', layer: 1, index: 0 });
    await tm().goTo(4);
    const call = vi.mocked(api.fetchFrame).mock.calls.at(-1)!;
    expect(call).toHaveLength(3);
    expect(JSON.stringify(call)).not.toContain('ablate');
  });
});

describe('playback', () => {
  it('plays through every checkpoint, ends on live and stops', async () => {
    await tm().first();
    tm().play();
    expect(tm().playing).toBe(true);
    const seen: (number | null)[] = [];
    for (let i = 0; i < 4; i++) {
      await vi.advanceTimersByTimeAsync(playbackInterval(1));
      seen.push(tm().cursor);
    }
    expect(seen).toEqual([1, 2, 4, null]);
    await vi.advanceTimersByTimeAsync(playbackInterval(1));
    expect(tm().playing).toBe(false);
  });

  it('play at the end restarts from the first checkpoint', async () => {
    tm().play();
    await flush();
    expect(tm().cursor).toBe(0);
    expect(tm().playing).toBe(true);
  });

  it('pause stops the playhead', async () => {
    await tm().first();
    tm().play();
    await vi.advanceTimersByTimeAsync(playbackInterval(1));
    tm().pause();
    await vi.advanceTimersByTimeAsync(playbackInterval(1) * 5);
    expect(tm().cursor).toBe(1);
    expect(tm().playing).toBe(false);
    tm().togglePlay();
    expect(tm().playing).toBe(true);
    tm().togglePlay();
    expect(tm().playing).toBe(false);
  });

  it('speed changes the interval between checkpoints', async () => {
    await tm().first();
    tm().setSpeed(4);
    tm().play();
    await vi.advanceTimersByTimeAsync(playbackInterval(4));
    expect(tm().cursor).toBe(1);
    tm().setSpeed(0.5);
    await vi.advanceTimersByTimeAsync(playbackInterval(4));
    expect(tm().cursor).toBe(1); // the slower interval was applied
    await vi.advanceTimersByTimeAsync(playbackInterval(0.5));
    expect(tm().cursor).toBe(2);
  });

  it('waits for a slow frame instead of skipping checkpoints', async () => {
    await tm().first();
    const slow = deferred<Frame>();
    vi.mocked(api.fetchFrame).mockImplementation(async (_id, epoch) => (epoch === 1 ? slow.promise : fakeFrame(epoch!)));
    tm().play();
    await vi.advanceTimersByTimeAsync(playbackInterval(1) * 4);
    expect(tm().cursor).toBe(1);
    expect(tm().frameLoading).toBe(true);
    slow.resolve(fakeFrame(1));
    await flush();
    expect(tm().frame!.epoch).toBe(1);
    await vi.advanceTimersByTimeAsync(playbackInterval(1));
    expect(tm().cursor).toBe(2);
  });

  it('manual navigation pauses playback', async () => {
    await tm().first();
    tm().play();
    await tm().goTo(4);
    expect(tm().playing).toBe(false);
  });
});

describe('microscope integration', () => {
  it('keeps the same selected neuron while the epoch changes', async () => {
    await useForgeStore.getState().select({ kind: 'neuron', layer: 2, index: 3 });
    for (const e of [0, 2, 4]) {
      await tm().goTo(e);
      await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS);
      const st = useForgeStore.getState();
      expect(st.selection).toEqual({ kind: 'neuron', layer: 2, index: 3 });
      expect(st.inspection).toEqual({ kind: 'neuron', epoch: e });
    }
  });

  it('what-if overlays the viewed checkpoint; undo/reset never move the playhead', async () => {
    await tm().goTo(2);
    await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS);
    await useForgeStore.getState().addIntervention({ type: 'set_bias', layer: 1, index: 0, value: 3 });
    let req = vi.mocked(api.fetchGraph).mock.calls.at(-1)![1];
    expect(req.checkpoint_epoch).toBe(2);
    expect(req.interventions).toHaveLength(1);
    await useForgeStore.getState().undo();
    req = vi.mocked(api.fetchGraph).mock.calls.at(-1)![1];
    expect(req).toMatchObject({ checkpoint_epoch: 2, interventions: [] });
    expect(tm().cursor).toBe(2);
  });

  it('loads the selected component history once per selection/probe', async () => {
    await useForgeStore.getState().select({ kind: 'neuron', layer: 1, index: 0 });
    await tm().loadHistory();
    await tm().loadHistory();
    expect(api.fetchComponentHistory).toHaveBeenCalledTimes(1);
    expect(tm().history).toEqual({ ref: { kind: 'neuron', layer: 1, index: 0 } });
    await useForgeStore.getState().select(null);
    await tm().loadHistory();
    expect(tm().history).toBeNull();
  });

  it('retraining resets the playhead to live and cancels a pending sync', async () => {
    await tm().goTo(1);
    await useForgeStore.getState().updateSession(session('s1', [0, 1, 2, 4, 8, 12]));
    await vi.advanceTimersByTimeAsync(SYNC_DELAY_MS * 2);
    expect(tm().cursor).toBeNull();
    expect(tm().frame).toBeNull();
    expect(useForgeStore.getState().checkpointEpoch).toBeNull();
  });
});

describe('epoch comparison', () => {
  it('defaults to first checkpoint vs live and sends the selected component', async () => {
    await useForgeStore.getState().select({ kind: 'layer', layer: 1 });
    await tm().setCompareMode(true);
    expect(tm()).toMatchObject({ compareMode: true, compareA: 0, compareB: null });
    expect(vi.mocked(api.fetchEpochComparison).mock.calls.at(-1)![1]).toEqual({
      epoch_a: 0, epoch_b: null, probe: { sample_index: 0 }, ref: { kind: 'layer', layer: 1 },
    });
  });

  it('uses the playhead as B when it is historical, and swaps', async () => {
    await tm().goTo(4);
    await tm().setCompareMode(true);
    expect(tm()).toMatchObject({ compareA: 0, compareB: 4 });
    await tm().swapCompare();
    expect(tm()).toMatchObject({ compareA: 4, compareB: 0 });
    await tm().setCompareEpochs(1, 8);
    expect(tm().compareB).toBeNull(); // 8 is the live epoch
  });

  it('ignores a stale comparison and cancels the superseded request', async () => {
    const slow = deferred<never>();
    let aborted = false;
    vi.mocked(api.fetchEpochComparison).mockImplementationOnce((_id, _b, signal) => {
      signal!.addEventListener('abort', () => { aborted = true; });
      return slow.promise;
    });
    const first = tm().setCompareMode(true);
    await tm().setCompareEpochs(1, 2);
    expect(aborted).toBe(true);
    slow.resolve({ body: 'stale' } as never);
    await first;
    expect(tm().comparison).toEqual({ body: { epoch_a: 1, epoch_b: 2, probe: { sample_index: 0 }, ref: null } });
  });

  it('snaps compare epochs to surviving checkpoints after retraining', async () => {
    await tm().setCompareMode(true);
    await tm().setCompareEpochs(1, 4);
    await useForgeStore.getState().updateSession(session('s1', [0, 2, 4, 8, 16]));
    expect(tm()).toMatchObject({ compareMode: true, compareA: 2, compareB: 4 });
    await useForgeStore.getState().setSession(session('s2'));
    expect(tm()).toMatchObject({ compareMode: false, compareA: null });
  });
});
