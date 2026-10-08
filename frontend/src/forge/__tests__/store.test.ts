import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ComponentRef, ExperimentRequest, SessionSummary } from '../types';

vi.mock('../api', () => ({
  fetchGraph: vi.fn(),
  compare: vi.fn(),
  fetchTrace: vi.fn(async (_id: string, req: { interventions: unknown[]; checkpoint_epoch: number | null }) => ({ tag: req.interventions.length, epoch: req.checkpoint_epoch })),
  inspect: vi.fn(),
  errorMessage: (e: unknown) => String(e),
}));

const api = await import('../api');
const { useForgeStore, onForgeGraph } = await import('../store');

const session = { session_id: 's1', structure: {}, epoch: 0 } as unknown as SessionSummary;

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.mocked(api.fetchGraph).mockReset().mockImplementation(async (_id, req: ExperimentRequest) =>
    ({ tag: req.interventions.length } as never));
  vi.mocked(api.compare).mockReset().mockImplementation(async (_id, req: ExperimentRequest) =>
    ({ tag: req.interventions.length } as never));
  vi.mocked(api.inspect).mockReset().mockImplementation(async (_id, ref: ComponentRef) => ({ kind: ref.kind } as never));
  useForgeStore.setState({ session: null, interventions: [], selection: null, graph: null, comparison: null, inspection: null });
});

describe('forge store', () => {
  it('sends probe, interventions and checkpoint with every request', async () => {
    await useForgeStore.getState().setSession(session);
    await useForgeStore.getState().setCheckpoint(3);
    await useForgeStore.getState().addIntervention({ type: 'ablate_neuron', layer: 1, index: 0 });
    const last = vi.mocked(api.fetchGraph).mock.calls.at(-1)![1];
    expect(last).toEqual({
      probe: { sample_index: 0 },
      interventions: [{ type: 'ablate_neuron', layer: 1, index: 0 }],
      checkpoint_epoch: 3,
    });
  });

  it('undo and reset change the request log', async () => {
    const s = useForgeStore.getState();
    await s.setSession(session);
    await s.addIntervention({ type: 'set_bias', layer: 1, index: 0, value: 1 });
    await s.addIntervention({ type: 'set_bias', layer: 1, index: 0, value: 2 });
    await useForgeStore.getState().undo();
    expect(useForgeStore.getState().interventions).toEqual([{ type: 'set_bias', layer: 1, index: 0, value: 1 }]);
    await useForgeStore.getState().reset();
    expect(useForgeStore.getState().interventions).toEqual([]);
    expect(useForgeStore.getState().comparison).toEqual({ tag: 0 });
  });

  it('a new session clears interventions and selection', async () => {
    await useForgeStore.getState().setSession(session);
    await useForgeStore.getState().select({ kind: 'layer', layer: 1 });
    await useForgeStore.getState().addIntervention({ type: 'ablate_neuron', layer: 1, index: 0 });
    await useForgeStore.getState().setSession({ ...session, session_id: 's2' });
    const st = useForgeStore.getState();
    expect(st.interventions).toEqual([]);
    expect(st.selection).toBeNull();
    expect(st.inspection).toBeNull();
  });

  it('ignores stale responses that arrive out of order', async () => {
    await useForgeStore.getState().setSession(session);
    const slow = deferred<never>();
    vi.mocked(api.fetchGraph).mockImplementationOnce(() => slow.promise);
    const first = useForgeStore.getState().addIntervention({ type: 'ablate_neuron', layer: 1, index: 0 });
    const second = useForgeStore.getState().reset();
    await second;
    expect(useForgeStore.getState().graph).toEqual({ tag: 0 });
    slow.resolve({ tag: 'stale' } as never);
    await first;
    expect(useForgeStore.getState().graph).toEqual({ tag: 0 });
  });

  it('notifies graph listeners and supports unsubscribe', async () => {
    const seen: unknown[] = [];
    const off = onForgeGraph((g) => seen.push(g));
    await useForgeStore.getState().setSession(session);
    off();
    await useForgeStore.getState().refresh();
    expect(seen).toHaveLength(1);
  });

  it('loads the inspection for the selected component', async () => {
    await useForgeStore.getState().setSession(session);
    await useForgeStore.getState().select({ kind: 'connection', layer: 1, source: 0, target: 0 });
    expect(useForgeStore.getState().inspection).toEqual({ kind: 'connection' });
    await useForgeStore.getState().select(null);
    expect(useForgeStore.getState().inspection).toBeNull();
  });
});

describe('forge store · trace', () => {
  it('fetches the computation trace with the same experiment and drops stale traces', async () => {
    await useForgeStore.getState().setSession(session);
    await useForgeStore.getState().setCheckpoint(3);
    const call = vi.mocked(api.fetchTrace).mock.calls.at(-1)!;
    expect(call[1].checkpoint_epoch).toBe(3);
    expect(call[1].learning_rate).toBe(useForgeStore.getState().previewLr);
    expect(useForgeStore.getState().trace).toMatchObject({ epoch: 3 });

    const slow = deferred<never>();
    vi.mocked(api.fetchTrace).mockImplementationOnce(() => slow.promise);
    const first = useForgeStore.getState().setCheckpoint(1);
    await useForgeStore.getState().setCheckpoint(2);
    slow.resolve({ epoch: 1 } as never);
    await first;
    expect(useForgeStore.getState().trace).toMatchObject({ epoch: 2 });
  });
});
