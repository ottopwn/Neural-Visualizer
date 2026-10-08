// Guided demos and the presentation journey.
//
// Every step calls the same actions a user would trigger by hand -- build,
// train, move the Time Machine playhead, select a neuron, add a what-if
// intervention, step the Pass Explorer -- so a demo always shows the real
// application state.  Nothing here fabricates values: the narration reads
// numbers from the stores after the real actions completed.

import { create } from 'zustand';
import * as api from '../forge/api';
import { useExplorer } from '../forge/explorer';
import { useForgeStore } from '../forge/store';
import { useTimeMachine } from '../forge/timeMachine';
import type { ComponentRef, ComputationTrace } from '../forge/types';
import { useNetworkStore } from '../store/networkStore';
import { useExperiment } from './experiment';
import { useWorkspace, type WorkspaceMode } from './workspace';
import { tr } from '../i18n';

/** The demo model: small enough to read, large enough to be interesting. */
export const DEMO_SETUP = {
  dataset: 'Circle' as const,
  noise: 10,
  neurons: [8, 8],
  activations: ['ReLU', 'ReLU'] as const,
  learning_rate: 0.01,
  epochs: 30,
};

const forge = () => useForgeStore.getState();
const ws = () => useWorkspace.getState();

function waitIdle(timeoutMs = 20000): Promise<void> {
  // The forge store refreshes graph/compare/trace asynchronously after an action.
  return new Promise((resolve) => {
    const start = Date.now();
    const tick = () => {
      const s = forge();
      if ((!s.loading && s.trace) || Date.now() - start > timeoutMs) resolve();
      else setTimeout(tick, 60);
    };
    tick();
  });
}

/** Configure the sidebar exactly as a user would, then Build. */
export async function buildDemoModel(): Promise<void> {
  const ns = useNetworkStore.getState();
  ns.setCustomDataset(null);
  ns.setNetworkConfig({
    model_type: 'ANN', n_layers: DEMO_SETUP.neurons.length,
    neurons: [...DEMO_SETUP.neurons, 8, 8, 8], activations: [...DEMO_SETUP.activations, 'ReLU', 'ReLU', 'ReLU'],
  });
  ns.setTrainingConfig({ dataset: DEMO_SETUP.dataset, noise: DEMO_SETUP.noise, learning_rate: DEMO_SETUP.learning_rate, epochs: DEMO_SETUP.epochs, batch_size: 32, reg_type: 'None' });
  const ok = await useExperiment.getState().build();
  if (!ok) throw new Error(useExperiment.getState().buildStatus.message ?? tr().status.buildFailed);
  await waitIdle();
}

export async function trainDemoModel(): Promise<void> {
  const ok = await useExperiment.getState().train(DEMO_SETUP.epochs);
  if (!ok) throw new Error(useExperiment.getState().trainStatus.message ?? tr().status.trainFailed);
  await waitIdle();
}

/** A trained real model; reuses the current one when it is already trained. */
export async function ensureTrained(): Promise<void> {
  const s = forge().session;
  if (!s || s.epoch === 0 || useNetworkStore.getState().graphSource !== 'model') {
    if (!s || useNetworkStore.getState().graphSource !== 'model') await buildDemoModel();
    await trainDemoModel();
  }
  if (forge().checkpointEpoch !== null) await useTimeMachine.getState().goLive();
  await waitIdle();
}

/**
 * A probe worth looking at: the sample the live model is least sure about
 * (lowest probability for its predicted class, read from the real frame).
 */
export async function pickUncertainProbe(): Promise<void> {
  const s = forge().session;
  if (!s) return;
  const frame = await api.fetchFrame(s.session_id, s.epoch, forge().probe);
  let best = 0;
  frame.confidence.forEach((c, i) => { if (c < frame.confidence[best]) best = i; });
  await forge().setProbe({ sample_index: best });
  await waitIdle();
}

/**
 * The hidden neuron whose removal hurts dataset accuracy most -- found by
 * actually disabling each hidden neuron (one real what-if comparison each).
 */
export async function mostDamagingNeuron(): Promise<{ ref: ComponentRef; before: number; after: number; tried: number } | null> {
  const f = forge();
  const s = f.session;
  if (!s) return null;
  const hidden = s.structure.layers.filter((l) => l.role === 'hidden');
  let best: { ref: ComponentRef; before: number; after: number; tried: number } | null = null;
  let tried = 0;
  for (const l of hidden) {
    for (let i = 0; i < l.size; i++) {
      tried++;
      const c = await api.compare(s.session_id, {
        probe: f.probe, checkpoint_epoch: f.checkpointEpoch,
        interventions: [{ type: 'ablate_neuron', layer: l.layer, index: i }],
      });
      const drop = c.baseline.dataset_accuracy - c.intervened.dataset_accuracy;
      if (!best || drop > best.before - best.after) {
        best = { ref: { kind: 'neuron', layer: l.layer, index: i }, before: c.baseline.dataset_accuracy, after: c.intervened.dataset_accuracy, tried: 0 };
      }
    }
  }
  return best && { ...best, tried };
}

let lastSearch: { name: string; before: number; after: number; tried: number } | null = null;

/** The most active neuron of the first hidden layer on the probe. */
export function mostActiveNeuron(t: ComputationTrace): ComponentRef {
  const a = t.layers[0].a;
  let best = 0;
  a.forEach((v, i) => { if (Math.abs(v) > Math.abs(a[best])) best = i; });
  return { kind: 'neuron', layer: 1, index: best };
}

function setMode(m: WorkspaceMode) {
  ws().setMode(m);
}

// ── one-click demos ─────────────────────────────────────────────────────────

export type DemoId = 'learn' | 'neuron' | 'break' | 'signal' | '3d';

/** A one-click demo; its title and description live in the i18n dictionary (demos[id]). */
export interface Demo {
  id: DemoId;
  run: () => Promise<void>;
}

export const DEMOS: Demo[] = [
  {
    id: 'learn',
    run: async () => {
      await buildDemoModel();
      await trainDemoModel();
      setMode('timemachine');
      await useTimeMachine.getState().first();
      useTimeMachine.getState().play();
    },
  },
  {
    id: 'neuron',
    run: async () => {
      await ensureTrained();
      await pickUncertainProbe();
      setMode('network');
      ws().setRightOpen(true);
      await forge().select(mostActiveNeuron(forge().trace!));
    },
  },
  {
    id: 'break',
    run: async () => {
      await ensureTrained();
      await forge().reset();
      await waitIdle();
      const found = await mostDamagingNeuron();
      setMode('network');
      ws().setRightOpen(true);
      if (found && found.ref.kind === 'neuron') {
        await forge().select(found.ref);
        await forge().addIntervention({ type: 'ablate_neuron', layer: found.ref.layer, index: found.ref.index });
      }
    },
  },
  {
    id: 'signal',
    run: async () => {
      await ensureTrained();
      setMode('explorer');
      const ex = useExplorer.getState();
      ex.setDir('forward');
      ex.first();
      ex.play();
    },
  },
  {
    id: '3d',
    run: async () => {
      await ensureTrained();
      setMode('3d');
    },
  },
];

interface GuideState {
  running: string | null;
  error: string | null;
  run: (demo: Demo) => Promise<void>;
  clearError: () => void;
}

export const useGuide = create<GuideState>((set, get) => ({
  running: null,
  error: null,
  run: async (demo) => {
    if (get().running) return;
    set({ running: demo.id, error: null });
    try {
      await demo.run();
      set({ running: null });
    } catch (err) {
      set({ running: null, error: err instanceof Error ? err.message : String(err) });
    }
  },
  clearError: () => set({ error: null }),
}));

// ── presentation journey ────────────────────────────────────────────────────

export interface JourneyStep {
  title: () => string;
  /** Narration, built from the real state after `enter` completed. */
  text: () => string;
  enter: () => Promise<void>;
}

function history() {
  const h = forge().session?.history ?? [];
  return { first: h[0], last: h[h.length - 1] };
}

function selectedName(): string {
  const sel = forge().selection;
  const t = forge().trace;
  if (!sel || sel.kind !== 'neuron' || !t) return tr().journey.selectedFallback;
  return sel.layer === 0 ? t.feature_names[sel.index] : t.layers[sel.layer - 1].neuron_names[sel.index];
}

export const JOURNEY: JourneyStep[] = [
  {
    title: () => tr().journey.init,
    enter: async () => { setMode('network'); await buildDemoModel(); },
    text: () => {
      const s = forge().session;
      return s ? tr().journey.initText(s.structure.param_count, s.dataset_X.length, s.dataset_name) : tr().journey.building;
    },
  },
  {
    title: () => tr().journey.train,
    enter: async () => {
      if ((forge().session?.epoch ?? 0) === 0) await trainDemoModel(); // going back does not retrain
      setMode('timemachine');
      await useTimeMachine.getState().first();
      useTimeMachine.getState().play();
    },
    text: () => {
      const { first, last } = history();
      return first && last
        ? tr().journey.trainText(last.epoch, first.accuracy, last.accuracy, first.loss, last.loss)
        : tr().journey.training;
    },
  },
  {
    title: () => tr().journey.rewind,
    enter: async () => {
      setMode('timemachine');
      const tm = useTimeMachine.getState();
      tm.pause();
      const eps = forge().session?.checkpoints.map((c) => c.epoch) ?? [];
      await tm.goTo(eps.find((e) => e >= 2) ?? eps[0] ?? null);
    },
    text: () => {
      const e = useTimeMachine.getState().cursor;
      const row = forge().session?.history.find((r) => r.epoch === e);
      return row && e !== null ? tr().journey.rewindText(e, row.accuracy) : tr().journey.travelling;
    },
  },
  {
    title: () => tr().journey.select,
    enter: async () => {
      await useTimeMachine.getState().goLive();
      await forge().reset();
      setMode('network');
      ws().setRightOpen(true);
      await pickUncertainProbe();
      const found = await mostDamagingNeuron();
      const ref = found?.ref ?? mostActiveNeuron(forge().trace!);
      await forge().select(ref);
      lastSearch = found && found.ref.kind === 'neuron' ? { name: selectedName(), before: found.before, after: found.after, tried: found.tried } : null;
    },
    text: () => (lastSearch
      ? tr().journey.selectSearch(lastSearch.tried, lastSearch.name)
      : tr().journey.selectPlain(selectedName())),
  },
  {
    title: () => tr().journey.inspect,
    enter: async () => { forge().setMode('lab'); },
    text: () => {
      const sel = forge().selection;
      const t = forge().trace;
      if (!sel || sel.kind !== 'neuron' || !t || sel.layer === 0) return tr().journey.inspectFallback;
      const d = t.layers[sel.layer - 1];
      return tr().journey.inspectText(selectedName(), d.z[sel.index].toFixed(4), d.activation, d.a[sel.index].toFixed(4), d.grad_z[sel.index].toFixed(4));
    },
  },
  {
    title: () => tr().journey.disable,
    enter: async () => {
      const sel = forge().selection;
      if (sel && sel.kind === 'neuron' && sel.layer > 0 && !forge().interventions.length) {
        await forge().addIntervention({ type: 'ablate_neuron', layer: sel.layer, index: sel.index });
      }
    },
    text: () => tr().journey.disableText(selectedName()),
  },
  {
    title: () => tr().journey.change,
    enter: async () => { setMode('network'); await waitIdle(); },
    text: () => {
      const c = forge().comparison;
      if (!c) return tr().journey.comparing;
      return tr().journey.changeText(c.baseline.dataset_accuracy, c.intervened.dataset_accuracy, c.dataset_flip_fraction, c.prediction_changed);
    },
  },
  {
    title: () => tr().journey.forward,
    enter: async () => {
      await forge().reset();
      await waitIdle();
      setMode('explorer');
      const ex = useExplorer.getState();
      ex.setDir('forward');
      ex.first();
      ex.play();
    },
    text: () => {
      const t = forge().trace;
      return t
        ? tr().journey.forwardText(t.input.map((v) => v.toFixed(2)).join(', '), t.class_names[t.predicted_class], t.probabilities[t.predicted_class])
        : tr().journey.loadingPass;
    },
  },
  {
    title: () => tr().journey.backward,
    enter: async () => {
      setMode('explorer');
      const ex = useExplorer.getState();
      ex.setDir('backward');
      ex.first();
      ex.play();
    },
    text: () => {
      const t = forge().trace;
      return t ? tr().journey.backwardText(t.loss.toFixed(4)) : '…';
    },
  },
  {
    title: () => tr().journey.threeD,
    enter: async () => {
      useExplorer.getState().pause();
      setMode('3d');
    },
    text: () => tr().journey.threeDText,
  },
];

interface PresentationState {
  active: boolean;
  index: number;
  busy: boolean;
  error: string | null;
  /** Panel visibility before presenting, restored on exit. */
  restore: { left: boolean; right: boolean } | null;
  start: () => Promise<void>;
  goTo: (i: number) => Promise<void>;
  next: () => Promise<void>;
  back: () => Promise<void>;
  exit: () => void;
}

export const usePresentation = create<PresentationState>((set, get) => ({
  active: false,
  index: 0,
  busy: false,
  error: null,
  restore: null,
  start: async () => {
    const w = ws();
    set({ active: true, restore: { left: w.leftOpen, right: w.rightOpen } });
    w.setLeftOpen(false);
    w.setRightOpen(true);
    w.setWelcomeOpen(false);
    await get().goTo(0);
  },
  goTo: async (i) => {
    if (get().busy || i < 0 || i >= JOURNEY.length) return;
    set({ index: i, busy: true, error: null });
    try {
      await JOURNEY[i].enter();
      set({ busy: false });
    } catch (err) {
      set({ busy: false, error: err instanceof Error ? err.message : String(err) });
    }
  },
  next: () => get().goTo(get().index + 1),
  back: () => get().goTo(get().index - 1),
  exit: () => {
    const r = get().restore;
    useExplorer.getState().pause();
    useTimeMachine.getState().pause();
    if (r) { ws().setLeftOpen(r.left); ws().setRightOpen(r.right); }
    set({ active: false, restore: null, busy: false, error: null });
  },
}));
