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
import { pct } from '../forge/format';
import { useForgeStore } from '../forge/store';
import { useTimeMachine } from '../forge/timeMachine';
import type { ComponentRef, ComputationTrace } from '../forge/types';
import { useNetworkStore } from '../store/networkStore';
import { useExperiment } from './experiment';
import { useWorkspace, type WorkspaceMode } from './workspace';

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
  if (!ok) throw new Error(useExperiment.getState().buildStatus.message ?? 'Build failed');
  await waitIdle();
}

export async function trainDemoModel(): Promise<void> {
  const ok = await useExperiment.getState().train(DEMO_SETUP.epochs);
  if (!ok) throw new Error(useExperiment.getState().trainStatus.message ?? 'Training failed');
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

export interface Demo {
  id: string;
  title: string;
  body: string;
  run: () => Promise<void>;
}

export const DEMOS: Demo[] = [
  {
    id: 'learn', title: 'Watch a network learn',
    body: 'Builds and trains a real 2→8→8→2 network on the Circle data, then replays its training checkpoint by checkpoint in the Time Machine.',
    run: async () => {
      await buildDemoModel();
      await trainDemoModel();
      setMode('timemachine');
      await useTimeMachine.getState().first();
      useTimeMachine.getState().play();
    },
  },
  {
    id: 'neuron', title: 'Inspect a neuron',
    body: 'Selects the most active hidden neuron for a sample and opens the Microscope: its inputs, weights, weighted sum, activation and gradients.',
    run: async () => {
      await ensureTrained();
      await pickUncertainProbe();
      setMode('network');
      ws().setRightOpen(true);
      await forge().select(mostActiveNeuron(forge().trace!));
    },
  },
  {
    id: 'break', title: 'Break the network',
    body: 'Tries disabling every hidden neuron, keeps the one whose loss hurts accuracy most, and shows the real predictions before and after.',
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
    id: 'signal', title: 'Follow a signal',
    body: 'Plays one input through the real forward pass: every weighted sum, activation, logit and probability — then you can run it backwards.',
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
    id: '3d', title: 'See it in 3D',
    body: 'Opens the real network in 3D and follows the backward pass: gradients flowing from the loss to every weight.',
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
    set({ running: demo.title, error: null });
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
  title: string;
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
  if (!sel || sel.kind !== 'neuron' || !t) return 'the selected neuron';
  return sel.layer === 0 ? t.feature_names[sel.index] : t.layers[sel.layer - 1].neuron_names[sel.index];
}

export const JOURNEY: JourneyStep[] = [
  {
    title: 'Initialise a real network',
    enter: async () => { setMode('network'); await buildDemoModel(); },
    text: () => {
      const s = forge().session;
      return s
        ? `A real PyTorch MLP with ${s.structure.param_count} parameters, randomly initialised, looking at ${s.dataset_X.length} points of the ${s.dataset_name} dataset. Edges show the real signal w·a on one input.`
        : 'Building…';
    },
  },
  {
    title: 'Watch it train',
    enter: async () => {
      if ((forge().session?.epoch ?? 0) === 0) await trainDemoModel(); // going back does not retrain
      setMode('timemachine');
      await useTimeMachine.getState().first();
      useTimeMachine.getState().play();
    },
    text: () => {
      const { first, last } = history();
      return first && last
        ? `${last.epoch} epochs of real Adam training. Accuracy went from ${pct(first.accuracy)} at epoch 0 to ${pct(last.accuracy)}; the loss from ${first.loss.toFixed(3)} to ${last.loss.toFixed(3)}. Every epoch was stored as a checkpoint.`
        : 'Training…';
    },
  },
  {
    title: 'Rewind with the Time Machine',
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
      return row
        ? `Back at epoch ${e}: the decision regions, accuracy (${pct(row.accuracy)}) and every weight are the stored checkpoint, not a reconstruction.`
        : 'Travelling…';
    },
  },
  {
    title: 'Select a neuron',
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
      ? `Back to the live model, on the sample it is least sure about. Neural Forge disabled each of the ${lastSearch.tried} hidden neurons in turn: ${lastSearch.name} matters most. The Microscope on the right shows what it computes.`
      : `Back to the live model. The Microscope on the right shows what ${selectedName()} computes.`),
  },
  {
    title: 'Inspect its real values',
    enter: async () => { forge().setMode('lab'); },
    text: () => {
      const sel = forge().selection;
      const t = forge().trace;
      if (!sel || sel.kind !== 'neuron' || !t || sel.layer === 0) return 'Lab mode shows the equations and raw numbers.';
      const d = t.layers[sel.layer - 1];
      return `Lab mode: ${selectedName()} computes z = Σ w·a + b = ${d.z[sel.index].toFixed(4)} and outputs ${d.activation}(z) = ${d.a[sel.index].toFixed(4)}. Its gradient dL/dz on this input is ${d.grad_z[sel.index].toFixed(4)}.`;
    },
  },
  {
    title: 'Disable it',
    enter: async () => {
      const sel = forge().selection;
      if (sel && sel.kind === 'neuron' && sel.layer > 0 && !forge().interventions.length) {
        await forge().addIntervention({ type: 'ablate_neuron', layer: sel.layer, index: sel.index });
      }
    },
    text: () => `What-if: ${selectedName()}'s output is forced to 0 for everything downstream. The stored weights are untouched — this is an overlay on the forward pass.`,
  },
  {
    title: 'See the output change',
    enter: async () => { setMode('network'); await waitIdle(); },
    text: () => {
      const c = forge().comparison;
      if (!c) return 'Comparing…';
      return `Dataset accuracy ${pct(c.baseline.dataset_accuracy)} → ${pct(c.intervened.dataset_accuracy)}; ${pct(c.dataset_flip_fraction)} of the predictions changed${c.prediction_changed ? ', including this probe' : ''}. Undo or Reset restores the original network.`;
    },
  },
  {
    title: 'Follow a forward pass',
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
        ? `The edit is reset. One input (${t.input.map((v) => v.toFixed(2)).join(', ')}) flows through the network step by step: weighted sums, activations, logits and softmax — ending at P(${t.class_names[t.predicted_class]}) = ${pct(t.probabilities[t.predicted_class])}.`
        : 'Loading the pass…';
    },
  },
  {
    title: 'Follow the gradient flow',
    enter: async () => {
      setMode('explorer');
      const ex = useExplorer.getState();
      ex.setDir('backward');
      ex.first();
      ex.play();
    },
    text: () => {
      const t = forge().trace;
      return t
        ? `Backward: the loss (${t.loss.toFixed(4)}) is differentiated through every layer — dL/dlogits = p − y, then Wᵀ·δ and the activation slopes — down to every weight's gradient.`
        : '…';
    },
  },
  {
    title: 'The real network in 3D',
    enter: async () => {
      useExplorer.getState().pause();
      setMode('3d');
    },
    text: () => 'The same model in 3D: spheres are neurons coloured by their real activation, lines are weights carrying real signal. Orbit, zoom, click any neuron — the Microscope follows.',
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
