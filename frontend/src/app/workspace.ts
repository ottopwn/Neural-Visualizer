// Workspace (UI) state: which instrument fills the centre, which side panels
// are open.  Experiment state lives in forge/store.ts; this store never holds
// model data.

import { create } from 'zustand';

export type WorkspaceMode = 'network' | 'timemachine' | 'explorer' | '3d' | 'transformer' | 'analysis';

export const MODES: { id: WorkspaceMode; label: string; short: string; hint: string }[] = [
  { id: 'network', label: 'Network', short: 'Network', hint: 'The live network: real signal on the probe input. Click any neuron, layer or connection.' },
  { id: 'timemachine', label: 'Time Machine', short: 'Time', hint: 'Travel through stored training checkpoints and compare epochs.' },
  { id: 'explorer', label: 'Forward / Backward', short: 'Pass', hint: 'Step through one forward and backward pass with every real number.' },
  { id: '3d', label: '3D', short: '3D', hint: 'The real network in 3D: activations, weights, gradients and signal flow.' },
  { id: 'transformer', label: 'Transformer Lab', short: 'Transformer', hint: 'A small, real Transformer trained locally: tokens, attention, next-token probabilities.' },
  { id: 'analysis', label: 'Analysis', short: 'More', hint: 'Secondary analyses, in-browser tools, illustrative diagrams and code export.' },
];

interface WorkspaceState {
  mode: WorkspaceMode;
  leftOpen: boolean;
  rightOpen: boolean;
  /** Welcome / guided-demo launcher. */
  welcomeOpen: boolean;
  setWelcomeOpen: (v: boolean) => void;
  setMode: (m: WorkspaceMode) => void;
  setLeftOpen: (v: boolean) => void;
  setRightOpen: (v: boolean) => void;
}

export const useWorkspace = create<WorkspaceState>((set) => ({
  mode: 'network',
  leftOpen: true,
  rightOpen: true,
  welcomeOpen: false,
  setWelcomeOpen: (welcomeOpen) => set({ welcomeOpen }),
  setMode: (mode) => set({ mode }),
  setLeftOpen: (leftOpen) => set({ leftOpen }),
  setRightOpen: (rightOpen) => set({ rightOpen }),
}));
