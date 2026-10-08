// Workspace (UI) state: which instrument fills the centre, which side panels
// are open.  Experiment state lives in forge/store.ts; this store never holds
// model data.

import { create } from 'zustand';

export type WorkspaceMode = 'network' | 'timemachine' | 'explorer' | '3d' | 'transformer' | 'analysis';

/** Workspace order in the top bar; labels and hints come from the i18n dictionary (modes.*). */
export const MODES: WorkspaceMode[] = ['network', 'timemachine', 'explorer', '3d', 'transformer', 'analysis'];

/**
 * Explore: the simplified beginner path (Build → Train → Understand) on the
 * same real model.  Lab: every instrument and panel.
 */
export type Experience = 'explore' | 'lab';
export const EXPERIENCE_KEY = 'nf-experience';

function readExperience(): Experience {
  try {
    const v = localStorage.getItem(EXPERIENCE_KEY);
    if (v === 'explore' || v === 'lab') return v;
  } catch { /* storage unavailable */ }
  return 'explore';
}

interface WorkspaceState {
  experience: Experience;
  setExperience: (e: Experience) => void;
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
  experience: readExperience(),
  setExperience: (experience) => {
    try { localStorage.setItem(EXPERIENCE_KEY, experience); } catch { /* storage unavailable */ }
    set({ experience });
  },
  mode: 'network',
  leftOpen: true,
  rightOpen: true,
  welcomeOpen: false,
  setWelcomeOpen: (welcomeOpen) => set({ welcomeOpen }),
  // Choosing a Lab instrument (from a demo, the tour or a link) leaves Explore.
  setMode: (mode) => {
    try { localStorage.setItem(EXPERIENCE_KEY, 'lab'); } catch { /* storage unavailable */ }
    set({ mode, experience: 'lab' });
  },
  setLeftOpen: (leftOpen) => set({ leftOpen }),
  setRightOpen: (rightOpen) => set({ rightOpen }),
}));
