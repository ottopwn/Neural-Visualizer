// Transformer Lab state.  The model lives on the backend (trained once per
// backend process); this store only holds the input text, the head-ablation
// list and the selection, and fetches traces (sequence-checked).

import { create } from 'zustand';
import * as api from './api';
import type { TransformerInfo, TransformerTrace } from './transformerTypes';

export type LabStage = 'tokens' | 'embeddings' | 'attention' | 'block' | 'next';

interface TransformerLabState {
  info: TransformerInfo | null;
  infoLoading: boolean;
  text: string;
  trace: TransformerTrace | null;
  loading: boolean;
  error: string | null;
  ablate: [number, number][];
  stage: LabStage;
  layer: number;
  head: number;
  /** Selected attention cell: query (destination) row and key (source) column. */
  query: number | null;
  key: number | null;

  loadInfo: () => Promise<void>;
  setText: (t: string) => void;
  run: (text?: string) => Promise<void>;
  toggleAblate: (layer: number, head: number) => Promise<void>;
  clearAblate: () => Promise<void>;
  setStage: (s: LabStage) => void;
  setHead: (layer: number, head: number) => void;
  selectCell: (query: number | null, key: number | null) => void;
}

let seq = 0;
let abort: AbortController | null = null;

export const useTransformerLab = create<TransformerLabState>((set, get) => ({
  info: null,
  infoLoading: false,
  text: 'the cat sat on the mat . then it',
  trace: null,
  loading: false,
  error: null,
  ablate: [],
  stage: 'attention',
  layer: 0,
  head: 0,
  query: null,
  key: null,

  loadInfo: async () => {
    if (get().info || get().infoLoading) return;
    set({ infoLoading: true, error: null });
    try {
      const info = await api.fetchTransformerInfo();
      set({ info, infoLoading: false });
    } catch (err) {
      set({ infoLoading: false, error: api.errorMessage(err) });
    }
  },
  setText: (text) => set({ text }),
  run: async (text) => {
    const t = (text ?? get().text).trim();
    if (text !== undefined) set({ text });
    if (!t) return;
    const s = ++seq;
    abort?.abort();
    const ctrl = new AbortController();
    abort = ctrl;
    set({ loading: true, error: null });
    try {
      const trace = await api.fetchTransformerTrace({ text: t, ablate_heads: get().ablate, top_k: 10 }, ctrl.signal);
      if (s !== seq) return;
      const n = trace.tokens.length;
      const q = get().query;
      // Default focus: the last position (the one that predicts the next token).
      set({ trace, loading: false, query: q !== null && q < n ? q : n - 1, key: get().key !== null && get().key! < n ? get().key : null });
    } catch (err) {
      if (api.isCancel(err)) return;
      if (s === seq) set({ loading: false, error: api.errorMessage(err) });
    }
  },
  toggleAblate: (layer, head) => {
    const cur = get().ablate;
    const on = cur.some(([l, h]) => l === layer && h === head);
    set({ ablate: on ? cur.filter(([l, h]) => !(l === layer && h === head)) : [...cur, [layer, head]] });
    return get().run();
  },
  clearAblate: () => { set({ ablate: [] }); return get().run(); },
  setStage: (stage) => set({ stage }),
  setHead: (layer, head) => set({ layer, head }),
  selectCell: (query, key) => set({ query, key }),
}));
